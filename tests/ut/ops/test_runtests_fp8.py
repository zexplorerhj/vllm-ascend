#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import pytest


OPS_ROOT = Path(__file__).resolve().parents[2] / "ops"
RUNNER = OPS_ROOT / "runtests_fp8.sh"


def _prepare_fake_suite(tmp_path: Path) -> Path:
    suite_dir = tmp_path / "fake ops"
    suite_dir.mkdir()
    runner = suite_dir / RUNNER.name
    shutil.copyfile(RUNNER, runner)

    (suite_dir / "run_tests.sh").write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"${FAKE_RUN_STATUS:-0}\" -ne 0 ]; then\n"
        "    exit \"$FAKE_RUN_STATUS\"\n"
        "fi\n",
        encoding="utf-8",
    )
    plot_spy = (
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "import sys\n"
        "\n"
        "with Path(os.environ['PLOT_LOG']).open(\n"
        "    'a', encoding='utf-8'\n"
        ") as output:\n"
        "    output.write(json.dumps({\n"
        "        'script': Path(__file__).name,\n"
        "        'args': sys.argv[1:],\n"
        "    }) + '\\n')\n"
    )
    (suite_dir / "plot_fp8_suite.py").write_text(
        plot_spy,
        encoding="utf-8",
    )
    (suite_dir / "plot_normquant_captured.py").write_text(
        plot_spy,
        encoding="utf-8",
    )
    return runner


def _run_fake_suite(
    tmp_path: Path,
    *extra_args: str,
    fake_run_status: int = 0,
    device: str = "npu:0",
    precisions: Optional[str] = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    runner = _prepare_fake_suite(tmp_path)
    output_dir = tmp_path / "results with spaces"
    plot_log = tmp_path / "plot-calls.jsonl"
    environment = os.environ.copy()
    environment["FAKE_RUN_STATUS"] = str(fake_run_status)
    environment["PLOT_LOG"] = str(plot_log)

    command = [
        "bash",
        str(runner),
        "--device",
        device,
        "--output-dir",
        str(output_dir),
    ]
    if precisions is not None:
        command.extend(("--precisions", precisions))
    command.extend(extra_args)

    completed = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    return completed, output_dir, plot_log


def test_dry_run_does_not_plot(tmp_path):
    completed, _, plot_log = _run_fake_suite(tmp_path, "--dry-run")

    assert completed.returncode == 0, completed.stderr
    assert "Plot:       skipped for dry-run" in completed.stdout
    assert "[plot/suite-overview] running" not in completed.stdout
    assert "[plot/normquant-captured] running" not in completed.stdout
    assert not plot_log.exists()


def test_failed_benchmark_does_not_plot(tmp_path):
    completed, _, plot_log = _run_fake_suite(
        tmp_path,
        fake_run_status=17,
    )

    assert completed.returncode != 0
    assert "Failed:" in completed.stdout
    assert "[plot/suite-overview] running" not in completed.stdout
    assert "[plot/normquant-captured] running" not in completed.stdout
    assert not plot_log.exists()


def test_default_npu_plots_suite_overview_once_with_output_dir(tmp_path):
    completed, output_dir, plot_log = _run_fake_suite(tmp_path)

    assert completed.returncode == 0, completed.stderr
    plot_calls = [
        json.loads(line)
        for line in plot_log.read_text(encoding="utf-8").splitlines()
    ]
    assert plot_calls == [
        {
            "script": "plot_fp8_suite.py",
            "args": [
                "--input-root",
                str(output_dir),
                "--output",
                str(output_dir / "fp8_suite_overview.png"),
            ],
        }
    ]
    assert completed.stdout.count("[plot/suite-overview] running") == 1
    assert "[plot/normquant-captured] running" not in completed.stdout


@pytest.mark.parametrize(
    ("device", "precisions", "expected_precisions"),
    [
        pytest.param("npu:0", "fp8,mxfp8", ["fp8", "mxfp8"], id="npu-subset"),
        pytest.param("cuda:0", None, ["fp8"], id="cuda-default"),
    ],
)
def test_subset_or_cuda_uses_normquant_plot(
    tmp_path,
    device,
    precisions,
    expected_precisions,
):
    completed, output_dir, plot_log = _run_fake_suite(
        tmp_path,
        device=device,
        precisions=precisions,
    )

    assert completed.returncode == 0, completed.stderr
    plot_calls = [
        json.loads(line)
        for line in plot_log.read_text(encoding="utf-8").splitlines()
    ]
    assert plot_calls == [
        {
            "script": "plot_normquant_captured.py",
            "args": [
                "--input-root",
                str(output_dir),
                "--output",
                str(output_dir / "normquant_captured_comparison.png"),
                "--precisions",
                *expected_precisions,
            ],
        }
    ]
    assert "[plot/suite-overview] running" not in completed.stdout
    assert completed.stdout.count("[plot/normquant-captured] running") == 1


def test_no_plot_disables_plot_after_success(tmp_path):
    completed, _, plot_log = _run_fake_suite(tmp_path, "--no-plot")

    assert completed.returncode == 0, completed.stderr
    assert "Plot:       disabled" in completed.stdout
    assert "[plot/suite-overview] running" not in completed.stdout
    assert "[plot/normquant-captured] running" not in completed.stdout
    assert not plot_log.exists()
