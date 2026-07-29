#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import shlex
import subprocess
from pathlib import Path

import pytest


OPS_ROOT = Path(__file__).resolve().parents[2] / "ops"
DISPATCHER = OPS_ROOT / "run_tests.sh"


def _dry_run(*args: str) -> list[list[str]]:
    completed = subprocess.run(
        ["bash", str(DISPATCHER), "--formal", *args, "--dry-run"],
        check=True,
        cwd=OPS_ROOT,
        capture_output=True,
        text=True,
    )
    commands = []
    for line in completed.stdout.splitlines():
        if line.startswith("DRY-RUN: "):
            commands.append(shlex.split(line.removeprefix("DRY-RUN: ")))
    return commands


def test_formal_all_quick_dispatches_every_precision_without_shapes(tmp_path):
    output_dir = tmp_path / "formal results"
    commands = _dry_run(
        "--operator",
        "all",
        "--device",
        "cuda:0",
        "--output-dir",
        str(output_dir),
        "--quick",
    )

    assert len(commands) == 10
    entry_names = [Path(command[1]).name for command in commands]
    assert entry_names == [
        "test_add.py",
        "test_linear.py",
        "test_linear.py",
        "test_rmsnorm.py",
        "test_flash_attention.py",
        "test_flash_attention.py",
        "test_groupgemm.py",
        "test_groupgemm.py",
        "test_paged_attention.py",
        "test_recurrent_gated_delta_rule.py",
    ]
    assert [
        command[command.index("--precision") + 1]
        for command in commands
        if "--precision" in command
    ] == ["fp16", "bf16", "fp16", "bf16", "bf16", "int8"]

    forbidden_shape_flags = {
        "--sizes",
        "--tflops-sizes",
        "--n-ctx-values",
        "--head-dims",
        "--causal-values",
        "--seq-lens",
        "--seqlen-start",
        "--seqlen-end",
        "--batch-min",
        "--batch-max",
        "--batch-seq-lens",
        "--modes",
        "--batches",
    }
    protocol_flags = {
        "--warmup",
        "--iterations",
        "--repeats",
        "--tflops-warmup",
        "--tflops-iterations",
        "--tflops-repeats",
    }
    for command in commands:
        assert command[0] == "python3"
        assert "--quick" in command
        assert "--shard-index" in command
        assert command[command.index("--shard-index") + 1] == "0"
        assert "--num-shards" in command
        assert command[command.index("--num-shards") + 1] == "1"
        assert not forbidden_shape_flags.intersection(command)
        present_protocol_flags = protocol_flags.intersection(command)
        assert present_protocol_flags in (
            {"--repeats"},
            {"--tflops-repeats"},
        )
        repeat_flag = next(iter(present_protocol_flags))
        assert command[command.index(repeat_flag) + 1] == "5"

        if Path(command[1]).name == "test_recurrent_gated_delta_rule.py":
            assert command[command.index("--output") + 1] == str(
                output_dir / "recurrent.csv"
            )
        else:
            assert command[command.index("--result-dir") + 1] == str(
                output_dir
            )


def test_formal_fp8_opt_in_dispatches_only_h20_linear_and_groupgemm(
    tmp_path,
):
    commands = _dry_run(
        "--operator",
        "all",
        "--precision",
        "fp8",
        "--device",
        "cuda:0",
        "--output-dir",
        str(tmp_path / "h20-fp8"),
    )

    assert len(commands) == 2
    assert [Path(command[1]).name for command in commands] == [
        "test_linear.py",
        "test_groupgemm.py",
    ]
    assert [command[command.index("--precision") + 1] for command in commands] == [
        "fp8",
        "fp8",
    ]


def test_formal_fp8_opt_in_dispatches_950pr_linear_and_groupgemm(
    tmp_path,
):
    commands = _dry_run(
        "--operator",
        "all",
        "--precision",
        "fp8",
        "--device",
        "npu:0",
        "--output-dir",
        str(tmp_path / "950pr-fp8"),
    )

    assert [Path(command[1]).name for command in commands] == [
        "test_linear.py",
        "test_groupgemm.py",
    ]
    assert [
        command[command.index("--precision") + 1] for command in commands
    ] == ["fp8", "fp8"]


def test_formal_mxfp8_opt_in_dispatches_950pr_linear_and_groupgemm(
    tmp_path,
):
    commands = _dry_run(
        "--operator",
        "all",
        "--precision",
        "mxfp8",
        "--device",
        "npu:0",
        "--output-dir",
        str(tmp_path / "950pr-mxfp8"),
    )

    assert [Path(command[1]).name for command in commands] == [
        "test_linear.py",
        "test_groupgemm.py",
    ]
    assert [
        command[command.index("--precision") + 1] for command in commands
    ] == ["mxfp8", "mxfp8"]


def test_formal_mxfp4_opt_in_dispatches_950pr_linear_and_groupgemm(tmp_path):
    commands = _dry_run(
        "--operator",
        "all",
        "--precision",
        "mxfp4",
        "--device",
        "npu:0",
        "--output-dir",
        str(tmp_path / "950pr-mxfp4"),
    )

    assert [Path(command[1]).name for command in commands] == [
        "test_linear.py",
        "test_groupgemm.py",
    ]
    assert [
        command[command.index("--precision") + 1] for command in commands
    ] == ["mxfp4", "mxfp4"]


@pytest.mark.parametrize(
    ("precision", "device"),
    [
        ("fp8", "cuda:0"),
        ("mxfp8", "npu:0"),
        ("mxfp4", "npu:0"),
    ],
)
def test_formal_quantized_linear_keeps_entry_owned_sparse_grid(
    tmp_path,
    precision,
    device,
):
    commands = _dry_run(
        "--operator",
        "linear",
        "--precision",
        precision,
        "--device",
        device,
        "--output-dir",
        str(tmp_path / precision),
    )

    assert len(commands) == 1
    command = commands[0]
    assert Path(command[1]).name == "test_linear.py"
    assert command[command.index("--precision") + 1] == precision
    assert not {
        "--tflops-start",
        "--tflops-end",
        "--tflops-step",
        "--tflops-sizes",
    }.intersection(command)
    assert "--tflops-iterations" not in command


@pytest.mark.parametrize("precision", ["mxfp8", "mxfp4"])
@pytest.mark.parametrize("device", ["auto", "cuda", "cuda:0"])
def test_formal_mx_precision_rejects_non_npu_devices_without_dispatch(
    tmp_path,
    precision,
    device,
):
    completed = subprocess.run(
        [
            "bash",
            str(DISPATCHER),
            "--formal",
            "--operator",
            "all",
            "--precision",
            precision,
            "--device",
            device,
            "--output-dir",
            str(tmp_path / f"invalid-{precision}"),
            "--dry-run",
        ],
        cwd=OPS_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert precision in completed.stderr.lower()
    assert "npu" in completed.stderr.lower()
    assert "DRY-RUN:" not in completed.stdout


def test_formal_default_npu_matrix_does_not_include_opt_in_fp8(tmp_path):
    commands = _dry_run(
        "--operator",
        "all",
        "--device",
        "npu:0",
        "--output-dir",
        str(tmp_path / "npu-default"),
    )

    assert len(commands) == 10
    assert all(
        command[command.index("--precision") + 1]
        not in {"fp8", "mxfp8", "mxfp4"}
        for command in commands
        if "--precision" in command
    )


def test_formal_mxfp4_groupgemm_dispatches_its_own_entry(tmp_path):
    commands = _dry_run(
        "--operator",
        "groupgemm",
        "--precision",
        "mxfp4",
        "--device",
        "npu:0",
        "--output-dir",
        str(tmp_path / "mxfp4-groupgemm"),
    )

    assert len(commands) == 1
    assert Path(commands[0][1]).name == "test_groupgemm.py"
    assert commands[0][commands[0].index("--precision") + 1] == "mxfp4"


def test_formal_quick_forwards_explicit_protocol_and_shard(tmp_path):
    output_dir = tmp_path / "group"
    commands = _dry_run(
        "--operator",
        "groupgemm",
        "--device",
        "npu:3",
        "--output-dir",
        str(output_dir),
        "--warmup",
        "7",
        "--iterations",
        "11",
        "--repeats",
        "5",
        "--shard-index",
        "2",
        "--num-shards",
        "4",
        "--quick",
    )

    assert len(commands) == 2
    for command in commands:
        assert command[command.index("--device") + 1] == "npu:3"
        assert command[command.index("--result-dir") + 1] == str(output_dir)
        assert command[command.index("--tflops-warmup") + 1] == "7"
        assert command[command.index("--tflops-iterations") + 1] == "11"
        assert command[command.index("--tflops-repeats") + 1] == "5"
        assert command[command.index("--shard-index") + 1] == "2"
        assert command[command.index("--num-shards") + 1] == "4"
        assert "--quick" in command


def test_formal_dispatcher_rejects_invalid_shard_without_running_python(
    tmp_path,
):
    completed = subprocess.run(
        [
            "bash",
            str(DISPATCHER),
            "--formal",
            "--operator",
            "add",
            "--device",
            "auto",
            "--output-dir",
            str(tmp_path),
            "--shard-index",
            "2",
            "--num-shards",
            "2",
            "--dry-run",
        ],
        cwd=OPS_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "shard" in completed.stderr.lower()
    assert "DRY-RUN:" not in completed.stdout


@pytest.mark.parametrize(
    "device",
    [
        "cuda:",
        "cuda:0junk",
        "cuda:1.2",
        "cuda:-1",
        "npu:",
        "npu:0junk",
        "npu:1.2",
        "npu:-1",
    ],
)
def test_formal_dispatcher_rejects_malformed_accelerator_device_suffix(
    tmp_path,
    device,
):
    completed = subprocess.run(
        [
            "bash",
            str(DISPATCHER),
            "--formal",
            "--operator",
            "add",
            "--device",
            device,
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ],
        cwd=OPS_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "device" in completed.stderr.lower()
    assert "DRY-RUN:" not in completed.stdout


def test_formal_dispatcher_unsets_task_queue_by_default(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python3"
    fake_python.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"${TASK_QUEUE_ENABLE-unset}\" > \"$TQ_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    task_queue_log = tmp_path / "task-queue.log"
    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    environment["TQ_LOG"] = str(task_queue_log)

    subprocess.run(
        [
            "bash",
            str(DISPATCHER),
            "--formal",
            "--operator",
            "add",
            "--device",
            "npu:0",
            "--output-dir",
            str(tmp_path / "output"),
            "--quick",
        ],
        check=True,
        cwd=OPS_ROOT,
        env=environment,
    )

    assert task_queue_log.read_text(encoding="utf-8").strip() == "unset"


def test_formal_dispatcher_exports_stabilization_repeats(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python3"
    fake_python.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$OPERATOR_TEST_STABILIZATION_REPEATS\" "
        "> \"$STABILIZATION_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    stabilization_log = tmp_path / "stabilization.log"
    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    environment["STABILIZATION_LOG"] = str(stabilization_log)

    subprocess.run(
        [
            "bash",
            str(DISPATCHER),
            "--formal",
            "--operator",
            "add",
            "--device",
            "npu:0",
            "--output-dir",
            str(tmp_path / "output"),
            "--stabilization-repeats",
            "3",
            "--quick",
        ],
        check=True,
        cwd=OPS_ROOT,
        env=environment,
    )

    assert stabilization_log.read_text(encoding="utf-8").strip() == "3"


def test_formal_dispatcher_exports_default_stabilization_repeats(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python3"
    fake_python.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$OPERATOR_TEST_STABILIZATION_REPEATS\" "
        "> \"$STABILIZATION_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    stabilization_log = tmp_path / "stabilization.log"
    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    environment["STABILIZATION_LOG"] = str(stabilization_log)

    subprocess.run(
        [
            "bash",
            str(DISPATCHER),
            "--formal",
            "--operator",
            "add",
            "--device",
            "npu:0",
            "--output-dir",
            str(tmp_path / "output"),
            "--quick",
        ],
        check=True,
        cwd=OPS_ROOT,
        env=environment,
    )

    assert stabilization_log.read_text(encoding="utf-8").strip() == "2"


def test_formal_dispatcher_rejects_noncanonical_stabilization_repeats(
    tmp_path,
):
    completed = subprocess.run(
        [
            "bash",
            str(DISPATCHER),
            "--formal",
            "--operator",
            "add",
            "--device",
            "npu:0",
            "--output-dir",
            str(tmp_path / "output"),
            "--stabilization-repeats",
            "02",
            "--dry-run",
        ],
        cwd=OPS_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "规范十进制" in completed.stderr
    assert "DRY-RUN:" not in completed.stdout


def test_formal_dispatcher_can_explicitly_unset_task_queue(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python3"
    fake_python.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"${TASK_QUEUE_ENABLE-unset}\" > \"$TQ_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    task_queue_log = tmp_path / "task-queue.log"
    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    environment["TQ_LOG"] = str(task_queue_log)
    environment["TASK_QUEUE_ENABLE"] = "1"

    subprocess.run(
        [
            "bash",
            str(DISPATCHER),
            "--formal",
            "--operator",
            "add",
            "--device",
            "npu:0",
            "--output-dir",
            str(tmp_path / "output"),
            "--task-queue",
            "unset",
            "--quick",
        ],
        check=True,
        cwd=OPS_ROOT,
        env=environment,
    )

    assert task_queue_log.read_text(encoding="utf-8").strip() == "unset"


@pytest.mark.parametrize("task_queue", ["0", "1", "2"])
def test_formal_dispatcher_forwards_explicit_task_queue(
    tmp_path,
    task_queue,
):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python3"
    fake_python.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"${TASK_QUEUE_ENABLE-unset}\" > \"$TQ_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    task_queue_log = tmp_path / "task-queue.log"
    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    environment["TQ_LOG"] = str(task_queue_log)

    subprocess.run(
        [
            "bash",
            str(DISPATCHER),
            "--formal",
            "--operator",
            "add",
            "--device",
            "npu:0",
            "--output-dir",
            str(tmp_path / "output"),
            "--task-queue",
            task_queue,
            "--quick",
        ],
        check=True,
        cwd=OPS_ROOT,
        env=environment,
    )

    assert task_queue_log.read_text(encoding="utf-8").strip() == task_queue


def test_formal_all_continues_after_independent_entry_failure(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python3"
    fake_python.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$DISPATCH_LOG\"\n"
        "case \"$1\" in\n"
        "  *test_linear.py) exit 9 ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    dispatch_log = tmp_path / "dispatch.log"
    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    environment["DISPATCH_LOG"] = str(dispatch_log)

    completed = subprocess.run(
        [
            "bash",
            str(DISPATCHER),
            "--formal",
            "--operator",
            "all",
            "--device",
            "auto",
            "--output-dir",
            str(tmp_path / "output"),
            "--quick",
        ],
        cwd=OPS_ROOT,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 1
    dispatched = dispatch_log.read_text(encoding="utf-8").splitlines()
    assert len(dispatched) == 10
    assert any("test_linear.py" in line for line in dispatched)
    assert "test_recurrent_gated_delta_rule.py" in dispatched[-1]


def test_relative_output_dir_is_resolved_from_callers_directory(tmp_path):
    completed = subprocess.run(
        [
            "bash",
            str(DISPATCHER),
            "--formal",
            "--operator",
            "add",
            "--device",
            "auto",
            "--output-dir",
            "caller-results",
            "--dry-run",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    line = next(
        line for line in completed.stdout.splitlines()
        if line.startswith("DRY-RUN: ")
    )
    command = shlex.split(line.removeprefix("DRY-RUN: "))

    assert command[command.index("--result-dir") + 1] == str(
        tmp_path / "caller-results"
    )


def test_no_argument_option_seven_still_lists_registered_operators(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python3"
    fake_python.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$DISPATCH_LOG\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    dispatch_log = tmp_path / "interactive.log"
    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
    environment["DISPATCH_LOG"] = str(dispatch_log)

    completed = subprocess.run(
        ["bash", str(DISPATCHER)],
        cwd=tmp_path,
        input="7\n",
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0
    assert dispatch_log.read_text(encoding="utf-8").splitlines() == [
        "test_main.py --list"
    ]
    assert "test_recurrent_gated_delta_rule.py" not in dispatch_log.read_text(
        encoding="utf-8"
    )
