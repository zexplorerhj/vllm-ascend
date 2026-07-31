from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest


OPS_ROOT = Path(__file__).resolve().parents[2] / "ops"
sys.path.insert(0, str(OPS_ROOT))

from operator_test_framework import PrecisionType  # noqa: E402
from tests.test_vector_fma import (  # noqa: E402
    build_event_row,
    build_formal_row,
    parse_args,
    precondition_device,
    run_formal_measurement,
)
from vector_fma.base import VectorFmaConfig, vector_fma_flops  # noqa: E402


def _args():
    return parse_args(
        [
            "--mode",
            "formal",
            "--device",
            "cuda:0",
            "--precision",
            "bf16",
            "--elements",
            "16",
            "--fma-depth",
            "32",
            "--accumulators",
            "4",
            "--block-size",
            "8",
            "--num-programs",
            "2",
        ]
    )


def _metrics():
    return SimpleNamespace(
        avg_time_ms=2.0,
        repeat_samples_ms=[2.0 + index * 0.001 for index in range(30)],
        framework_api=(
            "OperatorTestFramework.run_core_operator_performance_test_v2"
        ),
        protocol_version="operator-test-framework-v2-fresh-v6",
    )


def test_formal_defaults_are_exact():
    args = parse_args(["--mode", "formal", "--device", "cuda:0"])

    assert args.warmup == 20
    assert args.iterations == 1
    assert args.samples == 30
    assert args.process_index in (0, 1, 2)
    assert args.precondition_seconds == 5.0
    assert args.precondition_launches == 20


@pytest.mark.parametrize("value", ["-1", "3", "4"])
def test_process_index_is_limited_to_three_processes(value):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--mode",
                "formal",
                "--device",
                "cuda:0",
                "--process-index",
                value,
            ]
        )


def test_csv_rejects_unverified_tensor_or_cube_provider():
    event_row = {"provider": "candidate"}
    with pytest.raises(RuntimeError, match="profile verification"):
        build_formal_row(event_row, profile_status="unverified")
    with pytest.raises(RuntimeError, match="Tensor/Cube"):
        build_formal_row(
            event_row,
            profile_status="verified",
            tensor_or_cube_used=True,
        )


def test_raw_event_row_keeps_all_samples_and_is_not_publishable():
    args = _args()
    config = VectorFmaConfig(
        elements=16,
        fma_depth=32,
        accumulators=4,
        block_size=8,
        num_programs=2,
    )

    row = build_event_row(
        args,
        config,
        "triton_common_bf16_fma",
        _metrics(),
        {"device_name": "NVIDIA H20-3e", "compute_units": 78},
        {"seconds": 5.2, "launches": 260},
    )

    assert row["profile_status"] == "unverified"
    assert row["publishable_peak"] is False
    assert len(row["repeat_samples_ms"]) == 30
    assert row["flops"] == vector_fma_flops(config)
    assert row["tflops"] == pytest.approx(
        vector_fma_flops(config) / 0.002 / 1e12
    )
    assert row["process_index"] == 0


def test_precondition_requires_both_time_and_launch_thresholds():
    class FakeOperator:
        def __init__(self):
            self.launches = 0

        def _prepare_data_for_core_operator(self, *args):
            return {"prepared": True}

        def _execute_core_operator(self, prepared, implementation):
            assert prepared == {"prepared": True}
            assert implementation == "provider"
            self.launches += 1

    operator = FakeOperator()
    clock_values = iter([0.0] + [0.1] * 19 + [6.0] * 10)
    synchronizations = []

    result = precondition_device(
        operator,
        {},
        "cuda:0",
        PrecisionType.BF16,
        "provider",
        min_seconds=5.0,
        min_launches=20,
        time_fn=lambda: next(clock_values),
        synchronize_fn=lambda device: synchronizations.append(device),
    )

    assert result["launches"] == 20
    assert result["seconds"] >= 5.0
    assert operator.launches == 20
    assert synchronizations


def test_formal_measurement_uses_framework_v2_exact_protocol():
    args = _args()
    calls = []
    framework = SimpleNamespace(
        run_core_operator_performance_test_v2=lambda **kwargs: (
            calls.append(kwargs) or _metrics()
        )
    )
    operator = SimpleNamespace(provider_name="triton_common_bf16_fma")
    precondition_calls = []

    metrics, precondition = run_formal_measurement(
        args,
        operator,
        {"config": "data"},
        framework,
        PrecisionType.BF16,
        "triton_common_bf16_fma",
        precondition_fn=lambda *values, **kwargs: (
            precondition_calls.append((values, kwargs))
            or {"seconds": 5.1, "launches": 20}
        ),
    )

    assert metrics is not None
    assert precondition == {"seconds": 5.1, "launches": 20}
    assert len(precondition_calls) == 1
    assert len(calls) == 1
    call = calls[0]
    assert call["num_warmup"] == 20
    assert call["num_iterations"] == 1
    assert call["num_repeats"] == 30
    assert call["retain_outputs"] is True
    assert call["verify_independent_storage"] is True
    assert call["num_stabilization_repeats"] == 0
    assert call["dispatch_mode"] == "eager_direct"


def test_run_tests_dispatches_vector_fma_without_changing_protocol(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(OPS_ROOT / "run_tests.sh"),
            "--formal",
            "--operator",
            "vector-fma",
            "--device",
            "cuda:0",
            "--output-dir",
            str(tmp_path),
            "--precision",
            "bf16",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "tests/test_vector_fma.py" in result.stdout
    assert "--mode formal" in result.stdout
    assert "--warmup 20" in result.stdout
    assert "--iterations 1" in result.stdout
    assert "--samples 30" in result.stdout
    assert "--process-index 0" in result.stdout
