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
    event_row_sha256,
    parse_args,
    precondition_device,
    profile_identity,
    run_formal_measurement,
)
from vector_fma.base import VectorFmaConfig, vector_fma_flops  # noqa: E402


def _args():
    args = parse_args(
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
    args.run_id = "unit-run"
    return args


def _metrics():
    return SimpleNamespace(
        avg_time_ms=10.0,
        repeat_samples_ms=[10.0 + index * 0.001 for index in range(30)],
        framework_api=(
            "OperatorTestFramework.run_core_operator_performance_test_v2"
        ),
        protocol_version="operator-test-framework-v2-fresh-v6",
        timing_method="device_event",
        timing_semantics="device elapsed time",
        warmup_iterations=20,
        iterations=1,
        repeats=30,
        dispatch_loop_policy="python_direct_prepared_payload_loop",
    )


def test_formal_defaults_are_exact():
    args = parse_args(["--mode", "formal", "--device", "cuda:0"])

    assert args.warmup == 20
    assert args.iterations == 1
    assert args.samples == 30
    assert args.process_index in (0, 1, 2)
    assert args.precondition_seconds == 5.0
    assert args.precondition_launches == 20
    assert args.run_id is None


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
    event_row = {
        "run_id": "run",
        "process_index": 0,
        "device": "cuda:0",
        "device_uuid": "GPU-test",
        "device_name": "NVIDIA H20-3e",
        "precision": "bf16",
        "provider": "candidate",
        "elements": 16,
        "fma_depth": 32,
        "accumulators": 4,
        "block_size": 8,
        "num_programs": 2,
        "telemetry_complete": True,
    }
    with pytest.raises(RuntimeError, match="profile verification"):
        build_formal_row(
            event_row,
            profile_manifest={"status": "unverified"},
        )
    with pytest.raises(RuntimeError, match="required profile field"):
        build_formal_row(
            event_row,
            profile_manifest={"status": "verified"},
        )

    manifest = {
        "status": "verified",
        "identity": profile_identity(event_row),
        "event_row_sha256": event_row_sha256(event_row),
        "profile_launch_count": 3,
        "tensor_or_cube_used": True,
        "spilled": False,
        "profile_duration_is_diagnostic": True,
        "instruction_evidence": ["synthetic"],
        "profile_tool": "synthetic-profiler",
        "profile_artifacts": ["synthetic.txt"],
    }
    with pytest.raises(RuntimeError, match="Tensor/Cube"):
        build_formal_row(
            event_row,
            profile_manifest=manifest,
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
        {
            "device_name": "NVIDIA H20-3e",
            "device_uuid": "GPU-test",
            "compute_units": 78,
            "telemetry_status": "ok",
        },
        {
            "device_name": "NVIDIA H20-3e",
            "device_uuid": "GPU-test",
            "compute_units": 78,
            "telemetry_status": "ok",
        },
        {"seconds": 5.2, "launches": 260},
    )

    assert row["profile_status"] == "unverified"
    assert row["publishable_peak"] is False
    assert len(row["repeat_samples_ms"]) == 30
    assert row["flops"] == vector_fma_flops(config)
    assert row["tflops"] == pytest.approx(
        vector_fma_flops(config) / 0.010 / 1e12
    )
    assert row["process_index"] == 0
    assert row["run_id"] == "unit-run"
    assert row["device_state_before"]["telemetry_status"] == "ok"
    assert row["device_state_after"]["telemetry_status"] == "ok"


def test_verified_manifest_must_match_full_event_identity_and_hash():
    args = _args()
    config = VectorFmaConfig(
        elements=16,
        fma_depth=32,
        accumulators=4,
        block_size=8,
        num_programs=2,
    )
    device = {
        "device_name": "NVIDIA H20-3e",
        "device_uuid": "GPU-test",
        "compute_units": 78,
        "telemetry_status": "ok",
    }
    row = build_event_row(
        args,
        config,
        "triton_common_bf16_fma",
        _metrics(),
        device,
        device,
        {"seconds": 5.2, "launches": 20},
    )
    manifest = {
        "status": "verified",
        "identity": profile_identity(row),
        "event_row_sha256": event_row_sha256(row),
        "profile_launch_count": 3,
        "tensor_or_cube_used": False,
        "spilled": False,
        "profile_duration_is_diagnostic": True,
        "instruction_evidence": ["no tensor/cube; vector instructions found"],
        "profile_tool": "synthetic-profiler",
        "profile_artifacts": ["synthetic.txt"],
    }

    formal = build_formal_row(row, profile_manifest=manifest)

    assert formal["publishable_peak"] is True
    unrelated = dict(manifest)
    unrelated["identity"] = dict(manifest["identity"])
    unrelated["identity"]["fma_depth"] = 64
    with pytest.raises(RuntimeError, match="identity"):
        build_formal_row(row, profile_manifest=unrelated)


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


@pytest.mark.parametrize(
    "extra",
    [
        ["--precondition-seconds", "0"],
        ["--precondition-seconds", "nan"],
        ["--precondition-launches", "19"],
    ],
)
def test_cli_cannot_weaken_precondition_gate(extra):
    with pytest.raises(SystemExit):
        parse_args(
            ["--mode", "formal", "--device", "cuda:0", *extra]
        )


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


def test_formal_measurement_rejects_non_peak_latency():
    args = _args()
    framework = SimpleNamespace(
        run_core_operator_performance_test_v2=lambda **kwargs: SimpleNamespace(
            **{
                **_metrics().__dict__,
                "avg_time_ms": 2.0,
            }
        )
    )
    operator = SimpleNamespace(provider_name="triton_common_bf16_fma")

    with pytest.raises(RuntimeError, match="5-20 ms"):
        run_formal_measurement(
            args,
            operator,
            {},
            framework,
            PrecisionType.BF16,
            operator.provider_name,
            precondition_fn=lambda *values, **kwargs: {
                "seconds": 5.1,
                "launches": 20,
            },
        )


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
    assert result.stdout.count("tests/test_vector_fma.py") == 3
    assert result.stdout.count("--mode formal") == 3
    assert result.stdout.count("--warmup 20") == 3
    assert result.stdout.count("--iterations 1") == 3
    assert result.stdout.count("--samples 30") == 3
    for process_index in (0, 1, 2):
        assert f"--process-index {process_index}" in result.stdout


def test_run_tests_rejects_ignored_repeat_override(tmp_path):
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
            "--repeats",
            "7",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "repeats=30" in result.stderr
