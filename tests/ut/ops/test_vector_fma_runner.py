from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys

import pytest


OPS_ROOT = Path(__file__).resolve().parents[2] / "ops"
sys.path.insert(0, str(OPS_ROOT))

from operator_test_framework import PrecisionType  # noqa: E402
import tests.test_vector_fma as vector_fma_runner  # noqa: E402
from tests.test_vector_fma import (  # noqa: E402
    build_event_row,
    build_formal_row,
    collect_device_info,
    event_row_sha256,
    load_event_sidecar,
    main,
    parse_args,
    precondition_device,
    profile_identity,
    run_formal_measurement,
    write_event_sidecar,
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


def _device_info():
    return {
        "device_name": "NVIDIA H20-3e",
        "device_uuid": "GPU-test",
        "pci_bus_id": "0000:01:00.0",
        "compute_units": 78,
        "telemetry_status": "ok",
    }


def _event_row():
    args = _args()
    config = VectorFmaConfig(
        elements=16,
        fma_depth=32,
        accumulators=4,
        block_size=8,
        num_programs=2,
    )
    return build_event_row(
        args,
        config,
        "triton_common_bf16_fma",
        _metrics(),
        _device_info(),
        _device_info(),
        {"seconds": 5.2, "launches": 20},
    )


def _verified_manifest(event_row):
    return {
        "status": "verified",
        "identity": profile_identity(event_row),
        "event_row_sha256": event_row_sha256(event_row),
        "profile_launch_count": 3,
        "tensor_or_cube_used": False,
        "spilled": False,
        "profile_duration_is_diagnostic": True,
        "instruction_evidence": [
            "no tensor/cube; vector instructions found"
        ],
        "profile_tool": "synthetic-profiler",
        "profile_artifacts": ["synthetic.txt"],
    }


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
        _device_info(),
        _device_info(),
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
    assert row["telemetry_complete"] is True


def test_verified_manifest_must_match_full_event_identity_and_hash():
    row = _event_row()
    manifest = _verified_manifest(row)

    formal = build_formal_row(row, profile_manifest=manifest)

    assert formal["publishable_peak"] is True
    unrelated = dict(manifest)
    unrelated["identity"] = dict(manifest["identity"])
    unrelated["identity"]["fma_depth"] = 64
    with pytest.raises(RuntimeError, match="identity"):
        build_formal_row(row, profile_manifest=unrelated)
    tampered_hash = dict(manifest)
    tampered_hash["event_row_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="hash"):
        build_formal_row(row, profile_manifest=tampered_hash)


def test_event_sidecar_is_canonical_typed_and_hash_verified(tmp_path):
    row = _event_row()
    sidecar_path = tmp_path / "measurement_event.json"

    sidecar_hash = write_event_sidecar(sidecar_path, row)
    loaded_row, loaded_hash = load_event_sidecar(sidecar_path)

    assert loaded_row == row
    assert loaded_hash == event_row_sha256(row) == sidecar_hash
    on_disk = sidecar_path.read_text(encoding="utf-8")
    payload = json.loads(on_disk)
    assert on_disk == json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
    assert isinstance(payload["event_row"]["repeat_samples_ms"], list)
    assert isinstance(payload["event_row"]["publishable_peak"], bool)
    assert isinstance(payload["event_row"]["fma_depth"], int)
    assert isinstance(payload["event_row"]["median_latency_ms"], float)


def test_event_sidecar_tamper_fails_closed(tmp_path):
    row = _event_row()
    sidecar_path = tmp_path / "measurement_event.json"
    write_event_sidecar(sidecar_path, row)
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["event_row"]["repeat_samples_ms"][0] += 1.0
    sidecar_path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="hash"):
        load_event_sidecar(sidecar_path)


def test_promote_uses_persisted_event_without_remeasurement(
    tmp_path, monkeypatch
):
    row = _event_row()
    sidecar_path = tmp_path / "measurement_event.json"
    manifest_path = tmp_path / "measurement_profile.json"
    result_dir = tmp_path / "verified"
    write_event_sidecar(sidecar_path, row)
    manifest_path.write_text(
        json.dumps(
            _verified_manifest(row),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        vector_fma_runner,
        "run_formal_measurement",
        lambda *args, **kwargs: pytest.fail(
            "promote must not run a fresh measurement"
        ),
    )
    monkeypatch.setattr(
        vector_fma_runner,
        "collect_device_info",
        lambda *args, **kwargs: pytest.fail(
            "promote must not inspect a live device"
        ),
    )
    monkeypatch.setattr(
        vector_fma_runner,
        "TritonVectorFmaOperatorTest",
        lambda *args, **kwargs: pytest.fail(
            "promote must not initialize an operator"
        ),
    )

    result = main(
        [
            "--mode",
            "promote",
            "--event-json",
            str(sidecar_path),
            "--profile-manifest",
            str(manifest_path),
            "--result-dir",
            str(result_dir),
        ]
    )

    assert result == 0
    verified = list(result_dir.glob("*_verified.csv"))
    assert len(verified) == 1
    assert sidecar_path.read_text(encoding="utf-8")


def test_promote_tampered_manifest_fails_closed(tmp_path):
    row = _event_row()
    sidecar_path = tmp_path / "measurement_event.json"
    manifest_path = tmp_path / "measurement_profile.json"
    result_dir = tmp_path / "verified"
    write_event_sidecar(sidecar_path, row)
    manifest = _verified_manifest(row)
    manifest["event_row_sha256"] = "0" * 64
    manifest_path.write_text(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="bind"):
        main(
            [
                "--mode",
                "promote",
                "--event-json",
                str(sidecar_path),
                "--profile-manifest",
                str(manifest_path),
                "--result-dir",
                str(result_dir),
            ]
        )

    assert not list(result_dir.glob("*_verified.csv"))


def test_formal_cannot_promote_a_fresh_measurement():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--mode",
                "formal",
                "--device",
                "cuda:0",
                "--profile-manifest",
                "profile.json",
            ]
        )


@pytest.mark.parametrize(
    ("field", "before", "after"),
    [
        ("device_uuid", "", ""),
        ("device_name", None, None),
        ("pci_bus_id", None, None),
        ("device_uuid", "GPU-before", "GPU-after"),
        ("device_name", "H20-before", "H20-after"),
        ("pci_bus_id", "0000:01:00.0", "0000:02:00.0"),
    ],
)
def test_telemetry_complete_requires_stable_nonempty_identity(
    field, before, after
):
    args = _args()
    config = VectorFmaConfig(
        elements=16,
        fma_depth=32,
        accumulators=4,
        block_size=8,
        num_programs=2,
    )
    device_before = _device_info()
    device_after = _device_info()
    device_before[field] = before
    device_after[field] = after

    row = build_event_row(
        args,
        config,
        "triton_common_bf16_fma",
        _metrics(),
        device_before,
        device_after,
        {"seconds": 5.2, "launches": 20},
    )

    assert row["telemetry_complete"] is False


def test_950pr_device_info_uses_supported_queries_and_byte_units(monkeypatch):
    properties = SimpleNamespace(
        name="Ascend950PR_957b",
        uuid="0008d80a-d200-0000-0000-004dd5940000",
        vector_core_num=56,
        cube_core_num=28,
        total_memory=115_543_814_976,
    )
    fake_torch_npu = SimpleNamespace(
        npu=SimpleNamespace(
            get_device_properties=lambda index: (
                properties if index == 0 else pytest.fail("wrong device")
            )
        )
    )
    commands = []

    def fake_state(command):
        commands.append(command)
        if "board" in command:
            stdout = "PCIe Bus Info : 0000:11:00.0"
        else:
            stdout = "Aicore Freq(MHZ) : 1650"
        return {"exit_code": 0, "stdout": stdout, "stderr": ""}

    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    monkeypatch.setattr(vector_fma_runner, "_run_state_command", fake_state)

    result = collect_device_info("npu:0")

    assert result["telemetry_status"] == "ok"
    assert result["pci_bus_id"] == "0000:11:00.0"
    assert result["total_memory_bytes"] == properties.total_memory
    assert commands == [
        ["npu-smi", "info", "-t", "board", "-i", "0"],
        ["npu-smi", "info", "-t", "common", "-i", "0"],
    ]


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
