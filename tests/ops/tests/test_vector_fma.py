#!/usr/bin/env python3
"""Formal Event runner for the cross-platform Vector FMA microbenchmark."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import subprocess
import sys
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch


OPS_ROOT = Path(__file__).resolve().parents[1]
if str(OPS_ROOT) not in sys.path:
    sys.path.insert(0, str(OPS_ROOT))

from operator_test_framework import (  # noqa: E402
    OperatorTestFramework,
    PrecisionType,
)
from vector_fma.base import (  # noqa: E402
    VectorFmaConfig,
    vector_fma_flops,
)
from vector_fma.triton_impl import (  # noqa: E402
    TritonVectorFmaOperatorTest,
    validate_triton_launch_geometry,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _precondition_seconds(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 5.0:
        raise argparse.ArgumentTypeError(
            "precondition seconds must be finite and >= 5"
        )
    return parsed


def _precondition_launches(value: str) -> int:
    parsed = int(value)
    if parsed < 20:
        raise argparse.ArgumentTypeError(
            "precondition launches must be >= 20"
        )
    return parsed


def _run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise argparse.ArgumentTypeError(
            "run id must contain only letters, digits, dot, underscore, dash"
        )
    return value


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("formal", "probe", "profile", "promote"),
        default="formal",
    )
    parser.add_argument("--device")
    parser.add_argument(
        "--precision",
        choices=("fp16", "bf16"),
        default="bf16",
    )
    parser.add_argument("--provider", default="triton")
    parser.add_argument("--elements", type=_positive_int, default=4096)
    parser.add_argument("--fma-depth", type=_positive_int, default=65536)
    parser.add_argument(
        "--accumulators",
        type=int,
        choices=(4, 8, 16),
        default=8,
    )
    parser.add_argument("--block-size", type=_positive_int, default=256)
    parser.add_argument("--num-programs", type=_positive_int, default=16)
    parser.add_argument("--warmup", type=_non_negative_int, default=20)
    parser.add_argument("--iterations", type=_positive_int, default=1)
    parser.add_argument("--samples", type=_positive_int, default=30)
    parser.add_argument(
        "--process-index",
        type=int,
        choices=(0, 1, 2),
        default=0,
    )
    parser.add_argument(
        "--precondition-seconds",
        type=_precondition_seconds,
        default=5.0,
    )
    parser.add_argument(
        "--precondition-launches",
        type=_precondition_launches,
        default=20,
    )
    parser.add_argument("--run-id", type=_run_id)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("vector_fma_results"),
    )
    parser.add_argument("--profile-manifest", type=Path)
    parser.add_argument("--event-json", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "promote":
        if args.event_json is None or args.profile_manifest is None:
            parser.error(
                "promote mode requires --event-json and --profile-manifest"
            )
    elif args.device is None:
        parser.error(f"{args.mode} mode requires --device")
    if args.mode == "formal" and args.profile_manifest is not None:
        parser.error(
            "--profile-manifest can only publish a persisted Event row "
            "through --mode promote"
        )
    if args.mode != "promote" and args.event_json is not None:
        parser.error("--event-json is only valid with --mode promote")
    return args


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    elif device.startswith("npu"):
        import torch_npu

        torch_npu.npu.synchronize()
    else:
        raise ValueError("Vector FMA formal timing requires CUDA or NPU")


def precondition_device(
    operator: Any,
    data: Dict[str, Any],
    device: str,
    precision: PrecisionType,
    implementation: str,
    *,
    min_seconds: float = 5.0,
    min_launches: int = 20,
    time_fn: Callable[[], float] = time.perf_counter,
    synchronize_fn: Callable[[str], None] = _synchronize,
) -> Dict[str, Any]:
    """Warm one cached raw kernel until both launch and duration gates pass."""
    if not math.isfinite(min_seconds) or min_seconds < 5.0:
        raise ValueError("min_seconds must be finite and >= 5")
    if min_launches < 20:
        raise ValueError("min_launches must be >= 20")
    prepared = operator._prepare_data_for_core_operator(
        data,
        device,
        precision,
        implementation,
    )
    synchronize_fn(device)
    start = time_fn()
    launches = 0
    elapsed = 0.0
    while launches < min_launches or elapsed < min_seconds:
        operator._execute_core_operator(prepared, implementation)
        launches += 1
        synchronize_fn(device)
        elapsed = time_fn() - start
    return {
        "seconds": float(elapsed),
        "launches": launches,
        "timed": False,
        "policy": "cached_raw_kernel_until_time_and_launch_gates",
    }


def run_formal_measurement(
    args: argparse.Namespace,
    operator: Any,
    data: Dict[str, Any],
    framework: OperatorTestFramework,
    precision: PrecisionType,
    provider: str,
    *,
    precondition_fn: Callable[..., Dict[str, Any]] = precondition_device,
) -> Tuple[Any, Dict[str, Any]]:
    if args.warmup != 20 or args.iterations != 1 or args.samples != 30:
        raise ValueError("formal protocol is fixed at W20/I1/R30")
    precondition = precondition_fn(
        operator,
        data,
        args.device,
        precision,
        provider,
        min_seconds=args.precondition_seconds,
        min_launches=args.precondition_launches,
    )
    metrics = framework.run_core_operator_performance_test_v2(
        operator_test=operator,
        data=data,
        device=args.device,
        precision=precision,
        implementation=provider,
        num_warmup=args.warmup,
        num_iterations=args.iterations,
        num_repeats=args.samples,
        retain_outputs=True,
        verify_independent_storage=True,
        num_stabilization_repeats=0,
        dispatch_mode="eager_direct",
    )
    if len(metrics.repeat_samples_ms) != args.samples:
        raise RuntimeError(
            "Framework V2 did not return exactly 30 raw Event samples"
        )
    expected_fields = {
        "framework_api": (
            "OperatorTestFramework.run_core_operator_performance_test_v2"
        ),
        "protocol_version": "operator-test-framework-v2-fresh-v6",
        "timing_method": "device_event",
        "warmup_iterations": 20,
        "iterations": 1,
        "repeats": 30,
        "dispatch_loop_policy": "python_direct_prepared_payload_loop",
    }
    for field, expected in expected_fields.items():
        actual = getattr(metrics, field, None)
        if actual != expected:
            raise RuntimeError(
                f"Framework V2 provenance drift for {field}: "
                f"{actual!r} != {expected!r}"
            )
    if "device elapsed time" not in str(
        getattr(metrics, "timing_semantics", "")
    ):
        raise RuntimeError("Framework V2 did not report device elapsed time")
    if not 5.0 <= float(metrics.avg_time_ms) <= 20.0:
        raise RuntimeError(
            "selected Vector FMA kernel must be in the 5-20 ms gate"
        )
    return metrics, precondition


def build_event_row(
    args: argparse.Namespace,
    config: VectorFmaConfig,
    provider: str,
    metrics: Any,
    device_info_before: Dict[str, Any],
    device_info_after: Dict[str, Any],
    precondition: Dict[str, Any],
) -> Dict[str, Any]:
    samples = [float(value) for value in metrics.repeat_samples_ms]
    flops = vector_fma_flops(config)
    latency_ms = float(metrics.avg_time_ms)
    stable_telemetry_fields = ("device_uuid", "device_name", "pci_bus_id")
    telemetry_identity_stable = all(
        isinstance(device_info_before.get(field), str)
        and bool(device_info_before[field].strip())
        and isinstance(device_info_after.get(field), str)
        and bool(device_info_after[field].strip())
        and device_info_before[field] == device_info_after[field]
        for field in stable_telemetry_fields
    )
    row = {
        "status": "ok",
        "profile_status": "unverified",
        "publishable_peak": False,
        "process_index": args.process_index,
        "process_pid": os.getpid(),
        "host": platform.node(),
        "run_id": args.run_id,
        "device": args.device,
        "precision": args.precision,
        "provider": provider,
        "elements": config.elements,
        "fma_depth": config.fma_depth,
        "accumulators": config.accumulators,
        "block_size": config.block_size,
        "num_programs": config.num_programs,
        "active_scalar_lanes": config.elements * config.accumulators,
        "flops": flops,
        "median_latency_ms": latency_ms,
        "tflops": (
            flops / (latency_ms / 1000.0) / 1e12
            if latency_ms > 0
            else 0.0
        ),
        "repeat_samples_ms": samples,
        "sample_count": len(samples),
        "framework_api": getattr(metrics, "framework_api", None),
        "protocol_version": getattr(metrics, "protocol_version", None),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "samples": args.samples,
        "timing_method": metrics.timing_method,
        "timing_semantics": metrics.timing_semantics,
        "dispatch_mode": metrics.dispatch_loop_policy,
        "precondition_seconds": precondition["seconds"],
        "precondition_launches": precondition["launches"],
        "precondition_timed": precondition.get("timed", False),
        "device_name": device_info_before.get("device_name"),
        "device_uuid": device_info_before.get("device_uuid"),
        "pci_bus_id": device_info_before.get("pci_bus_id"),
        "compute_units": device_info_before.get("compute_units"),
        "cube_units": device_info_before.get("cube_units"),
        "total_memory_bytes": device_info_before.get("total_memory_bytes"),
        "telemetry_complete": bool(
            device_info_before.get("telemetry_status") == "ok"
            and device_info_after.get("telemetry_status") == "ok"
            and telemetry_identity_stable
        ),
        "device_state_before": dict(device_info_before),
        "device_state_after": dict(device_info_after),
    }
    return row


PROFILE_IDENTITY_FIELDS = (
    "run_id",
    "process_index",
    "device",
    "device_uuid",
    "device_name",
    "precision",
    "provider",
    "elements",
    "fma_depth",
    "accumulators",
    "block_size",
    "num_programs",
)


def profile_identity(event_row: Dict[str, Any]) -> Dict[str, Any]:
    missing = [key for key in PROFILE_IDENTITY_FIELDS if key not in event_row]
    if missing:
        raise RuntimeError(f"event row lacks profile identity fields: {missing}")
    return {key: event_row[key] for key in PROFILE_IDENTITY_FIELDS}


EVENT_SIDECAR_SCHEMA_VERSION = "vector-fma-event-row-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def event_row_sha256(event_row: Dict[str, Any]) -> str:
    encoded = _canonical_json(event_row).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_event_sidecar(path: Path, event_row: Dict[str, Any]) -> str:
    """Persist the exact typed Event row with its canonical JSON hash."""
    if not isinstance(event_row, dict):
        raise TypeError("event row must be a dictionary")
    row_hash = event_row_sha256(event_row)
    payload = {
        "schema_version": EVENT_SIDECAR_SCHEMA_VERSION,
        "event_row_sha256": row_hash,
        "event_row": event_row,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_canonical_json(payload) + "\n", encoding="utf-8")
    return row_hash


def load_event_sidecar(path: Path) -> Tuple[Dict[str, Any], str]:
    """Load a canonical sidecar, rejecting schema or hash drift."""
    try:
        encoded = path.read_text(encoding="utf-8")
        payload = json.loads(encoded)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load Event sidecar: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Event sidecar must contain a JSON object")
    expected_fields = {
        "schema_version",
        "event_row_sha256",
        "event_row",
    }
    if set(payload) != expected_fields:
        raise RuntimeError("Event sidecar schema fields do not match")
    if payload["schema_version"] != EVENT_SIDECAR_SCHEMA_VERSION:
        raise RuntimeError("Event sidecar schema version does not match")
    event_row = payload["event_row"]
    row_hash = payload["event_row_sha256"]
    if not isinstance(event_row, dict):
        raise RuntimeError("Event sidecar event_row must be a JSON object")
    if (
        not isinstance(row_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", row_hash) is None
    ):
        raise RuntimeError("Event sidecar hash is malformed")
    try:
        canonical_payload = _canonical_json(payload) + "\n"
        actual_hash = event_row_sha256(event_row)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Event sidecar is not canonical JSON: {exc}") from exc
    if encoded != canonical_payload:
        raise RuntimeError("Event sidecar is not canonical JSON")
    if not secrets.compare_digest(row_hash, actual_hash):
        raise RuntimeError("Event sidecar Event-row hash does not match")
    return event_row, row_hash


def build_formal_row(
    event_row: Dict[str, Any],
    *,
    profile_manifest: Dict[str, Any],
    profile_manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Promote one raw Event row only after a matching profile gate passes."""
    required_fields = (
        "status",
        "identity",
        "event_row_sha256",
        "profile_launch_count",
        "tensor_or_cube_used",
        "spilled",
        "profile_duration_is_diagnostic",
        "instruction_evidence",
        "profile_tool",
        "profile_artifacts",
    )
    if profile_manifest.get("status") != "verified":
        raise RuntimeError("profile verification is required for publication")
    missing = [key for key in required_fields if key not in profile_manifest]
    if missing:
        raise RuntimeError(f"required profile field missing: {missing}")
    if profile_manifest["identity"] != profile_identity(event_row):
        raise RuntimeError("profile identity does not match the Event row")
    if profile_manifest["event_row_sha256"] != event_row_sha256(event_row):
        raise RuntimeError("profile Event-row hash does not match")
    launch_count = profile_manifest["profile_launch_count"]
    if (
        not isinstance(launch_count, int)
        or isinstance(launch_count, bool)
        or launch_count < 3
    ):
        raise RuntimeError("profile verification requires at least 3 launches")
    for field in (
        "tensor_or_cube_used",
        "spilled",
        "profile_duration_is_diagnostic",
    ):
        if type(profile_manifest[field]) is not bool:
            raise RuntimeError(f"profile field {field} must be an exact bool")
    if profile_manifest["tensor_or_cube_used"]:
        raise RuntimeError("Tensor/Cube execution cannot be published")
    if profile_manifest["spilled"]:
        raise RuntimeError("spilled Vector FMA configuration cannot be published")
    if not profile_manifest["profile_duration_is_diagnostic"]:
        raise RuntimeError("profile duration must remain diagnostic only")
    evidence = profile_manifest["instruction_evidence"]
    artifacts = profile_manifest["profile_artifacts"]
    if (
        not isinstance(evidence, list)
        or not evidence
        or not all(isinstance(item, str) and item for item in evidence)
    ):
        raise RuntimeError("instruction evidence must be a non-empty string list")
    if (
        not isinstance(profile_manifest["profile_tool"], str)
        or not profile_manifest["profile_tool"]
        or not isinstance(artifacts, list)
        or not artifacts
        or not all(isinstance(item, str) and item for item in artifacts)
    ):
        raise RuntimeError("profile tool and artifacts must be explicit")
    if event_row.get("telemetry_complete") is not True:
        raise RuntimeError("complete before/after device telemetry is required")
    row = dict(event_row)
    row.update(
        {
            "profile_status": "verified",
            "publishable_peak": True,
            "profile_manifest": profile_manifest_path,
            "profile_tool": profile_manifest["profile_tool"],
            "profile_launch_count": launch_count,
        }
    )
    return row


def _device_index(device: str) -> int:
    return int(device.split(":", 1)[1]) if ":" in device else 0


def _run_state_command(command: Sequence[str]) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "exit_code": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def collect_device_info(device: str) -> Dict[str, Any]:
    index = _device_index(device)
    if device.startswith("cuda"):
        properties = torch.cuda.get_device_properties(index)
        state = _run_state_command(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=uuid,pci.bus_id,name,pstate,clocks.current.sm,"
                "clocks.current.memory,temperature.gpu,power.draw,"
                "memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ]
        )
        state_values = [
            value.strip() for value in state["stdout"].split(",")
        ]
        telemetry_ok = state["exit_code"] == 0 and len(state_values) == 10
        return {
            "device_name": properties.name,
            "device_uuid": state_values[0] if telemetry_ok else None,
            "pci_bus_id": state_values[1] if telemetry_ok else None,
            "compute_units": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": ".".join(
                str(value) for value in torch.cuda.get_device_capability(index)
            ),
            "telemetry_status": "ok" if telemetry_ok else "unavailable",
            "device_state_raw": state,
        }
    if device.startswith("npu"):
        import torch_npu

        properties = torch_npu.npu.get_device_properties(index)
        board_state = _run_state_command(
            ["npu-smi", "info", "-t", "board", "-i", str(index)]
        )
        common_state = _run_state_command(
            ["npu-smi", "info", "-t", "common", "-i", str(index)]
        )
        bus_match = re.search(
            r"\b[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:"
            r"[0-9A-Fa-f]{2}\.[0-7]\b",
            board_state["stdout"],
        )
        device_uuid = str(getattr(properties, "uuid", "")) or None
        telemetry_ok = (
            board_state["exit_code"] == 0
            and common_state["exit_code"] == 0
            and device_uuid is not None
            and bus_match is not None
        )
        return {
            "device_name": properties.name,
            "device_uuid": device_uuid,
            "pci_bus_id": bus_match.group(0) if bus_match else None,
            "compute_units": properties.vector_core_num,
            "cube_units": properties.cube_core_num,
            # torch_npu reports this property in bytes even though its repr
            # renders a human-readable MB suffix.
            "total_memory_bytes": int(properties.total_memory),
            "telemetry_status": "ok" if telemetry_ok else "unavailable",
            "device_state_raw": {
                "board": board_state,
                "common": common_state,
            },
        }
    raise ValueError("Vector FMA requires a CUDA or NPU device")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty Vector FMA CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if list(row) != fields:
                raise ValueError("Vector FMA rows must share one exact schema")
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _precision(value: str) -> PrecisionType:
    return (
        PrecisionType.FP16
        if value == "fp16"
        else PrecisionType.BF16
    )


def _config(args: argparse.Namespace) -> VectorFmaConfig:
    config = VectorFmaConfig(
        elements=args.elements,
        fma_depth=args.fma_depth,
        accumulators=args.accumulators,
        block_size=args.block_size,
        num_programs=args.num_programs,
    )
    validate_triton_launch_geometry(config)
    return config


def _load_profile_manifest(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise RuntimeError("profile manifest must contain a JSON object")
    return manifest


def _generated_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}-pid{os.getpid()}-{secrets.token_hex(4)}"


def promote_persisted_event(
    event_json_path: Path,
    profile_manifest_path: Path,
    result_dir: Path,
) -> Path:
    """Publish a persisted measurement without launching the operator again."""
    event_row, sidecar_hash = load_event_sidecar(event_json_path)
    manifest = _load_profile_manifest(profile_manifest_path)
    if manifest.get("event_row_sha256") != sidecar_hash:
        raise RuntimeError("profile manifest does not bind the Event sidecar")
    formal = build_formal_row(
        event_row,
        profile_manifest=manifest,
        profile_manifest_path=str(profile_manifest_path),
    )
    formal["event_json"] = str(event_json_path)
    formal["event_row_sha256"] = sidecar_hash
    formal_path = result_dir / f"{event_json_path.stem}_verified.csv"
    write_rows(formal_path, [formal])
    return formal_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.mode == "promote":
        formal_path = promote_persisted_event(
            args.event_json,
            args.profile_manifest,
            args.result_dir,
        )
        print(f"verified formal row: {formal_path}")
        return 0
    if args.run_id is None:
        args.run_id = _generated_run_id()
    if args.mode != "formal":
        raise RuntimeError(
            f"{args.mode} mode is reserved for the native/profile tasks"
        )
    if args.provider != "triton":
        raise RuntimeError(
            "Task 3 runner currently exposes only the common Triton provider"
        )
    config = _config(args)
    precision = _precision(args.precision)
    operator = TritonVectorFmaOperatorTest(precision, config)
    providers = operator.get_available_implementations(args.device)
    if operator.provider_name not in providers:
        operator._require_runtime(args.device)
        raise RuntimeError(f"{operator.provider_name} is unavailable")

    data = operator.generate_test_data(seed=17)
    framework = OperatorTestFramework(str(args.result_dir))
    device_info_before = collect_device_info(args.device)
    metrics, precondition = run_formal_measurement(
        args,
        operator,
        data,
        framework,
        precision,
        operator.provider_name,
    )
    device_info_after = collect_device_info(args.device)
    row = build_event_row(
        args,
        config,
        operator.provider_name,
        metrics,
        device_info_before,
        device_info_after,
        precondition,
    )
    safe_device = args.device.replace(":", "")
    stem = (
        f"vector_fma_{safe_device}_{args.precision}_{operator.provider_name}_"
        f"p{args.process_index}_n{config.elements}_r{config.fma_depth}_"
        f"a{config.accumulators}_b{config.block_size}_"
        f"g{config.num_programs}_{args.run_id}"
    )
    event_path = args.result_dir / f"{stem}_events.csv"
    event_json_path = args.result_dir / f"{stem}_event.json"
    row_hash = write_event_sidecar(event_json_path, row)
    write_rows(event_path, [row])
    print(f"raw Event row: {event_path}")
    print(f"typed Event JSON: {event_json_path} sha256={row_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
