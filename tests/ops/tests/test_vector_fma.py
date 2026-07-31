#!/usr/bin/env python3
"""Formal Event runner for the cross-platform Vector FMA microbenchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("formal", "probe", "profile"),
        default="formal",
    )
    parser.add_argument("--device", required=True)
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
    parser.add_argument("--precondition-seconds", type=float, default=5.0)
    parser.add_argument(
        "--precondition-launches",
        type=_positive_int,
        default=20,
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("vector_fma_results"),
    )
    parser.add_argument("--profile-manifest", type=Path)
    return parser.parse_args(argv)


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
    if min_seconds < 0:
        raise ValueError("min_seconds must be non-negative")
    if min_launches <= 0:
        raise ValueError("min_launches must be positive")
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
    return metrics, precondition


def build_event_row(
    args: argparse.Namespace,
    config: VectorFmaConfig,
    provider: str,
    metrics: Any,
    device_info: Dict[str, Any],
    precondition: Dict[str, Any],
) -> Dict[str, Any]:
    samples = [float(value) for value in metrics.repeat_samples_ms]
    flops = vector_fma_flops(config)
    latency_ms = float(metrics.avg_time_ms)
    row = {
        "status": "ok",
        "profile_status": "unverified",
        "publishable_peak": False,
        "process_index": args.process_index,
        "process_pid": os.getpid(),
        "host": platform.node(),
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
        "timing_method": "device_event",
        "dispatch_mode": "eager_direct",
        "precondition_seconds": precondition["seconds"],
        "precondition_launches": precondition["launches"],
        "precondition_timed": precondition.get("timed", False),
        **device_info,
    }
    return row


def build_formal_row(
    event_row: Dict[str, Any],
    *,
    profile_status: str,
    tensor_or_cube_used: bool = False,
    spilled: bool = False,
    profile_manifest: Optional[str] = None,
) -> Dict[str, Any]:
    """Promote one raw Event row only after a matching profile gate passes."""
    if profile_status != "verified":
        raise RuntimeError("profile verification is required for publication")
    if tensor_or_cube_used:
        raise RuntimeError("Tensor/Cube execution cannot be published")
    if spilled:
        raise RuntimeError("spilled Vector FMA configuration cannot be published")
    row = dict(event_row)
    row.update(
        {
            "profile_status": "verified",
            "publishable_peak": True,
            "profile_manifest": profile_manifest,
        }
    )
    return row


def _device_index(device: str) -> int:
    return int(device.split(":", 1)[1]) if ":" in device else 0


def _run_state_command(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"
    output = result.stdout.strip() or result.stderr.strip()
    return f"exit={result.returncode}; {output}"


def collect_device_info(device: str) -> Dict[str, Any]:
    index = _device_index(device)
    if device.startswith("cuda"):
        properties = torch.cuda.get_device_properties(index)
        state = _run_state_command(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=name,pstate,clocks.current.sm,"
                "clocks.current.memory,temperature.gpu,power.draw,"
                "memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ]
        )
        return {
            "device_name": properties.name,
            "compute_units": properties.multi_processor_count,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": ".".join(
                str(value) for value in torch.cuda.get_device_capability(index)
            ),
            "device_state_raw": state,
        }
    if device.startswith("npu"):
        import torch_npu

        properties = torch_npu.npu.get_device_properties(index)
        state = _run_state_command(
            ["npu-smi", "info", "-i", str(index), "-c", "0"]
        )
        return {
            "device_name": properties.name,
            "compute_units": properties.vector_core_num,
            "cube_units": properties.cube_core_num,
            "total_memory_bytes": int(properties.total_memory) * 1024 * 1024,
            "device_state_raw": state,
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
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
    metrics, precondition = run_formal_measurement(
        args,
        operator,
        data,
        framework,
        precision,
        operator.provider_name,
    )
    row = build_event_row(
        args,
        config,
        operator.provider_name,
        metrics,
        collect_device_info(args.device),
        precondition,
    )
    safe_device = args.device.replace(":", "")
    stem = (
        f"vector_fma_{safe_device}_{args.precision}_{operator.provider_name}_"
        f"p{args.process_index}_n{config.elements}_r{config.fma_depth}_"
        f"a{config.accumulators}"
    )
    event_path = args.result_dir / f"{stem}_events.csv"
    write_rows(event_path, [row])
    print(f"raw Event row: {event_path}")

    if args.profile_manifest is not None:
        manifest = _load_profile_manifest(args.profile_manifest)
        formal = build_formal_row(
            row,
            profile_status=str(manifest.get("status", "unverified")),
            tensor_or_cube_used=bool(
                manifest.get("tensor_or_cube_used", False)
            ),
            spilled=bool(manifest.get("spilled", False)),
            profile_manifest=str(args.profile_manifest),
        )
        formal_path = args.result_dir / f"{stem}_verified.csv"
        write_rows(formal_path, [formal])
        print(f"verified formal row: {formal_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
