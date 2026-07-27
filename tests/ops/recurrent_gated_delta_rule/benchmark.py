#!/usr/bin/env python3
"""Framework-V2 performance curves for Qwen3.5 recurrent GDN."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

OPS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPS_ROOT))

from operator_test_framework import (  # noqa: E402
    OperatorTestFramework,
    PrecisionType,
)
from recurrent_gated_delta_rule.base import (  # noqa: E402
    HEAD_DIM,
    NUM_KEY_HEADS,
    NUM_VALUE_HEADS,
    TOKEN_COUNTS,
    RecurrentGatedDeltaRuleOperatorTest,
)


DEFAULT_BATCHES = (1, 4, 8, 16, 32, 64, 128)
CSV_FIELDS = (
    "operator",
    "hardware",
    "device",
    "device_name",
    "provider",
    "mode",
    "batch_size",
    "tokens_per_sequence",
    "total_tokens",
    "num_key_heads",
    "num_value_heads",
    "head_dim",
    "query_key_dtype",
    "value_beta_state_dtype",
    "g_dtype",
    "time_ms",
    "recurrent_tokens_per_second",
    "warmup",
    "iterations",
    "repeats",
    "repeat_samples_ms",
    "protocol_version",
    "aggregation",
    "preallocated_invocations_per_repeat",
    "input_reuse_within_repeat",
    "input_storage_sets_verified",
    "input_storage_ptr_count",
    "output_storage_sets_verified",
    "output_storage_ptr_count",
    # Historical aliases retained for existing CSV consumers.
    "independent_storage_sets_verified",
    "storage_ptr_count",
    "output_storage_policy",
    "bytes_per_invocation",
    "timed_region",
    "framework_api",
    "seed",
    "correctness_cosine",
    "correctness_max_abs",
    "correctness_mean_abs",
    "state_correctness_cosine",
    "state_correctness_max_abs",
    "state_correctness_mean_abs",
    "correctness_pass",
    "status",
    "error",
    "point_index",
    "shard_index",
    "num_shards",
)


def parse_csv_strings(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("list must not be empty")
    return result


def parse_csv_ints(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return result


def resolve_device(requested: str) -> tuple[str, str]:
    if requested == "auto":
        try:
            import torch_npu  # noqa: F401

            if torch.npu.is_available():
                requested = "npu:0"
        except (ImportError, AttributeError):
            pass
        if requested == "auto" and torch.cuda.is_available():
            requested = "cuda:0"
    if requested == "cuda":
        requested = "cuda:0"
    if requested == "npu":
        requested = "npu:0"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.set_device(device)
        return requested, torch.cuda.get_device_name(device)
    if device.type == "npu":
        import torch_npu

        torch_npu.npu.set_compile_mode(jit_compile=False)
        torch.npu.set_device(device)
        return requested, torch.npu.get_device_name(device.index or 0)
    raise RuntimeError(f"only CUDA/NPU are supported, got {device}")


def cleanup(device: str) -> None:
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    else:
        torch.npu.synchronize()
        torch.npu.empty_cache()


def environment(device: str, device_name: str, provider: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "device": device,
        "device_name": device_name,
        "provider": provider,
        "framework": "tests/ops/operator_test_framework.py",
        "framework_api": "OperatorTestFramework.run_core_operator_performance_test_v2",
        "ascend_custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
    }
    if device.startswith("cuda"):
        result["cuda"] = torch.version.cuda
        try:
            import vllm

            result["vllm"] = getattr(vllm, "__version__", "unknown")
        except Exception as exc:
            result["vllm"] = f"unavailable: {exc}"
    else:
        import torch_npu

        result["torch_npu"] = torch_npu.__version__
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hardware", default="")
    parser.add_argument("--provider", default="default")
    parser.add_argument("--modes", type=parse_csv_strings, default=["decode", "mtp3"])
    parser.add_argument("--batches", type=parse_csv_ints, default=list(DEFAULT_BATCHES))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path)
    parser.add_argument("--skip-correctness", action="store_true")
    args = parser.parse_args(argv)

    unknown_modes = sorted(set(args.modes).difference(TOKEN_COUNTS))
    if unknown_modes:
        raise ValueError(f"unsupported modes: {unknown_modes}")
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        raise ValueError("warmup >= 0 and iterations/repeats > 0 are required")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("shard index must satisfy 0 <= index < num_shards")

    device, device_name = resolve_device(args.device)
    hardware = device_name if args.hardware in ("", "auto") else args.hardware
    operator = RecurrentGatedDeltaRuleOperatorTest()
    providers = operator.get_formal_implementations(device)
    provider = providers[0] if args.provider == "default" else args.provider
    if provider not in providers:
        raise RuntimeError(
            f"provider {provider!r} is unavailable for {device}; choices={providers}"
        )
    framework = OperatorTestFramework(result_dir=str(args.output.parent))
    framework.register_operator(operator)

    env = environment(device, device_name, provider)
    env["protocol"] = {
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "preallocation": "framework V2 prepares W+I independent data/state sets",
        "repeat_aggregation": "median of R framework-V2 averages",
        "output_policy": (
            "framework-preallocated output/state"
            if provider == "cuda_vllm_fla_direct_out"
            else "provider-managed return; recurrent state is preallocated and mutated in-place"
        ),
    }
    env_path = args.environment_output or args.output.with_suffix(".environment.json")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n")

    checks: dict[str, dict[str, float | bool]] = {}
    if not args.skip_correctness:
        for mode in args.modes:
            checks[mode] = operator.correctness(mode, device, provider)
            print("correctness", mode, json.dumps(checks[mode]), flush=True)
            if not checks[mode]["passed"]:
                raise RuntimeError(f"correctness failed for {mode}: {checks[mode]}")
            cleanup(device)

    points = [(mode, batch) for mode in args.modes for batch in args.batches]
    rows: list[dict[str, Any]] = []
    for point_index, (mode, batch_size) in enumerate(points):
        if point_index % args.num_shards != args.shard_index:
            continue
        tokens_per_sequence = TOKEN_COUNTS[mode]
        total_tokens = batch_size * tokens_per_sequence
        check = checks.get(mode, {})
        row: dict[str, Any] = {
            "operator": operator.operator_name,
            "hardware": hardware,
            "device": device,
            "device_name": device_name,
            "provider": provider,
            "mode": mode,
            "batch_size": batch_size,
            "tokens_per_sequence": tokens_per_sequence,
            "total_tokens": total_tokens,
            "num_key_heads": NUM_KEY_HEADS,
            "num_value_heads": NUM_VALUE_HEADS,
            "head_dim": HEAD_DIM,
            "query_key_dtype": "BF16",
            "value_beta_state_dtype": "BF16",
            "g_dtype": "FP32",
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "seed": "",
            "correctness_cosine": check.get("cosine", ""),
            "correctness_max_abs": check.get("max_abs", ""),
            "correctness_mean_abs": check.get("mean_abs", ""),
            "state_correctness_cosine": check.get("state_cosine", ""),
            "state_correctness_max_abs": check.get("state_max_abs", ""),
            "state_correctness_mean_abs": check.get("state_mean_abs", ""),
            "correctness_pass": check.get("passed", ""),
            "status": "pending",
            "error": "",
            "point_index": point_index,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        }
        try:
            data = operator.generate_test_data(
                mode=mode,
                batch_size=batch_size,
            )
            row["seed"] = data["metadata"]["seed"]
            metrics = framework.run_core_operator_performance_test_v2(
                operator_test=operator,
                data=data,
                device=device,
                precision=PrecisionType.BF16,
                implementation=provider,
                num_warmup=args.warmup,
                num_iterations=args.iterations,
                num_repeats=args.repeats,
                retain_outputs=True,
                verify_independent_storage=True,
            )
            provenance = framework.performance_provenance(metrics)
            median_ms = float(metrics.avg_time_ms)
            recurrent_tokens_per_second = (
                float(metrics.throughput)
                if metrics.throughput is not None
                else total_tokens * 1000.0 / median_ms
            )
            row.update(
                provenance,
                time_ms=median_ms,
                recurrent_tokens_per_second=recurrent_tokens_per_second,
                independent_storage_sets_verified=provenance[
                    "input_storage_sets_verified"
                ],
                storage_ptr_count=provenance[
                    "input_storage_ptr_count"
                ],
                bytes_per_invocation="",
                status="ok",
            )
            print(
                f"{hardware} {mode} B={batch_size}: {median_ms:.6f} ms, "
                f"{row['recurrent_tokens_per_second']:.0f} recurrent-token/s",
                flush=True,
            )
        except Exception as exc:
            row.update(
                time_ms="",
                recurrent_tokens_per_second="",
                repeat_samples_ms="",
                protocol_version="",
                aggregation="",
                preallocated_invocations_per_repeat="",
                input_reuse_within_repeat="",
                input_storage_sets_verified="",
                input_storage_ptr_count="",
                output_storage_sets_verified="",
                output_storage_ptr_count="",
                independent_storage_sets_verified="",
                storage_ptr_count="",
                output_storage_policy="",
                bytes_per_invocation="",
                timed_region="",
                framework_api="",
                status="error",
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )
            print(row["error"], file=sys.stderr, flush=True)
        rows.append(row)
        write_csv(args.output, rows)

    if not rows or any(row["status"] != "ok" for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
