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
    FRESH_ITERATION_PLAN_FIELDS,
    PERFORMANCE_PROVENANCE_FIELDS,
    OperatorTestFramework,
    PrecisionType,
    build_fresh_iteration_plan,
    build_curve_selection_provenance,
    finalize_curve_coverage,
)
from recurrent_gated_delta_rule.base import (  # noqa: E402
    HEAD_DIM,
    NUM_KEY_HEADS,
    NUM_VALUE_HEADS,
    TOKEN_COUNTS,
    RecurrentGatedDeltaRuleOperatorTest,
)


DEFAULT_BATCHES = (1, 4, 8, 16, 32, 64, 128)
RECURRENT_BASE_ITERATIONS = 20
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
    *FRESH_ITERATION_PLAN_FIELDS,
    *PERFORMANCE_PROVENANCE_FIELDS,
    # Historical aliases retained for existing CSV consumers.
    "independent_storage_sets_verified",
    "storage_ptr_count",
    "bytes_per_invocation",
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
    "selection_mode",
    "shape_matrix_source",
    "coverage_mode",
    "coverage_total_formal_points",
    "coverage_total_requested_points",
    "coverage_selected_points",
    "selection_covers_full_formal_matrix",
    "coverage_complete",
)


def canonical_recurrent_bytes_per_invocation(
    mode: str,
    batch_size: int,
) -> int:
    """Cross-provider peak-retained bytes for one recurrent invocation."""
    tokens_per_sequence = TOKEN_COUNTS[mode]
    total_tokens = batch_size * tokens_per_sequence
    return (
        534_628 * total_tokens
        + 524_288
        + 4 * (batch_size + 1)
        + (4 * batch_size if tokens_per_sequence > 1 else 0)
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
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=(
            "fixed measured iterations; omitted selects deterministic "
            "fresh-storage adaptive iterations"
        ),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    unknown_modes = sorted(set(args.modes).difference(TOKEN_COUNTS))
    if unknown_modes:
        raise ValueError(f"unsupported modes: {unknown_modes}")
    if (
        args.warmup < 0
        or (args.iterations is not None and args.iterations <= 0)
        or args.repeats <= 0
    ):
        raise ValueError(
            "warmup >= 0, optional iterations > 0, and repeats > 0 "
            "are required"
        )
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
        "requested_iterations": (
            args.iterations if args.iterations is not None else "auto"
        ),
        "base_iterations": RECURRENT_BASE_ITERATIONS,
        "iteration_selection": (
            "deterministic cross-provider fresh-storage adaptive"
            if args.iterations is None else "explicit fixed"
        ),
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
    correctness_errors: dict[str, str] = {}
    if not args.skip_correctness:
        for mode in args.modes:
            try:
                check = operator.correctness(mode, device, provider)
                checks[mode] = check
                print("correctness", mode, json.dumps(check), flush=True)
                if not check["passed"]:
                    correctness_errors[mode] = (
                        f"correctness failed for {mode}: {check}"
                    )
            except Exception as exc:
                correctness_errors[mode] = (
                    f"correctness for {mode} raised "
                    f"{type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
            try:
                cleanup(device)
            except Exception as exc:
                cleanup_error = (
                    f"correctness cleanup for {mode} raised "
                    f"{type(exc).__name__}: {exc}"
                )
                if mode in correctness_errors:
                    correctness_errors[mode] += f"\n{cleanup_error}"
                else:
                    correctness_errors[mode] = cleanup_error

    def mark_error(row: dict[str, Any], error: str) -> None:
        row.update(
            time_ms="",
            recurrent_tokens_per_second="",
            repeat_samples_ms="",
            repeat_min_ms="",
            repeat_median_ms="",
            repeat_max_ms="",
            repeat_spread_pct="",
            protocol_version="",
            aggregation="",
            preallocated_invocations_per_repeat="",
            input_reuse_within_repeat="",
            input_storage_sets_verified="",
            input_storage_ptr_count="",
            output_storage_sets_verified="",
            output_storage_ptr_count="",
            preallocated_output_aliases_verified="",
            preallocated_output_sets_verified="",
            output_tensors_per_set="",
            output_allocation_mode="",
            output_allocation_policy="",
            independent_storage_sets_verified="",
            storage_ptr_count="",
            output_storage_policy="",
            bytes_per_invocation="",
            timed_region="",
            framework_api="",
            status="error",
            error=error,
        )

    points = [(mode, batch) for mode in args.modes for batch in args.batches]
    indexed_points = list(enumerate(points))
    selected_points = [
        (point_index, mode, batch_size)
        for point_index, (mode, batch_size) in indexed_points
        if (
            not args.quick
            or point_index % len(args.batches) == 0
        )
        and point_index % args.num_shards == args.shard_index
    ]
    coverage_total_by_mode = {
        mode: sum(
            candidate_mode == mode
            for _, (candidate_mode, _) in indexed_points
        )
        for mode in args.modes
    }
    coverage_selected_by_mode = {
        mode: sum(
            candidate_mode == mode
            for _, candidate_mode, _ in selected_points
        )
        for mode in args.modes
    }
    canonical_modes = ["decode", "mtp3"]
    canonical_batches = list(DEFAULT_BATCHES)
    selection_by_mode = {
        mode: build_curve_selection_provenance(
            quick=args.quick,
            num_shards=args.num_shards,
            total_formal_points=len(canonical_batches),
            total_requested_points=coverage_total_by_mode[mode],
            selected_points=coverage_selected_by_mode[mode],
            uses_formal_shape_matrix=(
                args.modes == canonical_modes
                and args.batches == canonical_batches
            ),
        )
        for mode in args.modes
    }
    rows: list[dict[str, Any]] = []
    for point_index, mode, batch_size in selected_points:
        tokens_per_sequence = TOKEN_COUNTS[mode]
        total_tokens = batch_size * tokens_per_sequence
        iteration_plan = build_fresh_iteration_plan(
            num_warmup=args.warmup,
            requested_iterations=args.iterations,
            base_iterations=RECURRENT_BASE_ITERATIONS,
            estimated_unique_bytes_per_invocation=(
                canonical_recurrent_bytes_per_invocation(
                    mode,
                    batch_size,
                )
            ),
        )
        effective_iterations = int(
            iteration_plan["effective_iterations"]
        )
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
            "iterations": effective_iterations,
            "repeats": args.repeats,
            **iteration_plan,
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
            **selection_by_mode[mode],
        }
        if mode in correctness_errors:
            mark_error(row, correctness_errors[mode])
            print(row["error"], file=sys.stderr, flush=True)
            rows.append(row)
            write_csv(args.output, rows)
            continue
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
                num_iterations=effective_iterations,
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
            mark_error(
                row,
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )
            print(row["error"], file=sys.stderr, flush=True)
        rows.append(row)
        write_csv(args.output, rows)

    any_complete = False
    for mode in args.modes:
        any_complete = finalize_curve_coverage(
            [row for row in rows if row["mode"] == mode]
        ) or any_complete
    if any_complete:
        write_csv(args.output, rows)

    if not rows or any(row["status"] != "ok" for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
