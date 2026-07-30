"""Formal fused RMSNorm plus activation-quantization benchmark entry."""

import argparse
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import uuid

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from norm_quant import (  # noqa: E402
    NormQuantVariant,
    create_norm_quant_operator,
)
from operator_test_framework import (  # noqa: E402
    FRESH_ITERATION_PLAN_FIELDS,
    PERFORMANCE_PROVENANCE_FIELDS,
    GraphCaptureUnsupportedError,
    OperatorTestFramework,
    PrecisionType,
    build_curve_selection_provenance,
    build_memory_bounded_fresh_invocation_plan,
    finalize_curve_coverage,
)


FORMAL_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
FORMAL_HIDDEN_SIZES = [4096, 7168, 8192]
TOKEN_SWEEP_HIDDEN = 7168
HIDDEN_SWEEP_TOKENS = 128
PROFILE_TOKENS = [1, 128, 4096]
PROFILE_HIDDEN = 7168
NORM_QUANT_BASE_ITERATIONS = 20
NORM_QUANT_FRESH_STORAGE_HARD_LIMIT_BYTES = 40 * 1024**3
EVENT_WINDOW_TARGET_MIN_MS = 20.0
EVENT_WINDOW_TARGET_MS = 30.0
EVENT_WINDOW_TARGET_MAX_MS = 50.0
H20_DEVICE_MODEL = "NVIDIA H20-3e"
TERMINAL_STATUSES = frozenset({
    "ok",
    "unsupported",
    "unsupported_graph_capture",
    "error",
})
SUCCESS_STATUSES = frozenset({"ok"})
DISPATCH_MODES = {
    "eager": ("eager_direct",),
    "graph": ("captured_chain",),
    "both": ("eager_direct", "captured_chain"),
}
PRECISION_TYPES = {
    "fp8": PrecisionType.FP8,
    "mxfp8": PrecisionType.MXFP8,
    "mxfp4": PrecisionType.MXFP4,
}
FP8_VARIANTS = (
    NormQuantVariant.RMS_NORM_STATIC_FP8,
    NormQuantVariant.ADD_RMS_NORM_STATIC_FP8,
    NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8,
)
MX_VARIANTS = (
    NormQuantVariant.RMS_NORM_DYNAMIC_MX,
    NormQuantVariant.ADD_RMS_NORM_DYNAMIC_MX,
)

SELECTION_FIELDS = (
    "point_index",
    "shard_index",
    "num_shards",
    "selection_mode",
    "shape_matrix_source",
    "coverage_mode",
    "coverage_total_formal_points",
    "coverage_selected_points",
    "coverage_total_requested_points",
    "selection_covers_full_formal_matrix",
    "coverage_complete",
)
WINDOW_PLAN_FIELDS = (
    "event_window_target_min_ms",
    "event_window_target_ms",
    "event_window_target_max_ms",
    "calibration_iterations",
    "calibration_samples_ms",
    "calibration_is_diagnostic",
    "calibrated_iterations",
    "iteration_capacity",
    "window_target_capacity_limited",
    "window_target_feasible",
    "predicted_event_window_min_ms",
    "predicted_event_window_max_ms",
)
NORM_QUANT_FIELDS = (
    *SELECTION_FIELDS,
    "sweep",
    "variant",
    "native_op",
    "precision",
    "device",
    "device_model",
    "provider",
    "tokens",
    "hidden",
    "seed",
    "logical_bytes",
    "physical_bytes",
    "residual_semantics",
    "latency_ms",
    "effective_bandwidth_gb_s",
    "effective_bandwidth_semantics",
    "physical_bandwidth_gb_s",
    "status",
    "error",
    "capability_status",
    "graph_status",
    "torch_version",
    "torch_npu_version",
    "vllm_version",
    "flashinfer_version",
    *FRESH_ITERATION_PLAN_FIELDS,
    *WINDOW_PLAN_FIELDS,
    *PERFORMANCE_PROVENANCE_FIELDS,
)


def _safe_component(value: Any) -> str:
    component = re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower())
    return component.strip("-") or "unknown"


def _new_run_nonce() -> str:
    return f"{time.time_ns()}-p{os.getpid()}-{uuid.uuid4().hex[:12]}"


def build_artifact_identity(
    *,
    result_dir: Path,
    kind: str,
    precision: str,
    device: str,
    device_model: str,
    variant: str,
    provider: str,
    dispatch_mode: str,
    shard_index: int,
    num_shards: int,
    suffix: str,
    run_nonce: Optional[str] = None,
) -> Path:
    """Build a collision-safe path containing every formal identity axis."""
    nonce = run_nonce or _new_run_nonce()
    identity = "__".join([
        _safe_component(kind),
        _safe_component(precision),
        _safe_component(device),
        _safe_component(device_model),
        _safe_component(variant),
        _safe_component(provider),
        _safe_component(dispatch_mode),
        f"shard-{shard_index}-of-{num_shards}",
        _safe_component(nonce),
    ])
    return Path(result_dir) / f"{identity}{suffix}"


def _atomic_write_csv(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    fieldnames: Sequence[str] = NORM_QUANT_FIELDS,
) -> None:
    """Rewrite one checkpoint atomically so an interrupted run is resumable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(
            file_descriptor,
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def validate_formal_latency_rows(rows: Iterable[Dict[str, Any]]) -> None:
    """Reject diagnostic or non-numeric values from formal latency data."""
    for row in rows:
        if row.get("status") not in SUCCESS_STATUSES:
            continue
        diagnostic = row.get("profiler_is_diagnostic")
        if diagnostic is True or str(diagnostic).lower() == "true":
            raise ValueError("diagnostic profiler latency is not formal latency")
        latency = row.get("latency_ms")
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or float(latency) <= 0
        ):
            raise ValueError(f"invalid formal latency: {latency!r}")


def select_h20_best_envelope(
    rows: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Select the lowest successful provider latency for each H20 point."""
    candidates = [dict(row) for row in rows if row.get("status") == "ok"]
    validate_formal_latency_rows(candidates)
    non_h20_models = sorted({
        str(row.get("device_model", ""))
        for row in candidates
        if row.get("device_model") != H20_DEVICE_MODEL
    })
    if non_h20_models:
        raise ValueError(
            f"H20 envelope requires device_model={H20_DEVICE_MODEL!r}; "
            f"got {non_h20_models}"
        )
    best: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in candidates:
        key = (
            row.get("sweep", "tokens"),
            row["variant"],
            row["precision"],
            row["dispatch_mode"],
            int(row["tokens"]),
            int(row["hidden"]),
        )
        previous = best.get(key)
        if previous is None or float(row["latency_ms"]) < float(
            previous["latency_ms"]
        ):
            winner = dict(row)
            winner["winning_provider"] = row["provider"]
            best[key] = winner
    return [best[key] for key in sorted(best, key=str)]


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _resolve_device(device: str) -> str:
    if device == "cuda":
        return "cuda:0"
    if device == "npu":
        return "npu:0"
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    try:
        import torch_npu

        npu_api = getattr(torch, "npu", getattr(torch_npu, "npu", None))
        if npu_api is not None and npu_api.is_available():
            return "npu:0"
    except (ImportError, OSError, RuntimeError):
        pass
    raise RuntimeError("NormQuant formal entry requires CUDA or NPU")


def _device_model(device: str) -> str:
    try:
        if device.startswith("cuda"):
            return str(torch.cuda.get_device_name(device))
        if device.startswith("npu"):
            import torch_npu

            index = int(device.split(":", maxsplit=1)[1]) if ":" in device else 0
            return str(torch_npu.npu.get_device_name(index))
    except (
        AttributeError,
        AssertionError,
        ImportError,
        IndexError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        pass
    return f"unavailable-{device}"


def _torch_npu_version() -> str:
    try:
        import torch_npu

        return str(getattr(torch_npu, "__version__", "unknown"))
    except (ImportError, OSError, RuntimeError):
        return "not-installed"


def _validate_positive_ints(values: Sequence[int], label: str) -> List[int]:
    normalized = list(values)
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        for value in normalized
    ):
        raise ValueError(f"{label} must contain only positive integers")
    return normalized


def estimate_norm_quant_peak_retained_bytes(
    *,
    device: str,
    variant: NormQuantVariant,
    precision: PrecisionType,
    tokens: int,
    hidden: int,
) -> int:
    """Count peak retained device bytes for one complete prepared payload."""
    _validate_positive_ints([tokens], "tokens")
    _validate_positive_ints([hidden], "hidden")
    matrix_elements = tokens * hidden
    if device.startswith("cuda"):
        if precision is not PrecisionType.FP8:
            raise ValueError("CUDA retained-byte estimate supports only FP8")
        if variant is NormQuantVariant.RMS_NORM_STATIC_FP8:
            # x BF16 + weight BF16 + static scale FP32 + output FP8.
            return 3 * matrix_elements + 2 * hidden + 4
        if variant is NormQuantVariant.ADD_RMS_NORM_STATIC_FP8:
            # x/residual/residual-seed BF16 + weight + scale + FP8 output.
            return 7 * matrix_elements + 2 * hidden + 4
        if variant is NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8:
            # Static contract above without scale, plus FP32 token scales.
            return 7 * matrix_elements + 2 * hidden + 4 * tokens
        raise ValueError(f"unsupported CUDA NormQuant variant {variant.value}")
    if not device.startswith("npu"):
        raise ValueError(f"unsupported NormQuant device {device!r}")
    if variant is NormQuantVariant.RMS_NORM_STATIC_FP8:
        # x + (weight,beta,scale,offset) BF16 + allocating FP8 output.
        return 3 * matrix_elements + 8 * hidden
    if variant is NormQuantVariant.ADD_RMS_NORM_STATIC_FP8:
        # Inputs include residual seed; full output tuple includes BF16 x_out.
        return 9 * matrix_elements + 6 * hidden
    if variant is NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8:
        # Full five-output tuple includes FP8 y, BF16 x_out, FP32 scale.
        return 9 * matrix_elements + 2 * hidden + 4 * tokens
    if variant not in (
        NormQuantVariant.RMS_NORM_DYNAMIC_MX,
        NormQuantVariant.ADD_RMS_NORM_DYNAMIC_MX,
    ):
        raise ValueError(f"unsupported NPU NormQuant variant {variant.value}")
    if hidden % 64 != 0:
        raise ValueError("MX NormQuant hidden must be divisible by 64")
    quantized_bytes = (
        matrix_elements
        if precision is PrecisionType.MXFP8
        else matrix_elements // 2
    )
    pair_packed_scale_bytes = tokens * ((hidden + 63) // 64) * 2
    if variant is NormQuantVariant.RMS_NORM_DYNAMIC_MX:
        # x/weight plus primary, E8M0 scale, and zero-byte rstd.
        return (
            2 * matrix_elements
            + 2 * hidden
            + quantized_bytes
            + pair_packed_scale_bytes
        )
    # x/residual/residual-seed + weight, plus primary/scale/x_out/rstd.
    return (
        8 * matrix_elements
        + 2 * hidden
        + quantized_bytes
        + pair_packed_scale_bytes
    )


def build_norm_quant_window_plan(
    *,
    requested_warmup: int,
    requested_iterations: Optional[int],
    estimated_unique_bytes_per_invocation: int,
    fresh_storage_hard_limit_bytes: int,
    calibration_samples_ms: Dict[str, float],
    calibration_iterations: int,
) -> Dict[str, Any]:
    """Choose one memory-bounded W/I protocol shared by every formal cell."""
    if requested_warmup < 2:
        raise ValueError("NormQuant warmup must be >= 2")
    if requested_iterations is not None and requested_iterations <= 0:
        raise ValueError("requested iterations must be positive")
    if estimated_unique_bytes_per_invocation <= 0:
        raise ValueError("retained bytes per invocation must be positive")
    if fresh_storage_hard_limit_bytes <= 0:
        raise ValueError("fresh storage hard limit must be positive")
    if calibration_iterations <= 0:
        raise ValueError("calibration iterations must be positive")
    samples = {
        str(key): float(value)
        for key, value in calibration_samples_ms.items()
    }
    if any(
        not math.isfinite(value) or value <= 0
        for value in samples.values()
    ):
        raise ValueError("calibration samples must be positive and finite")
    if requested_iterations is None and not samples:
        raise ValueError("auto iteration planning requires calibration samples")

    capacity = (
        fresh_storage_hard_limit_bytes
        // estimated_unique_bytes_per_invocation
    )
    if capacity < 3:
        raise ValueError(
            "fresh-storage hard limit cannot satisfy minimum W2/I1"
        )
    effective_warmup = min(requested_warmup, capacity - 1)
    effective_warmup = max(2, effective_warmup)
    iteration_capacity = capacity - effective_warmup

    feasible = True
    if requested_iterations is not None:
        calibrated_iterations = requested_iterations
    else:
        latencies = list(samples.values())
        lower = max(
            math.ceil(EVENT_WINDOW_TARGET_MIN_MS / latency)
            for latency in latencies
        )
        upper = min(
            max(1, math.floor(EVENT_WINDOW_TARGET_MAX_MS / latency))
            for latency in latencies
        )
        feasible = lower <= upper
        target_from_fastest = max(
            1,
            round(EVENT_WINDOW_TARGET_MS / min(latencies)),
        )
        if feasible:
            calibrated_iterations = min(
                max(target_from_fastest, lower),
                upper,
            )
        else:
            calibrated_iterations = target_from_fastest
    effective_iterations = min(calibrated_iterations, iteration_capacity)
    capacity_limited = (
        effective_iterations < calibrated_iterations
        or effective_warmup < requested_warmup
    )
    framework_plan = build_memory_bounded_fresh_invocation_plan(
        requested_warmup=effective_warmup,
        requested_iterations=effective_iterations,
        base_iterations=NORM_QUANT_BASE_ITERATIONS,
        estimated_unique_bytes_per_invocation=(
            estimated_unique_bytes_per_invocation
        ),
        fresh_storage_hard_limit_bytes=fresh_storage_hard_limit_bytes,
        minimum_warmup=2,
        minimum_iterations=1,
    )
    framework_plan.update(
        iteration_selection_policy=(
            "explicit_fixed_hard_cap"
            if requested_iterations is not None
            else "diagnostic_calibrated_event_window_hard_cap"
        ),
        requested_iterations=(
            requested_iterations
            if requested_iterations is not None
            else "auto"
        ),
        requested_warmup=requested_warmup,
        event_window_target_min_ms=EVENT_WINDOW_TARGET_MIN_MS,
        event_window_target_ms=EVENT_WINDOW_TARGET_MS,
        event_window_target_max_ms=EVENT_WINDOW_TARGET_MAX_MS,
        calibration_iterations=calibration_iterations,
        calibration_samples_ms=json.dumps(samples, sort_keys=True),
        calibration_is_diagnostic=(requested_iterations is None),
        calibrated_iterations=calibrated_iterations,
        iteration_capacity=iteration_capacity,
        window_target_capacity_limited=capacity_limited,
        window_target_feasible=feasible,
        predicted_event_window_min_ms=(
            min(samples.values()) * effective_iterations if samples else ""
        ),
        predicted_event_window_max_ms=(
            max(samples.values()) * effective_iterations if samples else ""
        ),
    )
    return framework_plan


def build_norm_quant_shape_points(
    *,
    tokens: Sequence[int],
    hidden_sizes: Sequence[int],
    quick: bool,
    shard_index: int,
    num_shards: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build the exact token and hidden sweeps with stable global indices."""
    if (
        not isinstance(num_shards, int)
        or isinstance(num_shards, bool)
        or num_shards <= 0
    ):
        raise ValueError("num_shards must be a positive integer")
    if (
        not isinstance(shard_index, int)
        or isinstance(shard_index, bool)
        or not 0 <= shard_index < num_shards
    ):
        raise ValueError("shard_index must satisfy 0 <= index < num_shards")
    token_values = _validate_positive_ints(tokens, "tokens")
    hidden_values = _validate_positive_ints(hidden_sizes, "hidden_sizes")
    requested: List[Dict[str, Any]] = []
    for value in token_values:
        requested.append({
            "sweep": "tokens",
            "tokens": value,
            "hidden": TOKEN_SWEEP_HIDDEN,
        })
    for value in hidden_values:
        requested.append({
            "sweep": "hidden",
            "tokens": HIDDEN_SWEEP_TOKENS,
            "hidden": value,
        })
    if not requested:
        raise ValueError("at least one token or hidden-size point is required")
    for point_index, point in enumerate(requested):
        point["point_index"] = point_index

    selected = requested
    if quick:
        selected = []
        for sweep in ("tokens", "hidden"):
            first = next(
                (point for point in requested if point["sweep"] == sweep),
                None,
            )
            if first is not None:
                selected.append(first)
    selected = [
        point
        for point in selected
        if point["point_index"] % num_shards == shard_index
    ]
    if not selected:
        raise ValueError(
            "selected zero NormQuant points; choose a shard containing at "
            "least one requested point"
        )
    formal_requested = (
        token_values == FORMAL_TOKENS
        and hidden_values == FORMAL_HIDDEN_SIZES
    )
    selection = build_curve_selection_provenance(
        quick=quick,
        num_shards=num_shards,
        total_formal_points=(
            len(FORMAL_TOKENS) + len(FORMAL_HIDDEN_SIZES)
        ),
        total_requested_points=len(requested),
        selected_points=len(selected),
        uses_formal_shape_matrix=formal_requested,
    )
    return selected, selection


class NormQuantTestSuite:
    """Own formal correctness, Event curves, checkpoints, and diagnostics."""

    def __init__(
        self,
        *,
        precision: str = "fp8",
        device: str = "auto",
        operator_factory: Callable[
            [str, NormQuantVariant, PrecisionType], Any
        ] = create_norm_quant_operator,
    ):
        normalized_precision = precision.lower()
        if normalized_precision not in PRECISION_TYPES:
            raise ValueError(
                f"unsupported NormQuant precision: {normalized_precision}"
            )
        self.precision_name = normalized_precision
        self.precision = PRECISION_TYPES[normalized_precision]
        self.device = _resolve_device(device)
        if self.precision in (PrecisionType.MXFP8, PrecisionType.MXFP4):
            if not self.device.startswith("npu"):
                raise ValueError(
                    f"{normalized_precision.upper()} NormQuant requires NPU"
                )
        if self.device.startswith("cuda") and self.precision is not PrecisionType.FP8:
            raise ValueError("CUDA NormQuant supports only FP8")
        self.operator_factory = operator_factory
        self.framework: Optional[OperatorTestFramework] = None
        self.run_nonce = _new_run_nonce()

    @property
    def default_variants(self) -> Tuple[NormQuantVariant, ...]:
        return (
            FP8_VARIANTS
            if self.precision is PrecisionType.FP8
            else MX_VARIANTS
        )

    def _require_framework(self) -> OperatorTestFramework:
        if self.framework is None:
            raise RuntimeError("NormQuant suite requires a configured framework")
        return self.framework

    def _normalize_variants(
        self,
        variants: Optional[Sequence[Any]],
    ) -> List[NormQuantVariant]:
        selected = list(variants) if variants is not None else list(
            self.default_variants
        )
        normalized: List[NormQuantVariant] = []
        for value in selected:
            variant = (
                value
                if isinstance(value, NormQuantVariant)
                else NormQuantVariant(value)
            )
            if self.precision is PrecisionType.FP8 and variant not in FP8_VARIANTS:
                raise ValueError(
                    f"{variant.value} is not an FP8 NormQuant variant"
                )
            if self.precision in (
                PrecisionType.MXFP8,
                PrecisionType.MXFP4,
            ) and variant not in MX_VARIANTS:
                raise ValueError(
                    f"{variant.value} is not an MX NormQuant variant"
                )
            normalized.append(variant)
        if not normalized:
            raise ValueError("at least one NormQuant variant is required")
        return normalized

    @staticmethod
    def _declared_providers(operator: Any) -> List[str]:
        getter = getattr(operator, "get_declared_implementations", None)
        if callable(getter):
            declared = list(getter())
            if declared:
                return declared
        providers = getattr(operator, "_PROVIDERS", {})
        if isinstance(providers, dict):
            variant = getattr(operator, "variant", None)
            precision = getattr(operator, "precision", None)
            value = providers.get((variant, precision), providers.get(variant))
            if isinstance(value, str):
                return [value]
            if value:
                return list(value)
        return ["unresolved_native_provider"]

    @staticmethod
    def _native_op(operator: Any, provider: str) -> str:
        native_ops = getattr(operator, "_NATIVE_OPS", {})
        if provider in native_ops:
            return str(native_ops[provider])
        vllm_symbols = getattr(operator, "_VLLM_SYMBOLS", {})
        if provider in vllm_symbols:
            return f"torch.ops._C.{vllm_symbols[provider]}"
        flashinfer_symbols = getattr(operator, "_FLASHINFER_SYMBOLS", {})
        if provider in flashinfer_symbols:
            return f"flashinfer.{flashinfer_symbols[provider]}"
        return provider

    @staticmethod
    def _residual_semantics(variant: NormQuantVariant) -> str:
        if variant in (
            NormQuantVariant.ADD_RMS_NORM_STATIC_FP8,
            NormQuantVariant.ADD_RMS_NORM_DYNAMIC_FP8,
            NormQuantVariant.ADD_RMS_NORM_DYNAMIC_MX,
        ):
            return (
                "BF16 x1+x2 written to mutable residual/x_out; immutable "
                "per-payload residual seed restored outside timing"
            )
        return "no residual input or residual output"

    def _artifact_path(
        self,
        *,
        kind: str,
        variant: NormQuantVariant,
        provider: str,
        dispatch_mode: str,
        shard_index: int,
        num_shards: int,
        suffix: str,
    ) -> Path:
        framework = self._require_framework()
        return build_artifact_identity(
            result_dir=Path(framework.result_dir),
            kind=kind,
            precision=self.precision_name,
            device=self.device,
            device_model=_device_model(self.device),
            variant=variant.value,
            provider=provider,
            dispatch_mode=dispatch_mode,
            shard_index=shard_index,
            num_shards=num_shards,
            suffix=suffix,
            run_nonce=self.run_nonce,
        )

    def _base_row(
        self,
        *,
        point: Dict[str, Any],
        selection: Dict[str, Any],
        variant: NormQuantVariant,
        provider: str,
        dispatch_mode: str,
        operator: Any,
        data: Dict[str, Any],
        iteration_plan: Dict[str, Any],
        shard_index: int,
        num_shards: int,
    ) -> Dict[str, Any]:
        row = {field: "" for field in NORM_QUANT_FIELDS}
        row.update(
            point_index=point["point_index"],
            shard_index=shard_index,
            num_shards=num_shards,
            **selection,
            sweep=point["sweep"],
            variant=variant.value,
            native_op=self._native_op(operator, provider),
            precision=self.precision.name,
            device=self.device,
            device_model=_device_model(self.device),
            provider=provider,
            dispatch_mode=dispatch_mode,
            tokens=point["tokens"],
            hidden=point["hidden"],
            seed=0,
            logical_bytes=operator.logical_bytes(data),
            physical_bytes=operator.physical_bytes(data),
            residual_semantics=self._residual_semantics(variant),
            status="pending",
            capability_status="supported",
            graph_status=(
                "not_applicable"
                if dispatch_mode == "eager_direct"
                else "pending"
            ),
            torch_version=torch.__version__,
            torch_npu_version=_torch_npu_version(),
            vllm_version=_package_version("vllm"),
            flashinfer_version=_package_version("flashinfer-python"),
            task_queue_enable=os.environ.get("TASK_QUEUE_ENABLE", "unset"),
            **iteration_plan,
        )
        return row

    @staticmethod
    def _verify_performance_provenance(
        provenance: Dict[str, Any],
        metrics: Any,
        *,
        dispatch_mode: str,
        num_warmup: int,
        num_iterations: int,
        num_repeats: int,
        num_stabilization_repeats: int,
    ) -> None:
        def require_equal(field: str, actual: Any, expected: Any) -> None:
            if actual != expected:
                raise RuntimeError(
                    f"{field} mismatch: {actual!r} != {expected!r}"
                )

        def require_metric(
            field: str,
            attribute: str,
            expected: Any,
        ) -> None:
            if not hasattr(metrics, attribute):
                raise RuntimeError(
                    f"metrics.{attribute} is missing for {field}"
                )
            require_equal(
                f"metrics.{attribute}",
                getattr(metrics, attribute),
                expected,
            )

        def require_close(field: str, actual: Any, expected: float) -> None:
            if (
                isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or not math.isfinite(float(actual))
                or not math.isclose(
                    float(actual),
                    float(expected),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise RuntimeError(
                    f"{field} mismatch: {actual!r} != {expected!r}"
                )

        def parse_samples(field: str, count: int) -> List[float]:
            raw = provenance[field]
            try:
                values = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, ValueError) as error:
                raise RuntimeError(f"{field} is not valid JSON") from error
            if not isinstance(values, list) or len(values) != count:
                raise RuntimeError(
                    f"{field} must contain exactly {count} samples"
                )
            samples: List[float] = []
            for value in values:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0
                ):
                    raise RuntimeError(
                        f"{field} contains invalid sample {value!r}"
                    )
                samples.append(float(value))
            return samples

        def require_sample_match(
            field: str,
            actual: Any,
            expected: Sequence[float],
        ) -> None:
            if not isinstance(actual, (tuple, list)):
                raise RuntimeError(f"{field} must be a sample sequence")
            if len(actual) != len(expected):
                raise RuntimeError(
                    f"{field} must contain exactly {len(expected)} samples"
                )
            for index, (actual_value, expected_value) in enumerate(
                zip(actual, expected)
            ):
                require_close(
                    f"{field}[{index}]",
                    actual_value,
                    expected_value,
                )

        missing = [
            field
            for field in PERFORMANCE_PROVENANCE_FIELDS
            if field not in provenance
        ]
        if missing:
            raise RuntimeError(
                f"Framework V2 provenance missing fields: {missing}"
            )
        require_equal(
            "framework_api",
            provenance["framework_api"],
            "OperatorTestFramework.run_core_operator_performance_test_v2",
        )
        require_metric(
            "framework_api",
            "framework_api",
            provenance["framework_api"],
        )
        require_equal(
            "protocol_version",
            provenance["protocol_version"],
            "operator-test-framework-v2-fresh-v6",
        )
        require_metric(
            "protocol_version",
            "protocol_version",
            provenance["protocol_version"],
        )
        requested_counts = {
            "warmup": (num_warmup, "warmup_iterations"),
            "iterations": (num_iterations, "iterations"),
            "repeats": (num_repeats, "repeats"),
            "stabilization_repeats": (
                num_stabilization_repeats,
                "stabilization_repeats",
            ),
        }
        for field, (expected, attribute) in requested_counts.items():
            require_equal(field, provenance[field], expected)
            require_metric(field, attribute, expected)

        repeat_samples = parse_samples("repeat_samples_ms", num_repeats)
        stabilization_samples = parse_samples(
            "stabilization_repeat_samples_ms",
            num_stabilization_repeats,
        )
        require_sample_match(
            "metrics.repeat_samples_ms",
            getattr(metrics, "repeat_samples_ms", None),
            repeat_samples,
        )
        require_sample_match(
            "metrics.stabilization_repeat_samples_ms",
            getattr(metrics, "stabilization_repeat_samples_ms", None),
            stabilization_samples,
        )
        repeat_median = float(statistics.median(repeat_samples))
        require_close(
            "metrics.avg_time_ms",
            getattr(metrics, "avg_time_ms", None),
            repeat_median,
        )
        repeat_p25, _, repeat_p75 = (
            statistics.quantiles(
                repeat_samples,
                n=4,
                method="inclusive",
            )
            if len(repeat_samples) >= 2
            else (repeat_samples[0], repeat_samples[0], repeat_samples[0])
        )
        repeat_iqr_pct = (
            (repeat_p75 - repeat_p25) / repeat_median * 100.0
        )
        repeat_spread_pct = (
            (max(repeat_samples) / min(repeat_samples) - 1.0) * 100.0
        )
        for field, expected in {
            "repeat_min_ms": min(repeat_samples),
            "repeat_median_ms": repeat_median,
            "repeat_max_ms": max(repeat_samples),
            "repeat_p25_ms": repeat_p25,
            "repeat_p75_ms": repeat_p75,
            "repeat_iqr_pct": repeat_iqr_pct,
            "repeat_spread_pct": repeat_spread_pct,
        }.items():
            require_close(field, provenance[field], expected)

        event_samples = parse_samples(
            "event_window_samples_ms",
            num_repeats,
        )
        expected_event_samples = [
            sample * num_iterations for sample in repeat_samples
        ]
        require_sample_match(
            "event_window_samples_ms",
            event_samples,
            expected_event_samples,
        )
        for field, expected in {
            "event_window_min_ms": min(expected_event_samples),
            "event_window_median_ms": statistics.median(
                expected_event_samples
            ),
            "event_window_max_ms": max(expected_event_samples),
        }.items():
            require_close(field, provenance[field], expected)

        expected_aggregation = (
            "median_of_post_stabilization_repeat_means"
            if num_repeats > 1 and num_stabilization_repeats
            else "median_of_repeat_means"
            if num_repeats > 1
            else "single_post_stabilization_repeat_mean"
            if num_stabilization_repeats
            else "single_repeat_mean"
        )
        require_equal(
            "aggregation",
            provenance["aggregation"],
            expected_aggregation,
        )
        require_metric(
            "aggregation",
            "aggregation",
            expected_aggregation,
        )

        invocations = num_warmup + num_iterations
        require_equal(
            "preallocated_invocations_per_repeat",
            provenance["preallocated_invocations_per_repeat"],
            invocations,
        )
        require_metric(
            "preallocated_invocations_per_repeat",
            "preallocated_invocations_per_repeat",
            invocations,
        )
        audited_storage_sets = (
            num_iterations
            if dispatch_mode == "captured_chain"
            else invocations
        )
        for field, attribute in (
            ("input_storage_sets_verified", "input_storage_sets_verified"),
            ("output_storage_sets_verified", "output_storage_sets_verified"),
        ):
            require_equal(field, provenance[field], audited_storage_sets)
            require_metric(field, attribute, audited_storage_sets)
        for field, attribute in (
            ("input_storage_ptr_count", "input_storage_ptr_count"),
            ("output_storage_ptr_count", "output_storage_ptr_count"),
            ("output_tensor_count", "output_tensor_count"),
        ):
            actual = provenance[field]
            if (
                not isinstance(actual, int)
                or isinstance(actual, bool)
                or actual < audited_storage_sets
                or actual % audited_storage_sets != 0
            ):
                raise RuntimeError(
                    f"{field} is inconsistent with audited storage sets"
                )
            require_metric(field, attribute, actual)
        require_equal(
            "input_reuse_within_repeat",
            provenance["input_reuse_within_repeat"],
            False,
        )
        require_metric(
            "input_reuse_within_repeat",
            "input_reuse_within_repeat",
            False,
        )
        require_equal(
            "input_output_storage_disjoint",
            provenance["input_output_storage_disjoint"],
            True,
        )
        require_metric(
            "input_output_storage_disjoint",
            "input_output_storage_disjoint",
            True,
        )

        require_equal(
            "output_unique_storages_per_set",
            provenance["output_unique_storages_per_set"],
            provenance["output_storage_ptr_count"] // audited_storage_sets,
        )
        require_equal(
            "output_tensors_per_set",
            provenance["output_tensors_per_set"],
            provenance["output_tensor_count"] // audited_storage_sets,
        )
        require_equal(
            "output_allocation_policy",
            provenance["output_allocation_policy"],
            provenance["output_allocation_mode"],
        )
        has_preallocated_contract = (
            provenance["preallocated_output_contract"]
            == "declared_phase_invariant_out"
        )
        expected_output_aliases = num_warmup if has_preallocated_contract else 0
        require_equal(
            "preallocated_output_aliases_verified",
            provenance["preallocated_output_aliases_verified"],
            expected_output_aliases,
        )
        require_equal(
            "preallocated_output_sets_verified",
            provenance["preallocated_output_sets_verified"],
            expected_output_aliases,
        )
        require_metric(
            "preallocated_output_aliases_verified",
            "preallocated_output_aliases_verified",
            expected_output_aliases,
        )
        expected_output_allocation_mode = (
            "preallocated_output_contract_with_warmup_alias_probe"
            if has_preallocated_contract
            else "no_preallocated_output_buffer_verified"
        )
        require_equal(
            "output_allocation_mode",
            provenance["output_allocation_mode"],
            expected_output_allocation_mode,
        )
        require_metric(
            "output_allocation_mode",
            "output_allocation_mode",
            expected_output_allocation_mode,
        )
        require_metric(
            "output_storage_policy",
            "output_storage_policy",
            provenance["output_storage_policy"],
        )
        require_metric(
            "timed_output_capture_policy",
            "timed_output_capture_policy",
            provenance["timed_output_capture_policy"],
        )
        require_metric(
            "preallocated_output_contract",
            "preallocated_output_contract",
            provenance["preallocated_output_contract"],
        )
        require_metric(
            "output_alias_verification_scope",
            "output_alias_verification_scope",
            provenance["output_alias_verification_scope"],
        )
        expected_contract_invocations = (
            invocations if has_preallocated_contract else 0
        )
        require_equal(
            "preallocated_output_contract_invocations_per_repeat",
            provenance[
                "preallocated_output_contract_invocations_per_repeat"
            ],
            expected_contract_invocations,
        )
        require_metric(
            "preallocated_output_contract_invocations_per_repeat",
            "preallocated_output_contract_invocations_per_repeat",
            expected_contract_invocations,
        )
        require_equal(
            "output_verification_replay_invocations_per_repeat",
            provenance[
                "output_verification_replay_invocations_per_repeat"
            ],
            0,
        )
        require_metric(
            "output_verification_replay_invocations_per_repeat",
            "output_verification_replay_invocations_per_repeat",
            0,
        )
        require_equal(
            "workspace_allocation_policy",
            provenance["workspace_allocation_policy"],
            "not_audited",
        )
        require_metric(
            "workspace_allocation_policy",
            "workspace_allocation_policy",
            "not_audited",
        )
        require_metric(
            "task_queue_enable",
            "task_queue_enable",
            provenance["task_queue_enable"],
        )

        diagnostic = provenance["profiler_is_diagnostic"]
        if diagnostic is True or str(diagnostic).lower() == "true":
            raise RuntimeError(
                "diagnostic profiler result cannot enter formal latency"
            )
        require_metric(
            "profiler_is_diagnostic",
            "profiler_is_diagnostic",
            False,
        )
        require_equal("dispatch_mode", provenance["dispatch_mode"], dispatch_mode)
        require_metric("dispatch_mode", "dispatch_mode", dispatch_mode)

        is_graph = dispatch_mode == "captured_chain"
        dispatch_expectations = {
            "graph_capture_width": num_iterations if is_graph else 0,
            "graph_replays": 1 if is_graph else 0,
            "capture_timed": False,
            "mutable_inputs_restored": is_graph,
            "timing_method": (
                "device_event_graph_replay" if is_graph else "device_event"
            ),
            "timing_semantics": (
                "device elapsed time; capture and mutable restore excluded"
                if is_graph
                else "device elapsed time; includes stream-idle gaps "
                "between start/end events caused by host dispatch"
            ),
            "dispatch_loop_policy": (
                "captured_prepared_payload_chain_single_replay"
                if is_graph
                else "python_direct_prepared_payload_loop"
            ),
            "output_storage_policy": (
                "capture_outputs_retained_until_repeat_end"
                if is_graph
                else "retained_until_repeat_end"
            ),
            "timed_output_capture_policy": (
                "preallocated_output_contract_no_timed_return_capture"
                if has_preallocated_contract
                else "capture_returns_retained_outside_timed_replay"
                if is_graph
                else "retained_return_inside_timed_region"
            ),
            "output_alias_verification_scope": (
                "warmup_returns_only"
                if has_preallocated_contract
                else "warmup_and_capture_returns"
                if is_graph
                else "warmup_and_measured_returns"
            ),
        }
        for field, expected in dispatch_expectations.items():
            require_equal(field, provenance[field], expected)
            require_metric(field, field, expected)
        expected_calls = (
            num_warmup + 2 * num_iterations if is_graph else invocations
        )
        require_equal(
            "total_operator_calls_per_repeat",
            provenance["total_operator_calls_per_repeat"],
            expected_calls,
        )
        require_metric(
            "total_operator_calls_per_repeat",
            "total_operator_calls_per_repeat",
            expected_calls,
        )
        expected_timed_region = (
            "one graph replay containing I independent core invocations"
            if is_graph
            else "Python direct prepared-payload loop of "
            "_execute_core_operator; prepare excluded; timed Python returns "
            "discarded under declared out contract"
            if provenance["timed_output_capture_policy"]
            == "preallocated_output_contract_no_timed_return_capture"
            else "Python direct prepared-payload loop of "
            "_execute_core_operator plus return slot assignment; "
            "prepare excluded"
            if provenance["timed_output_capture_policy"]
            == "retained_return_inside_timed_region"
            else "Python direct prepared-payload loop of "
            "_execute_core_operator; prepare excluded"
        )
        require_equal(
            "timed_region",
            provenance["timed_region"],
            expected_timed_region,
        )
        require_metric("timed_region", "timed_region", expected_timed_region)
        expected_stabilization_policy = (
            "fresh_storage_full_window_priming_repeats"
            if num_stabilization_repeats
            else "none"
        )
        require_equal(
            "device_stabilization_policy",
            provenance["device_stabilization_policy"],
            expected_stabilization_policy,
        )
        require_equal(
            "device_stabilization_timed",
            provenance["device_stabilization_timed"],
            bool(num_stabilization_repeats),
        )
        require_metric(
            "device_stabilization_policy",
            "device_stabilization_policy",
            expected_stabilization_policy,
        )
        require_metric(
            "device_stabilization_timed",
            "device_stabilization_timed",
            bool(num_stabilization_repeats),
        )
        require_equal(
            "stabilization_operator_calls",
            provenance["stabilization_operator_calls"],
            num_stabilization_repeats * expected_calls,
        )
        require_metric(
            "stabilization_operator_calls",
            "stabilization_operator_calls",
            num_stabilization_repeats * expected_calls,
        )

    def _write_terminal_row(
        self,
        *,
        checkpoint_rows: Dict[Tuple[str, str, str], List[Dict[str, Any]]],
        checkpoint_files: Dict[Tuple[str, str, str], Path],
        key: Tuple[str, str, str],
        row: Dict[str, Any],
    ) -> None:
        checkpoint_rows.setdefault(key, []).append(row)
        _atomic_write_csv(checkpoint_files[key], checkpoint_rows[key])

    def _finalize_checkpoints(
        self,
        checkpoint_rows: Dict[Tuple[str, str, str], List[Dict[str, Any]]],
        checkpoint_files: Dict[Tuple[str, str, str], Path],
        expected_points: int,
    ) -> None:
        for key, rows in checkpoint_rows.items():
            if len(rows) != expected_points:
                raise RuntimeError(
                    f"formal checkpoint {key} has {len(rows)}/"
                    f"{expected_points} selected terminal rows"
                )
            if any(row["status"] not in TERMINAL_STATUSES for row in rows):
                raise RuntimeError(
                    f"formal checkpoint {key} contains non-terminal rows"
                )
            if finalize_curve_coverage(rows):
                _atomic_write_csv(checkpoint_files[key], rows)

    def run_curve_test(
        self,
        *,
        variants: Optional[Sequence[Any]] = None,
        tokens: Sequence[int] = FORMAL_TOKENS,
        hidden_sizes: Sequence[int] = FORMAL_HIDDEN_SIZES,
        num_warmup: int = 2,
        num_iterations: Optional[int] = None,
        num_repeats: int = 5,
        num_stabilization_repeats: int = 2,
        dispatch_mode: str = "both",
        quick: bool = False,
        shard_index: int = 0,
        num_shards: int = 1,
        plot_results: bool = True,
    ) -> Dict[str, Any]:
        """Run correctness first, then symmetric eager/graph Event curves."""
        framework = self._require_framework()
        if num_warmup < 2:
            raise ValueError("NormQuant formal warmup must be >= 2")
        if num_iterations is not None and num_iterations <= 0:
            raise ValueError("NormQuant iterations must be positive")
        if num_repeats <= 0:
            raise ValueError("NormQuant repeats must be positive")
        if num_stabilization_repeats < 0:
            raise ValueError("stabilization repeats must be non-negative")
        if dispatch_mode not in DISPATCH_MODES:
            raise ValueError(f"invalid NormQuant dispatch mode: {dispatch_mode}")
        dispatches = DISPATCH_MODES[dispatch_mode]
        points, selection = build_norm_quant_shape_points(
            tokens=tokens,
            hidden_sizes=hidden_sizes,
            quick=quick,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        selected_variants = self._normalize_variants(variants)
        all_rows: List[Dict[str, Any]] = []
        checkpoint_rows: Dict[
            Tuple[str, str, str], List[Dict[str, Any]]
        ] = {}
        checkpoint_files: Dict[Tuple[str, str, str], Path] = {}

        for variant in selected_variants:
            operator = self.operator_factory(
                self.device,
                variant,
                self.precision,
            )
            providers = list(operator.get_formal_implementations(self.device))
            capability_error = getattr(operator, "capability_error", None)
            if not providers:
                declared = self._declared_providers(operator)
                error = capability_error or RuntimeError(
                    f"{operator.operator_name} has no formal provider "
                    f"for {self.device}"
                )
                for provider in declared:
                    for mode in dispatches:
                        key = (variant.value, provider, mode)
                        checkpoint_files[key] = self._artifact_path(
                            kind="checkpoint",
                            variant=variant,
                            provider=provider,
                            dispatch_mode=mode,
                            shard_index=shard_index,
                            num_shards=num_shards,
                            suffix=".csv",
                        )
                        for point in points:
                            data = operator.generate_test_data(
                                tokens=point["tokens"],
                                hidden=point["hidden"],
                                seed=0,
                            )
                            estimated_bytes = (
                                estimate_norm_quant_peak_retained_bytes(
                                    device=self.device,
                                    variant=variant,
                                    precision=self.precision,
                                    tokens=point["tokens"],
                                    hidden=point["hidden"],
                                )
                            )
                            plan = build_norm_quant_window_plan(
                                requested_warmup=num_warmup,
                                requested_iterations=(
                                    num_iterations
                                    if num_iterations is not None
                                    else 1
                                ),
                                estimated_unique_bytes_per_invocation=(
                                    estimated_bytes
                                ),
                                fresh_storage_hard_limit_bytes=(
                                    NORM_QUANT_FRESH_STORAGE_HARD_LIMIT_BYTES
                                ),
                                calibration_samples_ms={},
                                calibration_iterations=(
                                    NORM_QUANT_BASE_ITERATIONS
                                ),
                            )
                            if num_iterations is None:
                                plan.update(
                                    requested_iterations="auto",
                                    iteration_selection_policy=(
                                        "unsupported_before_calibration"
                                    ),
                                    calibration_is_diagnostic=True,
                                    window_target_feasible=False,
                                )
                            row = self._base_row(
                                point=point,
                                selection=selection,
                                variant=variant,
                                provider=provider,
                                dispatch_mode=mode,
                                operator=operator,
                                data=data,
                                iteration_plan=plan,
                                shard_index=shard_index,
                                num_shards=num_shards,
                            )
                            row.update(
                                status="unsupported",
                                error=f"{type(error).__name__}: {error}",
                                capability_status="unsupported",
                                graph_status=(
                                    "unsupported_capability"
                                    if mode == "captured_chain"
                                    else "not_applicable"
                                ),
                            )
                            self._write_terminal_row(
                                checkpoint_rows=checkpoint_rows,
                                checkpoint_files=checkpoint_files,
                                key=key,
                                row=row,
                            )
                            all_rows.append(row)
                continue

            data_by_point_provider: Dict[Tuple[int, str], Dict[str, Any]] = {}
            correctness_errors: Dict[Tuple[int, str], Exception] = {}
            for point in points:
                point_index = int(point["point_index"])
                for provider in providers:
                    data = operator.generate_test_data(
                        tokens=point["tokens"],
                        hidden=point["hidden"],
                        seed=0,
                    )
                    data_by_point_provider[(point_index, provider)] = data
                    try:
                        operator.run_device_implementation(
                            data,
                            self.device,
                            self.precision,
                            provider,
                        )
                    except Exception as error:
                        correctness_errors[(point_index, provider)] = error

            plans_by_point: Dict[int, Dict[str, Any]] = {}
            calibration_errors: Dict[
                Tuple[int, str, str], Exception
            ] = {}
            for point in points:
                point_index = int(point["point_index"])
                estimated_bytes = estimate_norm_quant_peak_retained_bytes(
                    device=self.device,
                    variant=variant,
                    precision=self.precision,
                    tokens=point["tokens"],
                    hidden=point["hidden"],
                )
                calibration_samples: Dict[str, float] = {}
                calibration_width = NORM_QUANT_BASE_ITERATIONS
                if num_iterations is None:
                    calibration_plan = build_norm_quant_window_plan(
                        requested_warmup=num_warmup,
                        requested_iterations=NORM_QUANT_BASE_ITERATIONS,
                        estimated_unique_bytes_per_invocation=estimated_bytes,
                        fresh_storage_hard_limit_bytes=(
                            NORM_QUANT_FRESH_STORAGE_HARD_LIMIT_BYTES
                        ),
                        calibration_samples_ms={},
                        calibration_iterations=NORM_QUANT_BASE_ITERATIONS,
                    )
                    calibration_warmup = int(
                        calibration_plan["effective_warmup"]
                    )
                    calibration_width = int(
                        calibration_plan["effective_iterations"]
                    )
                    for provider in providers:
                        if (point_index, provider) in correctness_errors:
                            continue
                        for mode in dispatches:
                            try:
                                metrics = (
                                    framework
                                    .run_core_operator_performance_test_v2(
                                        operator_test=operator,
                                        data=data_by_point_provider[(
                                            point_index,
                                            provider,
                                        )],
                                        device=self.device,
                                        precision=self.precision,
                                        implementation=provider,
                                        num_warmup=calibration_warmup,
                                        num_iterations=calibration_width,
                                        num_repeats=1,
                                        retain_outputs=True,
                                        verify_independent_storage=True,
                                        num_stabilization_repeats=0,
                                        dispatch_mode=mode,
                                    )
                                )
                                latency = float(metrics.avg_time_ms)
                                if not math.isfinite(latency) or latency <= 0:
                                    raise RuntimeError(
                                        "invalid diagnostic calibration "
                                        f"latency: {latency}"
                                    )
                                calibration_samples[
                                    f"{provider}/{mode}"
                                ] = latency
                            except Exception as error:
                                calibration_errors[(
                                    point_index,
                                    provider,
                                    mode,
                                )] = error
                if num_iterations is not None:
                    plan = build_norm_quant_window_plan(
                        requested_warmup=num_warmup,
                        requested_iterations=num_iterations,
                        estimated_unique_bytes_per_invocation=estimated_bytes,
                        fresh_storage_hard_limit_bytes=(
                            NORM_QUANT_FRESH_STORAGE_HARD_LIMIT_BYTES
                        ),
                        calibration_samples_ms={},
                        calibration_iterations=NORM_QUANT_BASE_ITERATIONS,
                    )
                elif calibration_samples:
                    plan = build_norm_quant_window_plan(
                        requested_warmup=num_warmup,
                        requested_iterations=None,
                        estimated_unique_bytes_per_invocation=estimated_bytes,
                        fresh_storage_hard_limit_bytes=(
                            NORM_QUANT_FRESH_STORAGE_HARD_LIMIT_BYTES
                        ),
                        calibration_samples_ms=calibration_samples,
                        calibration_iterations=calibration_width,
                    )
                else:
                    plan = build_norm_quant_window_plan(
                        requested_warmup=num_warmup,
                        requested_iterations=1,
                        estimated_unique_bytes_per_invocation=estimated_bytes,
                        fresh_storage_hard_limit_bytes=(
                            NORM_QUANT_FRESH_STORAGE_HARD_LIMIT_BYTES
                        ),
                        calibration_samples_ms={},
                        calibration_iterations=calibration_width,
                    )
                    plan.update(
                        requested_iterations="auto",
                        iteration_selection_policy=(
                            "calibration_failed_minimum_protocol"
                        ),
                        calibration_is_diagnostic=True,
                        window_target_feasible=False,
                    )
                plans_by_point[point_index] = plan

            for provider in providers:
                for mode in dispatches:
                    key = (variant.value, provider, mode)
                    checkpoint_files[key] = self._artifact_path(
                        kind="checkpoint",
                        variant=variant,
                        provider=provider,
                        dispatch_mode=mode,
                        shard_index=shard_index,
                        num_shards=num_shards,
                        suffix=".csv",
                    )

                for point in points:
                    point_index = int(point["point_index"])
                    data = data_by_point_provider[(
                        point_index,
                        provider,
                    )]
                    plan = plans_by_point[point_index]
                    effective_warmup = int(
                        plan["effective_warmup"]
                    )
                    effective_iterations = int(
                        plan["effective_iterations"]
                    )
                    correctness_error = correctness_errors.get(
                        (point_index, provider)
                    )

                    for mode in dispatches:
                        key = (variant.value, provider, mode)
                        row = self._base_row(
                            point=point,
                            selection=selection,
                            variant=variant,
                            provider=provider,
                            dispatch_mode=mode,
                            operator=operator,
                            data=data,
                            iteration_plan=plan,
                            shard_index=shard_index,
                            num_shards=num_shards,
                        )
                        if correctness_error is not None:
                            row.update(
                                status="error",
                                error=(
                                    "correctness "
                                    f"{type(correctness_error).__name__}: "
                                    f"{correctness_error}"
                                ),
                                graph_status=(
                                    "not_attempted_correctness_failure"
                                    if mode == "captured_chain"
                                    else "not_applicable"
                                ),
                            )
                        elif (
                            point_index,
                            provider,
                            mode,
                        ) in calibration_errors:
                            calibration_error = calibration_errors[(
                                point_index,
                                provider,
                                mode,
                            )]
                            if isinstance(
                                calibration_error,
                                GraphCaptureUnsupportedError,
                            ):
                                row.update(
                                    status="unsupported_graph_capture",
                                    error=(
                                        f"{type(calibration_error).__name__}: "
                                        f"{calibration_error}"
                                    ),
                                    graph_status="capture_error",
                                )
                            else:
                                row.update(
                                    status="error",
                                    error=(
                                        f"{type(calibration_error).__name__}: "
                                        f"{calibration_error}"
                                    ),
                                    graph_status=(
                                        "calibration_error"
                                        if mode == "captured_chain"
                                        else "not_applicable"
                                    ),
                                )
                        else:
                            try:
                                metrics = (
                                    framework
                                    .run_core_operator_performance_test_v2(
                                        operator_test=operator,
                                        data=data,
                                        device=self.device,
                                        precision=self.precision,
                                        implementation=provider,
                                        num_warmup=effective_warmup,
                                        num_iterations=effective_iterations,
                                        num_repeats=num_repeats,
                                        retain_outputs=True,
                                        verify_independent_storage=True,
                                        num_stabilization_repeats=(
                                            num_stabilization_repeats
                                        ),
                                        dispatch_mode=mode,
                                    )
                                )
                            except Exception as error:
                                if isinstance(
                                    error,
                                    GraphCaptureUnsupportedError,
                                ):
                                    row.update(
                                        status="unsupported_graph_capture",
                                        error=f"{type(error).__name__}: {error}",
                                        graph_status="capture_error",
                                    )
                                else:
                                    row.update(
                                        status="error",
                                        error=f"{type(error).__name__}: {error}",
                                        graph_status=(
                                            "formal_execution_error"
                                            if mode == "captured_chain"
                                            else "not_applicable"
                                        ),
                                    )
                            else:
                                try:
                                    provenance = (
                                        framework.performance_provenance(
                                            metrics
                                        )
                                    )
                                    self._verify_performance_provenance(
                                        provenance,
                                        metrics,
                                        dispatch_mode=mode,
                                        num_warmup=effective_warmup,
                                        num_iterations=effective_iterations,
                                        num_repeats=num_repeats,
                                        num_stabilization_repeats=(
                                            num_stabilization_repeats
                                        ),
                                    )
                                    latency = float(metrics.avg_time_ms)
                                    if (
                                        not math.isfinite(latency)
                                        or latency <= 0
                                    ):
                                        raise RuntimeError(
                                            f"invalid Event latency: {latency}"
                                        )
                                except Exception as error:
                                    row.update(
                                        status="error",
                                        error=f"{type(error).__name__}: {error}",
                                        graph_status=(
                                            "formal_validation_error"
                                            if mode == "captured_chain"
                                            else "not_applicable"
                                        ),
                                    )
                                else:
                                    row.update(provenance)
                                    row.update(
                                        latency_ms=latency,
                                        effective_bandwidth_gb_s=(
                                            operator.calculate_bandwidth(
                                                data,
                                                latency,
                                            )
                                        ),
                                        effective_bandwidth_semantics=(
                                            "logical_bytes / Event latency"
                                        ),
                                        physical_bandwidth_gb_s=(
                                            operator.physical_bytes(data)
                                            / (latency * 1e6)
                                        ),
                                        status="ok",
                                        graph_status=(
                                            "captured"
                                            if mode == "captured_chain"
                                            else "not_applicable"
                                        ),
                                    )
                        self._write_terminal_row(
                            checkpoint_rows=checkpoint_rows,
                            checkpoint_files=checkpoint_files,
                            key=key,
                            row=row,
                        )
                        all_rows.append(row)

        self._finalize_checkpoints(
            checkpoint_rows,
            checkpoint_files,
            len(points),
        )
        validate_formal_latency_rows(all_rows)
        envelope_rows: List[Dict[str, Any]] = []
        envelope_files: List[str] = []
        if _device_model(self.device) == H20_DEVICE_MODEL:
            envelope_rows = select_h20_best_envelope(all_rows)
            for variant in selected_variants:
                for mode in dispatches:
                    selected = [
                        row
                        for row in envelope_rows
                        if row["variant"] == variant.value
                        and row["dispatch_mode"] == mode
                    ]
                    if not selected:
                        continue
                    path = self._artifact_path(
                        kind="h20-best-envelope",
                        variant=variant,
                        provider="winning-provider-per-point",
                        dispatch_mode=mode,
                        shard_index=shard_index,
                        num_shards=num_shards,
                        suffix=".csv",
                    )
                    envelope_fields = [
                        *NORM_QUANT_FIELDS,
                        "winning_provider",
                    ]
                    _atomic_write_csv(path, selected, envelope_fields)
                    envelope_files.append(str(path))

        plot_files: List[str] = []
        if plot_results:
            plot_files = self._plot_rows(
                all_rows,
                selected_variants,
                shard_index,
                num_shards,
            )
        failures = [
            row
            for row in all_rows
            if row["status"] not in SUCCESS_STATUSES
        ]
        if failures:
            statuses = sorted({row["status"] for row in failures})
            raise RuntimeError(
                f"{len(failures)} formal NormQuant terminal failure(s): "
                f"{', '.join(statuses)}; checkpoints retained"
            )
        return {
            "rows": all_rows,
            "checkpoint_files": {
                key: str(path) for key, path in checkpoint_files.items()
            },
            "envelope_rows": envelope_rows,
            "envelope_files": envelope_files,
            "plot_files": plot_files,
        }

    def run_accuracy_test(
        self,
        *,
        variants: Optional[Sequence[Any]] = None,
        tokens: Sequence[int] = FORMAL_TOKENS,
        hidden_sizes: Sequence[int] = FORMAL_HIDDEN_SIZES,
        quick: bool = False,
        shard_index: int = 0,
        num_shards: int = 1,
    ) -> Dict[str, Any]:
        """Run only full auxiliary correctness; never enter a timer."""
        self._require_framework()
        points, _ = build_norm_quant_shape_points(
            tokens=tokens,
            hidden_sizes=hidden_sizes,
            quick=quick,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        rows = []
        for variant in self._normalize_variants(variants):
            operator = self.operator_factory(
                self.device,
                variant,
                self.precision,
            )
            providers = list(operator.get_formal_implementations(self.device))
            if not providers:
                error = getattr(operator, "capability_error", None)
                raise RuntimeError(
                    f"unsupported {variant.value}: "
                    f"{type(error).__name__}: {error}"
                )
            for provider in providers:
                for point in points:
                    data = operator.generate_test_data(
                        tokens=point["tokens"],
                        hidden=point["hidden"],
                        seed=0,
                    )
                    operator.run_device_implementation(
                        data,
                        self.device,
                        self.precision,
                        provider,
                    )
                    rows.append({
                        **point,
                        "variant": variant.value,
                        "provider": provider,
                        "status": "ok",
                    })
        return {"rows": rows}

    def run_profile_test(
        self,
        *,
        variants: Optional[Sequence[Any]] = None,
        tokens: Sequence[int] = PROFILE_TOKENS,
        hidden: int = PROFILE_HIDDEN,
        num_warmup: int = 2,
        num_iterations: int = NORM_QUANT_BASE_ITERATIONS,
        shard_index: int = 0,
        num_shards: int = 1,
    ) -> Dict[str, Any]:
        """Write diagnostic manifests only; never expose profiler latency."""
        framework = self._require_framework()
        if num_warmup < 2:
            raise ValueError("NormQuant profile warmup must be >= 2")
        if num_iterations <= 0:
            raise ValueError("NormQuant profile iterations must be positive")
        token_values = _validate_positive_ints(tokens, "profile tokens")
        manifest_files: List[str] = []
        failures = []
        for variant in self._normalize_variants(variants):
            operator = self.operator_factory(
                self.device,
                variant,
                self.precision,
            )
            providers = list(operator.get_formal_implementations(self.device))
            if not providers:
                error = getattr(operator, "capability_error", None)
                raise RuntimeError(
                    f"unsupported {variant.value}: "
                    f"{type(error).__name__}: {error}"
                )
            for provider in providers:
                for point_index, token_count in enumerate(token_values):
                    if point_index % num_shards != shard_index:
                        continue
                    data = operator.generate_test_data(
                        tokens=token_count,
                        hidden=hidden,
                        seed=0,
                    )
                    operator.run_device_implementation(
                        data,
                        self.device,
                        self.precision,
                        provider,
                    )
                    trace_path = self._artifact_path(
                        kind="profile-trace",
                        variant=variant,
                        provider=provider,
                        dispatch_mode="eager-prepared-core",
                        shard_index=shard_index,
                        num_shards=num_shards,
                        suffix=(
                            f"-tokens-{token_count}-hidden-{hidden}"
                        ),
                    )
                    try:
                        result = framework.run_core_operator_profile_test_v2(
                            operator_test=operator,
                            data=data,
                            device=self.device,
                            precision=self.precision,
                            implementation=provider,
                            num_warmup=num_warmup,
                            num_iterations=num_iterations,
                            trace_file_path=str(trace_path),
                            verify_independent_storage=True,
                        )
                        diagnostic = result.get("profiler_is_diagnostic")
                        if diagnostic is not True:
                            raise RuntimeError(
                                "prepared-core profile must be diagnostic"
                            )
                        manifest = {
                            key: value
                            for key, value in result.items()
                            if key not in {"latency_ms", "avg_time_ms"}
                        }
                        manifest.update(
                            variant=variant.value,
                            native_op=self._native_op(operator, provider),
                            device_model=_device_model(self.device),
                            tokens=token_count,
                            hidden=hidden,
                            logical_invocation_count=num_iterations,
                            expected_fused_kernel_count=num_iterations,
                            trace_directory=str(trace_path),
                            profiler_is_diagnostic=True,
                            status="ok",
                        )
                    except Exception as error:
                        failures.append(error)
                        manifest = {
                            "variant": variant.value,
                            "provider": provider,
                            "device": self.device,
                            "device_model": _device_model(self.device),
                            "tokens": token_count,
                            "hidden": hidden,
                            "logical_invocation_count": num_iterations,
                            "expected_fused_kernel_count": num_iterations,
                            "trace_directory": str(trace_path),
                            "profiler_is_diagnostic": True,
                            "status": "error",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    manifest_path = self._artifact_path(
                        kind="profile-manifest",
                        variant=variant,
                        provider=provider,
                        dispatch_mode="eager-prepared-core",
                        shard_index=shard_index,
                        num_shards=num_shards,
                        suffix=f"-tokens-{token_count}.json",
                    )
                    _atomic_write_json(manifest_path, manifest)
                    manifest_files.append(str(manifest_path))
        if failures:
            raise RuntimeError(
                f"{len(failures)} NormQuant diagnostic profile(s) failed"
            )
        return {"manifest_files": manifest_files}

    def _plot_rows(
        self,
        rows: Sequence[Dict[str, Any]],
        variants: Sequence[NormQuantVariant],
        shard_index: int,
        num_shards: int,
    ) -> List[str]:
        """Plot connected numeric axes; unsupported points are annotations."""
        successful = [row for row in rows if row["status"] == "ok"]
        validate_formal_latency_rows(successful)
        if not successful:
            return []
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_files: List[str] = []
        for variant in variants:
            variant_rows = [
                row for row in rows if row["variant"] == variant.value
            ]
            for metric, ylabel in (
                ("latency_ms", "Latency (ms)"),
                ("effective_bandwidth_gb_s", "Effective bandwidth (GB/s)"),
                ("physical_bandwidth_gb_s", "Physical bandwidth (GB/s)"),
            ):
                figure, axes = plt.subplots(1, 2, figsize=(14, 6))
                for axis, sweep in zip(axes, ("tokens", "hidden")):
                    sweep_rows = [
                        row
                        for row in variant_rows
                        if row["sweep"] == sweep and row["status"] == "ok"
                    ]
                    series_keys = sorted({
                        (row["provider"], row["dispatch_mode"])
                        for row in sweep_rows
                    })
                    x_field = "tokens" if sweep == "tokens" else "hidden"
                    for provider, mode in series_keys:
                        series = sorted(
                            (
                                row
                                for row in sweep_rows
                                if row["provider"] == provider
                                and row["dispatch_mode"] == mode
                            ),
                            key=lambda row: int(row[x_field]),
                        )
                        axis.plot(
                            [int(row[x_field]) for row in series],
                            [float(row[metric]) for row in series],
                            linestyle=(
                                "-" if mode == "captured_chain" else "--"
                            ),
                            marker="o",
                            label=f"{provider} {mode}",
                        )
                    unsupported = [
                        row
                        for row in variant_rows
                        if row["sweep"] == sweep
                        and row["status"] != "ok"
                    ]
                    if unsupported:
                        summary = "\n".join(
                            f"{row[x_field]}: {row['status']}"
                            for row in unsupported[:8]
                        )
                        axis.text(
                            1.02,
                            0.5,
                            f"Unsupported/error\n{summary}",
                            transform=axis.transAxes,
                            va="center",
                            fontsize=8,
                        )
                    axis.set_xlabel(
                        "Tokens" if sweep == "tokens" else "Hidden size"
                    )
                    axis.set_ylabel(ylabel)
                    axis.grid(True, alpha=0.35)
                    if series_keys:
                        axis.legend(fontsize=7)
                figure.suptitle(
                    f"{variant.value} {self.precision.name} {ylabel}"
                )
                figure.tight_layout()
                path = self._artifact_path(
                    kind=f"plot-{metric}",
                    variant=variant,
                    provider="all-providers",
                    dispatch_mode="both",
                    shard_index=shard_index,
                    num_shards=num_shards,
                    suffix=".png",
                )
                figure.savefig(path, dpi=160)
                plt.close(figure)
                plot_files.append(str(path))
        return plot_files


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Formal fused RMSNorm plus FP8/MXFP8/MXFP4 quantization benchmark"
        )
    )
    parser.add_argument(
        "--mode",
        choices=("accuracy", "curve", "profile"),
        default="curve",
    )
    parser.add_argument(
        "--precision",
        choices=tuple(PRECISION_TYPES),
        default="fp8",
    )
    parser.add_argument(
        "--dispatch-mode",
        choices=tuple(DISPATCH_MODES),
        default="both",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[variant.value for variant in NormQuantVariant],
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        default=FORMAL_TOKENS,
    )
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        default=FORMAL_HIDDEN_SIZES,
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=(
            "explicit shared I; omitted calibrates one shared 20-50 ms "
            "Event window"
        ),
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cuda[:N], or npu[:N]",
    )
    parser.add_argument("--result-dir", default="test_results")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.warmup < 2:
            raise ValueError("--warmup must be >= 2")
        if args.iterations is not None and args.iterations <= 0:
            raise ValueError("--iterations must be positive")
        if args.repeats <= 0:
            raise ValueError("--repeats must be positive")
        variants = (
            [NormQuantVariant(value) for value in args.variants]
            if args.variants
            else None
        )
        framework = OperatorTestFramework(result_dir=args.result_dir)
        suite = NormQuantTestSuite(
            precision=args.precision,
            device=args.device,
        )
        suite.framework = framework
        stabilization_repeats = int(
            os.environ.get("OPERATOR_TEST_STABILIZATION_REPEATS", "2")
        )
        if args.mode == "curve":
            suite.run_curve_test(
                variants=variants,
                tokens=args.tokens,
                hidden_sizes=args.hidden_sizes,
                num_warmup=args.warmup,
                num_iterations=args.iterations,
                num_repeats=args.repeats,
                num_stabilization_repeats=stabilization_repeats,
                dispatch_mode=args.dispatch_mode,
                quick=args.quick,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                plot_results=not args.no_plot,
            )
        elif args.mode == "accuracy":
            suite.run_accuracy_test(
                variants=variants,
                tokens=args.tokens,
                hidden_sizes=args.hidden_sizes,
                quick=args.quick,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
        else:
            profile_tokens = [
                token for token in args.tokens if token in PROFILE_TOKENS
            ]
            if args.tokens == FORMAL_TOKENS:
                profile_tokens = PROFILE_TOKENS
            if not profile_tokens:
                raise ValueError(
                    "profile mode supports tokens 1, 128, and 4096 only"
                )
            suite.run_profile_test(
                variants=variants,
                tokens=profile_tokens,
                hidden=PROFILE_HIDDEN,
                num_warmup=args.warmup,
                num_iterations=(
                    args.iterations or NORM_QUANT_BASE_ITERATIONS
                ),
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
        return 0
    except Exception as error:
        print(
            f"NormQuant formal command failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
