#!/usr/bin/env python3
"""Plot the captured-chain NormQuant token sweep from one suite output.

The script is intentionally standalone so ``runtests_fp8.sh`` can invoke it
directly on both Ascend and CUDA machines.  It reads only the newest NormQuant
checkpoint run in each requested precision directory and writes one PNG.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator
except ImportError as exc:  # pragma: no cover - depends on the benchmark image
    raise SystemExit(
        "plotting requires matplotlib in the benchmark environment"
    ) from exc


PRECISION_ORDER = ("fp8", "mxfp8", "mxfp4")
EXPECTED_TOKEN_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
SUCCESS_STATUSES = {"ok", "success"}
THOUSANDS = FuncFormatter(lambda value, _: f"{int(value):,}")

VARIANT_LABELS = {
    "rms_norm_quant": "RmsNormQuant",
    "add_rms_norm_quant": "AddRmsNormQuant",
    "add_rms_norm_dynamic_quant": "AddRmsNormDynamicQuant",
    "rms_norm_dynamic_mx_quant": "RmsNormDynamicMxQuant",
    "add_rms_norm_dynamic_mx_quant": "AddRmsNormDynamicMxQuant",
}

SEMANTIC_COLORS = {
    "rms_native": "#0072B2",
    "add_native": "#D55E00",
    "add_triton_bf16": "#009E73",
    "add_triton_fp32": "#CC79A7",
}
FALLBACK_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#7E57C2",
    "#56B4E9",
    "#E69F00",
    "#4D4D4D",
)


class PlotDataError(RuntimeError):
    """The suite output is missing or is not safe to present as a full curve."""


@dataclass(frozen=True)
class Curve:
    precision: str
    variant: str
    provider: str
    device_model: str
    run_id: str
    path: Path
    rows: tuple[dict[str, str], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        required=True,
        type=Path,
        help="suite output directory containing fp8/, mxfp8/, and/or mxfp4/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG path (default: <input-root>/normquant_captured_comparison.png)",
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=PRECISION_ORDER,
        help="precision panels to render; defaults to discovered precision directories",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="allow quick or sharded checkpoints; intended only for diagnostics",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise PlotDataError(f"empty checkpoint: {path}")
    required = {
        "coverage_complete",
        "device_model",
        "dispatch_mode",
        "hidden",
        "latency_ms",
        "precision",
        "provider",
        "status",
        "sweep",
        "tokens",
        "variant",
    }
    missing = required - set(rows[0])
    if missing:
        raise PlotDataError(f"{path}: missing columns {sorted(missing)}")
    return rows


def _single_value(rows: Sequence[dict[str, str]], field: str, path: Path) -> str:
    values = {row[field] for row in rows}
    if len(values) != 1:
        raise PlotDataError(f"{path}: mixed {field} values {sorted(values)}")
    return next(iter(values))


def _run_id(path: Path) -> str:
    parts = path.stem.split("__")
    if len(parts) < 2:
        raise PlotDataError(f"unexpected checkpoint filename: {path.name}")
    return parts[-1]


def _newest_run(paths: Sequence[Path]) -> tuple[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(_run_id(path), []).append(path)
    if not grouped:
        raise PlotDataError("no NormQuant checkpoint runs found")
    run_id = max(
        grouped,
        key=lambda candidate: max(path.stat().st_mtime_ns for path in grouped[candidate]),
    )
    return run_id, sorted(grouped[run_id])


def _load_precision(
    input_root: Path,
    precision: str,
    *,
    allow_incomplete: bool,
) -> list[Curve]:
    precision_root = input_root / precision
    if not precision_root.is_dir():
        raise PlotDataError(f"missing precision directory: {precision_root}")

    paths = sorted(precision_root.glob("checkpoint__*.csv"))
    if not paths:
        raise PlotDataError(f"no NormQuant checkpoints below {precision_root}")
    run_id, run_paths = _newest_run(paths)

    curves: list[Curve] = []
    seen: set[tuple[str, str]] = set()
    for path in run_paths:
        rows = _read_csv(path)
        dispatch = _single_value(rows, "dispatch_mode", path)
        if dispatch != "captured_chain":
            continue
        actual_precision = _single_value(rows, "precision", path).lower()
        if actual_precision != precision:
            raise PlotDataError(
                f"{path}: precision {actual_precision!r} does not match directory {precision!r}"
            )
        variant = _single_value(rows, "variant", path)
        provider = _single_value(rows, "provider", path)
        device_model = _single_value(rows, "device_model", path)
        key = (variant, provider)
        if key in seen:
            raise PlotDataError(
                f"{precision}/{run_id}: duplicate captured curve {variant}/{provider}"
            )
        seen.add(key)

        token_rows = sorted(
            (row for row in rows if row["sweep"] == "tokens"),
            key=lambda row: int(row["tokens"]),
        )
        if not token_rows:
            raise PlotDataError(f"{path}: no token-sweep rows")
        bad = [
            row
            for row in token_rows
            if row["status"].strip().lower() not in SUCCESS_STATUSES
        ]
        if bad:
            details = ", ".join(
                f"tokens={row['tokens']} status={row['status']}" for row in bad[:4]
            )
            raise PlotDataError(f"{path}: unsuccessful rows: {details}")

        tokens = tuple(int(row["tokens"]) for row in token_rows)
        if len(tokens) != len(set(tokens)):
            raise PlotDataError(f"{path}: duplicate token points")
        if not allow_incomplete:
            if tokens != EXPECTED_TOKEN_SWEEP:
                raise PlotDataError(
                    f"{path}: expected full token sweep {EXPECTED_TOKEN_SWEEP}, got {tokens}"
                )
            if any(row["coverage_complete"] != "True" for row in token_rows):
                raise PlotDataError(f"{path}: coverage_complete is not True")

        curves.append(
            Curve(
                precision=precision,
                variant=variant,
                provider=provider,
                device_model=device_model,
                run_id=run_id,
                path=path,
                rows=tuple(token_rows),
            )
        )

    if not curves:
        raise PlotDataError(
            f"{precision}/{run_id}: newest run contains no captured-chain curves"
        )
    return sorted(curves, key=_curve_sort_key)


def _provider_label(provider: str) -> str:
    if "triton_flashinfer" in provider:
        return "Triton (FP32 h)"
    if "triton_fused" in provider:
        return "Triton (BF16 h)"
    if provider.startswith("npu_"):
        return "native"
    if "flashinfer" in provider:
        return "FlashInfer"
    if provider.startswith("cuda_vllm_"):
        return "vLLM"
    return provider


def _curve_label(curve: Curve) -> str:
    variant = VARIANT_LABELS.get(curve.variant, curve.variant)
    return f"{variant} · {_provider_label(curve.provider)}"


def _curve_sort_key(curve: Curve) -> tuple[int, int, str]:
    variant_order = 10 if curve.variant.startswith("rms_") else 0
    provider = curve.provider
    if provider.startswith("npu_") and "triton" not in provider:
        provider_order = 0
    elif "triton_fused" in provider:
        provider_order = 1
    elif "triton_flashinfer" in provider:
        provider_order = 2
    elif "flashinfer" in provider:
        provider_order = 3
    elif provider.startswith("cuda_vllm_"):
        provider_order = 4
    else:
        provider_order = 5
    return variant_order, provider_order, _curve_label(curve)


def _semantic_color(curve: Curve) -> str | None:
    provider = curve.provider
    if "triton_flashinfer" in provider:
        return SEMANTIC_COLORS["add_triton_fp32"]
    if "triton_fused" in provider:
        return SEMANTIC_COLORS["add_triton_bf16"]
    if provider.startswith("npu_"):
        key = "rms_native" if curve.variant.startswith("rms_") else "add_native"
        return SEMANTIC_COLORS[key]
    return None


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D8DCE2",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _machine_title(curves_by_precision: dict[str, list[Curve]]) -> str:
    models = {
        curve.device_model
        for curves in curves_by_precision.values()
        for curve in curves
        if curve.device_model
    }
    if len(models) == 1:
        model = next(iter(models)).replace("_", " ")
        return model.replace("Ascend950PR", "Ascend 950PR")
    if models:
        return " / ".join(sorted(models))
    return "Operator benchmark"


def plot(
    curves_by_precision: dict[str, list[Curve]],
    precisions: Sequence[str],
    output: Path,
) -> None:
    _configure_style()
    figure, axes_grid = plt.subplots(
        1,
        len(precisions),
        figsize=(6.0 * len(precisions), 5.5),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes_grid[0]

    for axis, precision in zip(axes, precisions):
        used_colors: dict[str, str] = {}
        for index, curve in enumerate(curves_by_precision[precision]):
            label = _curve_label(curve)
            color = _semantic_color(curve)
            if color is None:
                color = used_colors.setdefault(
                    label,
                    FALLBACK_COLORS[index % len(FALLBACK_COLORS)],
                )
            tokens = [int(row["tokens"]) for row in curve.rows]
            latency_us = [float(row["latency_ms"]) * 1000.0 for row in curve.rows]
            axis.plot(
                tokens,
                latency_us,
                color=color,
                linewidth=2.1,
                marker="o",
                markersize=3.8,
                label=label,
                zorder=3,
            )

        axis.set_title(f"{precision.upper()} · captured", loc="left")
        axis.set_xlabel("Tokens (hidden=7,168)")
        axis.set_ylabel("Device Event latency (us)")
        axis.set_xlim(0, 4250)
        axis.set_ylim(bottom=0)
        axis.set_xticks((0, 512, 1024, 2048, 3072, 4096))
        axis.xaxis.set_major_formatter(THOUSANDS)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=4))
        axis.set_axisbelow(True)
        axis.legend(loc="upper left", fontsize=8.1, handlelength=2.1)

    figure.suptitle(
        f"{_machine_title(curves_by_precision)} · RmsNormQuant / AddRmsNormQuant\n"
        "Captured-chain device-event latency · token sweep · hidden=7,168",
        fontsize=16,
        fontweight="semibold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    if not input_root.is_dir():
        raise PlotDataError(f"input root does not exist: {input_root}")

    if args.precisions:
        precisions = list(dict.fromkeys(args.precisions))
    else:
        precisions = [
            precision
            for precision in PRECISION_ORDER
            if (input_root / precision).is_dir()
        ]
    if not precisions:
        raise PlotDataError(f"no precision directories below {input_root}")

    output = (
        args.output.resolve()
        if args.output is not None
        else input_root / "normquant_captured_comparison.png"
    )
    curves_by_precision = {
        precision: _load_precision(
            input_root,
            precision,
            allow_incomplete=args.allow_incomplete,
        )
        for precision in precisions
    }
    plot(curves_by_precision, precisions, output)

    for precision in precisions:
        run_ids = {curve.run_id for curve in curves_by_precision[precision]}
        print(
            f"Plot input: {precision} run={','.join(sorted(run_ids))} "
            f"curves={len(curves_by_precision[precision])}"
        )
    print(f"Plot: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlotDataError as exc:
        print(f"plot error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
