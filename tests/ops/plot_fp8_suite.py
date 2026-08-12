#!/usr/bin/env python3
"""Draw the canonical five-panel FP8/MX operator-suite overview.

The input is the output directory produced by ``runtests_fp8.sh`` and must
contain complete ``fp8/``, ``mxfp8/``, and ``mxfp4/`` formal runs.  When an
output directory has been reused, only the newest run for each family is read;
failed or incomplete newest runs are rejected instead of silently falling back
to stale data.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Iterable, Sequence

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator
except ImportError as exc:  # pragma: no cover - benchmark image dependency
    raise SystemExit(
        "plotting requires matplotlib in the benchmark environment"
    ) from exc


PRECISIONS = ("fp8", "mxfp8", "mxfp4")
PRECISION_LABELS = {precision: precision.upper() for precision in PRECISIONS}
PRECISION_STYLES = {
    "fp8": {"color": "#E64B35", "marker": "o"},
    "mxfp8": {"color": "#009E73", "marker": "s"},
    "mxfp4": {"color": "#7E57C2", "marker": "D"},
}
PROVIDER_STYLES = {
    "rms_native": {"color": "#0072B2", "label": "RMS · native"},
    "add_native": {"color": "#D55E00", "label": "Add+RMS · native"},
    "add_triton_bf16": {
        "color": "#009E73",
        "label": "Add+RMS · Triton (BF16 h)",
    },
    "add_triton_fp32": {
        "color": "#CC79A7",
        "label": "Add+RMS · Triton (FP32 h)",
    },
}
SUCCESS_STATUSES = {"ok", "success"}
EXPECTED_TOKEN_SWEEP = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
)
THOUSANDS = FuncFormatter(lambda value, _: f"{int(value):,}")


class PlotDataError(RuntimeError):
    """The newest suite output is missing, failed, or incomplete."""


@dataclass(frozen=True)
class NormCurve:
    precision: str
    provider: str
    rows: tuple[dict[str, str], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        required=True,
        type=Path,
        help="suite output directory containing fp8/, mxfp8/, and mxfp4/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG path (default: <input-root>/fp8_suite_overview.png)",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise PlotDataError(f"empty CSV: {path}")
    return rows


def require_columns(
    rows: Sequence[dict[str, str]],
    columns: Iterable[str],
    path: Path,
) -> None:
    missing = set(columns) - set(rows[0])
    if missing:
        raise PlotDataError(f"{path}: missing columns {sorted(missing)}")


def require_success(rows: Sequence[dict[str, str]], path: Path) -> None:
    failures = [
        row
        for row in rows
        if row.get("status", "").strip().lower() not in SUCCESS_STATUSES
    ]
    if failures:
        details = ", ".join(
            f"point={row.get('point_index')} status={row.get('status')}"
            for row in failures[:4]
        )
        raise PlotDataError(f"{path}: unsuccessful rows: {details}")
    if any(row.get("coverage_complete") != "True" for row in rows):
        raise PlotDataError(f"{path}: coverage_complete is not True for every row")


def newest_file(pattern: str, root: Path) -> Path:
    matches = list(root.glob(pattern))
    if not matches:
        raise PlotDataError(f"no {pattern!r} below {root}")
    return max(matches, key=lambda path: (path.stat().st_mtime_ns, path.name))


def load_linear(input_root: Path) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for precision in PRECISIONS:
        precision_root = input_root / precision
        path = newest_file("linear_tflops_*.csv", precision_root)
        rows = read_csv(path)
        require_columns(
            rows,
            (
                "M",
                "N",
                "K",
                "device",
                "precision",
                "TFLOPS",
                "status",
                "coverage_complete",
            ),
            path,
        )
        require_success(rows, path)
        if len(rows) != 39:
            raise PlotDataError(f"{path}: expected 39 formal points, got {len(rows)}")
        if any(not (row["M"] == row["N"] == row["K"]) for row in rows):
            raise PlotDataError(f"{path}: expected square M=N=K matrices")
        actual = {row["precision"].lower() for row in rows}
        if actual != {precision}:
            raise PlotDataError(
                f"{path}: precision labels {sorted(actual)} do not match {precision}"
            )
        result[precision] = sorted(rows, key=lambda row: int(row["M"]))
    return result


def load_groupgemm(input_root: Path) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for precision in PRECISIONS:
        precision_root = input_root / precision
        path = newest_file("groupgemm_tflops_*.csv", precision_root)
        rows = read_csv(path)
        require_columns(
            rows,
            (
                "seq_len",
                "num_experts",
                "hidden_dim",
                "out_channel",
                "device",
                "precision",
                "throughput_trillion_ops_s",
                "status",
                "coverage_complete",
            ),
            path,
        )
        require_success(rows, path)
        if len(rows) != 10:
            raise PlotDataError(f"{path}: expected 10 formal points, got {len(rows)}")
        actual = {row["precision"].lower() for row in rows}
        if actual != {precision}:
            raise PlotDataError(
                f"{path}: precision labels {sorted(actual)} do not match {precision}"
            )
        fixed = {
            (row["num_experts"], row["hidden_dim"], row["out_channel"])
            for row in rows
        }
        if fixed != {("8", "7168", "4096")}:
            raise PlotDataError(f"{path}: unexpected fixed dimensions {fixed}")
        result[precision] = sorted(rows, key=lambda row: int(row["seq_len"]))
    return result


def run_id(path: Path) -> str:
    parts = path.stem.split("__")
    if len(parts) < 2:
        raise PlotDataError(f"unexpected checkpoint filename: {path.name}")
    return parts[-1]


def newest_checkpoint_run(paths: Sequence[Path]) -> list[Path]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(run_id(path), []).append(path)
    if not grouped:
        raise PlotDataError("no NormQuant checkpoint runs found")
    newest_id = max(
        grouped,
        key=lambda candidate: max(
            path.stat().st_mtime_ns for path in grouped[candidate]
        ),
    )
    return sorted(grouped[newest_id])


def single_value(rows: Sequence[dict[str, str]], field: str, path: Path) -> str:
    values = {row[field] for row in rows}
    if len(values) != 1:
        raise PlotDataError(f"{path}: mixed {field} values {sorted(values)}")
    return next(iter(values))


def provider_kind(provider: str, variant: str) -> str:
    if "triton_flashinfer" in provider:
        return "add_triton_fp32"
    if "triton_fused" in provider:
        return "add_triton_bf16"
    if provider.startswith("npu_"):
        return "rms_native" if variant.startswith("rms_") else "add_native"
    raise PlotDataError(f"unrecognized NormQuant provider: {provider}")


def load_norm(input_root: Path) -> dict[str, list[NormCurve]]:
    result: dict[str, list[NormCurve]] = {}
    for precision in PRECISIONS:
        precision_root = input_root / precision
        paths = newest_checkpoint_run(sorted(precision_root.glob("checkpoint__*.csv")))
        captured: list[NormCurve] = []
        seen: set[tuple[str, str, str]] = set()
        device_models: set[str] = set()
        formal_curve_count = 0
        for path in paths:
            rows = read_csv(path)
            require_columns(
                rows,
                (
                    "coverage_complete",
                    "device",
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
                ),
                path,
            )
            require_success(rows, path)
            if len(rows) != 16:
                raise PlotDataError(
                    f"{path}: expected 16 formal points, got {len(rows)}"
                )
            actual_precision = single_value(rows, "precision", path).lower()
            if actual_precision != precision:
                raise PlotDataError(
                    f"{path}: precision {actual_precision} does not match {precision}"
                )
            provider = single_value(rows, "provider", path)
            variant = single_value(rows, "variant", path)
            dispatch = single_value(rows, "dispatch_mode", path)
            if dispatch not in {"eager_direct", "captured_chain"}:
                raise PlotDataError(f"{path}: unsupported dispatch {dispatch}")
            device_models.add(single_value(rows, "device_model", path))
            key = (variant, provider, dispatch)
            if key in seen:
                raise PlotDataError(f"{path}: duplicate checkpoint curve {key}")
            seen.add(key)
            formal_curve_count += 1
            token_rows = sorted(
                (row for row in rows if row["sweep"] == "tokens"),
                key=lambda row: int(row["tokens"]),
            )
            hidden_rows = sorted(
                (row for row in rows if row["sweep"] == "hidden"),
                key=lambda row: int(row["hidden"]),
            )
            if len(token_rows) != 13 or len(hidden_rows) != 3:
                raise PlotDataError(
                    f"{path}: expected 13 token and 3 hidden sweep points"
                )
            tokens = tuple(int(row["tokens"]) for row in token_rows)
            if tokens != EXPECTED_TOKEN_SWEEP:
                raise PlotDataError(
                    f"{path}: expected token sweep {EXPECTED_TOKEN_SWEEP}, got {tokens}"
                )
            if {int(row["hidden"]) for row in token_rows} != {7168}:
                raise PlotDataError(f"{path}: token sweep must fix hidden=7168")
            if {int(row["tokens"]) for row in hidden_rows} != {128}:
                raise PlotDataError(f"{path}: hidden sweep must fix tokens=128")
            hidden_points = tuple(int(row["hidden"]) for row in hidden_rows)
            if hidden_points != (4096, 7168, 8192):
                raise PlotDataError(
                    f"{path}: expected hidden sweep (4096, 7168, 8192), "
                    f"got {hidden_points}"
                )
            if dispatch == "captured_chain":
                # Validate the provider now so a mislabeled curve cannot reach
                # the legend or silently receive the wrong color.
                provider_kind(provider, variant)
                captured.append(
                    NormCurve(
                        precision=precision,
                        provider=provider,
                        rows=tuple(token_rows),
                    )
                )
        expected_formal = 8 if precision == "fp8" else 4
        expected_captured = expected_formal // 2
        if formal_curve_count != expected_formal:
            raise PlotDataError(
                f"{precision}: expected {expected_formal} formal curves, "
                f"got {formal_curve_count}"
            )
        if len(captured) != expected_captured:
            raise PlotDataError(
                f"{precision}: expected {expected_captured} captured curves, "
                f"got {len(captured)}"
            )
        if len(device_models) != 1:
            raise PlotDataError(
                f"{precision}: expected one device_model, got "
                f"{sorted(device_models)}"
            )
        result[precision] = captured
    return result


def configure_style() -> None:
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


def finish_axis(
    axis: Axes,
    *,
    xlabel: str,
    ylabel: str,
    xlim: tuple[float, float],
) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_xscale("linear")
    axis.set_yscale("linear")
    axis.set_xlim(*xlim)
    axis.set_ylim(bottom=0)
    axis.xaxis.set_major_formatter(THOUSANDS)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=4))
    axis.tick_params(axis="both", which="major", labelsize=9.5)
    axis.set_axisbelow(True)


def plot_precision_panel(
    axis: Axes,
    data: dict[str, list[dict[str, str]]],
    *,
    x_field: str,
    y_field: str,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    maximum = 0.0
    for precision in PRECISIONS:
        rows = data[precision]
        x = [int(row[x_field]) for row in rows]
        y = [float(row[y_field]) for row in rows]
        maximum = max(maximum, max(y))
        style = PRECISION_STYLES[precision]
        axis.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            markersize=4.0,
            markevery=max(1, len(x) // 12),
            linewidth=2.2,
            label=PRECISION_LABELS[precision],
            zorder=3,
        )
    finish_axis(axis, xlabel=xlabel, ylabel=ylabel, xlim=(0, 34000))
    axis.set_ylim(0, maximum * 1.13)
    axis.set_xticks((0, 4096, 8192, 16384, 24576, 32768))
    axis.set_title(title, loc="left")
    axis.legend(loc="upper left", ncol=3, columnspacing=1.2, handlelength=2.4)


def plot_norm_panel(axis: Axes, precision: str, curves: Sequence[NormCurve]) -> None:
    handles: list[Line2D] = []
    labels: list[str] = []
    maximum = 0.0
    for curve in sorted(
        curves,
        key=lambda item: provider_kind(
            item.provider,
            single_value(item.rows, "variant", Path(item.provider)),
        ),
    ):
        variant = single_value(curve.rows, "variant", Path(curve.provider))
        kind = provider_kind(curve.provider, variant)
        style = PROVIDER_STYLES[kind]
        x = [int(row["tokens"]) for row in curve.rows]
        y = [float(row["latency_ms"]) * 1000.0 for row in curve.rows]
        maximum = max(maximum, max(y))
        axis.plot(
            x,
            y,
            color=style["color"],
            linewidth=2.0,
            marker="o",
            markersize=3.6,
            zorder=3,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linewidth=2.2,
                marker="o",
                markersize=4,
            )
        )
        labels.append(style["label"])
    finish_axis(
        axis,
        xlabel="Tokens (hidden=7,168)",
        ylabel="Device Event latency (us)",
        xlim=(0, 4250),
    )
    axis.set_ylim(0, maximum * 1.10)
    axis.set_xticks((0, 512, 1024, 2048, 3072, 4096))
    axis.set_title(f"Norm/Quant · {precision.upper()} · captured", loc="left")
    axis.legend(handles, labels, loc="upper left", fontsize=7.5, handlelength=2.0)


def collect_row_values(field: str, *datasets: object) -> set[str]:
    values: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, dict):
            if field in value and value[field]:
                values.add(str(value[field]))
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, NormCurve):
            visit(value.rows)

    for dataset in datasets:
        visit(dataset)
    return values


def machine_title(models: set[str]) -> str:
    if len(models) != 1:
        raise PlotDataError(
            f"expected exactly one device_model across NormQuant inputs, got "
            f"{sorted(models)}"
        )
    model = next(iter(models))
    if "950pr" in model.lower():
        return "Ascend 950PR"
    model = model.replace("_", " ")
    return re.sub(r"(?<=[a-z])(?=[A-Z0-9])", " ", model)


def save_overview(
    linear: dict[str, list[dict[str, str]]],
    groupgemm: dict[str, list[dict[str, str]]],
    norm: dict[str, list[NormCurve]],
    output: Path,
) -> None:
    configure_style()
    figure = plt.figure(figsize=(19, 10.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 6, height_ratios=(1.06, 1.0))
    linear_axis = figure.add_subplot(grid[0, :3])
    group_axis = figure.add_subplot(grid[0, 3:])
    norm_axes = [
        figure.add_subplot(grid[1, index * 2 : (index + 1) * 2])
        for index in range(3)
    ]
    plot_precision_panel(
        linear_axis,
        linear,
        x_field="M",
        y_field="TFLOPS",
        xlabel="Square matrix size (M=N=K)",
        ylabel="Throughput (TFLOPS)",
        title="Linear",
    )
    plot_precision_panel(
        group_axis,
        groupgemm,
        x_field="seq_len",
        y_field="throughput_trillion_ops_s",
        xlabel="Total tokens",
        ylabel="Throughput (trillion ops/s)",
        title="GroupGemm · E=8",
    )
    for axis, precision in zip(norm_axes, PRECISIONS):
        plot_norm_panel(axis, precision, norm[precision])

    devices = collect_row_values("device", linear, groupgemm, norm)
    if len(devices) != 1:
        raise PlotDataError(
            f"expected exactly one device across selected inputs, got "
            f"{sorted(devices)}"
        )
    title = machine_title(collect_row_values("device_model", norm))
    figure.suptitle(
        f"{title} · canonical FP8/MX operator suite\n"
        "Linear and GroupGemm compare precision; Norm/Quant summary shows "
        "captured token sweeps",
        fontsize=17,
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
    output = (
        args.output.resolve()
        if args.output is not None
        else input_root / "fp8_suite_overview.png"
    )
    linear = load_linear(input_root)
    groupgemm = load_groupgemm(input_root)
    norm = load_norm(input_root)
    save_overview(linear, groupgemm, norm, output)
    print(f"Plot: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlotDataError as exc:
        print(f"plot error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
