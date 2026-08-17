#!/usr/bin/env python3
"""Summarize GEANT Hattrick, BEST_MC, and SWAN results.

The script reads the original experiment text files, aligns every method to the
same test interval, derives per-class normalized fulfill ratios for the Gurobi
baselines, and writes auditable CSV/Markdown tables plus publication-style PNG
figures.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


SCHEMES = ("Hattrick", "BEST_MC", "SWAN")
PLOT_ORDER = ("BEST_MC", "SWAN", "Hattrick")
CLASSES = ("High", "Medium", "Low")
COLORS = {
    "Hattrick": "#8C2D26",
    "BEST_MC": "#111111",
    "SWAN": "#2563EB",
}
LINE_STYLES = {
    "Hattrick": "-",
    "BEST_MC": "-",
    "SWAN": "-",
}
STAT_COLUMNS = (
    "mean",
    "std",
    "p1",
    "p10",
    "p25",
    "median",
    "p75",
    "p90",
    "p95",
    "p99",
    "minimum",
    "maximum",
)


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate paper-style GEANT result tables and figures."
    )
    parser.add_argument("--project-root", type=Path, default=script_root)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--topology", default="geant")
    parser.add_argument("--num-paths", type=int, default=4)
    parser.add_argument("--cluster", type=int, default=0)
    parser.add_argument("--test-start", type=int, default=8618)
    parser.add_argument("--test-end", type=int, default=10772)
    parser.add_argument("--predictor", default="esm")
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required result file not found: {path}")
    return path


def load_vector(path: Path) -> np.ndarray:
    values = np.loadtxt(require_file(path), dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError(f"Result file is empty: {path}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Result file contains NaN or infinity: {path}")
    return values


def select_test_rows(
    rows: np.ndarray,
    test_start: int,
    test_end: int,
    expected: int,
    label: str,
) -> np.ndarray:
    """Accept either an all-snapshot array or an already sliced test array."""
    if rows.shape[0] == expected:
        selected = rows
    elif rows.shape[0] >= test_end:
        selected = rows[test_start:test_end]
    else:
        raise ValueError(
            f"{label} has {rows.shape[0]} snapshots; expected either "
            f"{expected} test snapshots or at least {test_end} total snapshots."
        )
    if selected.shape[0] != expected:
        raise ValueError(
            f"{label}: selected {selected.shape[0]} snapshots, expected {expected}."
        )
    return selected


def load_three_column_test_file(path: Path, expected: int, label: str) -> np.ndarray:
    values = load_vector(path)
    if values.size != expected * 3:
        raise ValueError(
            f"{label} has {values.size} values; expected {expected * 3} "
            f"({expected} snapshots x 3 classes)."
        )
    return values.reshape(expected, 3)


def load_six_column_baseline(
    path: Path,
    test_start: int,
    test_end: int,
    expected: int,
    label: str,
) -> np.ndarray:
    values = load_vector(path)
    if values.size % 6 != 0:
        raise ValueError(f"{label} line count is not divisible by 6: {values.size}")
    rows = values.reshape(-1, 6)
    return select_test_rows(rows, test_start, test_end, expected, label)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, label: str) -> np.ndarray:
    if numerator.shape != denominator.shape:
        raise ValueError(
            f"{label}: numerator shape {numerator.shape} does not match "
            f"denominator shape {denominator.shape}."
        )
    if np.any(denominator <= 0):
        indices = np.argwhere(denominator <= 0)[:5].tolist()
        raise ValueError(f"{label}: non-positive oracle denominator at {indices}")
    result = numerator / denominator
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label}: normalized result contains NaN or infinity.")
    return result


def load_oracle_columns(
    results_dir: Path,
    names: Sequence[str],
    test_start: int,
    test_end: int,
    expected: int,
    label: str,
) -> np.ndarray:
    columns = []
    for name in names:
        values = load_vector(results_dir / name).reshape(-1, 1)
        selected = select_test_rows(
            values, test_start, test_end, expected, f"{label}/{name}"
        )
        columns.append(selected[:, 0])
    return np.column_stack(columns)


def load_baseline_fulfill(
    sim_rows: np.ndarray, oracle_cumulative: np.ndarray, label: str
) -> np.ndarray:
    # Simulator columns are:
    # [MLU_h, cumulative_flow_h, MLU_hm, cumulative_flow_hm,
    #  MLU_hml, cumulative_flow_hml].
    cumulative = sim_rows[:, [1, 3, 5]]
    admitted_per_class = np.column_stack(
        (
            cumulative[:, 0],
            cumulative[:, 1] - cumulative[:, 0],
            cumulative[:, 2] - cumulative[:, 1],
        )
    )
    oracle_per_class = np.column_stack(
        (
            oracle_cumulative[:, 0],
            oracle_cumulative[:, 1] - oracle_cumulative[:, 0],
            oracle_cumulative[:, 2] - oracle_cumulative[:, 1],
        )
    )
    return safe_divide(admitted_per_class, oracle_per_class, label)


def summarize_vector(values: np.ndarray) -> Dict[str, float]:
    percentiles = np.percentile(values, [1, 10, 25, 50, 75, 90, 95, 99])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p1": float(percentiles[0]),
        "p10": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "median": float(percentiles[3]),
        "p75": float(percentiles[4]),
        "p90": float(percentiles[5]),
        "p95": float(percentiles[6]),
        "p99": float(percentiles[7]),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def summarize_class_metrics(
    data: Mapping[str, np.ndarray], scheme_order: Sequence[str] = SCHEMES
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for scheme in scheme_order:
        matrix = data[scheme]
        if matrix.ndim != 2 or matrix.shape[1] != 3:
            raise ValueError(f"{scheme} metrics must have shape [N, 3], got {matrix.shape}")
        for class_index, traffic_class in enumerate(CLASSES):
            row: Dict[str, object] = {
                "scheme": scheme,
                "traffic_class": traffic_class,
                "n": int(matrix.shape[0]),
            }
            row.update(summarize_vector(matrix[:, class_index]))
            rows.append(row)
    return rows


def summarize_runtime(data: Mapping[str, np.ndarray]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for scheme in SCHEMES:
        values = data[scheme]
        row: Dict[str, object] = {"scheme": scheme, "n": int(values.size)}
        row.update(summarize_vector(values))
        rows.append(row)
    return rows


def write_dict_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key, value in row.items():
                if isinstance(value, (float, np.floating)):
                    formatted[key] = f"{float(value):.10f}"
                else:
                    formatted[key] = value
            writer.writerow(formatted)


def write_per_snapshot_csv(
    path: Path,
    test_start: int,
    fulfill: Mapping[str, np.ndarray],
    mlu: Mapping[str, np.ndarray],
) -> None:
    expected = next(iter(fulfill.values())).shape[0]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "snapshot_index",
                "scheme",
                "traffic_class",
                "normalized_fulfill_ratio",
                "normalized_mlu",
            ]
        )
        for offset in range(expected):
            for scheme in SCHEMES:
                for class_index, traffic_class in enumerate(CLASSES):
                    writer.writerow(
                        [
                            test_start + offset,
                            scheme,
                            traffic_class,
                            f"{fulfill[scheme][offset, class_index]:.12f}",
                            (
                                f"{mlu[scheme][offset, class_index]:.12f}"
                                if scheme in mlu
                                else ""
                            ),
                        ]
                    )


def load_runtime_series(
    path: Path,
    test_start: int,
    test_end: int,
    expected: int,
    label: str,
) -> np.ndarray:
    values = load_vector(path).reshape(-1, 1)
    return select_test_rows(values, test_start, test_end, expected, label)[:, 0]


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, x.size + 1, dtype=np.float64) / x.size
    return x, y


def padded_limits(values: Iterable[np.ndarray]) -> tuple[float, float]:
    combined = np.concatenate([np.asarray(value).reshape(-1) for value in values])
    lower = float(np.min(combined))
    upper = float(np.max(combined))
    width = max(upper - lower, 0.01)
    return lower - 0.03 * width, upper + 0.03 * width


def plot_cdf_panels(
    data: Mapping[str, np.ndarray],
    output_path: Path,
    x_label: str,
    title: str,
    dpi: int,
    plot_order: Sequence[str] = PLOT_ORDER,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.55), sharey=True)
    for class_index, (axis, traffic_class) in enumerate(zip(axes, CLASSES)):
        for scheme in plot_order:
            x, y = empirical_cdf(data[scheme][:, class_index])
            axis.plot(
                x,
                y,
                color=COLORS[scheme],
                linestyle=LINE_STYLES[scheme],
                linewidth=1.8,
                label=scheme,
            )
        xmin, xmax = padded_limits(
            data[scheme][:, class_index] for scheme in plot_order
        )
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(0.0, 1.0)
        axis.axvline(1.0, color="#777777", linewidth=0.8, linestyle="--", alpha=0.7)
        axis.axhline(0.01, color="#B0B0B0", linewidth=0.6, linestyle=":")
        axis.axhline(0.10, color="#B0B0B0", linewidth=0.6, linestyle=":")
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.7)
        axis.set_xlabel(x_label)
        axis.set_title(f"({chr(97 + class_index)}) {traffic_class} class")
    axes[0].set_ylabel("CDF")
    if len(plot_order) > 1:
        axes[0].legend(loc="upper left", frameon=True, edgecolor="#BBBBBB")
    fig.suptitle(title, y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_fulfill_boxplot(
    data: Mapping[str, np.ndarray], output_path: Path, dpi: int
) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 4.6))
    width = 0.22
    centers = np.arange(len(CLASSES), dtype=float)
    offsets = {"Hattrick": -width, "BEST_MC": 0.0, "SWAN": width}
    for scheme in SCHEMES:
        positions = centers + offsets[scheme]
        box = axis.boxplot(
            [data[scheme][:, index] for index in range(3)],
            positions=positions,
            widths=width * 0.82,
            patch_artist=True,
            showfliers=False,
            whis=(1, 99),
            manage_ticks=False,
            medianprops={"color": "white", "linewidth": 1.4},
            whiskerprops={"color": COLORS[scheme], "linewidth": 1.0},
            capprops={"color": COLORS[scheme], "linewidth": 1.0},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(COLORS[scheme])
            patch.set_edgecolor(COLORS[scheme])
            patch.set_alpha(0.88)
    axis.axhline(1.0, color="#777777", linewidth=0.9, linestyle="--")
    axis.set_xticks(centers)
    axis.set_xticklabels(CLASSES)
    axis.set_xlabel("Traffic class")
    axis.set_ylabel("Normalized fulfill ratio")
    axis.set_title("GEANT normalized fulfill ratio (P1-P99 whiskers)")
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.5, alpha=0.7)
    axis.legend(
        handles=[Patch(facecolor=COLORS[name], label=name) for name in SCHEMES],
        loc="lower left",
        frameon=True,
        edgecolor="#BBBBBB",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_runtime(
    runtime: Mapping[str, np.ndarray], output_path: Path, dpi: int
) -> None:
    means = [float(np.mean(runtime[scheme])) for scheme in SCHEMES]
    fig, axis = plt.subplots(figsize=(6.8, 4.5))
    bars = axis.bar(
        np.arange(len(SCHEMES)),
        means,
        color=[COLORS[scheme] for scheme in SCHEMES],
        width=0.62,
    )
    axis.set_yscale("log")
    axis.set_xticks(np.arange(len(SCHEMES)))
    axis.set_xticklabels(SCHEMES)
    axis.set_ylabel("Average computation time (seconds / snapshot)")
    axis.set_title("GEANT recorded computation time\n(hardware not controlled)")
    axis.grid(axis="y", which="both", color="#D8D8D8", linewidth=0.5, alpha=0.7)
    for bar, value in zip(bars, means):
        axis.annotate(
            f"{value:.4f}s",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def rows_by_scheme_class(
    rows: Sequence[Mapping[str, object]],
) -> Dict[tuple[str, str], Mapping[str, object]]:
    return {(str(row["scheme"]), str(row["traffic_class"])): row for row in rows}


def plot_summary_table(
    rows: Sequence[Mapping[str, object]], output_path: Path, dpi: int
) -> None:
    lookup = rows_by_scheme_class(rows)
    table_rows = []
    row_labels = []
    for scheme in SCHEMES:
        for traffic_class in CLASSES:
            row = lookup[(scheme, traffic_class)]
            row_labels.append(f"{scheme} - {traffic_class}")
            table_rows.append(
                [
                    f"{float(row['mean']):.4f}",
                    f"{float(row['p1']):.4f}",
                    f"{float(row['p10']):.4f}",
                    f"{float(row['median']):.4f}",
                ]
            )
    fig, axis = plt.subplots(figsize=(8.3, 4.4))
    axis.axis("off")
    table = axis.table(
        cellText=table_rows,
        rowLabels=row_labels,
        colLabels=["Mean", "P1", "P10", "Median"],
        cellLoc="right",
        rowLoc="left",
        colLoc="center",
        bbox=[0.0, 0.0, 1.0, 0.91],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.6)
        if row_index == 0:
            cell.set_facecolor("#E8EDF3")
            cell.set_text_props(weight="bold", color="#111111")
        elif column_index == -1:
            scheme = row_labels[row_index - 1].split(" - ", 1)[0]
            cell.set_facecolor(COLORS[scheme])
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if row_index % 2 else "#F7F7F7")
    axis.set_title("GEANT normalized fulfill ratio summary", pad=12, fontsize=12)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def markdown_table(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[str],
    headers: Sequence[str],
    precision: int = 4,
) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.{precision}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    args: argparse.Namespace,
    expected: int,
    fulfill_rows: Sequence[Mapping[str, object]],
    mlu_rows: Sequence[Mapping[str, object]],
    runtime_rows: Sequence[Mapping[str, object]],
) -> None:
    compact_columns = ("scheme", "traffic_class", "mean", "p1", "p10", "median")
    compact_headers = ("Scheme", "Class", "Mean", "P1", "P10", "Median")
    runtime_columns = ("scheme", "mean", "median", "p95", "p99")
    runtime_headers = ("Scheme", "Mean (s)", "Median (s)", "P95 (s)", "P99 (s)")
    lookup = rows_by_scheme_class(fulfill_rows)

    observations = []
    for traffic_class in ("Medium", "Low"):
        h = float(lookup[("Hattrick", traffic_class)]["p10"])
        b = float(lookup[("BEST_MC", traffic_class)]["p10"])
        s = float(lookup[("SWAN", traffic_class)]["p10"])
        observations.append(
            f"- {traffic_class} P10: Hattrick={h:.4f}, "
            f"BEST_MC={b:.4f}, SWAN={s:.4f}."
        )

    contents = f"""# GEANT Hattrick experiment summary

## Experiment configuration

- Topology: `{args.topology}`
- Candidate paths per OD pair: `{args.num_paths}`
- Predictor: `{args.predictor}`
- Test interval: `[{args.test_start}, {args.test_end})`
- Test snapshots: `{expected}`
- Classes: High, Medium, Low
- Normalization oracle: Gurobi results computed with ground-truth traffic matrices

## Normalized fulfill ratio

{markdown_table(fulfill_rows, compact_columns, compact_headers)}

![Normalized fulfill ratio CDF](geant_fulfill_cdf.png)

![Normalized fulfill ratio boxplot](geant_fulfill_boxplot.png)

## Hattrick normalized MLU diagnostic

{markdown_table(mlu_rows, compact_columns, compact_headers)}

![Hattrick normalized MLU diagnostic CDF](geant_mlu_cdf.png)

## Computation time

{markdown_table(runtime_rows, runtime_columns, runtime_headers, precision=6)}

![Computation time](geant_runtime.png)

## Main observations

{chr(10).join(observations)}
- Values above 1 are retained rather than clipped, matching the paper's treatment of priority inversion and numerical tolerance.
- BEST_MC and SWAN runtimes are the sum of their three sequential optimization stages.
- Cross-scheme MLU is intentionally omitted: baseline simulator MLU is recorded after partial admission, so it is not directly comparable with Hattrick's normalized full-demand MLU diagnostic.

## Reproducibility notes

- Hattrick result files contain three interleaved values per test snapshot.
- BEST_MC and SWAN simulator files contain six values per snapshot: MLU and cumulative admitted flow after each priority stage.
- Baseline arrays cover all snapshots and are sliced to the same test interval before comparison.
- This run uses `K={args.num_paths}`; the paper's GEANT experiment uses `K=8`, so the figures reproduce the evaluation format and trend, not the exact published configuration.
- Runtime measurements were collected on the hardware used for each original run. Do not claim a hardware-controlled speedup unless all schemes are rerun under a comparable setup.
"""
    path.write_text(contents, encoding="utf-8")


def verify_outputs(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("Expected output files were not created: " + ", ".join(missing))
    empty = [str(path) for path in paths if path.stat().st_size == 0]
    if empty:
        raise RuntimeError("Output files are empty: " + ", ".join(empty))


def main() -> None:
    args = parse_args()
    if args.test_end <= args.test_start:
        raise ValueError("--test-end must be greater than --test-start")

    project_root = args.project_root.resolve()
    results_dir = (
        args.results_dir.resolve()
        if args.results_dir
        else project_root
        / "results"
        / args.topology
        / f"{args.num_paths}sp"
        / str(args.cluster)
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else project_root / "output"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    expected = args.test_end - args.test_start
    predictor = args.predictor

    oracle_flow_cumulative = load_oracle_columns(
        results_dir,
        (
            "gt_optimal_values_mf.txt",
            "gt_optimal_values_mf_mf.txt",
            "gt_optimal_values_mf_mf_mf.txt",
        ),
        args.test_start,
        args.test_end,
        expected,
        "flow oracle",
    )
    flexile_sim = load_six_column_baseline(
        results_dir / f"flexile_sim_results_{predictor}_mf_mf_mf.txt",
        args.test_start,
        args.test_end,
        expected,
        "BEST_MC simulator",
    )
    swan_sim = load_six_column_baseline(
        results_dir / f"swan_sim_results_{predictor}_mf_mf_mf.txt",
        args.test_start,
        args.test_end,
        expected,
        "SWAN simulator",
    )

    fulfill: Dict[str, np.ndarray] = {
        "Hattrick": load_three_column_test_file(
            results_dir / f"hattrick_values_{predictor}_sim_mlu_1.txt",
            expected,
            "Hattrick fulfill result",
        ),
        "BEST_MC": load_baseline_fulfill(
            flexile_sim, oracle_flow_cumulative, "BEST_MC fulfill ratio"
        ),
        "SWAN": load_baseline_fulfill(
            swan_sim, oracle_flow_cumulative, "SWAN fulfill ratio"
        ),
    }
    normalized_mlu: Dict[str, np.ndarray] = {
        "Hattrick": load_three_column_test_file(
            results_dir / f"hattrick_values_{predictor}_sim_mlu_0.txt",
            expected,
            "Hattrick MLU result",
        ),
    }

    hattrick_runtime_path = results_dir / "hattrick_runtime_flow.txt"
    if not hattrick_runtime_path.is_file():
        hattrick_runtime_path = results_dir / "hattrick_runtime.txt"
    runtime: Dict[str, np.ndarray] = {
        "Hattrick": load_runtime_series(
            hattrick_runtime_path,
            args.test_start,
            args.test_end,
            expected,
            "Hattrick runtime",
        )
    }
    for scheme, prefix in (("BEST_MC", "flexile"), ("SWAN", "swan")):
        stage_values = [
            load_runtime_series(
                results_dir / f"{prefix}_runtime_{stage}.txt",
                args.test_start,
                args.test_end,
                expected,
                f"{scheme} runtime stage {stage}",
            )
            for stage in (1, 2, 3)
        ]
        runtime[scheme] = stage_values[0] + stage_values[1] + stage_values[2]

    fulfill_rows = summarize_class_metrics(fulfill)
    mlu_rows = summarize_class_metrics(normalized_mlu, scheme_order=("Hattrick",))
    runtime_rows = summarize_runtime(runtime)

    per_snapshot_csv = output_dir / "geant_per_snapshot_metrics.csv"
    fulfill_csv = output_dir / "geant_fulfill_summary.csv"
    mlu_csv = output_dir / "geant_mlu_summary.csv"
    runtime_csv = output_dir / "geant_runtime_summary.csv"
    report_md = output_dir / "geant_report.md"
    fulfill_cdf_png = output_dir / "geant_fulfill_cdf.png"
    fulfill_boxplot_png = output_dir / "geant_fulfill_boxplot.png"
    mlu_cdf_png = output_dir / "geant_mlu_cdf.png"
    runtime_png = output_dir / "geant_runtime.png"
    table_png = output_dir / "geant_fulfill_table.png"

    write_per_snapshot_csv(
        per_snapshot_csv, args.test_start, fulfill, normalized_mlu
    )
    write_dict_csv(fulfill_csv, fulfill_rows)
    write_dict_csv(mlu_csv, mlu_rows)
    write_dict_csv(runtime_csv, runtime_rows)

    configure_plots()
    plot_cdf_panels(
        fulfill,
        fulfill_cdf_png,
        "Normalized fulfill ratio",
        "GEANT: Hattrick vs. BEST_MC and SWAN",
        args.dpi,
    )
    plot_fulfill_boxplot(fulfill, fulfill_boxplot_png, args.dpi)
    plot_cdf_panels(
        normalized_mlu,
        mlu_cdf_png,
        "Normalized MLU",
        "GEANT Hattrick normalized MLU diagnostic",
        args.dpi,
        plot_order=("Hattrick",),
    )
    plot_runtime(runtime, runtime_png, args.dpi)
    plot_summary_table(fulfill_rows, table_png, args.dpi)
    write_report(
        report_md,
        args,
        expected,
        fulfill_rows,
        mlu_rows,
        runtime_rows,
    )

    outputs = (
        per_snapshot_csv,
        fulfill_csv,
        mlu_csv,
        runtime_csv,
        report_md,
        fulfill_cdf_png,
        fulfill_boxplot_png,
        mlu_cdf_png,
        runtime_png,
        table_png,
    )
    verify_outputs(outputs)

    print(f"Results directory: {results_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"Test snapshots:    {expected}")
    print("Generated files:")
    for output in outputs:
        print(f"  {output.name}: {output.stat().st_size} bytes")


if __name__ == "__main__":
    main()
