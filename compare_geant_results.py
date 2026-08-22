#!/usr/bin/env python3
"""Build a unified GEANT comparison for baseline and optimized Hattrick."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = [
    "Hattrick (Baseline)",
    "Hattrick (Optimized)",
]
CLASSES = ["High", "Medium", "Low"]
COLORS = {
    "Hattrick (Baseline)": "#4C78A8",
    "Hattrick (Optimized)": "#F58518",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, default=Path("output/geant_k8"))
    parser.add_argument(
        "--optimized-dir", type=Path, default=Path("output/geant_k8_optimized")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/compare_on_geant")
    )
    return parser.parse_args()


def require_files(directory: Path, names: list[str]) -> None:
    missing = [str(directory / name) for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing comparison inputs: " + ", ".join(missing))


def baseline_hattrick(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[frame["scheme"] == "Hattrick"].copy()
    result["scheme"] = "Hattrick (Baseline)"
    return result


def optimized_hattrick(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[frame["scheme"] == "Hattrick"].copy()
    result["scheme"] = "Hattrick (Optimized)"
    return result


def load_inputs(baseline_dir: Path, optimized_dir: Path) -> dict[str, pd.DataFrame]:
    names = [
        "geant_fulfill_summary.csv",
        "geant_mlu_summary.csv",
        "geant_runtime_summary.csv",
        "geant_per_snapshot_metrics.csv",
    ]
    require_files(baseline_dir, names)
    require_files(optimized_dir, names)

    baseline = {name: pd.read_csv(baseline_dir / name) for name in names}
    optimized = {name: pd.read_csv(optimized_dir / name) for name in names}

    fulfill = pd.concat(
        [
            baseline_hattrick(baseline["geant_fulfill_summary.csv"]),
            optimized_hattrick(optimized["geant_fulfill_summary.csv"]),
        ],
        ignore_index=True,
    )
    runtime = pd.concat(
        [
            baseline_hattrick(baseline["geant_runtime_summary.csv"]),
            optimized_hattrick(optimized["geant_runtime_summary.csv"]),
        ],
        ignore_index=True,
    )
    mlu = pd.concat(
        [
            baseline_hattrick(baseline["geant_mlu_summary.csv"]),
            optimized_hattrick(optimized["geant_mlu_summary.csv"]),
        ],
        ignore_index=True,
    )
    per_snapshot = pd.concat(
        [
            baseline_hattrick(baseline["geant_per_snapshot_metrics.csv"]),
            optimized_hattrick(optimized["geant_per_snapshot_metrics.csv"]),
        ],
        ignore_index=True,
    )
    return {
        "fulfill": fulfill,
        "runtime": runtime,
        "mlu": mlu,
        "per_snapshot": per_snapshot,
    }


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["scheme"] = pd.Categorical(result["scheme"], METHODS, ordered=True)
    if "traffic_class" in result:
        result["traffic_class"] = pd.Categorical(
            result["traffic_class"], CLASSES, ordered=True
        )
        result = result.sort_values(["scheme", "traffic_class"])
    else:
        result = result.sort_values("scheme")
    result["scheme"] = result["scheme"].astype("object")
    if "traffic_class" in result:
        result["traffic_class"] = result["traffic_class"].astype("object")
    return result.reset_index(drop=True)


def save_combined_csvs(data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    for name in ("fulfill", "runtime", "mlu"):
        ordered(data[name]).to_csv(output_dir / f"geant_{name}_summary.csv", index=False)
    ordered(data["per_snapshot"]).to_csv(
        output_dir / "geant_per_snapshot_metrics.csv", index=False
    )

    rows = []
    for _, row in ordered(data["fulfill"]).iterrows():
        for statistic in ("mean", "p1", "p10", "median", "p95", "p99"):
            rows.append(
                {
                    "scheme": row["scheme"],
                    "traffic_class": row["traffic_class"],
                    "metric": "normalized_fulfill_ratio",
                    "statistic": statistic,
                    "value": row[statistic],
                }
            )
    for _, row in ordered(data["mlu"]).iterrows():
        for statistic in ("mean", "p1", "p10", "median", "p95", "p99"):
            rows.append(
                {
                    "scheme": row["scheme"],
                    "traffic_class": row["traffic_class"],
                    "metric": "normalized_mlu",
                    "statistic": statistic,
                    "value": row[statistic],
                }
            )
    for _, row in ordered(data["runtime"]).iterrows():
        for statistic in ("mean", "median", "p95", "p99"):
            rows.append(
                {
                    "scheme": row["scheme"],
                    "traffic_class": "All",
                    "metric": "runtime_seconds",
                    "statistic": statistic,
                    "value": row[statistic],
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "geant_all_key_metrics.csv", index=False)


def plot_fulfill_table(data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fulfill = ordered(data["fulfill"])
    rows = []
    row_labels = []
    for method in METHODS:
        short_name = "Baseline" if method.endswith("Baseline)") else "Optimized"
        for traffic_class in CLASSES:
            selected = fulfill.loc[
                (fulfill["scheme"] == method)
                & (fulfill["traffic_class"] == traffic_class)
            ].iloc[0]
            row_labels.append(f"{short_name} - {traffic_class}")
            rows.append(
                [
                    f"{selected['mean']:.6f}",
                    f"{selected['p1']:.6f}",
                    f"{selected['p10']:.6f}",
                    f"{selected['median']:.6f}",
                ]
            )
    figure, ax = plt.subplots(figsize=(9.2, 4.3))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=["Mean", "P1", "P10", "Median"],
        cellLoc="right",
        rowLoc="left",
        colLoc="center",
        bbox=[0.0, 0.0, 1.0, 0.91],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        if row == 0:
            cell.set_facecolor("#E8EDF3")
            cell.set_text_props(weight="bold")
        elif column == -1:
            method = METHODS[0] if row <= 3 else METHODS[1]
            cell.set_facecolor(COLORS[method])
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#F7F7F7")
    ax.set_title("GEANT normalized fulfill ratio summary", pad=12)
    figure.savefig(output_dir / "geant_fulfill_table.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_fulfill_cdf(data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    per_snapshot = ordered(data["per_snapshot"])
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), sharey=True)
    for ax, traffic_class in zip(axes, CLASSES):
        for method in METHODS:
            values = per_snapshot.loc[
                (per_snapshot["scheme"] == method)
                & (per_snapshot["traffic_class"] == traffic_class),
                "normalized_fulfill_ratio",
            ].dropna()
            values = np.sort(values.to_numpy(dtype=float))
            cumulative = np.arange(1, len(values) + 1) / len(values)
            ax.plot(values, cumulative, label=method, color=COLORS[method], linewidth=2)
        ax.axvline(1.0, color="#666666", linestyle="--", linewidth=1)
        ax.axhline(0.01, color="#AAAAAA", linestyle=":", linewidth=0.8)
        ax.axhline(0.10, color="#AAAAAA", linestyle=":", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_ylim(1.0 / 2154.0, 1.0)
        ax.ticklabel_format(style="plain", axis="x", useOffset=False)
        ax.set_title(f"{traffic_class} class")
        ax.set_xlabel("Normalized fulfill ratio")
        ax.grid(axis="y", which="both", alpha=0.25)
    axes[0].set_ylabel("CDF (log scale)")
    handles = [
        plt.Line2D((0, 1), (0, 0), color=COLORS[method], linewidth=2)
        for method in METHODS
    ]
    figure.suptitle("GEANT K=8 normalized fulfill ratio CDF", y=0.99)
    figure.legend(
        handles,
        METHODS,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.87))
    figure.savefig(output_dir / "geant_fulfill_cdf.png", dpi=180)
    plt.close(figure)


def get_value(
    frame: pd.DataFrame, scheme: str, statistic: str, traffic_class: str | None = None
) -> float:
    mask = frame["scheme"] == scheme
    if traffic_class is not None:
        mask &= frame["traffic_class"] == traffic_class
    return float(frame.loc[mask, statistic].iloc[0])


def build_change_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    before = "Hattrick (Baseline)"
    after = "Hattrick (Optimized)"
    for metric, frame, statistics, classes in (
        ("normalized_fulfill_ratio", data["fulfill"], ("mean", "p1", "p10", "median", "p95", "p99"), CLASSES),
        ("normalized_mlu", data["mlu"], ("mean", "median", "p95", "p99", "maximum"), CLASSES),
        ("runtime_seconds", data["runtime"], ("mean", "median", "p95", "p99", "maximum"), [None]),
    ):
        for traffic_class in classes:
            for statistic in statistics:
                baseline = get_value(frame, before, statistic, traffic_class)
                optimized = get_value(frame, after, statistic, traffic_class)
                rows.append(
                    {
                        "metric": metric,
                        "traffic_class": traffic_class or "All",
                        "statistic": statistic,
                        "baseline": baseline,
                        "optimized": optimized,
                        "absolute_change": optimized - baseline,
                        "relative_change_percent": (optimized / baseline - 1.0) * 100.0,
                    }
                )
    return pd.DataFrame(rows)


def plot_fulfill_boxplot(data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    per_snapshot = ordered(data["per_snapshot"])
    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.0))
    for ax, traffic_class in zip(axes, CLASSES):
        series = [
            per_snapshot.loc[
                (per_snapshot["scheme"] == method)
                & (per_snapshot["traffic_class"] == traffic_class),
                "normalized_fulfill_ratio",
            ].dropna().to_numpy(dtype=float)
            for method in METHODS
        ]
        box = ax.boxplot(
            series,
            positions=[0, 1],
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            whis=(1, 99),
            medianprops={"color": "white", "linewidth": 1.4},
        )
        for patch, method in zip(box["boxes"], METHODS):
            patch.set_facecolor(COLORS[method])
            patch.set_edgecolor(COLORS[method])
            patch.set_alpha(0.9)
        combined = np.concatenate(series)
        lower, upper = np.percentile(combined, [1, 99])
        padding = max((upper - lower) * 0.12, 1e-7)
        ax.set_ylim(lower - padding, upper + padding)
        ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1)
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)
        ax.set_xticks([0, 1], ["Baseline", "Optimized"])
        ax.set_title(f"{traffic_class} class")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Normalized fulfill ratio\n(zoomed to P1-P99)")
    figure.suptitle("GEANT normalized fulfill ratio (P1-P99 boxplot)", y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_dir / "geant_fulfill_boxplot.png", dpi=180)
    plt.close(figure)


def plot_mlu_cdf(data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    per_snapshot = ordered(data["per_snapshot"])
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), sharey=True)
    for ax, traffic_class in zip(axes, CLASSES):
        all_values = []
        for method in METHODS:
            values = per_snapshot.loc[
                (per_snapshot["scheme"] == method)
                & (per_snapshot["traffic_class"] == traffic_class),
                "normalized_mlu",
            ].dropna().to_numpy(dtype=float)
            values = (values - 1.0) * 1000.0
            all_values.append(values)
            values = np.sort(values)
            cumulative = np.arange(1, len(values) + 1) / len(values)
            ax.plot(values, cumulative, color=COLORS[method], linewidth=2)
        combined = np.concatenate(all_values)
        lower, upper = np.percentile(combined, [0.5, 99.5])
        padding = max((upper - lower) * 0.05, 0.01)
        ax.set_xlim(lower - padding, upper + padding)
        ax.axhline(0.01, color="#AAAAAA", linestyle=":", linewidth=0.8)
        ax.axhline(0.10, color="#AAAAAA", linestyle=":", linewidth=0.8)
        ax.set_title(f"{traffic_class} class")
        ax.set_xlabel("(Normalized MLU - 1) x 1000")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("CDF")
    handles = [
        plt.Line2D((0, 1), (0, 0), color=COLORS[method], linewidth=2)
        for method in METHODS
    ]
    figure.suptitle("GEANT normalized MLU CDF (P0.5-P99.5 x-axis)", y=0.99)
    figure.legend(
        handles,
        METHODS,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.87))
    figure.savefig(output_dir / "geant_mlu_cdf.png", dpi=180)
    plt.close(figure)


def plot_runtime(data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    statistics = ["mean", "median", "p95"]
    x = np.arange(len(statistics))
    width = 0.36
    figure, ax = plt.subplots(figsize=(7.2, 4.5))
    all_values = []
    for index, method in enumerate(METHODS):
        values = [get_value(data["runtime"], method, statistic) * 1000 for statistic in statistics]
        all_values.extend(values)
        bars = ax.bar(
            x + (index - 0.5) * width,
            values,
            width,
            color=COLORS[method],
            label=method,
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_ylim(min(all_values) - 0.8, max(all_values) + 0.8)
    ax.set_xticks(x, [statistic.upper() for statistic in statistics])
    ax.set_ylabel("Milliseconds per snapshot (zoomed axis)")
    ax.set_title("GEANT Hattrick inference time")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True, edgecolor="#BBBBBB")
    figure.tight_layout()
    figure.savefig(output_dir / "geant_runtime.png", dpi=180)
    plt.close(figure)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    data: dict[str, pd.DataFrame], changes: pd.DataFrame, output_dir: Path
) -> None:
    fulfill_rows = []
    for method in METHODS:
        for traffic_class in CLASSES:
            fulfill_rows.append(
                [
                    method,
                    traffic_class,
                    f"{get_value(data['fulfill'], method, 'mean', traffic_class):.6f}",
                    f"{get_value(data['fulfill'], method, 'p1', traffic_class):.6f}",
                    f"{get_value(data['fulfill'], method, 'p10', traffic_class):.6f}",
                    f"{get_value(data['fulfill'], method, 'median', traffic_class):.6f}",
                ]
            )
    runtime_rows = [
        [
            method,
            f"{get_value(data['runtime'], method, 'mean') * 1000:.3f}",
            f"{get_value(data['runtime'], method, 'median') * 1000:.3f}",
            f"{get_value(data['runtime'], method, 'p95') * 1000:.3f}",
        ]
        for method in METHODS
    ]
    mlu_rows = []
    for method in METHODS[:2]:
        for traffic_class in CLASSES:
            mlu_rows.append(
                [
                    method,
                    traffic_class,
                    f"{get_value(data['mlu'], method, 'mean', traffic_class):.6f}",
                    f"{get_value(data['mlu'], method, 'median', traffic_class):.6f}",
                    f"{get_value(data['mlu'], method, 'p95', traffic_class):.6f}",
                    f"{get_value(data['mlu'], method, 'p99', traffic_class):.6f}",
                    f"{get_value(data['mlu'], method, 'maximum', traffic_class):.6f}",
                ]
            )

    def delta(metric: str, cls: str, stat: str) -> float:
        return float(
            changes.loc[
                (changes["metric"] == metric)
                & (changes["traffic_class"] == cls)
                & (changes["statistic"] == stat),
                "absolute_change",
            ].iloc[0]
        )

    runtime_relative = float(
        changes.loc[
            (changes["metric"] == "runtime_seconds")
            & (changes["statistic"] == "mean"),
            "relative_change_percent",
        ].iloc[0]
    )
    report = f"""# GEANT K=8 Hattrick optimization comparison

本目录只比较原始 Hattrick 与 Optimized Hattrick。测试区间为 `[8618, 10772)`，共 2154 个快照。图表类型与基础实验/论文式汇总保持一致，并通过局部放大或对数坐标清晰展示两者差异。

## 一眼看结论

- 优化版平均推理时间相对原始 Hattrick 变化 `{runtime_relative:+.3f}%`，可视为基本不变。
- Medium 类 P1/P10 Fulfill Ratio 分别变化 `{delta('normalized_fulfill_ratio', 'Medium', 'p1'):+.6f}` / `{delta('normalized_fulfill_ratio', 'Medium', 'p10'):+.6f}`。
- Low 类 P10 提升 `{delta('normalized_fulfill_ratio', 'Low', 'p10'):+.6f}`，但更极端的 P1 变化 `{delta('normalized_fulfill_ratio', 'Low', 'p1'):+.6f}`，不能笼统声称所有尾部分位都提升。
- 三类平均 normalized MLU 分别变化：High `{delta('normalized_mlu', 'High', 'mean'):+.6f}`、Medium `{delta('normalized_mlu', 'Medium', 'mean'):+.6f}`、Low `{delta('normalized_mlu', 'Low', 'mean'):+.6f}`；负值表示更接近 Oracle。
- 优化版 High 类 MLU 的 P99/最大值变差，因此平均与常见尾部改善不代表最极端异常点同步改善。

![Fulfill-ratio CDF](geant_fulfill_cdf.png)

![Fulfill-ratio boxplot](geant_fulfill_boxplot.png)

![Fulfill-ratio table](geant_fulfill_table.png)

## Fulfill Ratio

{markdown_table(['Method', 'Class', 'Mean', 'P1', 'P10', 'Median'], fulfill_rows)}

## Hattrick normalized MLU

{markdown_table(['Method', 'Class', 'Mean', 'Median', 'P95', 'P99', 'Max'], mlu_rows)}

![Normalized MLU CDF](geant_mlu_cdf.png)

## 推理/求解时间

{markdown_table(['Method', 'Mean (ms)', 'Median (ms)', 'P95 (ms)'], runtime_rows)}

![Inference time](geant_runtime.png)

## 口径说明

- Fulfill Ratio 是相对 Gurobi ground-truth oracle 的归一化分级流量完成率，可用于观察接纳流量/最大流目标的接近程度。
- Fulfill CDF 使用对数纵轴突出 P1/P10 尾部；CDF 定义和原始数据没有变化。
- Fulfill 箱线图对每个流量类别分别缩放到 P1-P99；MLU CDF 的横轴缩放到 P0.5-P99.5，并保留完整 P99/Max 数值在表格中。
- Runtime 图使用局部放大的毫秒纵轴，柱顶标注真实数值，不能将视觉高度差直接解释为数量级加速或减速。
- 图表的源数据保存在本目录 CSV 文件中，可直接复核。
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")
    (output_dir / "geant_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stale_outputs = [
        "geant_four_method_fulfill_cdf.png",
        "geant_four_method_overview.png",
        "geant_fulfill_comparison.csv",
        "geant_hattrick_before_after.png",
        "geant_mlu_comparison.csv",
        "geant_runtime_comparison.csv",
    ]
    for name in stale_outputs:
        path = args.output_dir / name
        if path.is_file():
            path.unlink()
    data = load_inputs(args.baseline_dir, args.optimized_dir)
    save_combined_csvs(data, args.output_dir)
    changes = build_change_table(data)
    changes.to_csv(args.output_dir / "geant_hattrick_change_summary.csv", index=False)
    plot_fulfill_cdf(data, args.output_dir)
    plot_fulfill_boxplot(data, args.output_dir)
    plot_fulfill_table(data, args.output_dir)
    plot_mlu_cdf(data, args.output_dir)
    plot_runtime(data, args.output_dir)
    write_report(data, changes, args.output_dir)
    print(f"Wrote Hattrick before/after comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
