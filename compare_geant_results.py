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
    "BEST-MC",
    "SWAN",
]
CLASSES = ["High", "Medium", "Low"]
COLORS = {
    "Hattrick (Baseline)": "#4C78A8",
    "Hattrick (Optimized)": "#F58518",
    "BEST-MC": "#54A24B",
    "SWAN": "#E45756",
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


def rename_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["scheme"] = result["scheme"].replace(
        {"Hattrick": "Hattrick (Baseline)", "BEST_MC": "BEST-MC"}
    )
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
            rename_baselines(baseline["geant_fulfill_summary.csv"]),
            optimized_hattrick(optimized["geant_fulfill_summary.csv"]),
        ],
        ignore_index=True,
    )
    runtime = pd.concat(
        [
            rename_baselines(baseline["geant_runtime_summary.csv"]),
            optimized_hattrick(optimized["geant_runtime_summary.csv"]),
        ],
        ignore_index=True,
    )
    mlu = pd.concat(
        [
            rename_baselines(baseline["geant_mlu_summary.csv"]),
            optimized_hattrick(optimized["geant_mlu_summary.csv"]),
        ],
        ignore_index=True,
    )
    per_snapshot = pd.concat(
        [
            rename_baselines(baseline["geant_per_snapshot_metrics.csv"]),
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
        ordered(data[name]).to_csv(output_dir / f"geant_{name}_comparison.csv", index=False)

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


def grouped_bars(
    ax: plt.Axes,
    frame: pd.DataFrame,
    statistic: str,
    title: str,
    ylabel: str,
) -> None:
    x = np.arange(len(CLASSES))
    width = 0.19
    for index, method in enumerate(METHODS):
        values = []
        for traffic_class in CLASSES:
            match = frame.loc[
                (frame["scheme"] == method)
                & (frame["traffic_class"] == traffic_class),
                statistic,
            ]
            values.append(float(match.iloc[0]))
        ax.bar(
            x + (index - 1.5) * width,
            values,
            width,
            label=method,
            color=COLORS[method],
        )
    ax.set_xticks(x, CLASSES)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)


def plot_overview(data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    fulfill = ordered(data["fulfill"])
    runtime = ordered(data["runtime"])
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    grouped_bars(axes[0, 0], fulfill, "mean", "Mean fulfill ratio", "Ratio")
    grouped_bars(axes[0, 1], fulfill, "p10", "P10 fulfill ratio", "Ratio")
    grouped_bars(axes[1, 0], fulfill, "p1", "P1 fulfill ratio", "Ratio")

    values = [
        float(runtime.loc[runtime["scheme"] == method, "mean"].iloc[0]) * 1000
        for method in METHODS
    ]
    bars = axes[1, 1].bar(
        np.arange(len(METHODS)), values, color=[COLORS[method] for method in METHODS]
    )
    axes[1, 1].set_xticks(np.arange(len(METHODS)), METHODS, rotation=18, ha="right")
    axes[1, 1].set_title("Mean inference/optimization time")
    axes[1, 1].set_ylabel("Milliseconds per snapshot")
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].bar_label(bars, fmt="%.1f", padding=3, fontsize=9)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORS[method]) for method in METHODS
    ]
    figure.legend(handles, METHODS, loc="upper center", ncol=4, frameon=False)
    figure.suptitle("GEANT K=8: baseline methods and optimized Hattrick", y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_dir / "geant_four_method_overview.png", dpi=180)
    plt.close(figure)


def plot_cdf(data: dict[str, pd.DataFrame], output_dir: Path) -> None:
    per_snapshot = ordered(data["per_snapshot"])
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
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
        ax.set_title(traffic_class)
        ax.set_xlabel("Normalized fulfill ratio")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("CDF")
    figure.legend(loc="upper center", ncol=4, frameon=False)
    figure.suptitle("GEANT K=8 fulfill-ratio distributions", y=1.02)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output_dir / "geant_four_method_fulfill_cdf.png", dpi=180)
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


def plot_hattrick_change(
    data: dict[str, pd.DataFrame], changes: pd.DataFrame, output_dir: Path
) -> None:
    before = "Hattrick (Baseline)"
    after = "Hattrick (Optimized)"
    labels = ["Baseline", "Optimized"]
    colors = [COLORS[before], COLORS[after]]
    x = np.arange(len(CLASSES))
    width = 0.36
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, statistic, title in (
        (axes[0, 0], "mean", "Mean fulfill ratio"),
        (axes[0, 1], "p10", "P10 fulfill ratio"),
    ):
        for index, method in enumerate((before, after)):
            values = [get_value(data["fulfill"], method, statistic, cls) for cls in CLASSES]
            ax.bar(x + (index - 0.5) * width, values, width, label=labels[index], color=colors[index])
        ax.set_xticks(x, CLASSES)
        ax.set_title(title)
        ax.set_ylabel("Ratio")
        ax.grid(axis="y", alpha=0.25)

    mlu_stats = ["mean", "median", "p95"]
    mlu_x = np.arange(len(CLASSES) * len(mlu_stats))
    tick_labels = [f"{cls}\n{stat.upper()}" for cls in CLASSES for stat in mlu_stats]
    for index, method in enumerate((before, after)):
        values = [
            (get_value(data["mlu"], method, stat, cls) - 1.0) * 1000.0
            for cls in CLASSES
            for stat in mlu_stats
        ]
        axes[1, 0].bar(
            mlu_x + (index - 0.5) * width, values, width, label=labels[index], color=colors[index]
        )
    axes[1, 0].set_xticks(mlu_x, tick_labels)
    axes[1, 0].set_title("Normalized MLU excess over oracle (lower is better)")
    axes[1, 0].set_ylabel("(Normalized MLU - 1) x 1000")
    axes[1, 0].grid(axis="y", alpha=0.25)

    runtime_stats = ["mean", "median", "p95"]
    runtime_x = np.arange(len(runtime_stats))
    for index, method in enumerate((before, after)):
        values = [get_value(data["runtime"], method, stat) * 1000 for stat in runtime_stats]
        bars = axes[1, 1].bar(
            runtime_x + (index - 0.5) * width, values, width, label=labels[index], color=colors[index]
        )
        axes[1, 1].bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    axes[1, 1].set_xticks(runtime_x, [stat.upper() for stat in runtime_stats])
    axes[1, 1].set_title("Hattrick inference time")
    axes[1, 1].set_ylabel("Milliseconds per snapshot")
    axes[1, 1].grid(axis="y", alpha=0.25)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.suptitle("Hattrick before/after optimization", y=0.99)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_dir / "geant_hattrick_before_after.png", dpi=180)
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
    report = f"""# GEANT K=8 unified comparison

本目录把原始 Hattrick、Optimized Hattrick、BEST-MC 和 SWAN 放在同一口径下比较。测试区间为 `[8618, 10772)`，共 2154 个快照。

## 一眼看结论

- 优化版平均推理时间相对原始 Hattrick 变化 `{runtime_relative:+.3f}%`，可视为基本不变。
- Medium 类 P1/P10 Fulfill Ratio 分别变化 `{delta('normalized_fulfill_ratio', 'Medium', 'p1'):+.6f}` / `{delta('normalized_fulfill_ratio', 'Medium', 'p10'):+.6f}`。
- Low 类 P10 提升 `{delta('normalized_fulfill_ratio', 'Low', 'p10'):+.6f}`，但更极端的 P1 变化 `{delta('normalized_fulfill_ratio', 'Low', 'p1'):+.6f}`，不能笼统声称所有尾部分位都提升。
- 三类平均 normalized MLU 分别变化：High `{delta('normalized_mlu', 'High', 'mean'):+.6f}`、Medium `{delta('normalized_mlu', 'Medium', 'mean'):+.6f}`、Low `{delta('normalized_mlu', 'Low', 'mean'):+.6f}`；负值表示更接近 Oracle。
- 优化版 High 类 MLU 的 P99/最大值变差，因此平均与常见尾部改善不代表最极端异常点同步改善。

![Four-method overview](geant_four_method_overview.png)

![Fulfill-ratio CDF](geant_four_method_fulfill_cdf.png)

![Hattrick before and after](geant_hattrick_before_after.png)

## Fulfill Ratio

{markdown_table(['Method', 'Class', 'Mean', 'P1', 'P10', 'Median'], fulfill_rows)}

## Hattrick normalized MLU

{markdown_table(['Method', 'Class', 'Mean', 'Median', 'P95', 'P99', 'Max'], mlu_rows)}

## 推理/求解时间

{markdown_table(['Method', 'Mean (ms)', 'Median (ms)', 'P95 (ms)'], runtime_rows)}

## 口径说明

- Fulfill Ratio 是相对 Gurobi ground-truth oracle 的归一化分级流量完成率，可用于观察接纳流量/最大流目标的接近程度。
- Hattrick 的 MLU 是 full-demand 诊断值；SWAN/BEST-MC 的模拟器 MLU 在部分接纳后记录，口径不同，因此不做跨方法 MLU 柱状排名。
- BEST-MC 和 SWAN 时间为三个顺序优先级阶段的总和。
- 图表的源数据保存在本目录 CSV 文件中，可直接复核。
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_inputs(args.baseline_dir, args.optimized_dir)
    save_combined_csvs(data, args.output_dir)
    changes = build_change_table(data)
    changes.to_csv(args.output_dir / "geant_hattrick_change_summary.csv", index=False)
    plot_overview(data, args.output_dir)
    plot_cdf(data, args.output_dir)
    plot_hattrick_change(data, changes, args.output_dir)
    write_report(data, changes, args.output_dir)
    print(f"Wrote unified GEANT comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
