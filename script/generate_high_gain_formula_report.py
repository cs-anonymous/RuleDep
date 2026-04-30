#!/usr/bin/env python3
from __future__ import annotations

import os

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


REPORT_DIR = "/home/sy/RuleDep/reports/official_query_subset"
FEATURE_PATH = os.path.join(REPORT_DIR, "official_query_triple_features.csv")
PLOT_DIR = os.path.join(REPORT_DIR, "feature_plots")

COVERAGE_POINTS = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
CURVE_COVERAGES = np.arange(1, 101, dtype=float) / 100.0

FORMULA_FEATURES = [
    "topk_synergy",
    "pos_mass",
    "effective_candidates",
    "rule_dominance_ratio",
    "topk_redundancy",
    "syn_rule_ratio",
]
USECOLS = ["dataset", "raw_rr_stage1", "raw_rr_stage2"] + FORMULA_FEATURES


def percentile(df: pd.DataFrame, feature: str) -> np.ndarray:
    return df.groupby("dataset")[feature].rank(pct=True, method="average").to_numpy(dtype=float)


def inv_percentile(df: pd.DataFrame, feature: str) -> np.ndarray:
    return 1.0 - percentile(df, feature)


def build_scores(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, str]]:
    p_topk_synergy = percentile(df, "topk_synergy")
    p_pos_mass = percentile(df, "pos_mass")
    p_effective_candidates = percentile(df, "effective_candidates")
    low_rule_dominance = inv_percentile(df, "rule_dominance_ratio")
    low_topk_redundancy = inv_percentile(df, "topk_redundancy")
    p_syn_rule_ratio = percentile(df, "syn_rule_ratio")

    high_gain_score = (
        3.0 * p_topk_synergy
        + 1.0 * p_pos_mass
        + 3.0 * p_effective_candidates
        + 4.0 * low_rule_dominance
        + 0.25 * low_topk_redundancy
        + 1.0 * p_syn_rule_ratio
    ) / 12.25

    return {
        "compact_high_gain_score": (high_gain_score, "desc"),
        "topk_synergy": (df["topk_synergy"].to_numpy(dtype=float), "desc"),
        "pos_mass": (df["pos_mass"].to_numpy(dtype=float), "desc"),
        "effective_candidates": (df["effective_candidates"].to_numpy(dtype=float), "desc"),
        "rule_dominance_ratio": (df["rule_dominance_ratio"].to_numpy(dtype=float), "asc"),
        "topk_redundancy": (df["topk_redundancy"].to_numpy(dtype=float), "asc"),
        "syn_rule_ratio": (df["syn_rule_ratio"].to_numpy(dtype=float), "desc"),
    }


def curve_for_score(
    df: pd.DataFrame,
    score: np.ndarray,
    direction: str,
    coverages: np.ndarray,
) -> pd.DataFrame:
    rows = []
    datasets = sorted(df["dataset"].unique())
    dataset_values = df["dataset"].to_numpy()
    rr1 = df["raw_rr_stage1"].to_numpy(dtype=float)
    rr2 = df["raw_rr_stage2"].to_numpy(dtype=float)
    reverse = direction == "desc"

    for dataset in datasets:
        idx = np.flatnonzero(dataset_values == dataset)
        ordered = idx[np.argsort(score[idx], kind="mergesort")]
        if reverse:
            ordered = ordered[::-1]
        c1 = np.cumsum(rr1[ordered])
        c2 = np.cumsum(rr2[ordered])
        n_total = len(ordered)
        for coverage in coverages:
            n = max(1, int(round(n_total * float(coverage))))
            mrr1 = c1[n - 1] / n
            mrr2 = c2[n - 1] / n
            rows.append(
                {
                    "dataset": dataset,
                    "coverage": float(coverage),
                    "n": n,
                    "mrr_stage1": mrr1,
                    "mrr_stage2": mrr2,
                    "gain_pt": (mrr2 / mrr1 - 1.0) if mrr1 > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def signed_sqrt(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.sqrt(np.abs(values))


def plot_high_gain(curves: pd.DataFrame, out_path: str, mode: str = "full") -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    y_values = []
    palette = plt.get_cmap("tab10")
    use_sqrt = mode == "sqrt"
    clip50 = mode == "clip50"

    for i, (dataset, group) in enumerate(curves.groupby("dataset", sort=True)):
        group = group.sort_values("coverage")
        raw_y = (group["gain_pt"] * 100.0).to_numpy(dtype=float)
        y_values.extend(raw_y.tolist())
        plot_y = signed_sqrt(raw_y) if use_sqrt else raw_y
        ax.plot(
            group["coverage"] * 100.0,
            plot_y,
            linewidth=1.35,
            alpha=0.72,
            color=palette(i % 10),
            label=dataset,
        )

    macro = curves.groupby("coverage", as_index=False)["gain_pt"].mean().sort_values("coverage")
    macro_y = (macro["gain_pt"] * 100.0).to_numpy(dtype=float)
    y_values.extend(macro_y.tolist())
    macro_plot_y = signed_sqrt(macro_y) if use_sqrt else macro_y
    ax.plot(
        macro["coverage"] * 100.0,
        macro_plot_y,
        color="#111111",
        linewidth=2.8,
        marker="o",
        markersize=2.8,
        markevery=5,
        label="Macro average",
        zorder=5,
    )
    ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.7)
    ax.set_xlim(1, 100)
    if use_sqrt:
        ticks = [0, 1, 4, 9, 16, 25, 50, 100, 150, 200]
        tick_positions = signed_sqrt(np.array(ticks, dtype=float))
        hi = max(max(y_values), 0.0)
        shown = [pos for tick, pos in zip(ticks, tick_positions) if tick <= hi * 1.05 or tick <= 50]
        shown_labels = [tick for tick in ticks[: len(shown)]]
        ax.set_yticks(shown)
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _pos: f"{value * abs(value):.0f}")
        )
        ax.set_ylim(-1, signed_sqrt(np.array([max(hi, 50.0) * 1.08]))[0])
    elif clip50:
        ax.set_ylim(0, 50)
    elif y_values:
        lo = min(min(y_values), 0.0)
        hi = max(max(y_values), 0.0)
        pad = max((hi - lo) * 0.06, 1.0)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Coverage ranked by Compact High-gain Formula (%)")
    if use_sqrt:
        ax.set_ylabel("gain_pt in selected subset (%), sqrt-scaled axis")
        ax.set_title("Compact High-gain Formula (sqrt y-axis)")
    elif clip50:
        ax.set_ylabel("gain_pt in selected subset (%)")
        ax.set_title("Compact High-gain Formula (y-axis clipped at 50%)")
    else:
        ax.set_ylabel("gain_pt in selected subset (%)")
        ax.set_title("Compact High-gain Formula")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fixed_coverage_macro_table(df: pd.DataFrame, scores: dict[str, tuple[np.ndarray, str]]) -> pd.DataFrame:
    rows = []
    for name, (score, direction) in scores.items():
        curves = curve_for_score(df, score, direction, np.array(COVERAGE_POINTS, dtype=float))
        macro = curves.groupby("coverage")["gain_pt"].mean()
        rows.append(
            {
                "feature_or_component": name,
                "sort_direction": direction,
                "5%": macro.loc[0.05],
                "10%": macro.loc[0.10],
                "20%": macro.loc[0.20],
                "30%": macro.loc[0.30],
                "50%": macro.loc[0.50],
                "100%": macro.loc[1.00],
            }
        )
    return pd.DataFrame(rows)


def fixed_coverage_dataset_table(df: pd.DataFrame, scores: dict[str, tuple[np.ndarray, str]]) -> pd.DataFrame:
    rows = []
    for name, (score, direction) in scores.items():
        curves = curve_for_score(df, score, direction, np.array(COVERAGE_POINTS, dtype=float))
        for dataset, group in curves.groupby("dataset", sort=True):
            values = group.set_index("coverage")["gain_pt"]
            rows.append(
                {
                    "feature_or_component": name,
                    "sort_direction": direction,
                    "dataset": dataset,
                    "5%": values.loc[0.05],
                    "10%": values.loc[0.10],
                    "20%": values.loc[0.20],
                    "30%": values.loc[0.30],
                    "50%": values.loc[0.50],
                    "100%": values.loc[1.00],
                }
            )
    return pd.DataFrame(rows)


def format_table(table: pd.DataFrame) -> str:
    out = table.copy()
    for col in out.columns:
        if col not in {"dataset", "feature_or_component", "sort_direction"}:
            out[col] = out[col].map(lambda v: f"{float(v):.4f}")
    return out.to_markdown(index=False)


def write_markdown(curves: pd.DataFrame, macro_table: pd.DataFrame, out_path: str) -> None:
    fixed = curves[curves["coverage"].isin(COVERAGE_POINTS)]
    dataset_table = fixed.pivot(index="dataset", columns="coverage", values="gain_pt").reset_index()
    dataset_table.columns = ["dataset", "5%", "10%", "20%", "30%", "50%", "100%"]
    macro = fixed.groupby("coverage")["gain_pt"].mean()
    dataset_table.loc[len(dataset_table)] = [
        "macro",
        macro.loc[0.05],
        macro.loc[0.10],
        macro.loc[0.20],
        macro.loc[0.30],
        macro.loc[0.50],
        macro.loc[1.00],
    ]

    headline_10 = float(macro.loc[0.10])
    headline_20 = float(macro.loc[0.20])

    lines = [
        "# Compact High-gain Formula Report",
        "",
        (
            f"Headline result: at 10% coverage the selected subset has {headline_10 * 100:.1f}% "
            f"`gain_pt`; at 20% coverage it has {headline_20 * 100:.1f}% `gain_pt`."
        ),
        "",
        "Metric uses the raw official per-query reciprocal ranks, not relation-level calibrated deltas.",
        "",
        "Formula:",
        "",
        "```text",
        "score = (",
        "  3.0 * P_d(topk_synergy)",
        "  + 1.0 * P_d(pos_mass)",
        "  + 3.0 * P_d(effective_candidates)",
        "  + 4.0 * (1 - P_d(rule_dominance_ratio))",
        "  + 0.25 * (1 - P_d(topk_redundancy))",
        "  + 1.0 * P_d(syn_rule_ratio)",
        ") / 12.25",
        "```",
        "",
        "Plot: [`feature_plots/high_gain_formula__desc_sqrt.png`](feature_plots/high_gain_formula__desc_sqrt.png)",
        "",
        "Alternative full-scale plot: [`feature_plots/high_gain_formula__desc.png`](feature_plots/high_gain_formula__desc.png)",
        "",
        "Alternative 0%-50% zoom plot: [`feature_plots/high_gain_formula__desc_clip50.png`](feature_plots/high_gain_formula__desc_clip50.png)",
        "",
        "Curve data: [`high_gain_formula_curves.csv`](high_gain_formula_curves.csv)",
        "",
        "## Fixed Coverage Results",
        "",
        format_table(dataset_table),
        "",
        "## Formula Variables",
        "",
        "The table reports macro-average `gain_pt` when each formula variable is used alone with its formula-consistent sort direction. Per-dataset values are in [`high_gain_formula_variable_dataset_gain.csv`](high_gain_formula_variable_dataset_gain.csv).",
        "",
        format_table(macro_table),
        "",
        (
            f"Headline result repeated: 10% coverage gives {headline_10 * 100:.1f}% `gain_pt`; "
            f"20% coverage gives {headline_20 * 100:.1f}% `gain_pt`."
        ),
        "",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)
    df = pd.read_csv(FEATURE_PATH, usecols=USECOLS)
    scores = build_scores(df)

    high_score, high_direction = scores["compact_high_gain_score"]
    curves = curve_for_score(df, high_score, high_direction, CURVE_COVERAGES)
    curves.to_csv(os.path.join(REPORT_DIR, "high_gain_formula_curves.csv"), index=False)
    plot_high_gain(curves, os.path.join(PLOT_DIR, "high_gain_formula__desc.png"), mode="full")
    plot_high_gain(curves, os.path.join(PLOT_DIR, "high_gain_formula__desc_sqrt.png"), mode="sqrt")
    plot_high_gain(curves, os.path.join(PLOT_DIR, "high_gain_formula__desc_clip50.png"), mode="clip50")

    macro_table = fixed_coverage_macro_table(df, scores)
    macro_table.to_csv(os.path.join(REPORT_DIR, "high_gain_formula_variable_macro_gain.csv"), index=False)
    dataset_table = fixed_coverage_dataset_table(df, scores)
    dataset_table.to_csv(os.path.join(REPORT_DIR, "high_gain_formula_variable_dataset_gain.csv"), index=False)

    write_markdown(curves, macro_table, os.path.join(REPORT_DIR, "high_gain_formula_report.md"))

    fixed_macro = curves[curves["coverage"].isin([0.10, 0.20])].groupby("coverage")["gain_pt"].mean()
    print(f"10% coverage gain_pt={fixed_macro.loc[0.10]:.4f} ({fixed_macro.loc[0.10] * 100:.1f}%)")
    print(f"20% coverage gain_pt={fixed_macro.loc[0.20]:.4f} ({fixed_macro.loc[0.20] * 100:.1f}%)")
    print(os.path.join(PLOT_DIR, "high_gain_formula__desc.png"))
    print(os.path.join(PLOT_DIR, "high_gain_formula__desc_sqrt.png"))
    print(os.path.join(PLOT_DIR, "high_gain_formula__desc_clip50.png"))
    print(os.path.join(REPORT_DIR, "high_gain_formula_variable_macro_gain.csv"))
    print(os.path.join(REPORT_DIR, "high_gain_formula_variable_dataset_gain.csv"))


if __name__ == "__main__":
    main()
