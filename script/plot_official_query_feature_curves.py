#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_COLUMNS = {
    "raw_rr_stage1",
    "raw_rr_stage2",
    "raw_delta_rr",
    "rr_stage1",
    "rr_stage2",
    "delta_rr",
    "gain_pt",
    "official_relation_delta_mrr",
    "calibration_offset",
}

ID_COLUMNS = {
    "dataset",
    "experiment",
    "relation",
    "direction",
    "query",
    "filename",
    "case_id",
    "target_gt_entity",
}


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def numeric_feature_names(df: pd.DataFrame) -> list[str]:
    names = []
    for col in df.columns:
        if col in METRIC_COLUMNS or col in ID_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            names.append(col)
    return names


def subset_curve(group: pd.DataFrame, feature: str, direction: str) -> list[dict]:
    ordered = group.sort_values(feature, ascending=(direction == "asc")).reset_index(drop=True)
    n_total = len(ordered)
    rows = []
    for pct in range(1, 101):
        n = max(1, round(n_total * pct / 100.0))
        subset = ordered.iloc[:n]
        mrr_s1 = float(subset["rr_stage1"].mean())
        mrr_s2 = float(subset["rr_stage2"].mean())
        gain_pt = (mrr_s2 / mrr_s1 - 1.0) if mrr_s1 > 0 else 0.0
        rows.append({
            "dataset": str(group["dataset"].iloc[0]),
            "feature": feature,
            "sort_direction": direction,
            "coverage": pct / 100.0,
            "n": int(n),
            "threshold": float(subset[feature].iloc[-1]),
            "mrr_stage1": mrr_s1,
            "mrr_stage2": mrr_s2,
            "delta_mrr": mrr_s2 - mrr_s1,
            "gain_pt": gain_pt,
        })
    return rows


def build_curves(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        for direction in ("desc", "asc"):
            for _dataset, group in df.groupby("dataset", sort=True):
                rows.extend(subset_curve(group, feature, direction))
    return pd.DataFrame(rows)


def plot_feature(curves: pd.DataFrame, feature: str, direction: str, out_path: str):
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    subset = curves[(curves["feature"] == feature) & (curves["sort_direction"] == direction)]
    y_values = []
    for dataset, group in subset.groupby("dataset", sort=True):
        group = group.sort_values("coverage")
        ys = group["gain_pt"] * 100.0
        y_values.extend(float(v) for v in ys if pd.notna(v) and np.isfinite(v))
        ax.plot(
            group["coverage"] * 100.0,
            ys,
            linewidth=1.25,
            alpha=0.46,
            label=dataset,
        )

    macro = subset.groupby("coverage", as_index=False)["gain_pt"].mean().sort_values("coverage")
    macro_y = macro["gain_pt"] * 100.0
    y_values.extend(float(v) for v in macro_y if pd.notna(v) and np.isfinite(v))
    ax.plot(
        macro["coverage"] * 100.0,
        macro_y,
        color="#111111",
        linewidth=2.6,
        marker="o",
        markersize=2.6,
        markevery=5,
        label="ALL macro avg",
        zorder=5,
    )

    ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.7)
    ax.set_xlim(0, 100)
    if y_values:
        lo, hi = np.percentile(y_values, [2, 98])
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
        pad = max((hi - lo) * 0.08, 0.5)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"Data coverage (%) ranked by {feature} ({direction})")
    ax.set_ylabel("gain_pt in subset (%)")
    ax.set_title(f"{feature} ({direction})")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_feature_rankings(curves: pd.DataFrame, coverages: list[float]) -> pd.DataFrame:
    rows = []
    for coverage in coverages:
        subset = curves[curves["coverage"].round(8).eq(round(coverage, 8))]
        grouped = subset.groupby(["feature", "sort_direction"], as_index=False).agg(
            macro_gain_pt=("gain_pt", "mean"),
            macro_delta_mrr=("delta_mrr", "mean"),
            min_dataset_gain_pt=("gain_pt", "min"),
            max_dataset_gain_pt=("gain_pt", "max"),
            positive_dataset_count=("gain_pt", lambda s: int((s > 0).sum())),
        )
        grouped["coverage"] = coverage
        grouped = grouped.sort_values("macro_gain_pt", ascending=False).reset_index(drop=True)
        grouped["rank"] = grouped.index + 1
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def write_ranking_markdown(rankings: pd.DataFrame, out_path: str, topn: int = 30):
    lines = [
        "# Feature Rankings at Fixed Coverage",
        "",
        "Ranking score: macro-average `gain_pt` across datasets at the same coverage.",
        "",
        "`sort_direction=desc` keeps larger feature values first; `asc` keeps smaller feature values first.",
        "",
    ]
    for coverage, group in rankings.groupby("coverage", sort=True):
        lines.append(f"## Coverage {coverage:.0%}")
        lines.append("")
        lines.append("| rank | feature | order | macro gain_pt | positive datasets | min dataset gain | max dataset gain |")
        lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: |")
        for row in group.sort_values("rank").head(topn).to_dict("records"):
            lines.append(
                "| {rank} | `{feature}` | {sort_direction} | {macro_gain_pt:.4f} | "
                "{positive_dataset_count} | {min_dataset_gain_pt:.4f} | {max_dataset_gain_pt:.4f} |".format(**row)
            )
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_plot_index(out_dir: str, features: list[str]):
    lines = [
        "# Feature Curve Plots",
        "",
        "Each image fixes one feature and shows six dataset curves.",
        "",
        "- `__desc`: keep triples with larger feature values first.",
        "- `__asc`: keep triples with smaller feature values first.",
        "- x-axis: data coverage in the selected subset.",
        "- y-axis: `gain_pt = MRR_stage2 / MRR_stage1 - 1` in the selected subset.",
        "- `ALL macro avg` is the average of dataset-level `gain_pt` values at the same coverage point.",
        "",
        "## Feature Calculation",
        "",
        "All features are computed from the query/candidate-set graph before looking at which candidate is the GT. The threshold search does not use `target_gt_entity`, GT rank, GT score, or any GT-specific dependency value.",
        "",
        "### Basic Query Size",
        "",
        "- `num_candidates`: number of candidates retained for this test triple query.",
        "- `num_rule_nodes`: number of unique rule IDs appearing in any candidate's active rule list.",
        "- `num_dependency_edges`: number of unique displayed dependency pairs among active candidate rules.",
        "- `query_num_nodes`: node count stored in `queries.json`; should match the query graph's rule-node count used by the demo.",
        "- `query_num_edges`: edge count stored in `queries.json`; should match the query graph's dependency-edge count used by the demo.",
        "",
        "### Candidate Rule Support",
        "",
        "For each candidate, `rules` is the active rule list and `scoredRuleCount` is the number of scored rules.",
        "",
        "- `avg_rules_per_candidate`: mean `scoredRuleCount` over all candidates.",
        "- `max_rules_per_candidate`: max `scoredRuleCount` over all candidates.",
        "- `candidate_rule_coverage`: fraction of candidates with at least one active rule.",
        "",
        "### Candidate Dependency Activity",
        "",
        "For each candidate, `positiveDep`, `negativeDep`, and `dependencyScore` are candidate-level dependency aggregates from Stage2 scoring. These are aggregated over the whole candidate set, not over the GT.",
        "",
        "- `candidate_dep_coverage`: fraction of candidates with `positiveDep + negativeDep > 0`.",
        "- `sum_positive_dep`, `sum_negative_dep`: sums over all candidates.",
        "- `avg_positive_dep`, `avg_negative_dep`: means over all candidates.",
        "- `max_positive_dep`, `max_negative_dep`: maxima over all candidates.",
        "- `avg_candidate_dep_score`: mean candidate `dependencyScore`.",
        "- `max_candidate_dep_score`: max candidate `dependencyScore`.",
        "",
        "### Candidate Score Distribution",
        "",
        "These use candidate-set score distributions, not GT scores.",
        "",
        "- `avg_stage1_score`, `max_stage1_score`: mean/max official Stage1 candidate score.",
        "- `stage1_top_margin`: top-1 minus top-2 official Stage1 candidate score.",
        "- `avg_stage2_score`, `max_stage2_score`: mean/max official Stage2 candidate score.",
        "- `stage2_top_margin`: top-1 minus top-2 official Stage2 candidate score.",
        "",
        "### Dependency Edge Types",
        "",
        "`displayedDependencyPairs` gives active rule-rule pairs in candidates. The pair type and weight are looked up from `data/<dataset>/rules/synergy_filtered.txt` and `redundancy_filtered.txt`.",
        "",
        "- `unique_synergy_edges`: number of unique active dependency pairs found in `synergy_filtered.txt`.",
        "- `unique_redundancy_edges`: number of unique active dependency pairs found in `redundancy_filtered.txt`.",
        "",
        "### Rule Weights",
        "",
        "For each candidate, `maxplus` contains rule contribution values shown by the demo. The `rule_weight_*` features pool these values across all candidates in the query.",
        "",
        "- `rule_weight_topK_sum`: sum of the top K pooled rule contribution values, for K in `{1,3,5,10}`.",
        "- `rule_weight_topK_mean`: mean of the top K pooled rule contribution values.",
        "- `rule_weight_max`: max pooled rule contribution value.",
        "- `rule_weight_mean`: mean pooled rule contribution value.",
        "",
        "### Synergy and Redundancy Weights",
        "",
        "For active dependency pairs, the absolute filtered dependency weight is used. Synergy and redundancy are computed separately.",
        "",
        "- `synergy_weight_topK_sum`, `redundancy_weight_topK_sum`: sum of top K absolute active dependency weights.",
        "- `synergy_weight_topK_mean`, `redundancy_weight_topK_mean`: mean of top K absolute active dependency weights.",
        "- `synergy_weight_max`, `redundancy_weight_max`: max absolute active dependency weight.",
        "- `synergy_weight_mean`, `redundancy_weight_mean`: mean absolute active dependency weight.",
        "",
        "### Composite Features",
        "",
        "Composite features use dataset-wise percentile ranks so that features with different scales can be combined. A high percentile means the query is high on that feature within the dataset.",
        "",
        "- `combo_dependency_activity`: mean percentile rank of `num_dependency_edges`, `candidate_dep_coverage`, `sum_positive_dep`, `sum_negative_dep`, `unique_synergy_edges`, and `unique_redundancy_edges`.",
        "- `combo_candidate_complexity`: mean percentile rank of `num_candidates`, `avg_rules_per_candidate`, and `rule_weight_mean`.",
        "- `combo_stage1_uncertainty`: mean of `1 - percentile(max_stage1_score)` and `1 - percentile(stage1_top_margin)`.",
        "- `combo_low_rule_confidence`: `1 - percentile(rule_weight_max)`.",
        "- `combo_dep_activity_x_uncertainty`: `combo_dependency_activity * combo_stage1_uncertainty`.",
        "- `combo_complex_dep_low_conf`: mean of `combo_candidate_complexity`, `combo_dependency_activity`, and `combo_low_rule_confidence`.",
        "",
        "## Plot Links",
        "",
    ]
    for feature in features:
        slug = slugify(feature)
        lines.append(f"- `{feature}`: [{slug}__desc.png]({slug}__desc.png), [{slug}__asc.png]({slug}__asc.png)")
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", default="/home/sy/RuleDep/reports/0421/official_query_subset/official_query_triple_features.csv")
    parser.add_argument("--out-dir", default="/home/sy/RuleDep/reports/0421/official_query_subset")
    args = parser.parse_args()

    df = pd.read_csv(args.features_csv)
    features = numeric_feature_names(df)
    curves = build_curves(df, features)

    os.makedirs(args.out_dir, exist_ok=True)
    curves_path = os.path.join(args.out_dir, "feature_threshold_curves.csv")
    curves.to_csv(curves_path, index=False)

    plot_dir = os.path.join(args.out_dir, "feature_plots")
    os.makedirs(plot_dir, exist_ok=True)
    for feature in features:
        for direction in ("desc", "asc"):
            plot_feature(curves, feature, direction, os.path.join(plot_dir, f"{slugify(feature)}__{direction}.png"))
    write_plot_index(plot_dir, features)

    rankings = build_feature_rankings(curves, [0.10, 0.20])
    rankings.to_csv(os.path.join(args.out_dir, "feature_rankings_at_coverage.csv"), index=False)
    write_ranking_markdown(rankings, os.path.join(args.out_dir, "feature_rankings_at_coverage.md"))

    print(f"features={len(features)}")
    print(f"curves={len(curves)}")
    print(f"plots={len(features) * 2}")


if __name__ == "__main__":
    main()
