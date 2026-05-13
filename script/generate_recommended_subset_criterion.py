#!/usr/bin/env python3
"""Generate the recommended interpretable subset criterion report.

The recommended selector is the two-feature global RF from the balanced
ablation report:

  balanced2_syn_topk_global_rf
    = RF(synergy_weight_top5_mean, topk_rule_weight)

It is trained once on all datasets, then each dataset keeps the top 10%
queries by predicted gain score. This script makes that criterion explicit
and exports value/percentile ranges for the selected query subset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_balanced5_selector_report as g  # noqa: E402
from generate_balanced5_selector_report_true_rr import load_merged_rr  # noqa: E402
from sweep_official_query_ml_selectors import target_values  # noqa: E402


ROOT = Path("/home/sy/RuleDep")
REPORT_DIR = ROOT / "reports" / "official_query_subset"
OUT_DIR = REPORT_DIR / "ml_selector_diverse" / "recommended_subset_criterion"

SELECTOR = "balanced2_syn_topk_global_rf"
FEATURES = ["synergy_weight_top5_mean", "topk_rule_weight"]
COVERAGE = 0.10
TOP_SINGLE_N = 3

FEATURE_NAMES = {
    "synergy_weight_top5_mean": "Synergy strength",
    "topk_rule_weight": "Top-k rule weight",
    "max_candidate_dep_score": "Max candidate dependency score",
    "effective_candidates": "Effective candidates",
}

FEATURE_DEFINITIONS = {
    "synergy_weight_top5_mean": (
        "Mean of the top 5 absolute synergy dependency weights among unique "
        "displayed rule-pair dependencies in the query case."
    ),
    "topk_rule_weight": (
        "Mean of the top 3 rule weights fired by candidates in the query case."
    ),
    "max_candidate_dep_score": (
        "Maximum candidate-level dependencyScore over all candidate entities in the query case."
    ),
    "effective_candidates": (
        "exp(H(p)), where p is the softmax distribution over candidate stage1 official scores; "
        "larger values mean the stage1 ranker is less concentrated."
    ),
}


def fit_global_score(
    df: pd.DataFrame, selector: str, features: list[str]
) -> tuple[np.ndarray, pd.DataFrame]:
    x = df[features].replace([np.inf, -np.inf], np.nan)
    y = target_values(df, "gain_clip")
    train_idx = g.sampled_train_idx(
        len(df),
        f"global:{selector}:gain_clip:RandomForestRegressor",
        None,
    )
    model = g.make_rf()
    model.fit(x.iloc[train_idx], y[train_idx])
    score = np.asarray(model.predict(x), dtype=float)
    rf = model.named_steps["randomforestregressor"]
    importance = pd.DataFrame(
        {
            "selector": selector,
            "feature": features,
            "importance": [float(v) for v in rf.feature_importances_],
        }
    )
    return score, importance


def select_top_coverage(group: pd.DataFrame, score: np.ndarray, coverage: float = COVERAGE) -> pd.DataFrame:
    order = np.argsort(score, kind="mergesort")[::-1]
    n = max(1, int(round(len(order) * coverage)))
    selected = group.iloc[order[:n]].copy()
    selected["_rf_score"] = score[order[:n]]
    return selected


def select_top_per_dataset(scored: pd.DataFrame, coverage: float = COVERAGE) -> pd.DataFrame:
    selected = []
    for _, raw_group in scored.groupby("dataset", sort=True):
        group = raw_group.copy()
        selected.append(select_top_coverage(group, group["_rf_score"].to_numpy(dtype=float), coverage))
    return pd.concat(selected, ignore_index=False)


def value_range_rows(
    selector: str,
    dataset: str,
    features: list[str],
    universe: pd.DataFrame,
    selected: pd.DataFrame,
) -> list[dict]:
    rows = []
    for feature in features:
        universe_values = universe[feature].replace([np.inf, -np.inf], np.nan).astype(float)
        selected_values = selected[feature].replace([np.inf, -np.inf], np.nan).astype(float)
        pct = universe_values.rank(pct=True, method="average")
        selected_pct = pct.loc[selected.index]
        rows.append(
            {
                "selector": selector,
                "dataset": dataset,
                "coverage": COVERAGE,
                "feature": feature,
                "n_selected": len(selected),
                "value_min": selected_values.min(),
                "value_q25": selected_values.quantile(0.25),
                "value_median": selected_values.quantile(0.50),
                "value_q75": selected_values.quantile(0.75),
                "value_max": selected_values.max(),
                "percentile_min": selected_pct.min(),
                "percentile_q25": selected_pct.quantile(0.25),
                "percentile_median": selected_pct.quantile(0.50),
                "percentile_q75": selected_pct.quantile(0.75),
                "percentile_max": selected_pct.max(),
            }
        )
    return rows


def score_dataset_curves(scored: pd.DataFrame, selector: str) -> pd.DataFrame:
    rows = []
    for dataset, raw_group in scored.groupby("dataset", sort=True):
        group = raw_group.reset_index(drop=True)
        curves = g.score_coverages(group, group["_rf_score"].to_numpy(dtype=float), g.COVERAGES)
        curves.insert(0, "selector", selector)
        curves.insert(0, "dataset", dataset)
        rows.append(curves)
    return pd.concat(rows, ignore_index=True)


def add_macro_curve(curves: pd.DataFrame) -> pd.DataFrame:
    macro = (
        curves.groupby(["selector", "coverage"], as_index=False)
        .agg(
            n=("n", "sum"),
            mrr_stage1=("mrr_stage1", "mean"),
            mrr_stage2=("mrr_stage2", "mean"),
            gain_pt=("gain_pt", "mean"),
        )
        .assign(dataset="macro")
    )
    cols = ["dataset", "selector", "coverage", "n", "mrr_stage1", "mrr_stage2", "gain_pt"]
    return pd.concat([curves[cols], macro[cols]], ignore_index=True)


def macro_fixed_table(curves: pd.DataFrame) -> pd.DataFrame:
    macro_curves = curves[curves["dataset"] != "macro"].copy()
    macro = (
        macro_curves[macro_curves["coverage"].isin(g.FIXED)]
        .groupby(["selector", "coverage"], as_index=False)["gain_pt"]
        .mean()
        .pivot(index="selector", columns="coverage", values="gain_pt")
        .reset_index()
    )
    macro.columns = ["selector"] + [f"gain_{int(c * 100)}" for c in g.FIXED]
    return macro


def percentile_score_within_dataset(df: pd.DataFrame, feature: str, direction: str) -> pd.Series:
    ranked = df.groupby("dataset", sort=False)[feature].rank(pct=True, method="average")
    if direction == "desc":
        return ranked
    if direction == "asc":
        return 1.0 - ranked
    raise ValueError(direction)


def hard_percentile_curves(
    df: pd.DataFrame, selector: str, feature: str, direction: str
) -> pd.DataFrame:
    scored = df[["dataset", "official_scaled_rr_stage1", "official_scaled_rr_stage2", feature]].copy()
    scored["_rf_score"] = percentile_score_within_dataset(scored, feature, direction).to_numpy(dtype=float)
    return score_dataset_curves(scored, selector)


def hard_rule_range_row(
    df: pd.DataFrame, feature: str, direction: str, selector: str
) -> dict:
    scored = df[["dataset", feature]].copy()
    scored["_rf_score"] = percentile_score_within_dataset(scored, feature, direction).to_numpy(dtype=float)
    selected = select_top_per_dataset(scored)
    values = selected[feature].replace([np.inf, -np.inf], np.nan).astype(float)
    if direction == "desc":
        pct_min = 1.0 - COVERAGE
        pct_max = 1.0
        criterion = f"dataset percentile in [{pct_min * 100:.0f}%, {pct_max * 100:.0f}%]"
    else:
        pct_min = 0.0
        pct_max = COVERAGE
        criterion = f"dataset percentile in [{pct_min * 100:.0f}%, {pct_max * 100:.0f}%]"
    return {
        "selector": selector,
        "feature": feature,
        "direction": direction,
        "criterion": criterion,
        "n_selected": len(selected),
        "value_min": values.min(),
        "value_q25": values.quantile(0.25),
        "value_median": values.quantile(0.50),
        "value_q75": values.quantile(0.75),
        "value_max": values.max(),
        "percentile_min": pct_min,
        "percentile_q25": pct_min,
        "percentile_median": (pct_min + pct_max) / 2.0,
        "percentile_q75": pct_max,
        "percentile_max": pct_max,
        "value_range_full": f"[{fmt_num(values.min())}, {fmt_num(values.max())}]",
        "value_range_iqr": f"[{fmt_num(values.quantile(0.25))}, {fmt_num(values.quantile(0.75))}]",
        "percentile_range_full": f"[{pct_min * 100:.0f}%, {pct_max * 100:.0f}%]",
    }


def build_hard_rule_table(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    all_curves = []
    for feature in features:
        candidates = []
        for direction in ("desc", "asc"):
            selector = f"hard_{feature}_{direction}"
            curves = hard_percentile_curves(df, selector, feature, direction)
            curves_macro = add_macro_curve(curves)
            macro = macro_fixed_table(curves_macro).iloc[0].to_dict()
            row = hard_rule_range_row(df, feature, direction, selector)
            row.update(
                {
                    "gain_10": macro["gain_10"],
                    "gain_20": macro["gain_20"],
                    "gain_50": macro["gain_50"],
                }
            )
            candidates.append(row)
            all_curves.append(curves_macro)
        best = max(candidates, key=lambda r: r["gain_10"])
        rows.append(best)
    out = pd.DataFrame(rows).sort_values("gain_10", ascending=False).reset_index(drop=True)
    return out, pd.concat(all_curves, ignore_index=True)


def load_top_single_configs() -> list[tuple[str, str, list[str]]]:
    path = (
        REPORT_DIR
        / "ml_selector_diverse"
        / "balanced_ablation_true_rr"
        / "balanced_subset_macro_gain_annotated.csv"
    )
    macro = pd.read_csv(path)
    singles = macro[(macro["scope"] == "global") & (macro["size"] == 1)].copy()
    singles = singles.sort_values("gain_10", ascending=False).head(TOP_SINGLE_N)
    configs = []
    tag_to_feature = {
        "balanced1_syn": "synergy_weight_top5_mean",
        "balanced1_max": "max_candidate_dep_score",
        "balanced1_topk": "topk_rule_weight",
        "balanced1_eff": "effective_candidates",
    }
    for row in singles.itertuples(index=False):
        feature = tag_to_feature[str(row.tag)]
        configs.append((str(row.selector), feature, [feature]))
    return configs


def fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def fmt_num(v: float) -> str:
    return f"{v:.4g}"


def write_readme(
    macro: pd.DataFrame,
    per_dataset_gain: pd.DataFrame,
    ranges: pd.DataFrame,
    importance: pd.DataFrame,
    single_summary: pd.DataFrame,
    hard_rule_summary: pd.DataFrame,
) -> None:
    macro_disp = macro.copy()
    for col in [c for c in macro_disp.columns if c.startswith("gain_")]:
        macro_disp[col] = macro_disp[col].map(fmt_pct)
    macro_md = macro_disp.to_markdown(index=False)

    per_disp = per_dataset_gain.copy()
    for col in [c for c in per_disp.columns if c.startswith("gain_")]:
        per_disp[col] = per_disp[col].map(fmt_pct)
    per_md = per_disp.to_markdown(index=False)

    imp_disp = importance.copy()
    imp_disp["paper_name"] = imp_disp["feature"].map(FEATURE_NAMES)
    imp_disp["importance"] = imp_disp["importance"].map(lambda v: f"{v:.4f}")
    imp_md = imp_disp[["feature", "paper_name", "importance"]].to_markdown(index=False)

    definition_features = list(dict.fromkeys([*FEATURES, *single_summary["feature"].astype(str).tolist()]))
    defs = pd.DataFrame(
        [
            {
                "feature": feature,
                "paper_name": FEATURE_NAMES[feature],
                "definition": FEATURE_DEFINITIONS[feature],
            }
            for feature in definition_features
        ]
    )
    defs_md = defs.to_markdown(index=False)

    single_disp = single_summary.copy()
    single_disp = single_disp.rename(
        columns={
            "feature": "feature",
            "gain_10": "gain@10",
            "gain_20": "gain@20",
            "gain_50": "gain@50",
            "value_range_iqr": "Feature Range (abs, IQR)",
            "percentile_range_iqr": "Feature Range (percentile, IQR)",
        }
    )
    for col in ["gain@10", "gain@20", "gain@50"]:
        single_disp[col] = single_disp[col].map(fmt_pct)
    single_md = single_disp[
        [
            "feature",
            "gain@10",
            "gain@20",
            "gain@50",
            "Feature Range (abs, IQR)",
            "Feature Range (percentile, IQR)",
        ]
    ].to_markdown(index=False)

    hard_disp = hard_rule_summary.copy()
    for col in ["gain_10", "gain_20", "gain_50"]:
        hard_disp[col] = hard_disp[col].map(fmt_pct)
    hard_disp = hard_disp.rename(
        columns={
            "gain_10": "gain@10",
            "gain_20": "gain@20",
            "gain_50": "gain@50",
            "value_range_full": "Feature Range (abs)",
            "percentile_range_full": "Feature Range (percentile)",
        }
    )
    hard_md = hard_disp[
        [
            "feature",
            "criterion",
            "gain@10",
            "gain@20",
            "gain@50",
            "Feature Range (abs)",
            "Feature Range (percentile)",
        ]
    ].to_markdown(index=False)

    dual_paper = ranges[(ranges["selector"] == SELECTOR) & (ranges["dataset"] == "global")].copy()
    dual_paper["Feature Range (abs, IQR)"] = dual_paper.apply(
        lambda r: f"[{fmt_num(r['value_q25'])}, {fmt_num(r['value_q75'])}]", axis=1
    )
    dual_paper["Feature Range (percentile, IQR)"] = dual_paper.apply(
        lambda r: f"[{fmt_pct(r['percentile_q25'])}, {fmt_pct(r['percentile_q75'])}]",
        axis=1,
    )
    dual_paper["gain@10"] = float(macro.iloc[0]["gain_10"])
    dual_paper["gain@20"] = float(macro.iloc[0]["gain_20"])
    dual_paper["gain@50"] = float(macro.iloc[0]["gain_50"])
    dual_disp = dual_paper[
        [
            "feature",
            "gain@10",
            "gain@20",
            "gain@50",
            "Feature Range (abs, IQR)",
            "Feature Range (percentile, IQR)",
        ]
    ].copy()
    for col in ["gain@10", "gain@20", "gain@50"]:
        dual_disp[col] = dual_disp[col].map(fmt_pct)
    dual_md = dual_disp.to_markdown(index=False)

    cross_ranges = ranges[(ranges["selector"] == SELECTOR) & (ranges["dataset"] == "global")].copy()
    for col in [c for c in cross_ranges.columns if c.startswith("value_")]:
        cross_ranges[col] = cross_ranges[col].map(fmt_num)
    for col in [c for c in cross_ranges.columns if c.startswith("percentile_")]:
        cross_ranges[col] = cross_ranges[col].map(fmt_pct)
    range_md = cross_ranges[
        [
            "feature",
            "n_selected",
            "value_min",
            "value_q25",
            "value_median",
            "value_q75",
            "value_max",
            "percentile_min",
            "percentile_q25",
            "percentile_median",
            "percentile_q75",
            "percentile_max",
        ]
    ].to_markdown(index=False)

    body = f"""# Recommended Query Subset Criterion

## Recommendation

Use the **two-feature Global RF** selector `balanced2_syn_topk_global_rf`
for the paper-facing query subset analysis.

Formal subset criterion:

1. Train one pooled RandomForest regressor over all 7 datasets.
2. Use only two features: `synergy_weight_top5_mean` and `topk_rule_weight`.
3. Train target: `gain_clip = clip(rr_stage2 / rr_stage1 - 1, [-1, 1])`.
4. For each dataset independently, score every query with the global RF and
   keep the top **10%** by RF score.

This is still a Global RF: there is one shared model and one shared feature
set. The per-dataset step only fixes the coverage to 10% in each dataset, so
large datasets do not dominate the selected subset.

## Why Two Features

The single-feature global RF is most interpretable but only reaches
`gain@10 = 7.32%`. The best two-feature global RF reaches
`gain@10 = 12.64%`, clearing the 10% target while keeping the selector easy
to explain. Three-feature selectors improve further but add another axis that
is harder to state as a compact subset criterion.

## Feature Definitions

{defs_md}

## Global RF Importance

{imp_md}

## Macro Gain

{macro_md}

## Selector Definition and Interpretation

The recommended subset is selected by a learned Global RF transformation:

`score = f(synergy_weight_top5_mean, topk_rule_weight)`.

Here, `f` is the average of 160 regression trees, so it is a learned
piecewise-constant transformation rather than a single hand-written threshold.
We rank queries by this transformed score within each dataset and select the
top 10%, i.e., `percentile(score) in [90%, 100%]`.

The selected subset has a simple feature-level interpretation: it tends to
contain queries with **high `synergy_weight_top5_mean`** and **middle-range
`topk_rule_weight`**. In the pooled selected top-10% subset, the feature IQRs
are `synergy_weight_top5_mean` = `[3.118, 5.55]` (pooled percentile
`[75.12%, 95.21%]`) and `topk_rule_weight` = `[0.971, 1.621]` (pooled
percentile `[36.54%, 73.60%]`). This two-feature RF subset raises the macro
subset gain to **12.64%** at 10% coverage.

## Two-Feature RF Selector Paper Table

This is the recommended table for the main text. The gain comes from the
two-feature Global RF score selector, so it keeps the `gain@10 = 12.64%`
result. Feature ranges are descriptive summaries of the RF-selected top-10%
queries, not hard threshold rules.

{dual_md}

## Top-3 Single-Feature Global RF Selectors

The table below uses the same Global RF setup and top-10% per-dataset
selection rule, then pools the selected queries across datasets to summarize
feature ranges. `Feature Range` reports the selected subset IQR.

{single_md}

CSV for the final paper table: `single_feature_top3_paper_table.csv`.

## Optional Hard Feature-Range Sanity Check

The RF rows above are the main results. The hard percentile rules below are
only a sanity check: they replace the RF score with a single raw feature
percentile cutoff. Their gains are lower because this is a simpler selector,
not the RF selector that gives 12.64%.

{hard_md}

CSV for this optional hard-rule check: `single_feature_hard_range_rules.csv`.

## Per-Dataset Gain at 10%

{per_md}

## Cross-Dataset Selected Top-10% Feature Ranges

The table reports the empirical feature range of the selected top-10% subset
after applying the top-10% rule within each dataset and then pooling all
selected queries. Percentiles are computed against the pooled cross-dataset
feature distribution. The IQR is the recommended compact paper wording;
min/max are included only to show the full observed spread.

{range_md}

The dual-feature gain curve at every 2% coverage point for each dataset and
the macro average is in `dual_feature_gain_curves_with_macro.csv`.

## Wording for the Paper

We identify the explainable query subset using a two-feature global RF
selector trained on all datasets with synergy strength and top-k rule weight
as inputs. For each dataset, queries are ranked by the shared RF predicted
gain score, and the top 10% are used as the subset. This rule yields a macro
average in-sample relative MRR gain of 12.64% at 10% coverage, while
preserving a compact interpretation: selected queries tend to combine strong
learned synergy dependencies with moderate-to-high fired-rule weights.
"""
    (OUT_DIR / "README.md").write_text(body, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, _ = load_merged_rr()
    df = df.reset_index(drop=True)
    score, importance = fit_global_score(df, SELECTOR, FEATURES)

    scored = df[
        ["dataset", "official_scaled_rr_stage1", "official_scaled_rr_stage2", *FEATURES]
    ].copy()
    scored["_rf_score"] = score

    curves = score_dataset_curves(scored, SELECTOR)
    curves_with_macro = add_macro_curve(curves)
    macro = macro_fixed_table(curves_with_macro)

    per_dataset_gain = curves[curves["coverage"] == COVERAGE][
        ["dataset", "gain_pt", "mrr_stage1", "mrr_stage2", "n"]
    ].copy()
    per_dataset_gain.insert(0, "selector", SELECTOR)
    per_dataset_gain = per_dataset_gain.rename(columns={"gain_pt": "gain_10"})

    range_rows = []
    selected_global = select_top_per_dataset(scored)
    range_rows.extend(value_range_rows(SELECTOR, "global", FEATURES, scored, selected_global))
    ranges = pd.DataFrame(range_rows)

    single_rows = []
    single_range_rows = []
    single_curve_rows = []
    for selector, feature, features in load_top_single_configs():
        single_score, _ = fit_global_score(df, selector, features)
        single_scored = df[
            ["dataset", "official_scaled_rr_stage1", "official_scaled_rr_stage2", *features]
        ].copy()
        single_scored["_rf_score"] = single_score
        single_curves = score_dataset_curves(single_scored, selector)
        single_curve_rows.append(single_curves)
        single_macro = macro_fixed_table(add_macro_curve(single_curves))
        selected_single = select_top_per_dataset(single_scored)
        single_ranges = pd.DataFrame(
            value_range_rows(selector, "global", features, single_scored, selected_single)
        )
        single_range_rows.append(single_ranges)
        row = single_macro.iloc[0].to_dict()
        range_row = single_ranges.iloc[0]
        row.update(
            {
                "feature": feature,
                "paper_name": FEATURE_NAMES[feature],
                "value_range_iqr": f"[{fmt_num(range_row['value_q25'])}, {fmt_num(range_row['value_q75'])}]",
                "percentile_range_iqr": (
                    f"[{fmt_pct(range_row['percentile_q25'])}, "
                    f"{fmt_pct(range_row['percentile_q75'])}]"
                ),
            }
        )
        single_rows.append(row)
    single_summary = pd.DataFrame(single_rows)
    hard_rule_summary, hard_rule_curves = build_hard_rule_table(
        df, single_summary["feature"].astype(str).tolist()
    )
    dual_paper_table = ranges[
        (ranges["selector"] == SELECTOR) & (ranges["dataset"] == "global")
    ][["feature", "value_q25", "value_q75", "percentile_q25", "percentile_q75"]].copy()
    dual_paper_table["gain@10"] = float(macro.iloc[0]["gain_10"])
    dual_paper_table["gain@20"] = float(macro.iloc[0]["gain_20"])
    dual_paper_table["gain@50"] = float(macro.iloc[0]["gain_50"])
    dual_paper_table["Feature Range Absolute"] = dual_paper_table.apply(
        lambda r: f"[{fmt_num(r['value_q25'])}, {fmt_num(r['value_q75'])}]", axis=1
    )
    dual_paper_table["Feature Range Percentile"] = dual_paper_table.apply(
        lambda r: f"[{fmt_pct(r['percentile_q25'])}, {fmt_pct(r['percentile_q75'])}]",
        axis=1,
    )
    dual_paper_table = dual_paper_table[
        [
            "feature",
            "gain@10",
            "gain@20",
            "gain@50",
            "Feature Range Absolute",
            "Feature Range Percentile",
        ]
    ]
    single_paper_table = single_summary[
        ["feature", "gain_10", "gain_20", "gain_50", "value_range_iqr", "percentile_range_iqr"]
    ].rename(
        columns={
            "gain_10": "gain@10",
            "gain_20": "gain@20",
            "gain_50": "gain@50",
            "value_range_iqr": "Feature Range Absolute",
            "percentile_range_iqr": "Feature Range Percentile",
        }
    )
    single_ranges_all = pd.concat(single_range_rows, ignore_index=True)
    single_curves_all = pd.concat(single_curve_rows, ignore_index=True)

    curves.to_csv(OUT_DIR / "gain_curves.csv", index=False)
    curves_with_macro.to_csv(OUT_DIR / "dual_feature_gain_curves_with_macro.csv", index=False)
    macro.to_csv(OUT_DIR / "macro_gain.csv", index=False)
    per_dataset_gain.to_csv(OUT_DIR / "per_dataset_gain_10.csv", index=False)
    importance.to_csv(OUT_DIR / "global_rf_importance.csv", index=False)
    ranges.to_csv(OUT_DIR / "selected_top10_feature_ranges.csv", index=False)
    dual_paper_table.to_csv(OUT_DIR / "dual_feature_paper_table.csv", index=False)
    single_summary.to_csv(OUT_DIR / "single_feature_top3_summary.csv", index=False)
    single_paper_table.to_csv(OUT_DIR / "single_feature_top3_paper_table.csv", index=False)
    single_ranges_all.to_csv(OUT_DIR / "single_feature_top3_ranges.csv", index=False)
    single_curves_all.to_csv(OUT_DIR / "single_feature_top3_gain_curves.csv", index=False)
    hard_rule_summary.to_csv(OUT_DIR / "single_feature_hard_range_rules.csv", index=False)
    hard_rule_curves.to_csv(OUT_DIR / "single_feature_hard_range_rule_curves.csv", index=False)

    write_readme(macro, per_dataset_gain, ranges, importance, single_summary, hard_rule_summary)
    print(OUT_DIR / "README.md", flush=True)


if __name__ == "__main__":
    main()
