#!/usr/bin/env python3
"""Balanced4 selector report (drops `num_neg_dep` from Balanced5).

Same true-per-query-RR data plumbing as `generate_balanced5_selector_report_true_rr.py`
(re-uses `load_merged_rr`). Trains:
  - balanced4_rf:        RF per dataset on 4 features.
  - balanced4_global_rf: one RF on all data pooled, evaluated per dataset.
  - single-feature RF baselines for each of the 4 features.
  - balanced5_rf / balanced5_global_rf are also re-fit here so the report
    contains a like-for-like 4 vs 5 side-by-side comparison.

Beyond the existing pipeline, this report exports **global RF feature
importance** in addition to per-dataset importance (the original
`generate_balanced5_selector_report.py` only emitted per-dataset importance).

Outputs:
  reports/official_query_subset/ml_selector_diverse/balanced4_report_true_rr/
    balanced4_gain_curves.csv
    balanced4_macro_gain_table.csv
    balanced4_rf_feature_importance.csv      # per-dataset and global, both balanced4 + balanced5
    balanced4_vs_5_comparison.csv            # head-to-head gain at fixed coverages
    rr_source_summary.csv
    figures/<selector>.png
    README.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_balanced5_selector_report as g  # noqa: E402
from generate_balanced5_selector_report_true_rr import load_merged_rr  # noqa: E402
from sweep_official_query_ml_selectors import target_values  # noqa: E402


ROOT = Path("/home/sy/RuleDep")
REPORT_DIR = ROOT / "reports" / "official_query_subset"
OUT_DIR = REPORT_DIR / "ml_selector_diverse" / "balanced4_report_true_rr"
FIG_DIR = OUT_DIR / "figures"

BALANCED4 = [
    "synergy_weight_top5_mean",
    "max_candidate_dep_score",
    "topk_rule_weight",
    "effective_candidates",
]
DROPPED_FEATURE = "num_neg_dep"


def _per_dataset_rf(
    df: pd.DataFrame,
    features: list[str],
    *,
    selector_name: str,
    seed_base: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    importance_rows = []
    for dataset, raw_group in df.groupby("dataset", sort=True):
        group = raw_group.reset_index(drop=True)
        x = group[features].replace([np.inf, -np.inf], np.nan)
        y = target_values(group, "gain_clip")
        train_idx = g.sampled_train_idx(
            len(group),
            f"{dataset}:{seed_base}:gain_clip:RandomForestRegressor",
            80_000,
        )
        model = g.make_rf()
        model.fit(x.iloc[train_idx], y[train_idx])
        score = np.asarray(model.predict(x), dtype=float)
        rf = model.named_steps["randomforestregressor"]
        for feature, imp in zip(features, rf.feature_importances_):
            importance_rows.append(
                {
                    "selector": selector_name,
                    "dataset": str(dataset),
                    "feature": feature,
                    "importance": float(imp),
                }
            )
        curves = g.score_coverages(group, score, g.COVERAGES)
        curves.insert(0, "selector", selector_name)
        curves.insert(0, "dataset", dataset)
        rows.append(curves)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(importance_rows)


def _global_rf(
    df: pd.DataFrame,
    features: list[str],
    *,
    selector_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df[features].replace([np.inf, -np.inf], np.nan)
    y = target_values(df, "gain_clip")
    train_idx = g.sampled_train_idx(
        len(df),
        f"global:{selector_name}:gain_clip:RandomForestRegressor",
        None,
    )
    model = g.make_rf()
    model.fit(x.iloc[train_idx], y[train_idx])
    score = np.asarray(model.predict(x), dtype=float)
    rf = model.named_steps["randomforestregressor"]
    importance_rows = [
        {
            "selector": selector_name,
            "dataset": "global",
            "feature": feature,
            "importance": float(imp),
        }
        for feature, imp in zip(features, rf.feature_importances_)
    ]

    rows = []
    scored = df[["dataset", "official_scaled_rr_stage1", "official_scaled_rr_stage2", *features]].copy()
    scored["_score"] = score
    for dataset, group in scored.groupby("dataset", sort=True):
        gp = group.reset_index(drop=True)
        curves = g.score_coverages(gp, gp["_score"].to_numpy(dtype=float), g.COVERAGES)
        curves.insert(0, "selector", selector_name)
        curves.insert(0, "dataset", dataset)
        rows.append(curves)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(importance_rows)


def _single_feature_rf(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        for dataset, group in df.groupby("dataset", sort=True):
            gp = group.reset_index(drop=True)
            x = gp[[feature]].replace([np.inf, -np.inf], np.nan)
            y = target_values(gp, "gain_clip")
            score = g.fit_rf_score(
                x,
                y,
                seed_key=f"{dataset}:{feature}:gain_clip:RandomForestRegressor",
                max_train_rows=80_000,
            )
            curves = g.score_coverages(gp, score, g.COVERAGES)
            curves.insert(0, "selector", feature)
            curves.insert(0, "dataset", dataset)
            rows.append(curves)
    return pd.concat(rows, ignore_index=True)


def macro_from_curves(curves: pd.DataFrame, ordered_selectors: list[str]) -> pd.DataFrame:
    fixed = curves[curves["coverage"].isin(g.FIXED)].copy()
    macro = (
        fixed.groupby(["selector", "coverage"], as_index=False)["gain_pt"]
        .mean()
        .pivot(index="selector", columns="coverage", values="gain_pt")
        .reset_index()
    )
    macro.columns = ["selector"] + [f"gain_{int(c * 100)}" for c in g.FIXED]
    order_map = {name: i for i, name in enumerate(ordered_selectors)}
    macro["order"] = macro["selector"].map(lambda s: order_map.get(s, len(order_map) + 1))
    return macro.sort_values("order").drop(columns=["order"]).reset_index(drop=True)


def comparison_table(macro: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("balanced4_rf", "balanced5_rf", "per-dataset"),
        ("balanced4_global_rf", "balanced5_global_rf", "global"),
    ]
    rows = []
    for b4, b5, label in pairs:
        r4 = macro[macro["selector"] == b4].iloc[0]
        r5 = macro[macro["selector"] == b5].iloc[0]
        for col in ["gain_10", "gain_20", "gain_30", "gain_50", "gain_100"]:
            rows.append(
                {
                    "scope": label,
                    "metric": col,
                    "balanced5": float(r5[col]),
                    "balanced4": float(r4[col]),
                    "delta_pt": float(r4[col]) - float(r5[col]),
                }
            )
    return pd.DataFrame(rows)


def write_readme(
    macro: pd.DataFrame,
    importance: pd.DataFrame,
    comparison: pd.DataFrame,
    rr_source_summary: pd.DataFrame,
) -> None:
    def fmt(df: pd.DataFrame) -> str:
        out = df.copy()
        for col in out.columns:
            if pd.api.types.is_float_dtype(out[col]):
                out[col] = out[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
        return out.to_markdown(index=False)

    macro_md = fmt(macro)

    comp_pivot_rows = []
    for scope in comparison["scope"].unique():
        sub = comparison[comparison["scope"] == scope]
        row = {"scope": scope}
        for _, r in sub.iterrows():
            row[f"balanced5_{r['metric']}"] = r["balanced5"]
            row[f"balanced4_{r['metric']}"] = r["balanced4"]
            row[f"delta_{r['metric']}"] = r["delta_pt"]
        comp_pivot_rows.append(row)
    comp_long = comparison.copy()
    comp_long["balanced5"] = comp_long["balanced5"].map(lambda v: f"{v * 100:.2f}%")
    comp_long["balanced4"] = comp_long["balanced4"].map(lambda v: f"{v * 100:.2f}%")
    comp_long["delta_pt"] = comp_long["delta_pt"].map(lambda v: f"{v * 100:+.2f} pt")
    comp_md = comp_long.to_markdown(index=False)

    global_imp = (
        importance[importance["dataset"] == "global"]
        .pivot(index="selector", columns="feature", values="importance")
        .reset_index()
    )
    cols = ["selector"] + [c for c in BALANCED4 if c in global_imp.columns]
    extra = [c for c in global_imp.columns if c not in cols and c != "selector"]
    global_imp = global_imp[cols + extra]
    global_imp_md = fmt(global_imp)

    per_imp = (
        importance[(importance["dataset"] != "global") & (importance["selector"] == "balanced4_rf")]
        .pivot(index="dataset", columns="feature", values="importance")
        .reindex(columns=BALANCED4)
        .reset_index()
    )
    per_imp_md = fmt(per_imp)

    fallback = rr_source_summary[
        (rr_source_summary["rr_source_stage1"] == "scaled_fallback")
        | (rr_source_summary["rr_source_stage2"] == "scaled_fallback")
    ]
    if not fallback.empty:
        fb_md = (
            fallback.groupby(["dataset", "rr_source_stage1", "rr_source_stage2"])
            .agg(n_rows=("n_rows", "sum"))
            .reset_index()
            .to_markdown(index=False)
        )
    else:
        fb_md = "_No fallback rows; every case used true per-query filtered RR._"

    feature_lines = "\n".join(f"- `{f}`: {g.FEATURE_LABELS.get(f, f)}" for f in BALANCED4)

    body = f"""# Balanced4 Selector Report (true per-query RR)

Drops `{DROPPED_FEATURE}` from the original Balanced5 set. The remaining 4
attributes are:

{feature_lines}

The dropped attribute had the lowest single-feature RF gain (6.76% @10%
coverage vs 10.6-11.8% for the others), the lowest mean per-dataset RF
importance (~0.116 vs 0.17-0.24), and a sign-flipping Ridge coefficient
across datasets.

## Macro Gain Table

{macro_md}

## Balanced4 vs Balanced5 head-to-head

Both are re-fit on the same data with stable seeds in this report so the
comparison is like-for-like.

{comp_md}

## Global RF Feature Importance (one ranking, all datasets pooled)

This is new in this report. The original Balanced5 report only emitted
per-dataset RF importance; here we expose the importance of the single
unified RF used as the selector.

{global_imp_md}

## Per-dataset RF Feature Importance (Balanced4)

{per_imp_md}

Per-dataset importances are noisy across datasets; the global RF importance
above is the one we recommend citing in a paper.

## RR Source (true per-query vs scaled fallback)

Same data plumbing as `balanced5_report_true_rr`. Fallback rows by dataset:

{fb_md}

Full breakdown is in `rr_source_summary.csv`.

## Figures

"""
    selectors = macro["selector"].astype(str).tolist()
    fig_lines = [f"- `{s}`: [figures/{s}.png](figures/{s}.png)" for s in selectors]
    body += "\n".join(fig_lines) + "\n"

    (OUT_DIR / "README.md").write_text(body, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df, rr_source_summary = load_merged_rr()

    b4_per_curves, b4_per_imp = _per_dataset_rf(
        df, BALANCED4, selector_name="balanced4_rf", seed_base="balanced4"
    )
    b4_global_curves, b4_global_imp = _global_rf(
        df, BALANCED4, selector_name="balanced4_global_rf"
    )
    single_curves = _single_feature_rf(df, BALANCED4)

    b5_per_curves, b5_per_imp = _per_dataset_rf(
        df, g.BALANCED5, selector_name="balanced5_rf", seed_base="balanced5"
    )
    b5_global_curves, b5_global_imp = _global_rf(
        df, g.BALANCED5, selector_name="balanced5_global_rf"
    )

    importance = pd.concat(
        [b4_per_imp, b4_global_imp, b5_per_imp, b5_global_imp], ignore_index=True
    )
    curves = pd.concat(
        [
            b4_per_curves,
            b4_global_curves,
            single_curves,
            b5_per_curves,
            b5_global_curves,
        ],
        ignore_index=True,
    )

    selectors_order = (
        ["balanced4_rf", "balanced4_global_rf", "balanced5_rf", "balanced5_global_rf"]
        + BALANCED4
    )
    macro = macro_from_curves(curves, selectors_order)
    comparison = comparison_table(macro)

    curves.to_csv(OUT_DIR / "balanced4_gain_curves.csv", index=False)
    macro.to_csv(OUT_DIR / "balanced4_macro_gain_table.csv", index=False)
    importance.to_csv(OUT_DIR / "balanced4_rf_feature_importance.csv", index=False)
    comparison.to_csv(OUT_DIR / "balanced4_vs_5_comparison.csv", index=False)
    rr_source_summary.to_csv(OUT_DIR / "rr_source_summary.csv", index=False)

    selectors = macro["selector"].astype(str).tolist()
    for selector in selectors:
        g.plot_selector(curves, selector, FIG_DIR / f"{selector}.png")

    write_readme(macro, importance, comparison, rr_source_summary)
    print(OUT_DIR / "README.md", flush=True)


if __name__ == "__main__":
    main()
