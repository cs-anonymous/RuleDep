#!/usr/bin/env python3
"""
Complete comp/rule/ratio analysis using TRUE RR (merged with true_official_per_query_rr_wide).

9 single-feature Global RF selectors:
  comp1, comp3, comp5, rule1, rule3, rule5, ratio1, ratio3, ratio5

Outputs:
  1. Macro gain@10/20/50 for each selector
  2. Per-selector top-10% feature value q10-q90 range
  3. ratio3 coverage curve CSV (0.02-1.00, per-dataset + macro)
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import regenerate_official_query_subset_reports as base

ROOT = Path("/home/sy/RuleDep")
REPORT_DIR = ROOT / "reports" / "official_query_subset"
TRUE_RR_WIDE = REPORT_DIR / "true_official_per_query_rr" / "true_official_per_query_rr_wide.csv"
JOIN_KEYS = ["dataset", "experiment", "relation", "direction", "query", "target_gt_entity"]

RANDOM_STATE = 20260430
COVERAGES = [0.10, 0.20, 0.50]
COVERAGE_CURVE_STEPS = np.arange(0.02, 1.001, 0.02)


# ---------------------------------------------------------------------------
# Data loading (true RR merge)
# ---------------------------------------------------------------------------
def load_merged_rr() -> pd.DataFrame:
    print("Loading features...")
    feat = pd.read_csv(REPORT_DIR / "official_query_triple_features.csv")
    feat = base.add_official_scaled_rr(feat)

    print("Loading true RR...")
    wide = pd.read_csv(TRUE_RR_WIDE)
    keep = JOIN_KEYS + ["true_official_rr_stage1", "true_official_rr_stage2"]
    wide = wide[keep]

    print("Merging...")
    merged = feat.merge(wide, on=JOIN_KEYS, how="left", validate="one_to_one")

    merged["official_scaled_rr_stage1"] = merged["true_official_rr_stage1"].fillna(
        merged["official_scaled_rr_stage1"]
    )
    merged["official_scaled_rr_stage2"] = merged["true_official_rr_stage2"].fillna(
        merged["official_scaled_rr_stage2"]
    )
    return merged


# ---------------------------------------------------------------------------
# Feature setup
# ---------------------------------------------------------------------------
FEATURE_COLS = {
    "comp1": "synergy_weight_top1_mean",
    "comp3": "synergy_weight_top3_mean",
    "comp5": "synergy_weight_top5_mean",
    "rule1": "rule_weight_top1_mean",
    "rule3": "rule_weight_top3_mean",
    "rule5": "rule_weight_top5_mean",
}

ALL_NAMES = ["comp1", "comp3", "comp5", "rule1", "rule3", "rule5", "ratio1", "ratio3", "ratio5"]


def feature_col(name: str) -> str:
    if name.startswith("ratio"):
        return name
    return FEATURE_COLS[name]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ratio1"] = df["synergy_weight_top1_mean"] / (df["rule_weight_top1_mean"] + 1e-9)
    df["ratio3"] = df["synergy_weight_top3_mean"] / (df["rule_weight_top3_mean"] + 1e-9)
    df["ratio5"] = df["synergy_weight_top5_mean"] / (df["rule_weight_top5_mean"] + 1e-9)
    return df


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
def compute_gain_clip_target(df: pd.DataFrame) -> np.ndarray:
    s1 = df["official_scaled_rr_stage1"].to_numpy(dtype=float)
    s2 = df["official_scaled_rr_stage2"].to_numpy(dtype=float)
    gain = np.where(s1 > 0, s2 / s1 - 1.0, 0.0)
    return np.clip(gain, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Global RF training
# ---------------------------------------------------------------------------
def make_rf():
    return RandomForestRegressor(
        n_estimators=160,
        max_depth=10,
        min_samples_leaf=20,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def train_global_rf(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    x = df[features].replace([np.inf, -np.inf], np.nan)
    y = compute_gain_clip_target(df)
    model = make_pipeline(SimpleImputer(strategy="median"), make_rf())
    model.fit(x, y)
    return model.predict(x)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_per_dataset(df: pd.DataFrame, score_col: str, coverage: float):
    """Return (per_dataset_rows, macro_gain_pt_pct)."""
    rows = []
    for dataset, group in df.groupby("dataset", sort=True):
        scores = group[score_col].to_numpy()
        s1 = group["official_scaled_rr_stage1"].to_numpy(dtype=float)
        s2 = group["official_scaled_rr_stage2"].to_numpy(dtype=float)

        order = np.argsort(scores, kind="mergesort")[::-1]
        n_sel = max(1, int(round(len(order) * coverage)))
        selected = order[:n_sel]

        m1 = float(s1[selected].mean())
        m2 = float(s2[selected].mean())
        gain_pt = (m2 / m1 - 1.0) * 100.0 if m1 > 0 else 0.0

        rows.append({
            "dataset": dataset,
            "coverage": coverage,
            "n": n_sel,
            "mrr_stage1": m1,
            "mrr_stage2": m2,
            "gain_pt": gain_pt,
        })

    macro_gain = np.mean([r["gain_pt"] for r in rows])
    return rows, macro_gain


def feature_q_range(df: pd.DataFrame, score_col: str, feat_col: str, coverage: float):
    """q10-q90 of feature values for top-coverage selected queries (across all datasets)."""
    all_feats = []
    for dataset, group in df.groupby("dataset", sort=True):
        scores = group[score_col].to_numpy()
        feats = group[feat_col].to_numpy(dtype=float)

        order = np.argsort(scores, kind="mergesort")[::-1]
        n_sel = max(1, int(round(len(order) * coverage)))
        selected = order[:n_sel]

        all_feats.extend(feats[selected].tolist())

    arr = np.array(all_feats)
    return float(np.percentile(arr, 10)), float(np.percentile(arr, 90))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_merged_rr()
    df = add_derived(df)

    # Verify all columns
    for name in ALL_NAMES:
        col = feature_col(name)
        assert col in df.columns, f"Missing: {col}"

    print(f"\nTotal queries: {len(df):,}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")

    # =====================================================================
    # Part 1: Single-feature Global RF — macro gain + feature ranges
    # =====================================================================
    print("\n" + "=" * 80)
    print("SINGLE-FEATURE GLOBAL RF (TRUE RR)")
    print("=" * 80)

    macro_summary = []
    range_summary = []
    # Store per-selector scores for later use
    score_cols = {}

    for name in ALL_NAMES:
        col = feature_col(name)
        sc = f"rf_{name}"
        score_cols[name] = sc
        print(f"\n  [{name}] training Global RF ({col})...", flush=True)
        scores = train_global_rf(df, [col])
        df[sc] = scores

        row = {"feature": name}
        for cov in COVERAGES:
            _, macro_gain = score_per_dataset(df, sc, cov)
            row[f"gain@{int(cov * 100)}"] = f"{macro_gain:.2f}%"
        macro_summary.append(row)
        print(f"    {'  '.join(f'{k}: {v}' for k, v in row.items())}")

        # Top-10% feature value q10-q90
        q10, q90 = feature_q_range(df, sc, col, 0.10)
        range_summary.append({
            "feature": name,
            "feature_col": col,
            "coverage": 0.10,
            "q10": f"{q10:.6f}",
            "q90": f"{q90:.6f}",
        })
        print(f"    top-10% {col} range: [{q10:.6f}, {q90:.6f}]")

    # Print tables
    macro_df = pd.DataFrame(macro_summary)
    print("\n\n=== Macro Gain Summary ===")
    print(macro_df.to_markdown(index=False))

    range_df = pd.DataFrame(range_summary)
    print("\n=== Top-10% Feature Value q10-q90 Range ===")
    print(range_df.to_markdown(index=False))

    # =====================================================================
    # Part 2: ratio3 coverage curve
    # =====================================================================
    print("\n" + "=" * 80)
    print("RATIO3 COVERAGE CURVE (0.02 → 1.00)")
    print("=" * 80)

    curve_rows = []
    ratio3_sc = score_cols["ratio3"]

    for dataset, group in df.groupby("dataset", sort=True):
        scores = group[ratio3_sc].to_numpy()
        s1 = group["official_scaled_rr_stage1"].to_numpy(dtype=float)
        s2 = group["official_scaled_rr_stage2"].to_numpy(dtype=float)

        order = np.argsort(scores, kind="mergesort")[::-1]

        for cov in COVERAGE_CURVE_STEPS:
            n_sel = max(1, int(round(len(order) * cov)))
            selected = order[:n_sel]

            m1 = float(s1[selected].mean())
            m2 = float(s2[selected].mean())
            gain_pt = (m2 / m1 - 1.0) * 100.0 if m1 > 0 else 0.0

            curve_rows.append({
                "dataset": dataset,
                "coverage": round(float(cov), 4),
                "n": n_sel,
                "mrr_stage1": round(m1, 6),
                "mrr_stage2": round(m2, 6),
                "gain_pt": round(gain_pt, 4),
            })

    # Macro row: mean gain_pt, mrr_s1, mrr_s2 across datasets at each coverage
    curve_tmp = pd.DataFrame(curve_rows)
    macro_curve = (
        curve_tmp.groupby("coverage")
        .agg(
            n=("n", "mean"),
            mrr_stage1=("mrr_stage1", "mean"),
            mrr_stage2=("mrr_stage2", "mean"),
            gain_pt=("gain_pt", "mean"),
        )
        .reset_index()
    )
    macro_curve["dataset"] = "macro"
    macro_curve = macro_curve[["dataset", "coverage", "n", "mrr_stage1", "mrr_stage2", "gain_pt"]]
    macro_curve["n"] = macro_curve["n"].astype(int)
    macro_curve["coverage"] = macro_curve["coverage"].round(4)
    macro_curve["mrr_stage1"] = macro_curve["mrr_stage1"].round(6)
    macro_curve["mrr_stage2"] = macro_curve["mrr_stage2"].round(6)
    macro_curve["gain_pt"] = macro_curve["gain_pt"].round(4)

    full_curve = pd.concat([curve_tmp, macro_curve], ignore_index=True)
    full_curve = full_curve[["dataset", "coverage", "n", "mrr_stage1", "mrr_stage2", "gain_pt"]]

    out_curve = REPORT_DIR / "ratio3_coverage_curve_true_rr.csv"
    full_curve.to_csv(out_curve, index=False)
    print(f"\nCoverage curve saved to: {out_curve}")
    print(f"  Rows: {len(full_curve)} ({len(COVERAGE_CURVE_STEPS)} per-dataset × {df['dataset'].nunique()} + {len(COVERAGE_CURVE_STEPS)} macro)")
    print("\nMacro preview (first 10):")
    print(macro_curve.head(10).to_string(index=False))

    # =====================================================================
    # Save summaries
    # =====================================================================
    out_macro = REPORT_DIR / "comp_rule_rf_complete_true_rr.csv"
    macro_df.to_csv(out_macro, index=False)
    print(f"\nMacro summary saved to: {out_macro}")

    out_range = REPORT_DIR / "comp_rule_rf_feature_range_true_rr.csv"
    range_df.to_csv(out_range, index=False)
    print(f"Feature range saved to: {out_range}")


if __name__ == "__main__":
    main()
