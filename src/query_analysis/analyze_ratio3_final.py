#!/usr/bin/env python3
"""
Final ratio3_global_rf analysis with TRUE RR.
Produces: macro gains, per-dataset gains, q10-q90 ranges, coverage curve for Fig. 5B,
and macro table for all 9 single features.
"""
from __future__ import annotations

import sys
import zlib
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
MAIN_COVERAGES = np.array([0.10, 0.20, 0.50])
FIG5B_COVERAGES = np.arange(0.02, 1.001, 0.02)  # 0.02, 0.04, ..., 1.00


def stable_seed(value: str) -> int:
    return RANDOM_STATE + zlib.crc32(value.encode("utf-8")) % 100_000


def make_rf():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=160,
            max_depth=10,
            min_samples_leaf=20,
            max_features="sqrt",
            bootstrap=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    )


def load_merged_rr() -> pd.DataFrame:
    print("Loading features...")
    feat = pd.read_csv(REPORT_DIR / "official_query_triple_features.csv")
    feat = base.add_official_scaled_rr(feat)

    print("Loading true RR...")
    wide = pd.read_csv(TRUE_RR_WIDE)
    keep = JOIN_KEYS + ["true_official_rr_stage1", "true_official_rr_stage2"]
    wide = wide[keep]

    print("Merging features + true RR...")
    merged = feat.merge(wide, on=JOIN_KEYS, how="left", validate="one_to_one")

    # TRUE RR where available, fallback to official_scaled_rr
    merged["rr_stage1"] = merged["true_official_rr_stage1"].fillna(merged["official_scaled_rr_stage1"])
    merged["rr_stage2"] = merged["true_official_rr_stage2"].fillna(merged["official_scaled_rr_stage2"])

    # Report coverage of true RR
    true_count = merged["true_official_rr_stage1"].notna().sum()
    print(f"  True RR available: {true_count:,} / {len(merged):,} ({true_count/len(merged)*100:.1f}%)")

    return merged


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all ratio features."""
    df = df.copy()
    for k in [1, 3, 5]:
        df[f"ratio{k}"] = df[f"synergy_weight_top{k}_mean"] / (df[f"rule_weight_top{k}_mean"] + 1e-9)
    return df


def target_gain_clip(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    gain = np.where(s1 > 0, s2 / s1 - 1.0, 0.0)
    return np.clip(gain, -1.0, 1.0)


def fit_rf_score(features_df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    x = features_df.replace([np.inf, -np.inf], np.nan)
    model = make_rf()
    model.fit(x, y)
    return np.asarray(model.predict(x), dtype=float)


def score_at_coverage(scores: np.ndarray, s1: np.ndarray, s2: np.ndarray, coverage: float) -> dict:
    order = np.argsort(scores, kind="mergesort")[::-1]
    n = max(1, int(round(len(order) * coverage)))
    selected = order[:n]
    m1 = float(s1[selected].mean())
    m2 = float(s2[selected].mean())
    gain_pt = (m2 / m1 - 1.0) if m1 > 0 else 0.0
    return {"n": n, "mrr_stage1": m1, "mrr_stage2": m2, "gain_pt": gain_pt}


def train_and_evaluate_global(
    df: pd.DataFrame,
    selector: str,
    features: list[str],
    coverages: np.ndarray,
) -> pd.DataFrame:
    """Train one global RF, evaluate per-dataset + macro."""
    y = target_gain_clip(
        df["rr_stage1"].to_numpy(dtype=float),
        df["rr_stage2"].to_numpy(dtype=float),
    )
    score = fit_rf_score(df[features], y)

    rows = []
    for dataset, group in df.groupby("dataset", sort=True):
        idx = group.index
        ds_scores = score[idx]
        ds_s1 = group["rr_stage1"].to_numpy(dtype=float)
        ds_s2 = group["rr_stage2"].to_numpy(dtype=float)

        for cov in coverages:
            r = score_at_coverage(ds_scores, ds_s1, ds_s2, float(cov))
            rows.append({
                "dataset": dataset,
                "selector": selector,
                "coverage": float(cov),
                **r,
            })

    return pd.DataFrame(rows)


def main():
    df = load_merged_rr()
    df = add_features(df)

    print(f"\nTotal queries: {len(df):,}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")

    # =====================================================================
    # 1. All 9 single-feature macro gains
    # =====================================================================
    single_feature_map = {
        "comp1": ["synergy_weight_top1_mean"],
        "comp3": ["synergy_weight_top3_mean"],
        "comp5": ["synergy_weight_top5_mean"],
        "rule1": ["rule_weight_top1_mean"],
        "rule3": ["rule_weight_top3_mean"],
        "rule5": ["rule_weight_top5_mean"],
        "ratio1": ["ratio1"],
        "ratio3": ["ratio3"],
        "ratio5": ["ratio5"],
    }

    print("\n" + "=" * 80)
    print("1. SINGLE-FEATURE GLOBAL RF MACRO GAINS (TRUE RR)")
    print("=" * 80)

    all_curves_list = []
    macro_rows = []

    for name, features in single_feature_map.items():
        print(f"\n  Training: {name} ...", end=" ", flush=True)
        curves = train_and_evaluate_global(df, name, features, MAIN_COVERAGES)
        all_curves_list.append(curves)
        print("done")

        # Macro average at 10/20/50
        macro_gains = {}
        for cov in MAIN_COVERAGES:
            key = int(cov * 100)
            val = curves[curves["coverage"] == cov]["gain_pt"].mean()
            macro_gains[f"gain@{key}"] = val

        macro_rows.append({"feature": name, **macro_gains})
        print(f"    gain@10={macro_gains['gain@10']:.2f}%, gain@20={macro_gains['gain@20']:.2f}%, gain@50={macro_gains['gain@50']:.2f}%")

    macro_df = pd.DataFrame(macro_rows)
    macro_df_fmt = macro_df.copy()
    for col in ["gain@10", "gain@20", "gain@50"]:
        macro_df_fmt[col] = macro_df_fmt[col].map(lambda x: f"{x*100:.2f}%")
    print("\n  " + macro_df_fmt.to_markdown(index=False))

    # =====================================================================
    # 2. ratio3_global_rf specific results
    # =====================================================================
    print("\n" + "=" * 80)
    print("2. ratio3_global_rf PER-DATASET GAINS @ 10%")
    print("=" * 80)

    # Re-train ratio3 specifically (already done above, extract)
    ratio3_curves = [c for c in all_curves_list if c["selector"].iloc[0] == "ratio3"][0]
    ratio3_10 = ratio3_curves[ratio3_curves["coverage"] == 0.10].copy()
    ratio3_10["gain_pt"] = ratio3_10["gain_pt"].map(lambda x: f"{x*100:.2f}%")
    ratio3_10["mrr_stage1"] = ratio3_10["mrr_stage1"].map(lambda x: f"{x:.6f}")
    ratio3_10["mrr_stage2"] = ratio3_10["mrr_stage2"].map(lambda x: f"{x:.6f}")
    print("\n  " + ratio3_10[["dataset", "n", "mrr_stage1", "mrr_stage2", "gain_pt"]].to_markdown(index=False))

    # Macro gain@10 for ratio3
    macro_gain10 = ratio3_curves[ratio3_curves["coverage"] == 0.10]["gain_pt"].mean()
    print(f"\n  ratio3_global_rf macro gain@10 = {macro_gain10*100:.2f}%")

    # =====================================================================
    # 3. ratio3_global_rf top-10% q10-q90 ranges
    # =====================================================================
    print("\n" + "=" * 80)
    print("3. ratio3_global_rf TOP-10% SELECTED QUERY RANGES")
    print("=" * 80)

    # Get the scores for ratio3
    y = target_gain_clip(
        df["rr_stage1"].to_numpy(dtype=float),
        df["rr_stage2"].to_numpy(dtype=float),
    )
    ratio3_scores = fit_rf_score(df[["ratio3"]], y)

    for dataset, group in df.groupby("dataset", sort=True):
        idx = group.index
        ds_scores = ratio3_scores[idx]
        order = np.argsort(ds_scores, kind="mergesort")[::-1]
        n = max(1, int(round(len(order) * 0.10)))
        selected_idx = order[:n]

        sel_comp3 = group.iloc[selected_idx]["synergy_weight_top3_mean"].values
        sel_rule3 = group.iloc[selected_idx]["rule_weight_top3_mean"].values
        sel_ratio3 = sel_comp3 / (sel_rule3 + 1e-9)

        def q(arr, p):
            return float(np.percentile(arr, p))

        print(f"\n  {dataset} (n={n}):")
        for name, vals in [("comp3", sel_comp3), ("rule3", sel_rule3), ("ratio3", sel_ratio3)]:
            print(f"    {name}: q10={q(vals,10):.6f}  q25={q(vals,25):.6f}  median={q(vals,50):.6f}  q75={q(vals,75):.6f}  q90={q(vals,90):.6f}")

    # =====================================================================
    # 4. Coverage curve for Fig. 5B (ratio3 only, all datasets + macro)
    # =====================================================================
    print("\n" + "=" * 80)
    print("4. BUILDING COVERAGE CURVE FOR FIG. 5B ...", end=" ", flush=True)

    fig5b_rows = []
    macro_by_cov = {}

    for dataset, group in df.groupby("dataset", sort=True):
        idx = group.index
        ds_scores = ratio3_scores[idx]
        ds_s1 = group["rr_stage1"].to_numpy(dtype=float)
        ds_s2 = group["rr_stage2"].to_numpy(dtype=float)

        for cov in FIG5B_COVERAGES:
            r = score_at_coverage(ds_scores, ds_s1, ds_s2, float(cov))
            fig5b_rows.append({
                "dataset": dataset,
                "selector": "ratio3_global_rf",
                "coverage": round(float(cov), 4),
                **r,
            })

    # Macro average across datasets
    for cov in FIG5B_COVERAGES:
        cov_val = round(float(cov), 4)
        ds_rows = [r for r in fig5b_rows if r["coverage"] == cov_val]
        macro_n = int(np.mean([r["n"] for r in ds_rows]))
        macro_m1 = float(np.mean([r["mrr_stage1"] for r in ds_rows]))
        macro_m2 = float(np.mean([r["mrr_stage2"] for r in ds_rows]))
        macro_gain = float(np.mean([r["gain_pt"] for r in ds_rows]))
        fig5b_rows.append({
            "dataset": "macro",
            "selector": "ratio3_global_rf",
            "coverage": cov_val,
            "n": macro_n,
            "mrr_stage1": macro_m1,
            "mrr_stage2": macro_m2,
            "gain_pt": macro_gain,
        })

    fig5b_df = pd.DataFrame(fig5b_rows)
    out_curve = REPORT_DIR / "ratio3_rf_gain_curves_true_rr.csv"
    fig5b_df.to_csv(out_curve, index=False)
    print("done")
    print(f"  Saved: {out_curve}")
    print(f"  Rows: {len(fig5b_df)} (7 datasets × {len(FIG5B_COVERAGES)} coverages + {len(FIG5B_COVERAGES)} macro rows)")

    # =====================================================================
    # 5. Save macro table for all 9 features
    # =====================================================================
    out_macro = REPORT_DIR / "ratio3_rf_all9_macro.csv"
    macro_numeric = pd.DataFrame(macro_rows)
    for col in ["gain@10", "gain@20", "gain@50"]:
        macro_numeric[col] = macro_numeric[col].map(lambda x: f"{x*100:.2f}%")
    macro_numeric.to_csv(out_macro, index=False)
    print(f"\n  Macro table saved to: {out_macro}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
