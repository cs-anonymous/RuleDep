#!/usr/bin/env python3
"""Analyze comp3, comp5, rule3, rule5, ratio3, ratio5 using TRUE RR data (same as balanced5)."""
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
COVERAGES = np.array([0.10, 0.20, 0.50])


def stable_seed(value: str) -> int:
    return RANDOM_STATE + zlib.crc32(value.encode("utf-8")) % 100_000


def make_rf(n_estimators: int = 160, max_depth: int = 10, min_samples_leaf: int = 20):
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            bootstrap=True,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    )


def load_merged_rr() -> pd.DataFrame:
    """Load and merge features with true RR (same as balanced5_true_rr)."""
    print("Loading features...")
    feat = pd.read_csv(REPORT_DIR / "official_query_triple_features.csv")
    feat = base.add_official_scaled_rr(feat)

    print("Loading true RR...")
    wide = pd.read_csv(TRUE_RR_WIDE)
    keep = JOIN_KEYS + ["true_official_rr_stage1", "true_official_rr_stage2"]
    wide = wide[keep]

    print("Merging...")
    merged = feat.merge(wide, on=JOIN_KEYS, how="left", validate="one_to_one")

    # Replace scaled RR with true RR (fallback to scaled if true is missing)
    merged["official_scaled_rr_stage1"] = merged["true_official_rr_stage1"].fillna(
        merged["official_scaled_rr_stage1"]
    )
    merged["official_scaled_rr_stage2"] = merged["true_official_rr_stage2"].fillna(
        merged["official_scaled_rr_stage2"]
    )

    return merged


def target_values(group: pd.DataFrame, target: str) -> np.ndarray:
    s1 = group["official_scaled_rr_stage1"].to_numpy(dtype=float)
    s2 = group["official_scaled_rr_stage2"].to_numpy(dtype=float)
    if target == "gain_clip":
        gain = np.where(s1 > 0, s2 / s1 - 1.0, 0.0)
        return np.clip(gain, -1.0, 1.0)
    raise ValueError(target)


def score_coverages(group: pd.DataFrame, score: np.ndarray, coverages: np.ndarray) -> pd.DataFrame:
    order = np.argsort(score, kind="mergesort")[::-1]
    s1 = group["official_scaled_rr_stage1"].to_numpy(dtype=float)
    s2 = group["official_scaled_rr_stage2"].to_numpy(dtype=float)
    rows = []
    for coverage in coverages:
        n = max(1, int(round(len(order) * float(coverage))))
        selected = order[:n]
        m1 = float(s1[selected].mean())
        m2 = float(s2[selected].mean())
        rows.append({
            "coverage": float(coverage),
            "n": n,
            "mrr_stage1": m1,
            "mrr_stage2": m2,
            "gain_pt": (m2 / m1 - 1.0) if m1 > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def fit_rf_score(
    x: pd.DataFrame,
    y: np.ndarray,
    seed_key: str,
    max_train_rows: int | None,
) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(seed_key))
    train_idx = np.arange(len(x))
    if max_train_rows is not None and len(train_idx) > max_train_rows:
        train_idx = np.sort(rng.choice(train_idx, size=max_train_rows, replace=False))
    model = make_rf()
    model.fit(x.iloc[train_idx], y[train_idx])
    return np.asarray(model.predict(x), dtype=float)


def train_global_scores(df: pd.DataFrame, selector: str, features: list[str]) -> pd.DataFrame:
    """Train ONE global RF on all datasets, then evaluate per-dataset."""
    y = target_values(df, "gain_clip")
    x = df[features].replace([np.inf, -np.inf], np.nan)
    score = fit_rf_score(
        x,
        y,
        seed_key=f"global:{selector}:gain_clip:RandomForestRegressor",
        max_train_rows=None,
    )

    scored = df[["dataset", "official_scaled_rr_stage1", "official_scaled_rr_stage2"]].copy()
    scored["_score"] = score

    rows = []
    for dataset, group in scored.groupby("dataset", sort=True):
        g = group.reset_index(drop=True)
        curves = score_coverages(g, g["_score"].to_numpy(dtype=float), COVERAGES)
        curves.insert(0, "selector", selector)
        curves.insert(0, "dataset", dataset)
        rows.append(curves)

    return pd.concat(rows, ignore_index=True)


def main():
    # Load merged data with true RR
    df = load_merged_rr()

    # Verify columns
    required = [
        "synergy_weight_top1_mean",
        "synergy_weight_top3_mean",
        "synergy_weight_top5_mean",
        "rule_weight_top1_mean",
        "rule_weight_top3_mean",
        "rule_weight_top5_mean",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        return

    # Add ratio features
    print("Adding ratio features...")
    df["ratio1"] = df["synergy_weight_top1_mean"] / (df["rule_weight_top1_mean"] + 1e-9)
    df["ratio3"] = df["synergy_weight_top3_mean"] / (df["rule_weight_top3_mean"] + 1e-9)
    df["ratio5"] = df["synergy_weight_top5_mean"] / (df["rule_weight_top5_mean"] + 1e-9)

    print(f"Total queries: {len(df):,}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")

    # Feature configs
    feature_map = {
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

    # Single-feature Global RF
    print("\n" + "="*80)
    print("SINGLE-FEATURE GLOBAL RF (using TRUE RR data)")
    print("="*80)

    all_curves = []
    for name, features in feature_map.items():
        print(f"\nTraining: {name}")
        curves = train_global_scores(df, name, features)
        all_curves.append(curves)

    # Dual-feature Global RF
    print("\n" + "="*80)
    print("DUAL-FEATURE GLOBAL RF")
    print("="*80)

    dual_configs = {
        "comp1+rule1": ["synergy_weight_top1_mean", "rule_weight_top1_mean"],
        "comp3+rule3": ["synergy_weight_top3_mean", "rule_weight_top3_mean"],
        "comp5+rule5": ["synergy_weight_top5_mean", "rule_weight_top5_mean"],
    }

    for name, features in dual_configs.items():
        print(f"\nTraining: {name}")
        curves = train_global_scores(df, name, features)
        all_curves.append(curves)

    # Combine and compute macro
    all_curves_df = pd.concat(all_curves, ignore_index=True)

    # Macro average
    macro = (
        all_curves_df.groupby(["selector", "coverage"], as_index=False)["gain_pt"]
        .mean()
        .pivot(index="selector", columns="coverage", values="gain_pt")
        .reset_index()
    )
    macro.columns = ["selector"] + [f"gain_{int(c * 100)}" for c in COVERAGES]

    # Format as percentages
    for col in macro.columns:
        if col.startswith("gain_"):
            macro[col] = macro[col].map(lambda x: f"{x*100:.2f}%")

    print("\n" + "="*80)
    print("MACRO AVERAGE (7 datasets) - TRUE RR")
    print("="*80)
    print(macro.to_markdown(index=False))

    # Save
    out_csv = REPORT_DIR / "comp_rule_rf_analysis_true_rr.csv"
    all_curves_df.to_csv(out_csv, index=False)
    print(f"\n\nDetailed curves saved to: {out_csv}")

    # Save macro table
    out_macro = REPORT_DIR / "comp_rule_rf_macro_true_rr.csv"
    macro_numeric = (
        all_curves_df.groupby(["selector", "coverage"], as_index=False)["gain_pt"]
        .mean()
        .pivot(index="selector", columns="coverage", values="gain_pt")
        .reset_index()
    )
    macro_numeric.columns = ["selector"] + [f"gain_{int(c * 100)}" for c in COVERAGES]
    macro_numeric.to_csv(out_macro, index=False)
    print(f"Macro table saved to: {out_macro}")


if __name__ == "__main__":
    main()
