#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline

import regenerate_official_query_subset_reports as base
from optimize_official_query_dataset_formulas import eligible_features


REPORT_DIR = Path("/home/sy/RuleDep/reports/official_query_subset")
OUT_DIR = REPORT_DIR / "ml_selectors"
RANDOM_STATE = 20260430
REPORT_COVERAGES = [0.10, 0.20, 0.30, 0.50, 1.00]


def make_rf(n_estimators: int, max_depth: int | None, min_samples_leaf: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def target_values(group: pd.DataFrame, target: str) -> np.ndarray:
    s1 = group["official_scaled_rr_stage1"].to_numpy(dtype=float)
    s2 = group["official_scaled_rr_stage2"].to_numpy(dtype=float)
    if target == "delta":
        return s2 - s1
    if target == "gain_clip":
        gain = np.where(s1 > 0, s2 / s1 - 1.0, 0.0)
        return np.clip(gain, -1.0, 1.0)
    raise ValueError(target)


def score_coverages(group: pd.DataFrame, score: np.ndarray, coverages: list[float]) -> dict[float, dict[str, float]]:
    order = np.argsort(score, kind="mergesort")[::-1]
    s1 = group["official_scaled_rr_stage1"].to_numpy(dtype=float)
    s2 = group["official_scaled_rr_stage2"].to_numpy(dtype=float)
    out = {}
    for coverage in coverages:
        n = max(1, int(round(len(order) * coverage)))
        selected = order[:n]
        m1 = float(s1[selected].mean())
        m2 = float(s2[selected].mean())
        out[coverage] = {
            "n": n,
            "mrr_stage1": m1,
            "mrr_stage2": m2,
            "gain_pt": (m2 / m1 - 1.0) if m1 > 0 else 0.0,
        }
    return out


def fit_predict_in_sample(
    x: pd.DataFrame,
    y: np.ndarray,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_train_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(x) > max_train_rows:
        train_idx = rng.choice(len(x), size=max_train_rows, replace=False)
    else:
        train_idx = np.arange(len(x))
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        make_rf(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf),
    )
    model.fit(x.iloc[train_idx], y[train_idx])
    return model.predict(x)


def fit_predict_oof(
    x: pd.DataFrame,
    y: np.ndarray,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_train_rows: int,
    folds: int,
    rng: np.random.Generator,
) -> np.ndarray:
    preds = np.zeros(len(x), dtype=float)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x), start=1):
        if len(train_idx) > max_train_rows:
            train_idx = rng.choice(train_idx, size=max_train_rows, replace=False)
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            make_rf(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf),
        )
        print(f"    fold {fold}/{folds} train={len(train_idx):,} valid={len(valid_idx):,}", flush=True)
        model.fit(x.iloc[train_idx], y[train_idx])
        preds[valid_idx] = model.predict(x.iloc[valid_idx])
    return preds


def run_dataset(
    dataset: str,
    group: pd.DataFrame,
    features: list[str],
    targets: list[str],
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_train_rows: int,
    folds: int,
) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(RANDOM_STATE + abs(hash(dataset)) % 10000)
    x = group[features].replace([np.inf, -np.inf], np.nan)
    summary_rows = []
    curve_rows = []
    for target in targets:
        y = target_values(group, target)
        for mode in ("in_sample", "oof"):
            print(f"[{dataset}] RF target={target} mode={mode}", flush=True)
            if mode == "in_sample":
                score = fit_predict_in_sample(x, y, n_estimators, max_depth, min_samples_leaf, max_train_rows, rng)
            else:
                score = fit_predict_oof(x, y, n_estimators, max_depth, min_samples_leaf, max_train_rows, folds, rng)
            result = score_coverages(group, score, REPORT_COVERAGES)
            row = {
                "dataset": dataset,
                "model": "RandomForestRegressor",
                "target": target,
                "mode": mode,
                "n": len(group),
            }
            for coverage, values in result.items():
                key = int(round(coverage * 100))
                row[f"gain_{key}"] = values["gain_pt"]
                row[f"mrr_stage1_{key}"] = values["mrr_stage1"]
                row[f"mrr_stage2_{key}"] = values["mrr_stage2"]
            summary_rows.append(row)
            for coverage, values in result.items():
                curve_rows.append(
                    {
                        "dataset": dataset,
                        "model": "RandomForestRegressor",
                        "target": target,
                        "mode": mode,
                        "coverage": coverage,
                        **values,
                    }
                )
    return summary_rows, curve_rows


def write_report(summary: pd.DataFrame, out_path: Path) -> None:
    macro = (
        summary.groupby(["model", "target", "mode"])[["gain_10", "gain_20", "gain_30", "gain_50", "gain_100"]]
        .mean()
        .reset_index()
        .sort_values(["mode", "gain_10", "gain_20"], ascending=[True, False, False])
    )
    best_oof = (
        summary[summary["mode"] == "oof"]
        .sort_values(["dataset", "gain_10", "gain_20"], ascending=[True, False, False])
        .groupby("dataset", as_index=False)
        .head(1)
    )
    best_insample = (
        summary[summary["mode"] == "in_sample"]
        .sort_values(["dataset", "gain_10", "gain_20"], ascending=[True, False, False])
        .groupby("dataset", as_index=False)
        .head(1)
    )

    def md_table(df: pd.DataFrame, cols: list[str]) -> str:
        out = df[cols].copy()
        for col in out.columns:
            if pd.api.types.is_numeric_dtype(out[col]):
                out[col] = out[col].map(lambda v: f"{float(v):.4f}")
        return out.to_markdown(index=False)

    lines = [
        "# ML Query Selector Results",
        "",
        "Models use only pre-stage2 raw query attributes and official-scaled targets. 100% coverage is aligned with official `metric-*.json` MRR.",
        "",
        "- `in_sample`: diagnostic upper bound; train and rank on the same query set.",
        "- `oof`: out-of-fold scores; each query is ranked by a model that did not train on that query.",
        "",
        "## Macro Results",
        "",
        md_table(macro, ["model", "target", "mode", "gain_10", "gain_20", "gain_30", "gain_50", "gain_100"]),
        "",
        "## Best OOF Model per Dataset",
        "",
        md_table(best_oof, ["dataset", "target", "gain_10", "gain_20", "gain_30", "gain_50", "gain_100"]),
        "",
        "## Best In-sample Model per Dataset",
        "",
        md_table(best_insample, ["dataset", "target", "gain_10", "gain_20", "gain_30", "gain_50", "gain_100"]),
        "",
        "## Caution",
        "",
        "In-sample numbers are not paper-safe as test claims. OOF numbers are more defensible, but a final paper claim should ideally tune model/hyperparameters on validation queries and report test queries once.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=160)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--max-train-rows", type=int, default=120000)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--targets", nargs="+", default=["delta", "gain_clip"])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[load]", flush=True)
    df = pd.read_csv(REPORT_DIR / "official_query_triple_features.csv")
    df = base.add_official_scaled_rr(df)
    features = eligible_features(df)
    print(f"[features] {len(features)}", flush=True)

    all_summary = []
    all_curves = []
    for dataset, group in df.groupby("dataset", sort=True):
        print(f"[dataset] {dataset} rows={len(group):,}", flush=True)
        summary, curves = run_dataset(
            dataset=str(dataset),
            group=group.reset_index(drop=True),
            features=features,
            targets=args.targets,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_train_rows=args.max_train_rows,
            folds=args.folds,
        )
        all_summary.extend(summary)
        all_curves.extend(curves)
        pd.DataFrame(all_summary).to_csv(OUT_DIR / "rf_selector_summary.csv", index=False)
        pd.DataFrame(all_curves).to_csv(OUT_DIR / "rf_selector_curves.csv", index=False)

    summary_df = pd.DataFrame(all_summary)
    write_report(summary_df, OUT_DIR / "README.md")
    print(OUT_DIR / "README.md", flush=True)


if __name__ == "__main__":
    main()
