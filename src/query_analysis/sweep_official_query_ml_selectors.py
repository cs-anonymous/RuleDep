#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import math
import zlib
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import regenerate_official_query_subset_reports as base
from optimize_official_query_dataset_formulas import eligible_features


ROOT = Path("/home/sy/RuleDep")
REPORT_DIR = ROOT / "reports" / "official_query_subset"
OUT_DIR = REPORT_DIR / "ml_selector_sweep"
RANDOM_STATE = 20260430
REPORT_COVERAGES = [0.10, 0.20, 0.30, 0.50, 1.00]
TARGET_COVERAGES = [0.10, 0.20]


def stable_seed(value: str) -> int:
    return RANDOM_STATE + zlib.crc32(value.encode("utf-8")) % 100_000


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


def percentile(values: np.ndarray, direction: str) -> np.ndarray:
    series = pd.Series(values)
    pct = series.rank(pct=True, method="average").to_numpy(dtype=float)
    return pct if direction == "desc" else 1.0 - pct


def single_feature_rankings(group: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        values = group[feature].to_numpy(dtype=float)
        for direction in ("desc", "asc"):
            score = percentile(values, direction)
            result = score_coverages(group, score, TARGET_COVERAGES)
            g10 = result[0.10]["gain_pt"]
            g20 = result[0.20]["gain_pt"]
            rows.append(
                {
                    "feature": feature,
                    "direction": direction,
                    "gain_10": g10,
                    "gain_20": g20,
                    "objective": g10 + g20,
                    "min_gain": min(g10, g20),
                }
            )
    return pd.DataFrame(rows).sort_values(["objective", "min_gain"], ascending=False)


def make_feature_sets(
    group: pd.DataFrame,
    features: list[str],
    requested_sizes: list[int],
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    singles = single_feature_rankings(group, features)
    ordered_features: list[str] = []
    for row in singles.itertuples(index=False):
        feature = str(row.feature)
        if feature not in ordered_features:
            ordered_features.append(feature)

    sets: dict[str, list[str]] = {}
    for size in requested_sizes:
        if size <= 0:
            continue
        actual = min(size, len(ordered_features))
        sets[f"top{actual}"] = ordered_features[:actual]
    sets[f"all{len(features)}"] = features
    return sets, singles


def global_feature_ranking(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    for dataset, raw_group in df.groupby("dataset", sort=True):
        group = raw_group.reset_index(drop=True)
        singles = single_feature_rankings(group, features)
        best_per_feature = (
            singles.sort_values(["feature", "objective", "min_gain"], ascending=[True, False, False])
            .groupby("feature", as_index=False)
            .head(1)
        )
        for row in best_per_feature.itertuples(index=False):
            detail_rows.append({"dataset": dataset, **row._asdict()})

    detail = pd.DataFrame(detail_rows)
    ranking = (
        detail.groupby("feature", as_index=False)
        .agg(
            macro_gain_10=("gain_10", "mean"),
            macro_gain_20=("gain_20", "mean"),
            macro_objective=("objective", "mean"),
            macro_min_gain=("min_gain", "mean"),
            datasets=("dataset", "nunique"),
        )
        .sort_values(["macro_objective", "macro_min_gain", "macro_gain_10"], ascending=False)
    )
    return ranking, detail


def make_model_factories() -> dict[str, Callable[[], object]]:
    factories: dict[str, Callable[[], object]] = {
        "Ridge": lambda: make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=1.0, random_state=RANDOM_STATE),
        ),
        "RandomForestRegressor": lambda: make_pipeline(
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
        ),
    }
    try:
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover - depends on local environment
        print(f"[warn] xgboost unavailable: {exc}", flush=True)
    else:
        factories["XGBRegressor"] = lambda: XGBRegressor(
            n_estimators=160,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=8,
            random_state=RANDOM_STATE,
        )
    return factories


def sample_train_idx(train_idx: np.ndarray, max_train_rows: int, rng: np.random.Generator) -> np.ndarray:
    if len(train_idx) <= max_train_rows:
        return train_idx
    return np.sort(rng.choice(train_idx, size=max_train_rows, replace=False))


def fit_predict_in_sample(
    model: object,
    x: pd.DataFrame,
    y: np.ndarray,
    max_train_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    train_idx = sample_train_idx(np.arange(len(x)), max_train_rows, rng)
    fitted = clone(model)
    fitted.fit(x.iloc[train_idx], y[train_idx])
    return np.asarray(fitted.predict(x), dtype=float)


def fit_predict_oof(
    model: object,
    x: pd.DataFrame,
    y: np.ndarray,
    max_train_rows: int,
    folds: int,
    rng: np.random.Generator,
) -> np.ndarray:
    preds = np.zeros(len(x), dtype=float)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x), start=1):
        train_idx = sample_train_idx(train_idx, max_train_rows, rng)
        fitted = clone(model)
        print(f"      fold {fold}/{folds} train={len(train_idx):,} valid={len(valid_idx):,}", flush=True)
        fitted.fit(x.iloc[train_idx], y[train_idx])
        preds[valid_idx] = np.asarray(fitted.predict(x.iloc[valid_idx]), dtype=float)
    return preds


def add_result_rows(
    rows: list[dict],
    curve_rows: list[dict],
    *,
    dataset: str,
    model_name: str,
    target: str,
    mode: str,
    feature_set: str,
    feature_count: int,
    group: pd.DataFrame,
    score: np.ndarray,
) -> None:
    result = score_coverages(group, score, REPORT_COVERAGES)
    row = {
        "dataset": dataset,
        "model": model_name,
        "target": target,
        "mode": mode,
        "feature_set": feature_set,
        "feature_count": feature_count,
        "n": len(group),
    }
    for coverage, values in result.items():
        key = int(round(coverage * 100))
        row[f"gain_{key}"] = values["gain_pt"]
        row[f"mrr_stage1_{key}"] = values["mrr_stage1"]
        row[f"mrr_stage2_{key}"] = values["mrr_stage2"]
    rows.append(row)
    for coverage, values in result.items():
        curve_rows.append(
            {
                "dataset": dataset,
                "model": model_name,
                "target": target,
                "mode": mode,
                "feature_set": feature_set,
                "feature_count": feature_count,
                "coverage": coverage,
                **values,
            }
        )


def write_report(summary: pd.DataFrame, feature_rows: list[dict], out_path: Path) -> None:
    gain_cols = ["gain_10", "gain_20", "gain_30", "gain_50", "gain_100"]
    macro = (
        summary.groupby(["model", "target", "mode", "feature_set", "feature_count"], dropna=False)[gain_cols]
        .mean()
        .reset_index()
    )
    macro["objective_10_plus_20"] = macro["gain_10"] + macro["gain_20"]
    macro = macro.sort_values(["mode", "objective_10_plus_20", "gain_10"], ascending=[True, False, False])

    def md_table(df: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
        out = df[cols].head(max_rows).copy() if max_rows else df[cols].copy()
        for col in out.columns:
            if pd.api.types.is_numeric_dtype(out[col]):
                out[col] = out[col].map(lambda v: f"{float(v):.4f}")
        return out.to_markdown(index=False)

    def best_rows_for_mode(mode: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        data = summary[summary["mode"] == mode].copy()
        if not len(data):
            return data, pd.Series(dtype=float), data, pd.Series(dtype=float)
        data["objective_10_plus_20"] = data["gain_10"] + data["gain_20"]
        best = (
            data.sort_values(["dataset", "objective_10_plus_20", "gain_10"], ascending=[True, False, False])
            .groupby("dataset", as_index=False)
            .head(1)
        )
        all_names = {f"all{count}" for count in sorted(summary["feature_count"].unique())}
        sparse = data[~data["feature_set"].isin(all_names)].copy()
        best_sparse = (
            sparse.sort_values(["dataset", "objective_10_plus_20", "feature_count"], ascending=[True, False, True])
            .groupby("dataset", as_index=False)
            .head(1)
            if len(sparse)
            else sparse
        )
        return best, best[gain_cols].mean(), best_sparse, best_sparse[gain_cols].mean() if len(best_sparse) else pd.Series(dtype=float)

    lines = [
        "# ML Selector Sweep",
        "",
        "Dataset-specific linear/RF/XGB selectors trained from pre-stage2 raw query attributes.",
        "",
        "- `oof`: each query is scored by a model trained on other folds from the same dataset.",
        "- Feature sets are selected per dataset by single-attribute diagnostic ranking, then used by the model.",
        "- Selection of the best row below is still diagnostic because model/feature-set choice is made on this same official-query table.",
        "",
    ]
    for mode, title in [("oof", "OOF"), ("in_sample", "In-sample")]:
        best, best_macro, best_sparse, best_sparse_macro = best_rows_for_mode(mode)
        if not len(best):
            continue
        lines.extend(
            [
                f"## Best {title} per Dataset",
                "",
                md_table(
                    best,
                    [
                        "dataset",
                        "model",
                        "target",
                        "feature_set",
                        "feature_count",
                        "gain_10",
                        "gain_20",
                        "gain_30",
                        "gain_50",
                        "gain_100",
                    ],
                ),
                "",
                f"Macro over the best {title} row per dataset:",
                "",
                "| coverage | macro gain_pt |",
                "| ---: | ---: |",
            ]
        )
        for key in gain_cols:
            lines.append(f"| {key.removeprefix('gain_')}% | {best_macro[key]:.4f} |")

        if len(best_sparse):
            lines.extend(
                [
                    "",
                    f"## Best Sparse {title} per Dataset",
                    "",
                    md_table(
                        best_sparse,
                        [
                            "dataset",
                            "model",
                            "target",
                            "feature_set",
                            "feature_count",
                            "gain_10",
                            "gain_20",
                            "gain_30",
                            "gain_50",
                            "gain_100",
                        ],
                    ),
                    "",
                    f"Macro over the best sparse {title} row per dataset:",
                    "",
                    "| coverage | macro gain_pt |",
                    "| ---: | ---: |",
                ]
            )
            for key in gain_cols:
                lines.append(f"| {key.removeprefix('gain_')}% | {best_sparse_macro[key]:.4f} |")
        lines.append("")

    lines.extend(
        [
            "## Top Macro Rows",
            "",
            md_table(
                macro,
                ["model", "target", "mode", "feature_set", "feature_count", "gain_10", "gain_20", "gain_30", "gain_50", "gain_100"],
                max_rows=20,
            ),
            "",
            "## Files",
            "",
            "- `ml_selector_sweep_summary.csv`: per dataset/model/target/mode/feature-set results.",
            "- `ml_selector_sweep_curves.csv`: long-form coverage curves.",
            "- `ml_selector_sweep_feature_sets.csv`: selected features in each per-dataset feature set.",
            "- `single_feature_rankings.csv`: diagnostic single-attribute ranking used to build sparse feature sets.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-sizes", nargs="+", type=int, default=[4, 8, 16, 32])
    parser.add_argument("--same-feature-sizes", nargs="+", type=int, default=[])
    parser.add_argument("--only-same", action="store_true")
    parser.add_argument("--targets", nargs="+", default=["delta", "gain_clip"])
    parser.add_argument("--models", nargs="+", default=["Ridge", "RandomForestRegressor", "XGBRegressor"])
    parser.add_argument("--modes", nargs="+", default=["oof", "in_sample"])
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-train-rows", type=int, default=120_000)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[load]", flush=True)
    df = pd.read_csv(REPORT_DIR / "official_query_triple_features.csv")
    df = base.add_official_scaled_rr(df)
    features = eligible_features(df)
    print(f"[features] eligible={len(features)}", flush=True)

    factories = make_model_factories()
    requested_models = [name for name in args.models if name in factories]
    missing = sorted(set(args.models) - set(requested_models))
    if missing:
        print(f"[warn] skipping unavailable models: {', '.join(missing)}", flush=True)

    summary_rows: list[dict] = []
    curve_rows: list[dict] = []
    feature_rows: list[dict] = []
    single_rows: list[dict] = []
    same_feature_sets: dict[str, list[str]] = {}

    if args.same_feature_sizes:
        print(f"[same-feature-ranking] sizes={args.same_feature_sizes}", flush=True)
        same_ranking, same_detail = global_feature_ranking(df, features)
        same_ranking.to_csv(out_dir / "same_feature_global_ranking.csv", index=False)
        same_detail.to_csv(out_dir / "same_feature_dataset_detail.csv", index=False)
        ordered_same_features = same_ranking["feature"].astype(str).tolist()
        for size in args.same_feature_sizes:
            if size <= 0:
                continue
            actual = min(size, len(ordered_same_features))
            same_feature_sets[f"same_top{actual}"] = ordered_same_features[:actual]

    for dataset, raw_group in df.groupby("dataset", sort=True):
        group = raw_group.reset_index(drop=True)
        print(f"[dataset] {dataset} rows={len(group):,}", flush=True)
        if args.only_same:
            feature_sets = dict(same_feature_sets)
        else:
            feature_sets, singles = make_feature_sets(group, features, args.feature_sizes)
            feature_sets.update(same_feature_sets)
            for rank, row in enumerate(singles.itertuples(index=False), start=1):
                single_rows.append({"dataset": dataset, "rank": rank, **row._asdict()})
        for feature_set, selected_features in feature_sets.items():
            for rank, feature in enumerate(selected_features, start=1):
                feature_rows.append(
                    {
                        "dataset": dataset,
                        "feature_set": feature_set,
                        "feature_count": len(selected_features),
                        "rank": rank,
                        "feature": feature,
                    }
                )

        for feature_set, selected_features in feature_sets.items():
            x = group[selected_features].replace([np.inf, -np.inf], np.nan)
            for target in args.targets:
                y = target_values(group, target)
                for model_name in requested_models:
                    base_model = factories[model_name]()
                    rng = np.random.default_rng(stable_seed(f"{dataset}:{feature_set}:{target}:{model_name}"))
                    for mode in args.modes:
                        print(
                            f"  [{feature_set:>5}] {model_name} target={target} mode={mode}",
                            flush=True,
                        )
                        if mode == "oof":
                            score = fit_predict_oof(
                                base_model,
                                x,
                                y,
                                max_train_rows=args.max_train_rows,
                                folds=args.folds,
                                rng=rng,
                            )
                        elif mode == "in_sample":
                            score = fit_predict_in_sample(base_model, x, y, max_train_rows=args.max_train_rows, rng=rng)
                        else:
                            raise ValueError(mode)
                        add_result_rows(
                            summary_rows,
                            curve_rows,
                            dataset=str(dataset),
                            model_name=model_name,
                            target=target,
                            mode=mode,
                            feature_set=feature_set,
                            feature_count=len(selected_features),
                            group=group,
                            score=score,
                        )

        pd.DataFrame(summary_rows).to_csv(out_dir / "ml_selector_sweep_summary.csv", index=False)
        pd.DataFrame(curve_rows).to_csv(out_dir / "ml_selector_sweep_curves.csv", index=False)
        pd.DataFrame(feature_rows).to_csv(out_dir / "ml_selector_sweep_feature_sets.csv", index=False)
        pd.DataFrame(single_rows).to_csv(out_dir / "single_feature_rankings.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    write_report(summary, feature_rows, out_dir / "README.md")
    print(out_dir / "README.md", flush=True)


if __name__ == "__main__":
    main()
