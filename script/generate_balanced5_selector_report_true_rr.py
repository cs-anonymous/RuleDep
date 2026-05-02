#!/usr/bin/env python3
"""Balanced5 selector report using TRUE per-query filtered RR (with scaled fallback).

Differences from `generate_balanced5_selector_report.py`:
- Joins per-query true official filtered RR from
  `reports/official_query_subset/true_official_per_query_rr/true_official_per_query_rr_wide.csv`
  on (dataset, experiment, relation, direction, query, target_gt_entity).
- For rows whose true_official_rr_stage{1,2} is missing (notably hetionet
  relation 1 = AeG stage2, where the rerun OOM'd), falls back to the legacy
  per-relation `official_scaled_rr_stage{1,2}` value.
- Records the per-row provenance in `rr_source_stage{1,2}` columns and writes
  a `rr_source_summary.csv` so reviewers can see the share of fallback rows.
- In addition to the per-dataset / global RandomForest selectors, also fits
  per-dataset and global Ridge regressors (`balanced5_ridge`,
  `balanced5_global_ridge`) on standardized features. Standardized coefficients
  are exported as `balanced5_ridge_coefficients.csv` and rendered in the README
  so the linear selector can be reported as an interpretable formula alongside
  the RF ceiling.
- Output goes to `reports/official_query_subset/ml_selector_diverse/balanced5_report_true_rr/`.

The downstream coverage / range / importance / plotting / README pipeline is
reused from `generate_balanced5_selector_report` by overwriting the
`official_scaled_rr_stage{1,2}` columns in-place with the (true || scaled)
values before calling the helpers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_balanced5_selector_report as g  # noqa: E402
import regenerate_official_query_subset_reports as base  # noqa: E402
from sweep_official_query_ml_selectors import RANDOM_STATE, stable_seed, target_values  # noqa: E402


ROOT = Path("/home/sy/RuleDep")
REPORT_DIR = ROOT / "reports" / "official_query_subset"
TRUE_RR_WIDE = REPORT_DIR / "true_official_per_query_rr" / "true_official_per_query_rr_wide.csv"
OUT_DIR = REPORT_DIR / "ml_selector_diverse" / "balanced5_report_true_rr"
FIG_DIR = OUT_DIR / "figures"

JOIN_KEYS = ["dataset", "experiment", "relation", "direction", "query", "target_gt_entity"]


def make_ridge() -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=1.0, random_state=RANDOM_STATE),
    )


def _extract_ridge_coefs(pipeline) -> tuple[np.ndarray, float]:
    ridge = pipeline.named_steps["ridge"]
    return np.asarray(ridge.coef_, dtype=float), float(ridge.intercept_)


def train_ridge_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    coef_rows = []
    for dataset, raw_group in df.groupby("dataset", sort=True):
        group = raw_group.reset_index(drop=True)
        x = group[g.BALANCED5].replace([np.inf, -np.inf], np.nan)
        y = target_values(group, "gain_clip")
        train_idx = g.sampled_train_idx(
            len(group),
            f"{dataset}:balanced5_ridge:gain_clip:Ridge",
            80_000,
        )
        model = make_ridge()
        model.fit(x.iloc[train_idx], y[train_idx])
        score = np.asarray(model.predict(x), dtype=float)
        coefs, intercept = _extract_ridge_coefs(model)
        for feature, coef in zip(g.BALANCED5, coefs):
            coef_rows.append(
                {
                    "selector": "balanced5_ridge",
                    "dataset": str(dataset),
                    "feature": feature,
                    "std_coef": float(coef),
                    "intercept": intercept,
                }
            )
        curves = g.score_coverages(group, score, g.COVERAGES)
        curves.insert(0, "selector", "balanced5_ridge")
        curves.insert(0, "dataset", dataset)
        rows.append(curves)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(coef_rows)


def train_global_ridge_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    coef_rows = []
    x = df[g.BALANCED5].replace([np.inf, -np.inf], np.nan)
    y = target_values(df, "gain_clip")
    train_idx = g.sampled_train_idx(
        len(df),
        "global:balanced5_global_ridge:gain_clip:Ridge",
        None,
    )
    model = make_ridge()
    model.fit(x.iloc[train_idx], y[train_idx])
    score = np.asarray(model.predict(x), dtype=float)
    coefs, intercept = _extract_ridge_coefs(model)
    for feature, coef in zip(g.BALANCED5, coefs):
        coef_rows.append(
            {
                "selector": "balanced5_global_ridge",
                "dataset": "global",
                "feature": feature,
                "std_coef": float(coef),
                "intercept": intercept,
            }
        )

    scored = df[["dataset", "official_scaled_rr_stage1", "official_scaled_rr_stage2", *g.BALANCED5]].copy()
    scored["_score"] = score
    for dataset, group in scored.groupby("dataset", sort=True):
        gp = group.reset_index(drop=True)
        curves = g.score_coverages(gp, gp["_score"].to_numpy(dtype=float), g.COVERAGES)
        curves.insert(0, "selector", "balanced5_global_ridge")
        curves.insert(0, "dataset", dataset)
        rows.append(curves)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(coef_rows)


def load_merged_rr() -> tuple[pd.DataFrame, pd.DataFrame]:
    feat = pd.read_csv(REPORT_DIR / "official_query_triple_features.csv")
    feat = base.add_official_scaled_rr(feat)

    wide = pd.read_csv(TRUE_RR_WIDE)
    keep = JOIN_KEYS + ["true_official_rr_stage1", "true_official_rr_stage2"]
    wide = wide[keep]

    feat_dup = int(feat.duplicated(JOIN_KEYS).sum())
    wide_dup = int(wide.duplicated(JOIN_KEYS).sum())
    if feat_dup or wide_dup:
        raise RuntimeError(
            f"unexpected duplicates on join keys: feat={feat_dup}, wide={wide_dup}"
        )

    merged = feat.merge(wide, on=JOIN_KEYS, how="left", validate="one_to_one")

    merged["rr_source_stage1"] = np.where(
        merged["true_official_rr_stage1"].notna(), "true", "scaled_fallback"
    )
    merged["rr_source_stage2"] = np.where(
        merged["true_official_rr_stage2"].notna(), "true", "scaled_fallback"
    )

    merged["scaled_rr_stage1_legacy"] = merged["official_scaled_rr_stage1"]
    merged["scaled_rr_stage2_legacy"] = merged["official_scaled_rr_stage2"]
    merged["official_scaled_rr_stage1"] = merged["true_official_rr_stage1"].fillna(
        merged["official_scaled_rr_stage1"]
    )
    merged["official_scaled_rr_stage2"] = merged["true_official_rr_stage2"].fillna(
        merged["official_scaled_rr_stage2"]
    )

    summary = (
        merged.groupby(["dataset", "relation", "rr_source_stage1", "rr_source_stage2"])
        .size()
        .reset_index(name="n_rows")
        .sort_values(["dataset", "relation", "rr_source_stage1", "rr_source_stage2"])
        .reset_index(drop=True)
    )
    return merged, summary


def write_readme_with_caveat(
    definitions: pd.DataFrame,
    macro: pd.DataFrame,
    ranges: pd.DataFrame,
    importances: pd.DataFrame,
    curves: pd.DataFrame,
    rr_source_summary: pd.DataFrame,
    ridge_coefficients: pd.DataFrame,
) -> None:
    g.OUT_DIR = OUT_DIR
    g.FIG_DIR = FIG_DIR
    g.write_readme(definitions, macro, ranges, importances, curves)

    readme_path = OUT_DIR / "README.md"
    body = readme_path.read_text(encoding="utf-8")

    fallback = rr_source_summary[
        (rr_source_summary["rr_source_stage1"] == "scaled_fallback")
        | (rr_source_summary["rr_source_stage2"] == "scaled_fallback")
    ].copy()
    if not fallback.empty:
        fallback_per_dataset = (
            fallback.groupby(["dataset", "rr_source_stage1", "rr_source_stage2"])
            .agg(n_rows=("n_rows", "sum"))
            .reset_index()
        )
        fallback_md = fallback_per_dataset.to_markdown(index=False)
    else:
        fallback_md = "_No fallback rows; every case used true per-query filtered RR._"

    caveat = (
        "\n## RR Source (true per-query vs scaled fallback)\n\n"
        "This variant of the Balanced5 report uses the **true official filtered RR per query** "
        "(merged from `reports/official_query_subset/true_official_per_query_rr/true_official_per_query_rr_wide.csv`) "
        "as both the RF training target and the coverage-curve evaluation target. "
        "When a row has no matching true per-query RR (test triple absent from the rerun export), "
        "the value falls back to the legacy relation-level `official_scaled_rr` "
        "(`raw_rr * official_relation_MRR / mean(raw_rr_within_relation)`). "
        "Per-row provenance is in `rr_source_stage1` and `rr_source_stage2` of the joined data and aggregated below.\n\n"
        "Known fallback sources:\n"
        "- **hetionet `AeG` (relation 1) stage2**: stage2 dependency-stage rerun OOM'd in evaluation "
        "(`active_matrix = zeros((eval_batch, 1.75M rules))` -> 17.13 GiB single allocation). "
        "Stage1 RR is the true value; stage2 RR is the scaled fallback for these 106,954 rows.\n"
        "- **WN18RR `_hypernym` tail (~4,854 rows)**: the legacy "
        "`official_query_triple_features.csv` was generated against a slightly different test sample "
        "set than the rerun export, so a subset of `_hypernym` tail queries has no matching true RR.\n\n"
        "### Fallback rows by dataset and stage\n\n"
        f"{fallback_md}\n\n"
        "Full breakdown (one row per `dataset x relation x stage1_source x stage2_source`) is in `rr_source_summary.csv`.\n"
    )

    coefs_global = ridge_coefficients[ridge_coefficients["selector"] == "balanced5_global_ridge"].copy()
    coefs_per = ridge_coefficients[ridge_coefficients["selector"] == "balanced5_ridge"].copy()

    coefs_global_md = (
        coefs_global[["feature", "std_coef", "intercept"]]
        .assign(std_coef=lambda d: d["std_coef"].map(lambda v: f"{v:+.4f}"))
        .assign(intercept=lambda d: d["intercept"].map(lambda v: f"{v:+.4f}"))
        .to_markdown(index=False)
    )
    coefs_per_pivot = (
        coefs_per.pivot(index="dataset", columns="feature", values="std_coef")
        .reindex(columns=g.BALANCED5)
        .reset_index()
    )
    for col in g.BALANCED5:
        coefs_per_pivot[col] = coefs_per_pivot[col].map(lambda v: f"{v:+.4f}" if pd.notna(v) else "")
    coefs_per_md = coefs_per_pivot.to_markdown(index=False)

    ridge_section = (
        "\n## Ridge Linear Selector (interpretable formula)\n\n"
        "We additionally fit `Ridge(alpha=1.0)` on `SimpleImputer(median) -> StandardScaler -> Ridge` "
        "for two configurations:\n\n"
        "- `balanced5_ridge`: one Ridge model **per dataset** (5 standardized coefficients per dataset).\n"
        "- `balanced5_global_ridge`: one Ridge model trained on **all datasets pooled** (one set of 5 standardized coefficients), "
        "then evaluated within each dataset.\n\n"
        "Coefficients are on **standardized features** (mean 0, std 1 per training distribution), "
        "so the magnitude is directly comparable across features within a model. The sign indicates "
        "whether higher feature value pushes the predicted `gain_clip = clip(rr_stage2 / rr_stage1 - 1, [-1, 1])` up or down.\n\n"
        "### Global Ridge coefficients (one formula for all datasets)\n\n"
        f"{coefs_global_md}\n\n"
        "### Per-dataset Ridge coefficients\n\n"
        f"{coefs_per_md}\n\n"
        "Full coefficients (with intercepts) are in `balanced5_ridge_coefficients.csv`.\n"
        "Coverage curves and per-dataset gains for the Ridge selectors are in `balanced5_gain_curves.csv` "
        "and figures `figures/balanced5_ridge.png` / `figures/balanced5_global_ridge.png`.\n"
    )

    readme_path.write_text(body + caveat + ridge_section, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df, rr_source_summary = load_merged_rr()

    definitions = pd.DataFrame(g.FEATURE_DEFINITIONS)
    definitions.to_csv(OUT_DIR / "balanced5_feature_definitions.csv", index=False)

    balanced_curves, ranges, importances = g.train_balanced5_scores(df)
    single_curves = g.train_single_feature_scores(df)
    global_curves, global_ranges = g.train_global_scores(df)
    ridge_curves, ridge_coefs_per = train_ridge_scores(df)
    global_ridge_curves, ridge_coefs_global = train_global_ridge_scores(df)
    ridge_coefficients = pd.concat([ridge_coefs_per, ridge_coefs_global], ignore_index=True)
    curves = pd.concat(
        [balanced_curves, global_curves, ridge_curves, global_ridge_curves, single_curves],
        ignore_index=True,
    )
    macro = g.macro_table(curves)

    curves.to_csv(OUT_DIR / "balanced5_gain_curves.csv", index=False)
    stale_directions = OUT_DIR / "balanced5_single_feature_directions.csv"
    if stale_directions.exists():
        stale_directions.unlink()
    ranges.to_csv(OUT_DIR / "balanced5_selected_feature_ranges.csv", index=False)
    global_ranges.to_csv(OUT_DIR / "balanced5_global_selected_feature_ranges.csv", index=False)
    importances.to_csv(OUT_DIR / "balanced5_rf_feature_importance.csv", index=False)
    macro.to_csv(OUT_DIR / "balanced5_macro_gain_table.csv", index=False)
    rr_source_summary.to_csv(OUT_DIR / "rr_source_summary.csv", index=False)
    ridge_coefficients.to_csv(OUT_DIR / "balanced5_ridge_coefficients.csv", index=False)

    selectors = macro["selector"].astype(str).tolist()
    for selector in selectors:
        g.plot_selector(curves, selector, FIG_DIR / f"{selector}.png")

    write_readme_with_caveat(
        definitions,
        macro,
        pd.concat([ranges, global_ranges], ignore_index=True),
        importances,
        curves,
        rr_source_summary,
        ridge_coefficients,
    )
    print(OUT_DIR / "README.md", flush=True)


if __name__ == "__main__":
    main()
