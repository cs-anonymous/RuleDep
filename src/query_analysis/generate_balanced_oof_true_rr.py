#!/usr/bin/env python3
"""5-fold random OOF evaluation for the 12 paper-relevant selectors.

For each selector, generate out-of-fold RF predictions via random K-fold
(K=5), then score coverage gain on the OOF predictions. This is the
honest "what gain would a fresh test query receive?" estimate, in
contrast to the in-sample numbers in `balanced_ablation_true_rr/`.

Selectors covered (12):

global RF (8):
  - balanced4_global_rf                : main paper method
  - balanced5_global_rf                : 5-feature comparison
  - balanced3_max_topk_eff_global_rf   : best balanced3 by in-sample gain@10
  - balanced2_syn_topk_global_rf       : best balanced2 by in-sample gain@10
  - balanced1_syn_global_rf            : single feature: synergy
  - balanced1_max_global_rf            : single feature: max_candidate_dep_score
  - balanced1_topk_global_rf           : single feature: topk_rule_weight
  - balanced1_eff_global_rf            : single feature: effective_candidates

per-dataset RF (4):
  - balanced4_rf                       : 4-feature ceiling
  - balanced3_syn_topk_eff_rf          : best per-dataset balanced3
  - balanced2_syn_topk_rf              : best per-dataset balanced2
  - balanced1_eff_rf                   : best per-dataset balanced1

Per-dataset OOF: 5-fold random split *within* each dataset.
Global OOF: 5-fold random split across the pooled 596k rows.

Outputs:
  reports/official_query_subset/ml_selector_diverse/balanced_oof_true_rr/
    balanced_oof_macro_gain.csv
    balanced_oof_per_dataset_gain.csv
    balanced_oof_vs_insample.csv     # head-to-head against in-sample numbers
    README.md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_balanced5_selector_report as g  # noqa: E402
from generate_balanced5_selector_report_true_rr import load_merged_rr  # noqa: E402
from sweep_official_query_ml_selectors import stable_seed, target_values  # noqa: E402


ROOT = Path("/home/sy/RuleDep")
REPORT_DIR = ROOT / "reports" / "official_query_subset"
OUT_DIR = REPORT_DIR / "ml_selector_diverse" / "balanced_oof_true_rr"

N_SPLITS = 5

BALANCED4 = [
    "synergy_weight_top5_mean",
    "max_candidate_dep_score",
    "topk_rule_weight",
    "effective_candidates",
]

GLOBAL_SELECTORS: list[tuple[str, list[str]]] = [
    ("balanced4_global_rf", BALANCED4),
    ("balanced5_global_rf", g.BALANCED5),
    (
        "balanced3_max_topk_eff_global_rf",
        ["max_candidate_dep_score", "topk_rule_weight", "effective_candidates"],
    ),
    ("balanced2_syn_topk_global_rf", ["synergy_weight_top5_mean", "topk_rule_weight"]),
    ("balanced1_syn_global_rf", ["synergy_weight_top5_mean"]),
    ("balanced1_max_global_rf", ["max_candidate_dep_score"]),
    ("balanced1_topk_global_rf", ["topk_rule_weight"]),
    ("balanced1_eff_global_rf", ["effective_candidates"]),
]

PER_DATASET_SELECTORS: list[tuple[str, list[str]]] = [
    ("balanced4_rf", BALANCED4),
    (
        "balanced3_syn_topk_eff_rf",
        ["synergy_weight_top5_mean", "topk_rule_weight", "effective_candidates"],
    ),
    ("balanced2_syn_topk_rf", ["synergy_weight_top5_mean", "topk_rule_weight"]),
    ("balanced1_eff_rf", ["effective_candidates"]),
]


def kfold_seed(label: str) -> int:
    return stable_seed(f"5foldOOF:{label}") % (2**31)


def oof_global(df: pd.DataFrame, features: list[str], selector: str) -> np.ndarray:
    n = len(df)
    score = np.full(n, np.nan, dtype=float)
    x = df[features].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
    y = target_values(df, "gain_clip")
    seed = kfold_seed(selector)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    t0 = time.time()
    for fold_ix, (train_idx, test_idx) in enumerate(kf.split(x)):
        t1 = time.time()
        model = g.make_rf()
        model.fit(x.iloc[train_idx], y[train_idx])
        score[test_idx] = model.predict(x.iloc[test_idx])
        print(
            f"    [{selector}] fold {fold_ix + 1}/{N_SPLITS} done in {time.time() - t1:.1f}s "
            f"(cumulative {time.time() - t0:.1f}s)",
            flush=True,
        )
    if np.any(np.isnan(score)):
        raise RuntimeError(f"{selector}: OOF score contains NaN")
    return score


def oof_per_dataset(df: pd.DataFrame, features: list[str], selector: str) -> pd.Series:
    out_score = pd.Series(np.full(len(df), np.nan, dtype=float), index=df.index)
    t0 = time.time()
    for dataset, raw_group in df.groupby("dataset", sort=True):
        group = raw_group.reset_index()
        x = group[features].replace([np.inf, -np.inf], np.nan)
        y = target_values(group, "gain_clip")
        seed = kfold_seed(f"{dataset}:{selector}")
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        scores_local = np.full(len(group), np.nan, dtype=float)
        t1 = time.time()
        for train_idx, test_idx in kf.split(x):
            model = g.make_rf()
            model.fit(x.iloc[train_idx], y[train_idx])
            scores_local[test_idx] = model.predict(x.iloc[test_idx])
        if np.any(np.isnan(scores_local)):
            raise RuntimeError(f"{selector}/{dataset}: OOF NaN")
        out_score.iloc[group["index"].to_numpy()] = scores_local
        print(
            f"    [{selector}] dataset={dataset} ({len(group)} rows) {N_SPLITS}-fold OOF in "
            f"{time.time() - t1:.1f}s (cumulative {time.time() - t0:.1f}s)",
            flush=True,
        )
    return out_score


def gain_curves(
    df: pd.DataFrame, score: np.ndarray | pd.Series, selector: str
) -> pd.DataFrame:
    rows = []
    s_array = np.asarray(score, dtype=float)
    work = df.reset_index(drop=True).copy()
    work["_oof_score"] = s_array
    for dataset, group in work.groupby("dataset", sort=True):
        gp = group.reset_index(drop=True)
        s = gp["_oof_score"].to_numpy(dtype=float)
        curves = g.score_coverages(gp, s, g.COVERAGES)
        curves.insert(0, "selector", selector)
        curves.insert(0, "dataset", dataset)
        rows.append(curves)
    return pd.concat(rows, ignore_index=True)


def macro_table(curves: pd.DataFrame, ordered_selectors: list[str]) -> pd.DataFrame:
    fixed = curves[curves["coverage"].isin(g.FIXED)].copy()
    macro = (
        fixed.groupby(["selector", "coverage"], as_index=False)["gain_pt"]
        .mean()
        .pivot(index="selector", columns="coverage", values="gain_pt")
        .reset_index()
    )
    macro.columns = ["selector"] + [f"gain_{int(c * 100)}" for c in g.FIXED]
    order = {name: i for i, name in enumerate(ordered_selectors)}
    macro["order"] = macro["selector"].map(lambda s: order.get(s, len(order) + 1))
    return macro.sort_values("order").drop(columns=["order"]).reset_index(drop=True)


def load_insample_baseline(selectors: list[str]) -> pd.DataFrame:
    sources = {
        "balanced4_global_rf": "balanced_ablation_true_rr/balanced_subset_macro_gain_annotated.csv",
        "balanced5_global_rf": "balanced_ablation_true_rr/balanced_subset_macro_gain_annotated.csv",
        "balanced3_max_topk_eff_global_rf": "balanced_ablation_true_rr/balanced_subset_macro_gain_annotated.csv",
        "balanced2_syn_topk_global_rf": "balanced_ablation_true_rr/balanced_subset_macro_gain_annotated.csv",
        "balanced1_syn_global_rf": "balanced_ablation_true_rr/balanced_subset_macro_gain_annotated.csv",
        "balanced1_max_global_rf": "balanced_ablation_true_rr/balanced_subset_macro_gain_annotated.csv",
        "balanced1_topk_global_rf": "balanced_ablation_true_rr/balanced_subset_macro_gain_annotated.csv",
        "balanced1_eff_global_rf": "balanced_ablation_true_rr/balanced_subset_macro_gain_annotated.csv",
        "balanced4_rf": "balanced_ablation_true_rr/balanced_subset_macro_gain_annotated.csv",
        "balanced3_syn_topk_eff_rf": "balanced_ablation_true_rr/balanced_subset_macro_gain_annotated.csv",
        "balanced2_syn_topk_rf": "balanced_ablation_true_rr/balanced_subset_macro_gain_annotated.csv",
        "balanced1_eff_rf": "balanced_ablation_true_rr/balanced_subset_macro_gain_annotated.csv",
    }
    rows = []
    cache: dict[str, pd.DataFrame] = {}
    for selector in selectors:
        rel = sources.get(selector)
        if rel is None:
            continue
        path = REPORT_DIR / "ml_selector_diverse" / rel
        if rel not in cache:
            cache[rel] = pd.read_csv(path)
        df = cache[rel]
        match = df[df["selector"] == selector]
        if not match.empty:
            row = match.iloc[0]
            rows.append(
                {
                    "selector": selector,
                    "insample_gain_10": float(row["gain_10"]),
                    "insample_gain_20": float(row["gain_20"]),
                    "insample_gain_50": float(row["gain_50"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[oof] loading data ...", flush=True)
    df, _ = load_merged_rr()
    print(f"[oof] {len(df)} rows loaded", flush=True)

    all_selectors_order = [name for name, _ in GLOBAL_SELECTORS] + [
        name for name, _ in PER_DATASET_SELECTORS
    ]

    all_curves = []

    print("[oof] running global RF selectors ({} total)".format(len(GLOBAL_SELECTORS)), flush=True)
    for selector, features in GLOBAL_SELECTORS:
        print(f"  -> {selector} ({len(features)} features)", flush=True)
        score = oof_global(df, features, selector)
        all_curves.append(gain_curves(df, score, selector))

    print("[oof] running per-dataset RF selectors ({} total)".format(len(PER_DATASET_SELECTORS)), flush=True)
    for selector, features in PER_DATASET_SELECTORS:
        print(f"  -> {selector} ({len(features)} features)", flush=True)
        score = oof_per_dataset(df, features, selector)
        all_curves.append(gain_curves(df, score.values, selector))

    curves = pd.concat(all_curves, ignore_index=True)
    macro = macro_table(curves, all_selectors_order)
    per_dataset = curves.pivot_table(
        index=["selector", "dataset"], columns="coverage", values="gain_pt"
    )
    per_dataset.columns = [f"gain_{int(c * 100)}" for c in per_dataset.columns]
    per_dataset = per_dataset.reset_index()

    insample = load_insample_baseline([s for s, _ in GLOBAL_SELECTORS] + [s for s, _ in PER_DATASET_SELECTORS])
    head_to_head = macro.merge(insample, on="selector", how="left")
    for col in ["gain_10", "gain_20", "gain_50"]:
        head_to_head[f"oof_minus_insample_{col}"] = head_to_head[col] - head_to_head[f"insample_{col}"]

    curves.to_csv(OUT_DIR / "balanced_oof_gain_curves.csv", index=False)
    macro.to_csv(OUT_DIR / "balanced_oof_macro_gain.csv", index=False)
    per_dataset.to_csv(OUT_DIR / "balanced_oof_per_dataset_gain.csv", index=False)
    head_to_head.to_csv(OUT_DIR / "balanced_oof_vs_insample.csv", index=False)

    write_readme(macro, per_dataset, head_to_head)
    print(OUT_DIR / "README.md", flush=True)


def write_readme(macro: pd.DataFrame, per_dataset: pd.DataFrame, head_to_head: pd.DataFrame) -> None:
    def fmt_pct(v):
        if pd.isna(v):
            return ""
        return f"{v * 100:.2f}%"

    def fmt_pt(v):
        if pd.isna(v):
            return ""
        return f"{v * 100:+.2f} pt"

    macro_disp = macro.copy()
    for col in ["gain_10", "gain_20", "gain_30", "gain_50", "gain_100"]:
        macro_disp[col] = macro_disp[col].map(fmt_pct)
    macro_md = macro_disp.to_markdown(index=False)

    h2h = head_to_head.copy()
    for col in ["gain_10", "gain_20", "gain_50"]:
        h2h[col] = h2h[col].map(fmt_pct)
        h2h[f"insample_{col}"] = h2h[f"insample_{col}"].map(fmt_pct)
        h2h[f"oof_minus_insample_{col}"] = h2h[f"oof_minus_insample_{col}"].map(fmt_pt)
    h2h_md = h2h[
        [
            "selector",
            "insample_gain_10",
            "gain_10",
            "oof_minus_insample_gain_10",
            "insample_gain_20",
            "gain_20",
            "oof_minus_insample_gain_20",
            "insample_gain_50",
            "gain_50",
            "oof_minus_insample_gain_50",
        ]
    ].rename(
        columns={
            "gain_10": "oof_gain_10",
            "gain_20": "oof_gain_20",
            "gain_50": "oof_gain_50",
        }
    ).to_markdown(index=False)

    per_disp = per_dataset.copy()
    for col in [c for c in per_disp.columns if c.startswith("gain_")]:
        per_disp[col] = per_disp[col].map(fmt_pct)
    per_md = per_disp.to_markdown(index=False)

    body = f"""# 5-fold Random OOF Evaluation (true per-query RR)

For each of the 12 paper-relevant selectors below, RF predictions are
generated via `KFold(n_splits=5, shuffle=True)` and the coverage gain
metric is computed on those out-of-fold scores. This is the honest
fresh-query estimate; the in-sample numbers in
`balanced_ablation_true_rr/` train and predict on the same rows.

- **Per-dataset selectors** (`*_rf`): K-fold split *within* each dataset.
- **Global selectors** (`*_global_rf`): K-fold split across the pooled
  596,060 rows; predictions are then grouped by dataset for coverage
  computation.

The macro `gain_X` is the mean across 7 datasets at coverage X%.

## Macro OOF gain

{macro_md}

## OOF vs in-sample (head-to-head)

`oof_minus_insample_*` is the OOF gain minus the in-sample gain reported
in `balanced_ablation_true_rr/`. Negative numbers indicate the in-sample
estimate was optimistic; magnitudes near zero indicate the model
generalises cleanly under random K-fold.

{h2h_md}

## Per-dataset OOF gain

{per_md}
"""
    (OUT_DIR / "README.md").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()
