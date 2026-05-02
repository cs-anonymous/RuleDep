#!/usr/bin/env python3
"""Greedy + exhaustive feature-subset ablation on top of Balanced4.

For each subset, fits per-dataset RF and a single global RF on the
true-per-query-RR data. Outputs:
- balanced_subset_macro_gain.csv         : selector x gain@coverage
- balanced_subset_global_importance.csv  : global RF importance per subset
- balanced_subset_per_dataset_importance.csv
- balanced_subset_winners.csv            : best 4 / 3 / 2 / 1 by gain@10%
- README.md

Subsets explored:
- balanced4: all 4 features                (1 subset)
- balanced3*: every C(4, 3)                (4 subsets)
- balanced2*: every C(4, 2)                (6 subsets)
- balanced1: each single feature           (4 subsets, identical to single-feat baseline)

Total subsets: 15. Reusing a stable seed scheme keyed on subset tag so
re-running the script reproduces the same numbers.
"""
from __future__ import annotations

import itertools
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
OUT_DIR = REPORT_DIR / "ml_selector_diverse" / "balanced_ablation_true_rr"
FIG_DIR = OUT_DIR / "figures"

BALANCED4 = [
    "synergy_weight_top5_mean",
    "max_candidate_dep_score",
    "topk_rule_weight",
    "effective_candidates",
]

SHORT = {
    "synergy_weight_top5_mean": "syn",
    "max_candidate_dep_score": "max",
    "topk_rule_weight": "topk",
    "effective_candidates": "eff",
}


def subset_tag(features: list[str]) -> str:
    return "_".join(SHORT[f] for f in features)


def all_subsets() -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for size in (4, 3, 2, 1):
        for combo in itertools.combinations(BALANCED4, size):
            features = list(combo)
            tag = f"balanced{size}_{subset_tag(features)}"
            out.append((tag, features))
    return out


def fit_per_dataset(
    df: pd.DataFrame, features: list[str], tag: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    importance_rows = []
    for dataset, raw_group in df.groupby("dataset", sort=True):
        group = raw_group.reset_index(drop=True)
        x = group[features].replace([np.inf, -np.inf], np.nan)
        y = target_values(group, "gain_clip")
        train_idx = g.sampled_train_idx(
            len(group),
            f"{dataset}:{tag}_rf:gain_clip:RandomForestRegressor",
            80_000,
        )
        model = g.make_rf()
        model.fit(x.iloc[train_idx], y[train_idx])
        score = np.asarray(model.predict(x), dtype=float)
        rf = model.named_steps["randomforestregressor"]
        for feature, imp in zip(features, rf.feature_importances_):
            importance_rows.append(
                {
                    "tag": tag,
                    "selector": f"{tag}_rf",
                    "dataset": str(dataset),
                    "feature": feature,
                    "importance": float(imp),
                }
            )
        curves = g.score_coverages(group, score, g.COVERAGES)
        curves.insert(0, "selector", f"{tag}_rf")
        curves.insert(0, "dataset", dataset)
        rows.append(curves)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(importance_rows)


def fit_global(
    df: pd.DataFrame, features: list[str], tag: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selector = f"{tag}_global_rf"
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
        [
            {
                "tag": tag,
                "selector": selector,
                "dataset": "global",
                "feature": feature,
                "importance": float(imp),
            }
            for feature, imp in zip(features, rf.feature_importances_)
        ]
    )

    rows = []
    scored = df[["dataset", "official_scaled_rr_stage1", "official_scaled_rr_stage2", *features]].copy()
    scored["_score"] = score
    for dataset, group in scored.groupby("dataset", sort=True):
        gp = group.reset_index(drop=True)
        curves = g.score_coverages(gp, gp["_score"].to_numpy(dtype=float), g.COVERAGES)
        curves.insert(0, "selector", selector)
        curves.insert(0, "dataset", dataset)
        rows.append(curves)
    return pd.concat(rows, ignore_index=True), importance


def macro_table(curves: pd.DataFrame) -> pd.DataFrame:
    fixed = curves[curves["coverage"].isin(g.FIXED)].copy()
    macro = (
        fixed.groupby(["selector", "coverage"], as_index=False)["gain_pt"]
        .mean()
        .pivot(index="selector", columns="coverage", values="gain_pt")
        .reset_index()
    )
    macro.columns = ["selector"] + [f"gain_{int(c * 100)}" for c in g.FIXED]
    return macro


def annotate_macro(macro: pd.DataFrame) -> pd.DataFrame:
    out = macro.copy()
    out["scope"] = out["selector"].map(
        lambda s: "global" if s.endswith("_global_rf") else "per_dataset"
    )
    out["tag"] = out["selector"].str.replace("_rf$", "", regex=True).str.replace(
        "_global$", "", regex=True
    )
    out["size"] = out["tag"].str.extract(r"balanced(\d)").astype(int)
    return out.sort_values(["scope", "size", "gain_10"], ascending=[True, False, False]).reset_index(
        drop=True
    )


def winners(macro: pd.DataFrame) -> pd.DataFrame:
    annotated = annotate_macro(macro)
    rows = []
    for scope in ["global", "per_dataset"]:
        for size in sorted(annotated["size"].unique(), reverse=True):
            sub = annotated[(annotated["scope"] == scope) & (annotated["size"] == size)]
            if sub.empty:
                continue
            best = sub.iloc[0]
            rows.append(
                {
                    "scope": scope,
                    "size": int(size),
                    "selector": best["selector"],
                    "tag": best["tag"],
                    "gain_10": float(best["gain_10"]),
                    "gain_20": float(best["gain_20"]),
                    "gain_50": float(best["gain_50"]),
                }
            )
    return pd.DataFrame(rows)


def write_readme(macro: pd.DataFrame, winners_df: pd.DataFrame, importance: pd.DataFrame) -> None:
    annotated = annotate_macro(macro)

    def fmt_pct(v):
        return f"{v * 100:.2f}%" if pd.notna(v) else ""

    annotated["gain_10"] = annotated["gain_10"].map(fmt_pct)
    annotated["gain_20"] = annotated["gain_20"].map(fmt_pct)
    annotated["gain_50"] = annotated["gain_50"].map(fmt_pct)
    annotated["gain_30"] = annotated["gain_30"].map(fmt_pct)
    annotated["gain_100"] = annotated["gain_100"].map(fmt_pct)

    full_md = annotated[["scope", "size", "selector", "gain_10", "gain_20", "gain_30", "gain_50"]].to_markdown(
        index=False
    )

    winners_disp = winners_df.copy()
    for col in ["gain_10", "gain_20", "gain_50"]:
        winners_disp[col] = winners_disp[col].map(lambda v: f"{v * 100:.2f}%")
    winners_md = winners_disp.to_markdown(index=False)

    global_imp = importance[importance["dataset"] == "global"].copy()
    global_imp["importance"] = global_imp["importance"].map(lambda v: f"{v:.4f}")
    global_imp_md = (
        global_imp[["tag", "feature", "importance"]]
        .pivot(index="tag", columns="feature", values="importance")
        .reset_index()
        .to_markdown(index=False)
    )

    body = f"""# Balanced Subset Ablation (true per-query RR)

Greedy and exhaustive feature-subset ablation built on top of Balanced4
(`synergy_weight_top5_mean`, `max_candidate_dep_score`, `topk_rule_weight`,
`effective_candidates`). For each non-empty subset of these 4 features we
fit:

- A per-dataset RandomForest (`<tag>_rf`).
- A single global RandomForest pooled across all 7 datasets (`<tag>_global_rf`).

The training distribution and seeds match the rest of the report family
(`generate_balanced5_selector_report_true_rr.py`).

## Best subset at each size

Rows below show the top-`gain_10` subset at each cardinality, separately
for the global selector (the recommended paper method) and the per-dataset
oracle ceiling.

{winners_md}

## Full macro gain across all 15 subsets x 2 scopes

{full_md}

## Global RF feature importance per subset

{global_imp_md}

## Notes

- `balanced1_syn_global_rf` should equal the single-feature `synergy_weight_top5_mean_global_rf`
  number reported in `balanced5_report_true_rr` modulo seed-key noise.
- For the paper, look at the `global` rows of the winners table: those tell
  you whether shrinking from 4 to 3 features (or 2, or 1) costs more or
  less gain than you can defend, and which subset is best at each size.
"""
    (OUT_DIR / "README.md").write_text(body, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df, _ = load_merged_rr()

    subsets = all_subsets()
    print(f"[ablation] running {len(subsets)} subsets x (per-dataset + global) RF", flush=True)

    all_curves = []
    all_per_dataset_imp = []
    all_global_imp = []

    for tag, features in subsets:
        print(f"  - {tag}: {features}", flush=True)
        per_curves, per_imp = fit_per_dataset(df, features, tag)
        glob_curves, glob_imp = fit_global(df, features, tag)
        all_curves.append(per_curves)
        all_curves.append(glob_curves)
        all_per_dataset_imp.append(per_imp)
        all_global_imp.append(glob_imp)

    curves = pd.concat(all_curves, ignore_index=True)
    per_dataset_imp = pd.concat(all_per_dataset_imp, ignore_index=True)
    global_imp = pd.concat(all_global_imp, ignore_index=True)
    importance = pd.concat([per_dataset_imp, global_imp], ignore_index=True)

    macro = macro_table(curves)
    winners_df = winners(macro)

    curves.to_csv(OUT_DIR / "balanced_subset_gain_curves.csv", index=False)
    macro.to_csv(OUT_DIR / "balanced_subset_macro_gain.csv", index=False)
    annotate_macro(macro).to_csv(OUT_DIR / "balanced_subset_macro_gain_annotated.csv", index=False)
    importance.to_csv(OUT_DIR / "balanced_subset_per_dataset_importance.csv", index=False)
    global_imp.to_csv(OUT_DIR / "balanced_subset_global_importance.csv", index=False)
    winners_df.to_csv(OUT_DIR / "balanced_subset_winners.csv", index=False)

    for selector in winners_df["selector"]:
        try:
            g.plot_selector(curves, selector, FIG_DIR / f"{selector}.png")
        except (KeyError, IndexError, ValueError):
            pass

    write_readme(macro, winners_df, importance)
    print(OUT_DIR / "README.md", flush=True)


if __name__ == "__main__":
    main()
