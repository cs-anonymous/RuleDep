#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import regenerate_official_query_subset_reports as base
from sweep_official_query_ml_selectors import RANDOM_STATE, stable_seed, target_values


ROOT = Path("/home/sy/RuleDep")
REPORT_DIR = ROOT / "reports" / "official_query_subset"
OUT_DIR = REPORT_DIR / "ml_selector_diverse" / "balanced5_report"
FIG_DIR = OUT_DIR / "figures"

BALANCED5 = [
    "synergy_weight_top5_mean",
    "max_candidate_dep_score",
    "topk_rule_weight",
    "num_neg_dep",
    "effective_candidates",
]

FEATURE_LABELS = {
    "balanced5_rf": "Balanced5 RF score",
    "balanced5_global_rf": "Balanced5 global RF score",
    "synergy_weight_top5_mean": "Synergy strength",
    "synergy_weight_top5_mean_global_rf": "Synergy strength global RF score",
    "max_candidate_dep_score": "Max candidate dependency score",
    "max_candidate_dep_score_global_rf": "Max candidate dependency score global RF score",
    "topk_rule_weight": "Top-k rule weight",
    "topk_rule_weight_global_rf": "Top-k rule weight global RF score",
    "num_neg_dep": "Number of negative dependencies",
    "num_neg_dep_global_rf": "Number of negative dependencies global RF score",
    "effective_candidates": "Effective candidates",
    "effective_candidates_global_rf": "Effective candidates global RF score",
}

FEATURE_DEFINITIONS = [
    {
        "feature": "synergy_weight_top5_mean",
        "paper_name": "Synergy strength",
        "definition": "Mean of the top 5 absolute synergy dependency weights among unique displayed rule-pair dependencies in the query case.",
        "source": "src/query_analysis/analyze_official_query_subsets.py: synergy_values from synergy_filtered.txt weights; topk_stats(..., prefix='synergy_weight').",
    },
    {
        "feature": "max_candidate_dep_score",
        "paper_name": "Max candidate dependency score",
        "definition": "Maximum candidate-level dependencyScore over all candidate entities in the query case.",
        "source": "src/query_analysis/analyze_official_query_subsets.py: max(dep_scores), where dep_scores collect candidate['dependencyScore'].",
    },
    {
        "feature": "topk_rule_weight",
        "paper_name": "Top-k rule weight",
        "definition": "Mean of the top 3 rule weights fired by candidates in the query case.",
        "source": "src/query_analysis/analyze_official_query_subsets.py: safe_mean(sorted(rule_values, reverse=True)[:3]), where rule_values collect candidate['maxplus'].",
    },
    {
        "feature": "num_neg_dep",
        "paper_name": "Number of negative dependencies",
        "definition": "Number of unique displayed dependency rule-pairs whose learned dependency type is redundancy.",
        "source": "src/query_analysis/analyze_official_query_subsets.py: len(redundancy_values).",
    },
    {
        "feature": "effective_candidates",
        "paper_name": "Effective candidates",
        "definition": "exp(H(p)), where p is the softmax distribution over candidate stage1 official scores; larger values mean the stage1 ranker is less concentrated.",
        "source": "src/query_analysis/analyze_official_query_subsets.py: exp(-sum p log p) from stage1_scores.",
    },
]

COVERAGES = np.round(np.arange(0.02, 1.0001, 0.02), 2)
FIXED = [0.10, 0.20, 0.30, 0.50, 1.00]
ANNOTATE = [0.10, 0.20, 0.50]


def make_rf() -> object:
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


def fit_rf_score(
    x: pd.DataFrame,
    y: np.ndarray,
    *,
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
        rows.append(
            {
                "coverage": float(coverage),
                "n": n,
                "mrr_stage1": m1,
                "mrr_stage2": m2,
                "gain_pt": (m2 / m1 - 1.0) if m1 > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def sampled_train_idx(n_rows: int, seed_key: str, max_train_rows: int | None) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(seed_key))
    train_idx = np.arange(n_rows)
    if max_train_rows is not None and len(train_idx) > max_train_rows:
        train_idx = np.sort(rng.choice(train_idx, size=max_train_rows, replace=False))
    return train_idx


def selected_feature_ranges(dataset: str, group: pd.DataFrame, score: np.ndarray, selector: str) -> list[dict]:
    rows = []
    order = np.argsort(score, kind="mergesort")[::-1]
    for coverage in ANNOTATE:
        n = max(1, int(round(len(order) * coverage)))
        selected = group.iloc[order[:n]]
        for feature in BALANCED5:
            values = selected[feature].replace([np.inf, -np.inf], np.nan).astype(float)
            rows.append(
                {
                    "selector": selector,
                    "dataset": dataset,
                    "coverage": coverage,
                    "feature": feature,
                    "n": n,
                    "min": values.min(),
                    "q25": values.quantile(0.25),
                    "median": values.quantile(0.50),
                    "q75": values.quantile(0.75),
                    "max": values.max(),
                    "mean": values.mean(),
                }
            )
    return rows


def train_balanced5_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    range_rows = []
    importance_rows = []
    for dataset, raw_group in df.groupby("dataset", sort=True):
        group = raw_group.reset_index(drop=True)
        x = group[BALANCED5].replace([np.inf, -np.inf], np.nan)
        y = target_values(group, "gain_clip")
        model = make_rf()
        train_idx = sampled_train_idx(
            len(group),
            f"{dataset}:balanced5:gain_clip:RandomForestRegressor",
            80_000,
        )
        model.fit(x.iloc[train_idx], y[train_idx])
        score = np.asarray(model.predict(x), dtype=float)
        rf = model.named_steps["randomforestregressor"]
        for feature, importance in zip(BALANCED5, rf.feature_importances_):
            importance_rows.append({"dataset": dataset, "feature": feature, "importance": float(importance)})

        curves = score_coverages(group, score, COVERAGES)
        curves.insert(0, "selector", "balanced5_rf")
        curves.insert(0, "dataset", dataset)
        rows.append(curves)
        range_rows.extend(selected_feature_ranges(str(dataset), group, score, "balanced5_rf"))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(range_rows), pd.DataFrame(importance_rows)


def train_global_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    range_rows = []
    y = target_values(df, "gain_clip")
    configs = [("balanced5_global_rf", BALANCED5)]
    configs.extend((f"{feature}_global_rf", [feature]) for feature in BALANCED5)

    for selector, features in configs:
        x = df[features].replace([np.inf, -np.inf], np.nan)
        score = fit_rf_score(
            x,
            y,
            seed_key=f"global:{selector}:gain_clip:RandomForestRegressor",
            max_train_rows=None,
        )
        scored = df[["dataset", "official_scaled_rr_stage1", "official_scaled_rr_stage2", *BALANCED5]].copy()
        scored["_score"] = score
        if selector == "balanced5_global_rf":
            range_rows.extend(selected_feature_ranges("global", scored.reset_index(drop=True), score, selector))
        for dataset, group in scored.groupby("dataset", sort=True):
            g = group.reset_index(drop=True)
            curves = score_coverages(g, g["_score"].to_numpy(dtype=float), COVERAGES)
            curves.insert(0, "selector", selector)
            curves.insert(0, "dataset", dataset)
            rows.append(curves)
            if selector == "balanced5_global_rf":
                range_rows.extend(selected_feature_ranges(str(dataset), g, g["_score"].to_numpy(dtype=float), selector))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(range_rows)


def train_single_feature_scores(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in BALANCED5:
        for dataset, group in df.groupby("dataset", sort=True):
            g = group.reset_index(drop=True)
            x = g[[feature]].replace([np.inf, -np.inf], np.nan)
            y = target_values(g, "gain_clip")
            score = fit_rf_score(
                x,
                y,
                seed_key=f"{dataset}:{feature}:gain_clip:RandomForestRegressor",
                max_train_rows=80_000,
            )
            curves = score_coverages(g, score, COVERAGES)
            curves.insert(0, "selector", feature)
            curves.insert(0, "dataset", dataset)
            rows.append(curves)
    return pd.concat(rows, ignore_index=True)


def macro_table(curves: pd.DataFrame) -> pd.DataFrame:
    fixed = curves[curves["coverage"].isin(FIXED)].copy()
    macro = (
        fixed.groupby(["selector", "coverage"], as_index=False)["gain_pt"]
        .mean()
        .pivot(index="selector", columns="coverage", values="gain_pt")
        .reset_index()
    )
    macro.columns = ["selector"] + [f"gain_{int(c * 100)}" for c in FIXED]
    order = ["balanced5_rf", "balanced5_global_rf"]
    for feature in BALANCED5:
        order.extend([feature, f"{feature}_global_rf"])
    macro["order"] = macro["selector"].map({name: i for i, name in enumerate(order)})
    return macro.sort_values("order").drop(columns=["order"])


def plot_selector(curves: pd.DataFrame, selector: str, out_path: Path) -> None:
    subset = curves[(curves["selector"] == selector) & (curves["coverage"] >= 0.10)].copy()
    macro = subset.groupby("coverage", as_index=False)["gain_pt"].mean()

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=140)
    palette = plt.get_cmap("tab10")
    for i, (dataset, g) in enumerate(subset.groupby("dataset", sort=True)):
        g = g.sort_values("coverage")
        ax.plot(
            g["coverage"] * 100,
            g["gain_pt"] * 100,
            color=palette(i % 10),
            alpha=0.55,
            linewidth=1.1,
            label=dataset,
        )
    ax.plot(
        macro["coverage"] * 100,
        macro["gain_pt"] * 100,
        color="#111111",
        linewidth=2.5,
        marker="o",
        markersize=2.5,
        label="macro",
    )

    for cov in ANNOTATE:
        row = macro[np.isclose(macro["coverage"], cov)].iloc[0]
        x = cov * 100
        y = float(row["gain_pt"]) * 100
        ax.scatter([x], [y], color="#111111", zorder=5, s=26)
        ax.annotate(
            f"gain({int(cov * 100)}%) = {y:.2f}%",
            xy=(x, y),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=8,
        )

    title = FEATURE_LABELS.get(selector, selector)
    ax.set_title(title)
    ax.set_xlabel(f"Coverage ranked by {selector} score")
    ax.set_ylabel("gain_pt (%)")
    ax.axhline(0, color="#777777", linewidth=0.8, alpha=0.7)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_xlim(10, 100)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_readme(
    definitions: pd.DataFrame,
    macro: pd.DataFrame,
    ranges: pd.DataFrame,
    importances: pd.DataFrame,
    curves: pd.DataFrame,
) -> None:
    def fmt_table(df: pd.DataFrame) -> str:
        out = df.copy()
        for col in out.columns:
            if pd.api.types.is_float_dtype(out[col]):
                out[col] = out[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "")
        return out.to_markdown(index=False)

    def compact_range_table(range_df: pd.DataFrame) -> pd.DataFrame:
        range10 = range_df[range_df["coverage"] == 0.10].copy()
        range10["range"] = range10.apply(lambda r: f"[{r['q25']:.4g}, {r['q75']:.4g}] median={r['median']:.4g}", axis=1)
        compact = range10.pivot(index="dataset", columns="feature", values="range").reset_index()
        compact = compact[["dataset", *BALANCED5]]
        if "global" in set(compact["dataset"]):
            compact["_order"] = compact["dataset"].map(lambda value: 0 if value == "global" else 1)
            compact = compact.sort_values(["_order", "dataset"]).drop(columns=["_order"]).reset_index(drop=True)
        return compact

    compact_range = compact_range_table(ranges[ranges["selector"] == "balanced5_rf"])
    compact_global_range = compact_range_table(ranges[ranges["selector"] == "balanced5_global_rf"])

    importance_table = importances.pivot(index="dataset", columns="feature", values="importance").reset_index()
    importance_table = importance_table[["dataset", *BALANCED5]]

    fixed_curves = curves[curves["coverage"].isin(FIXED)].copy()
    per_dataset = fixed_curves[fixed_curves["selector"] == "balanced5_rf"].pivot(
        index="dataset", columns="coverage", values="gain_pt"
    )
    per_dataset.columns = [f"gain_{int(c * 100)}" for c in per_dataset.columns]
    per_dataset = per_dataset.reset_index()

    lines = [
        "# Balanced5 Selector Report",
        "",
        "Balanced5 shared attributes:",
        "",
        *[f"- `{feature}`: {FEATURE_LABELS[feature]}" for feature in BALANCED5],
        "",
        "## Attribute Definitions",
        "",
        fmt_table(definitions[["feature", "paper_name", "definition"]]),
        "",
        "## Dataset-specific and Global RF Selectors",
        "",
        "`balanced5_rf` and the plain feature-name rows train one RF per dataset. Rows ending in `_global_rf` train one pooled RF over all datasets, then evaluate coverage separately within each dataset.",
    ]
    lines.extend(
        [
        "",
        "## Macro Gain Table",
        "",
        fmt_table(macro),
        "",
        "## Balanced5 Per-dataset Gain",
        "",
        fmt_table(per_dataset),
        "",
        "## Balanced5 RF Feature Importance",
        "",
        "Each dataset trains its own RF model with the same five attributes. Values are impurity-based RF feature importances and sum to 1 within each dataset.",
        "",
        fmt_table(importance_table),
        "",
        "## Balanced5 Top-10% Attribute Ranges",
        "",
        "Ranges are empirical IQRs among query cases selected by the dataset-specific RF score at 10% coverage.",
        "",
        fmt_table(compact_range),
        "",
        "## Balanced5 Global-RF Top-10% Attribute Ranges",
        "",
        "Ranges are empirical IQRs among query cases selected by the pooled global RF score at 10% coverage. The `global` row pools the selected cases before splitting by dataset.",
        "",
        fmt_table(compact_global_range),
        "",
        "Full top-10/20/50 range statistics are in `balanced5_selected_feature_ranges.csv` and `balanced5_global_selected_feature_ranges.csv`.",
        "",
        "## Figures",
        "",
        "Plots start at 10% coverage and follow the same dataset-color style as `feature_plots/`; black is the macro average.",
        "",
        ]
    )
    for selector in macro["selector"].astype(str):
        filename = f"{selector}.png"
        lines.append(f"- `{selector}`: [figures/{filename}](figures/{filename})")
    lines.append("")
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(REPORT_DIR / "official_query_triple_features.csv")
    df = base.add_official_scaled_rr(df)

    definitions = pd.DataFrame(FEATURE_DEFINITIONS)
    definitions.to_csv(OUT_DIR / "balanced5_feature_definitions.csv", index=False)

    balanced_curves, ranges, importances = train_balanced5_scores(df)
    single_curves = train_single_feature_scores(df)
    global_curves, global_ranges = train_global_scores(df)
    curves = pd.concat([balanced_curves, global_curves, single_curves], ignore_index=True)
    macro = macro_table(curves)

    curves.to_csv(OUT_DIR / "balanced5_gain_curves.csv", index=False)
    stale_directions = OUT_DIR / "balanced5_single_feature_directions.csv"
    if stale_directions.exists():
        stale_directions.unlink()
    ranges.to_csv(OUT_DIR / "balanced5_selected_feature_ranges.csv", index=False)
    global_ranges.to_csv(OUT_DIR / "balanced5_global_selected_feature_ranges.csv", index=False)
    importances.to_csv(OUT_DIR / "balanced5_rf_feature_importance.csv", index=False)
    macro.to_csv(OUT_DIR / "balanced5_macro_gain_table.csv", index=False)

    selectors = macro["selector"].astype(str).tolist()
    for selector in selectors:
        plot_selector(curves, selector, FIG_DIR / f"{selector}.png")

    write_readme(definitions, macro, pd.concat([ranges, global_ranges], ignore_index=True), importances, curves)
    print(OUT_DIR / "README.md", flush=True)


if __name__ == "__main__":
    main()
