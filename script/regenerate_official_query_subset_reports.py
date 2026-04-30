#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import math
import os
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/sy/RuleDep")
REPORT_DIR = ROOT / "reports" / "official_query_subset"
FEATURE_CSV = REPORT_DIR / "official_query_triple_features.csv"
PLOT_DIR = REPORT_DIR / "feature_plots"
DATA_ROOT = ROOT / "data"

COVERAGES = np.arange(0.02, 1.0001, 0.02)
FIXED_COVERAGES = [0.10, 0.20]

OUTCOME_COLUMNS = {
    "raw_rr_stage1",
    "raw_rr_stage2",
    "raw_delta_rr",
    "rr_stage1",
    "rr_stage2",
    "delta_rr",
    "gain_pt",
    "official_relation_delta_mrr",
    "calibration_offset",
    "official_scaled_rr_stage1",
    "official_scaled_rr_stage2",
    "official_scaled_delta_rr",
    "official_scaled_gain_pt",
    "official_relation_stage1_mrr",
    "official_relation_stage2_mrr",
    "official_relation_gain_pt",
    "relation_scale_stage1",
    "relation_scale_stage2",
    "num_test_samples",
}

COMPOSITE_PREFIXES = ("combo_",)


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def read_relation_names(dataset: str) -> list[str]:
    path = DATA_ROOT / dataset / "relation_ids.del"
    names = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            names.append(parts[1] if len(parts) == 2 else parts[0])
    return names


def load_official_relation_metrics(df: pd.DataFrame) -> dict[tuple[str, str, str], tuple[float, float, float]]:
    needed = sorted(set(zip(df["dataset"], df["experiment"])))
    out: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    for dataset, experiment in needed:
        relation_names = read_relation_names(str(dataset))
        exp_dir = DATA_ROOT / str(dataset) / "aggregation" / str(experiment)
        for metric_path in exp_dir.glob("metric-*.json"):
            try:
                metric = json.loads(metric_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            relation_id = int(metric["relation"])
            if relation_id >= len(relation_names):
                continue
            stage1 = metric.get("test_after_stage1") or {}
            stage2 = metric.get("test_after_stage2") or stage1
            if "mrr" not in stage1 or "mrr" not in stage2:
                continue
            out[(str(dataset), str(experiment), relation_names[relation_id])] = (
                float(stage1["mrr"]),
                float(stage2["mrr"]),
                float(metric.get("num_test_samples") or 0.0),
            )
    return out


def add_official_scaled_rr(df: pd.DataFrame) -> pd.DataFrame:
    official = load_official_relation_metrics(df)
    group_cols = ["dataset", "experiment", "relation"]
    raw = (
        df.groupby(group_cols, dropna=False)
        .agg(raw_s1=("raw_rr_stage1", "mean"), raw_s2=("raw_rr_stage2", "mean"), n=("raw_rr_stage1", "size"))
        .reset_index()
    )
    official_rows = []
    for row in raw.itertuples(index=False):
        key = (str(row.dataset), str(row.experiment), str(row.relation))
        off_s1, off_s2, num_test = official.get(key, (float(row.raw_s1), float(row.raw_s2), float(row.n)))
        scale1 = (off_s1 / float(row.raw_s1)) if float(row.raw_s1) > 0 else 1.0
        scale2 = (off_s2 / float(row.raw_s2)) if float(row.raw_s2) > 0 else 1.0
        official_rows.append(
            {
                "dataset": row.dataset,
                "experiment": row.experiment,
                "relation": row.relation,
                "official_relation_stage1_mrr": off_s1,
                "official_relation_stage2_mrr": off_s2,
                "official_relation_gain_pt": (off_s2 / off_s1 - 1.0) if off_s1 > 0 else 0.0,
                "relation_scale_stage1": scale1,
                "relation_scale_stage2": scale2,
                "num_test_samples": num_test,
            }
        )
    scale_df = pd.DataFrame(official_rows)
    out = df.merge(scale_df, on=group_cols, how="left")
    out["official_scaled_rr_stage1"] = out["raw_rr_stage1"] * out["relation_scale_stage1"].fillna(1.0)
    out["official_scaled_rr_stage2"] = out["raw_rr_stage2"] * out["relation_scale_stage2"].fillna(1.0)
    out["official_scaled_delta_rr"] = out["official_scaled_rr_stage2"] - out["official_scaled_rr_stage1"]
    out["official_scaled_gain_pt"] = np.where(
        out["official_scaled_rr_stage1"] > 0,
        out["official_scaled_rr_stage2"] / out["official_scaled_rr_stage1"] - 1.0,
        0.0,
    )
    return out


def raw_feature_names(df: pd.DataFrame) -> list[str]:
    numeric = set(df.select_dtypes(include=[np.number]).columns)
    names = []
    for col in df.columns:
        if col not in numeric:
            continue
        if col in OUTCOME_COLUMNS:
            continue
        if any(col.startswith(prefix) for prefix in COMPOSITE_PREFIXES):
            continue
        names.append(col)
    return names


def curve_for_values(
    data: pd.DataFrame,
    values: np.ndarray,
    direction: str,
    name: str,
    coverages: np.ndarray = COVERAGES,
) -> pd.DataFrame:
    rows = []
    reverse = direction == "desc"
    datasets = data["dataset"].to_numpy()
    s1 = data["official_scaled_rr_stage1"].to_numpy(dtype=float)
    s2 = data["official_scaled_rr_stage2"].to_numpy(dtype=float)
    for dataset in sorted(data["dataset"].unique()):
        idx = np.flatnonzero(datasets == dataset)
        order = np.argsort(values[idx], kind="mergesort")
        if reverse:
            order = order[::-1]
        ordered = idx[order]
        c1 = np.cumsum(s1[ordered])
        c2 = np.cumsum(s2[ordered])
        n_total = len(ordered)
        for cov in coverages:
            n = max(1, int(round(n_total * float(cov))))
            m1 = c1[n - 1] / n
            m2 = c2[n - 1] / n
            rows.append(
                {
                    "dataset": dataset,
                    "feature": name,
                    "sort_direction": direction,
                    "coverage": round(float(cov), 2),
                    "n": n,
                    "threshold": float(values[ordered[n - 1]]),
                    "mrr_stage1": float(m1),
                    "mrr_stage2": float(m2),
                    "delta_mrr": float(m2 - m1),
                    "gain_pt": float((m2 / m1 - 1.0) if m1 > 0 else 0.0),
                }
            )
    return pd.DataFrame(rows)


def build_feature_curves(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    pieces = []
    for i, feature in enumerate(features, start=1):
        print(f"[curves] {i}/{len(features)} {feature}", flush=True)
        values = df[feature].to_numpy(dtype=float)
        pieces.append(curve_for_values(df, values, "desc", feature))
        pieces.append(curve_for_values(df, values, "asc", feature))
    return pd.concat(pieces, ignore_index=True)


def feature_rankings(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for coverage in FIXED_COVERAGES:
        subset = curves[np.isclose(curves["coverage"], coverage)]
        grouped = subset.groupby(["feature", "sort_direction"], as_index=False).agg(
            macro_gain_pt=("gain_pt", "mean"),
            positive_datasets=("gain_pt", lambda x: int((x > 0).sum())),
            min_dataset_gain=("gain_pt", "min"),
            max_dataset_gain=("gain_pt", "max"),
        )
        grouped = grouped.sort_values("macro_gain_pt", ascending=False).reset_index(drop=True)
        grouped.insert(0, "rank", np.arange(1, len(grouped) + 1))
        grouped.insert(0, "coverage", coverage)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def best_summary(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, feature, direction), group in curves.groupby(["dataset", "feature", "sort_direction"], sort=True):
        full = group.loc[group["coverage"].sub(1.0).abs().idxmin()]
        eligible = group[group["coverage"] >= 0.20]
        best = eligible.loc[eligible["gain_pt"].idxmax()]
        rows.append(
            {
                "dataset": dataset,
                "feature": feature,
                "sort_direction": direction,
                "full_gain_pt": full["gain_pt"],
                "best_coverage_ge_20": best["coverage"],
                "best_n": int(best["n"]),
                "best_threshold": best["threshold"],
                "best_mrr_stage1": best["mrr_stage1"],
                "best_mrr_stage2": best["mrr_stage2"],
                "best_delta_mrr": best["delta_mrr"],
                "best_gain_pt": best["gain_pt"],
                "gain_lift_vs_full": best["gain_pt"] - full["gain_pt"],
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "best_gain_pt"], ascending=[True, False])


def plot_curves(curves: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = curves[["feature", "sort_direction"]].drop_duplicates().itertuples(index=False)
    palette = plt.get_cmap("tab10")
    for feature, direction in keys:
        group = curves[(curves["feature"] == feature) & (curves["sort_direction"] == direction)]
        fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=140)
        for i, (dataset, dg) in enumerate(group.groupby("dataset", sort=True)):
            dg = dg.sort_values("coverage")
            ax.plot(dg["coverage"] * 100, dg["gain_pt"] * 100, color=palette(i % 10), alpha=0.55, linewidth=1.1, label=dataset)
        macro = group.groupby("coverage", as_index=False)["gain_pt"].mean().sort_values("coverage")
        ax.plot(macro["coverage"] * 100, macro["gain_pt"] * 100, color="#111111", linewidth=2.5, marker="o", markersize=2.5, label="macro")
        for cov in FIXED_COVERAGES:
            row = macro[np.isclose(macro["coverage"], cov)]
            if not row.empty:
                y = float(row.iloc[0]["gain_pt"] * 100)
                ax.scatter([cov * 100], [y], color="#111111", s=26, zorder=5)
                ax.annotate(f"{int(cov*100)}%: {y:.2f}%", (cov * 100, y), textcoords="offset points", xytext=(6, 7), fontsize=8)
        ax.axhline(0, color="#777777", linewidth=0.8, alpha=0.7)
        ax.set_xlabel(f"Coverage ranked by {feature} ({direction})")
        ax.set_ylabel("gain_pt (%)")
        ax.set_title(f"{feature} ({direction})")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"{slugify(feature)}__{direction}.png")
        plt.close(fig)


def percentile_score(df: pd.DataFrame, feature: str, direction: str) -> np.ndarray:
    pct = df.groupby("dataset")[feature].rank(pct=True, method="average").to_numpy(dtype=float)
    return pct if direction == "desc" else 1.0 - pct


def formula_search(df: pd.DataFrame, rankings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[str, float]], str]:
    # Use the strongest non-outcome raw features at either 10% or 20% as the search pool.
    rank_pool = rankings[rankings["coverage"].isin(FIXED_COVERAGES)].copy()
    rank_pool["score"] = rank_pool["macro_gain_pt"]
    rank_pool = rank_pool.sort_values("score", ascending=False)
    candidates: list[tuple[str, str]] = []
    for row in rank_pool.itertuples(index=False):
        key = (str(row.feature), str(row.sort_direction))
        if key not in candidates:
            candidates.append(key)
        if len(candidates) >= 8:
            break

    components = {f"{feature}__{direction}": percentile_score(df, feature, direction) for feature, direction in candidates}
    names = list(components.keys())

    best_score = -math.inf
    best_names: tuple[str, ...] = ()
    best_weights: tuple[float, ...] = ()
    best_op = "linear"
    def weight_patterns(size: int) -> list[tuple[float, ...]]:
        patterns = {tuple([1.0] * size)}
        for i in range(size):
            for weight in (0.5, 2.0, 3.0):
                values = [1.0] * size
                values[i] = weight
                patterns.add(tuple(values))
        return sorted(patterns)

    def objective(score: np.ndarray) -> tuple[float, float, float]:
        curve = curve_for_values(df, score, "desc", "_formula_", np.array(FIXED_COVERAGES))
        macro = curve.groupby("coverage")["gain_pt"].mean()
        g10 = float(macro.loc[0.10])
        g20 = float(macro.loc[0.20])
        return g10 + g20, g10, g20

    # Compact but flexible: search 1-4 components with a small interpretable
    # positive-weight pattern set.
    for size in (1, 2, 3, 4):
        for subset in itertools.combinations(names, size):
            arrays = [components[name] for name in subset]
            for weights in weight_patterns(size):
                score = np.zeros(len(df), dtype=float)
                denom = 0.0
                for arr, w in zip(arrays, weights):
                    score += float(w) * arr
                    denom += float(w)
                score /= denom
                obj, _g10, _g20 = objective(score)
                if obj > best_score:
                    best_score = obj
                    best_names = subset
                    best_weights = tuple(float(w) for w in weights)
                    best_op = "linear"

            if size >= 2:
                score = np.maximum.reduce(arrays)
                obj, _g10, _g20 = objective(score)
                if obj > best_score:
                    best_score = obj
                    best_names = subset
                    best_weights = tuple([1.0] * size)
                    best_op = "max"

    formula_components: dict[str, tuple[str, float]] = {}
    for name, weight in zip(best_names, best_weights):
        feature, direction = name.rsplit("__", 1)
        formula_components[feature] = (direction, weight)

    if best_op == "max":
        score = np.maximum.reduce([components[name] for name in best_names])
        formula_text = "compact_score = max(\n"
        formula_text += "\n".join(
            f"  {'P_d' if direction == 'desc' else '(1 - P_d)'}({feature}),"
            for feature, (direction, _weight) in formula_components.items()
        ).rstrip(",")
        formula_text += "\n)"
    else:
        score = np.zeros(len(df), dtype=float)
        denom = sum(best_weights)
        for name, weight in zip(best_names, best_weights):
            score += weight * components[name]
        score /= denom
        formula_text = " + ".join(
            f"{weight:g} * {'P_d' if direction == 'desc' else '(1 - P_d)'}({feature})"
            for feature, (direction, weight) in formula_components.items()
        )
        formula_text = f"compact_score = ({formula_text}) / {denom:g}"

    curves = curve_for_values(df, score, "desc", "compact_formula", COVERAGES)
    return curves, formula_components, formula_text


def formula_variable_tables(df: pd.DataFrame, formula_curves: pd.DataFrame, components: dict[str, tuple[str, float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_macro = []
    rows_dataset = []
    items = [("compact_formula", "desc", None)] + [(feature, direction, feature) for feature, (direction, _w) in components.items()]
    for name, direction, feature in items:
        curves = formula_curves if feature is None else curve_for_values(df, df[feature].to_numpy(dtype=float), direction, name, np.array([0.10, 0.20, 1.0]))
        macro = curves.groupby("coverage")["gain_pt"].mean()
        rows_macro.append(
            {
                "feature_or_component": name,
                "sort_direction": direction,
                "10%": float(macro.loc[0.10]),
                "20%": float(macro.loc[0.20]),
                "100%": float(macro.loc[1.0]),
            }
        )
        for dataset, group in curves.groupby("dataset", sort=True):
            vals = group.set_index("coverage")["gain_pt"]
            rows_dataset.append(
                {
                    "feature_or_component": name,
                    "sort_direction": direction,
                    "dataset": dataset,
                    "10%": float(vals.loc[0.10]),
                    "20%": float(vals.loc[0.20]),
                    "100%": float(vals.loc[1.0]),
                }
            )
    return pd.DataFrame(rows_macro), pd.DataFrame(rows_dataset)


def format_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    out = df.head(max_rows).copy() if max_rows else df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda x: f"{float(x):.4f}")
    return out.to_markdown(index=False)


def write_rankings_md(rankings: pd.DataFrame, path: Path, topn: int = 30) -> None:
    lines = ["# Feature Rankings at Fixed Coverage", "", "Metric: official-scaled `gain_pt`, with coverage sampled every 2%.", ""]
    for coverage in FIXED_COVERAGES:
        lines.extend([f"## Coverage {int(coverage * 100)}%", ""])
        sub = rankings[np.isclose(rankings["coverage"], coverage)][
            ["rank", "feature", "sort_direction", "macro_gain_pt", "positive_datasets", "min_dataset_gain", "max_dataset_gain"]
        ]
        lines.append(format_table(sub, topn))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_formula_report(curves: pd.DataFrame, macro_table: pd.DataFrame, dataset_table: pd.DataFrame, formula: str, path: Path) -> None:
    fixed = curves[curves["coverage"].isin([0.10, 0.20, 1.0])]
    macro = fixed.groupby("coverage")["gain_pt"].mean()
    pivot = fixed.pivot(index="dataset", columns="coverage", values="gain_pt").reset_index()
    pivot.columns = ["dataset", "10%", "20%", "100%"]
    pivot.loc[len(pivot)] = ["macro", macro.loc[0.10], macro.loc[0.20], macro.loc[1.0]]
    lines = [
        "# Compact High-gain Formula Report",
        "",
        f"Headline: 10% coverage gives {macro.loc[0.10] * 100:.2f}% `gain_pt`; 20% coverage gives {macro.loc[0.20] * 100:.2f}% `gain_pt`.",
        "",
        "Formula:",
        "",
        "```text",
        formula,
        "```",
        "",
        "Metric: official-scaled per-query RR; 100% coverage is aligned to `metric-*.json` official Stage1/Stage2 MRR.",
        "",
        "Plot: [`feature_plots/high_gain_formula__desc.png`](feature_plots/high_gain_formula__desc.png)",
        "",
        "## Fixed Coverage Results",
        "",
        format_table(pivot),
        "",
        "## Formula Variables",
        "",
        format_table(macro_table),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_formula(curves: pd.DataFrame) -> None:
    plot_curves(curves.assign(feature="high_gain_formula", sort_direction="desc"), PLOT_DIR)


def write_readme(
    df: pd.DataFrame,
    features: list[str],
    rankings: pd.DataFrame,
    formula_curves: pd.DataFrame,
    formula_text: str,
) -> None:
    fixed_formula = formula_curves[formula_curves["coverage"].isin([0.10, 0.20, 1.0])]
    formula_macro = fixed_formula.groupby("coverage")["gain_pt"].mean()
    dataset_full = (
        df.groupby("dataset")
        .agg(mrr_stage1=("official_scaled_rr_stage1", "mean"), mrr_stage2=("official_scaled_rr_stage2", "mean"))
        .reset_index()
    )
    dataset_full["gain_pt"] = dataset_full["mrr_stage2"] / dataset_full["mrr_stage1"] - 1.0
    lines = [
        "# Official-aligned Query Subset Feature Analysis",
        "",
        "Metric: `gain_pt = MRR_stage2 / MRR_stage1 - 1`, computed from per-query RR after per-relation multiplicative scaling. This preserves query-level subset variation while making 100% coverage match the official `metric-*.json` Stage1/Stage2 MRR.",
        "",
        "Coverage grid: 2%, 4%, ..., 100%. Ranking features are raw query/candidate-set attributes; outcome fields, calibration fields, official-scale diagnostic fields, and `combo_*` features are excluded from the raw-attribute rankings.",
        "",
        "## Files",
        "",
        "- [`official_query_triple_features.csv`](official_query_triple_features.csv): query-level feature table used by the analyses.",
        "- [`feature_threshold_curves.csv`](feature_threshold_curves.csv): raw-attribute coverage curves at 2% increments.",
        "- [`feature_rankings_at_coverage.csv`](feature_rankings_at_coverage.csv): raw-attribute macro rankings at 10% and 20% coverage.",
        "- [`feature_rankings_at_coverage.md`](feature_rankings_at_coverage.md): readable top rankings.",
        "- [`best_feature_threshold_summary.csv`](best_feature_threshold_summary.csv): best per-dataset raw-attribute thresholds with coverage >=20%.",
        "- [`high_gain_formula_report.md`](high_gain_formula_report.md): selected compact formula and fixed-coverage results.",
        "- [`feature_plots/`](feature_plots/): per-feature coverage-gain plots.",
        "",
        "## Data Coverage",
        "",
        f"- Datasets: {', '.join(sorted(df['dataset'].unique()))}.",
        f"- Samples: {len(df):,} per-GT cases.",
        f"- Raw attributes ranked: {len(features)}.",
        "",
        "## 100% Official Alignment Check",
        "",
        format_table(dataset_full[["dataset", "mrr_stage1", "mrr_stage2", "gain_pt"]]),
        "",
        "## Main Takeaways",
        "",
        f"1. The new paper-facing compact formula gives {formula_macro.loc[0.10] * 100:.2f}% macro `gain_pt` at 10% coverage and {formula_macro.loc[0.20] * 100:.2f}% at 20% coverage.",
        "2. Monotonicity is not enforced; the formula is selected for high 10%/20% fixed-coverage gain.",
        "3. All raw-attribute plots and rankings use the 100%-aligned official-scaled metric, avoiding the earlier CODEX-L 100% mismatch.",
        "",
        "## Paper-facing Compact Formula",
        "",
        "```text",
        formula_text,
        "```",
        "",
        "See [`high_gain_formula_report.md`](high_gain_formula_report.md) for per-dataset values and component ablations.",
        "",
        "## Fixed-coverage Rankings",
        "",
    ]
    for coverage in FIXED_COVERAGES:
        sub = rankings[np.isclose(rankings["coverage"], coverage)][
            ["rank", "feature", "sort_direction", "macro_gain_pt", "positive_datasets", "min_dataset_gain", "max_dataset_gain"]
        ]
        lines.extend([f"### Coverage {int(coverage * 100)}%", "", format_table(sub, 20), ""])
    REPORT_DIR.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    print("[load] feature csv", flush=True)
    df = pd.read_csv(FEATURE_CSV)
    print("[metric] official scaling", flush=True)
    df = add_official_scaled_rr(df)
    features = raw_feature_names(df)
    print(f"[features] {len(features)} raw attributes", flush=True)

    curves = build_feature_curves(df, features)
    curves.to_csv(REPORT_DIR / "feature_threshold_curves.csv", index=False)
    rankings = feature_rankings(curves)
    rankings.to_csv(REPORT_DIR / "feature_rankings_at_coverage.csv", index=False)
    write_rankings_md(rankings, REPORT_DIR / "feature_rankings_at_coverage.md")
    best_summary(curves).to_csv(REPORT_DIR / "best_feature_threshold_summary.csv", index=False)

    print("[plots] feature curves", flush=True)
    plot_curves(curves, PLOT_DIR)

    print("[formula] search", flush=True)
    formula_curves, components, formula_text = formula_search(df, rankings)
    formula_curves.to_csv(REPORT_DIR / "high_gain_formula_curves.csv", index=False)
    formula_macro, formula_dataset = formula_variable_tables(df, formula_curves, components)
    formula_macro.to_csv(REPORT_DIR / "high_gain_formula_variable_macro_gain.csv", index=False)
    formula_dataset.to_csv(REPORT_DIR / "high_gain_formula_variable_dataset_gain.csv", index=False)
    formula_plot_curves = formula_curves.copy()
    formula_plot_curves["feature"] = "high_gain_formula"
    formula_plot_curves["sort_direction"] = "desc"
    plot_curves(formula_plot_curves, PLOT_DIR)
    write_formula_report(formula_curves, formula_macro, formula_dataset, formula_text, REPORT_DIR / "high_gain_formula_report.md")

    (REPORT_DIR / "universal_combo_formula.md").write_text(
        "# Universal Combo Selector Proposal\n\n"
        "The current paper-facing selector is the compact formula selected in "
        "[`high_gain_formula_report.md`](high_gain_formula_report.md). It is optimized for 10% and 20% "
        "official-scaled `gain_pt`; monotonicity is not enforced.\n\n"
        "```text\n"
        f"{formula_text}\n"
        "```\n",
        encoding="utf-8",
    )
    write_readme(df, features, rankings, formula_curves, formula_text)
    print("[done]", REPORT_DIR, flush=True)


if __name__ == "__main__":
    main()
