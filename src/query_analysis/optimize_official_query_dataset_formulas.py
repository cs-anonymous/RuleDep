#!/usr/bin/env python3
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import regenerate_official_query_subset_reports as base


ROOT = Path("/home/sy/RuleDep")
REPORT_DIR = ROOT / "reports" / "official_query_subset"
OUT_DIR = REPORT_DIR / "dataset_specific_formula"

TARGET_COVERAGES = [0.10, 0.20]
REPORT_COVERAGES = [0.10, 0.20, 0.30, 0.50, 1.00]

EXCLUDE_PREFIXES = ("stage2_",)
EXCLUDE_FEATURES = {
    # These are available only after stage2 or are direct diagnostics/outcomes.
    "avg_stage2_score",
    "max_stage2_score",
    "stage2_top_margin",
}


def eligible_features(df: pd.DataFrame) -> list[str]:
    features = []
    for feature in base.raw_feature_names(df):
        if feature in EXCLUDE_FEATURES:
            continue
        if any(feature.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
            continue
        features.append(feature)
    return features


def percentile(values: np.ndarray, direction: str) -> np.ndarray:
    series = pd.Series(values)
    pct = series.rank(pct=True, method="average").to_numpy(dtype=float)
    return pct if direction == "desc" else 1.0 - pct


def gain_for_score(group: pd.DataFrame, score: np.ndarray, coverage: float) -> tuple[float, float, float, int]:
    order = np.argsort(score, kind="mergesort")[::-1]
    n = max(1, int(round(len(order) * coverage)))
    selected = order[:n]
    m1 = float(group["official_scaled_rr_stage1"].to_numpy(dtype=float)[selected].mean())
    m2 = float(group["official_scaled_rr_stage2"].to_numpy(dtype=float)[selected].mean())
    gain = (m2 / m1 - 1.0) if m1 > 0 else 0.0
    return m1, m2, gain, n


def evaluate_formula(group: pd.DataFrame, score: np.ndarray, coverages: list[float]) -> dict[float, dict[str, float]]:
    out = {}
    for coverage in coverages:
        m1, m2, gain, n = gain_for_score(group, score, coverage)
        out[coverage] = {"n": n, "mrr_stage1": m1, "mrr_stage2": m2, "gain_pt": gain}
    return out


def single_feature_rankings(group: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        values = group[feature].to_numpy(dtype=float)
        for direction in ("desc", "asc"):
            score = percentile(values, direction)
            result = evaluate_formula(group, score, TARGET_COVERAGES)
            rows.append(
                {
                    "feature": feature,
                    "direction": direction,
                    "gain_10": result[0.10]["gain_pt"],
                    "gain_20": result[0.20]["gain_pt"],
                    "objective": result[0.10]["gain_pt"] + result[0.20]["gain_pt"],
                    "min_gain": min(result[0.10]["gain_pt"], result[0.20]["gain_pt"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["objective", "min_gain"], ascending=False)


def weight_patterns(size: int) -> list[tuple[float, ...]]:
    patterns = {tuple([1.0] * size)}
    for i in range(size):
        for weight in (0.5, 2.0, 3.0, 4.0):
            values = [1.0] * size
            values[i] = weight
            patterns.add(tuple(values))
    if size >= 3:
        for i, j in itertools.combinations(range(size), 2):
            for weight in (2.0, 3.0):
                values = [1.0] * size
                values[i] = weight
                values[j] = weight
                patterns.add(tuple(values))
    return sorted(patterns)


def optimize_dataset(group: pd.DataFrame, features: list[str], pool_size: int, max_terms: int) -> tuple[dict, pd.DataFrame]:
    singles = single_feature_rankings(group, features)
    candidates: list[tuple[str, str]] = []
    for row in singles.itertuples(index=False):
        key = (str(row.feature), str(row.direction))
        if key not in candidates:
            candidates.append(key)
        if len(candidates) >= pool_size:
            break

    scores = {
        (feature, direction): percentile(group[feature].to_numpy(dtype=float), direction)
        for feature, direction in candidates
    }

    best = None
    for size in range(1, max_terms + 1):
        for subset in itertools.combinations(candidates, size):
            arrays = [scores[key] for key in subset]
            for weights in weight_patterns(size):
                score = sum(weight * array for weight, array in zip(weights, arrays)) / sum(weights)
                result = evaluate_formula(group, score, TARGET_COVERAGES)
                g10 = result[0.10]["gain_pt"]
                g20 = result[0.20]["gain_pt"]
                objective = g10 + g20
                min_gain = min(g10, g20)
                # Prefer high sum; tie-break toward balanced 10/20 coverage.
                item = (objective, min_gain, g10, g20, subset, weights, score)
                if best is None or item[:4] > best[:4]:
                    best = item

    assert best is not None
    objective, min_gain, g10, g20, subset, weights, score = best
    full = evaluate_formula(group, score, REPORT_COVERAGES)
    formula = {
        "objective": objective,
        "min_gain": min_gain,
        "gain_10": g10,
        "gain_20": g20,
        "terms": subset,
        "weights": weights,
        "score": score,
        "coverage_results": full,
        "candidate_pool": candidates,
    }
    return formula, singles


def formula_text(terms: tuple[tuple[str, str], ...], weights: tuple[float, ...]) -> str:
    parts = []
    for (feature, direction), weight in zip(terms, weights):
        expr = f"P_d({feature})" if direction == "desc" else f"(1 - P_d({feature}))"
        parts.append(f"{weight:g} * {expr}")
    denom = sum(weights)
    return f"score = ({' + '.join(parts)}) / {denom:g}"


def write_outputs(summary_rows: list[dict], pool_rows: list[dict], detail_rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "dataset_specific_formula_summary.csv", index=False)
    pd.DataFrame(pool_rows).to_csv(OUT_DIR / "dataset_specific_candidate_pool.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(OUT_DIR / "dataset_specific_formula_curves.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    macro = summary[["gain_10", "gain_20", "gain_30", "gain_50", "gain_100"]].mean()

    lines = [
        "# Dataset-specific Formula Search",
        "",
        "Goal: optimize a small dataset-specific linear selector using only pre-stage2 raw query attributes.",
        "",
        "Metric: official-scaled `gain_pt`; 100% coverage is aligned with `metric-*.json`.",
        "",
        "Search space: for each dataset, first rank all eligible single raw attributes, keep a small candidate pool, then search 1-4 term nonnegative linear formulas over dataset-wise percentile ranks.",
        "",
        "## Macro Result",
        "",
        "| coverage | macro gain_pt |",
        "| ---: | ---: |",
    ]
    for cov, key in [(0.10, "gain_10"), (0.20, "gain_20"), (0.30, "gain_30"), (0.50, "gain_50"), (1.00, "gain_100")]:
        lines.append(f"| {int(cov * 100)}% | {macro[key]:.4f} |")

    lines.extend(
        [
            "",
            "## Per-dataset Formulas",
            "",
            "| dataset | formula | 10% | 20% | 30% | 50% | 100% |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | `{row['formula']}` | {row['gain_10']:.4f} | {row['gain_20']:.4f} | "
            f"{row['gain_30']:.4f} | {row['gain_50']:.4f} | {row['gain_100']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a diagnostic upper-bound style analysis unless the formulas are selected on validation data.",
            "- It shows whether the 10% / 20% targets are attainable with a small number of raw attributes per dataset.",
            "- The selected formulas should be treated as dataset-specific selectors, not a universal paper-facing rule.",
            "",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-size", type=int, default=10)
    parser.add_argument("--max-terms", type=int, default=4)
    args = parser.parse_args()

    print("[load]", flush=True)
    df = pd.read_csv(REPORT_DIR / "official_query_triple_features.csv")
    df = base.add_official_scaled_rr(df)
    features = eligible_features(df)
    print(f"[features] eligible={len(features)}", flush=True)

    summary_rows = []
    pool_rows = []
    detail_rows = []

    for dataset, group in df.groupby("dataset", sort=True):
        print(f"[dataset] {dataset}", flush=True)
        formula, singles = optimize_dataset(group.reset_index(drop=True), features, args.pool_size, args.max_terms)
        text = formula_text(formula["terms"], formula["weights"])
        cov = formula["coverage_results"]
        summary_rows.append(
            {
                "dataset": dataset,
                "formula": text,
                "objective_10_plus_20": formula["objective"],
                "min_gain_10_20": formula["min_gain"],
                "gain_10": cov[0.10]["gain_pt"],
                "gain_20": cov[0.20]["gain_pt"],
                "gain_30": cov[0.30]["gain_pt"],
                "gain_50": cov[0.50]["gain_pt"],
                "gain_100": cov[1.00]["gain_pt"],
            }
        )
        for rank, row in enumerate(singles.head(args.pool_size).itertuples(index=False), start=1):
            pool_rows.append(
                {
                    "dataset": dataset,
                    "rank": rank,
                    "feature": row.feature,
                    "direction": row.direction,
                    "gain_10": row.gain_10,
                    "gain_20": row.gain_20,
                    "objective": row.objective,
                }
            )
        for coverage, result in cov.items():
            detail_rows.append(
                {
                    "dataset": dataset,
                    "coverage": coverage,
                    "n": result["n"],
                    "mrr_stage1": result["mrr_stage1"],
                    "mrr_stage2": result["mrr_stage2"],
                    "gain_pt": result["gain_pt"],
                    "formula": text,
                }
            )

    write_outputs(summary_rows, pool_rows, detail_rows)
    print(OUT_DIR / "README.md", flush=True)


if __name__ == "__main__":
    main()
