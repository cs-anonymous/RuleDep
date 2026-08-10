#!/usr/bin/env python3
"""Paired query-level inference for official filtered Stage 1 and final ranks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT
    / "reports"
    / "official_query_subset"
    / "true_official_per_query_rr"
    / "main_table_per_query_rr_20260809"
    / "true_official_per_query_rr_wide.csv"
)
DEFAULT_OUTPUT = ROOT / "reports" / "query_level_paired_test" / "official_filtered"


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues, dtype=float)
    running = 0.0
    m = len(pvalues)
    for position, index in enumerate(order):
        running = max(running, min((m - position) * float(pvalues[index]), 1.0))
        adjusted[index] = running
    return adjusted


def grouped_bootstrap_means(
    values: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
    batch_size: int = 200,
) -> np.ndarray:
    unique, counts = np.unique(values, return_counts=True)
    probabilities = counts.astype(float) / counts.sum()
    samples = np.empty(repetitions, dtype=float)
    n = len(values)
    for start in range(0, repetitions, batch_size):
        stop = min(start + batch_size, repetitions)
        sampled_counts = rng.multinomial(n, probabilities, size=stop - start)
        samples[start:stop] = sampled_counts @ unique / n
    return samples


def grouped_sign_flip_pvalue(
    values: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
    batch_size: int = 500,
) -> float:
    nonzero = values[values != 0.0]
    if len(nonzero) == 0:
        return 1.0
    unique, counts = np.unique(nonzero, return_counts=True)
    observed = abs(float(values.mean()))
    exceedances = 0
    for start in range(0, repetitions, batch_size):
        size = min(batch_size, repetitions - start)
        positive_counts = rng.binomial(counts, 0.5, size=(size, len(counts)))
        signed_sums = (2 * positive_counts - counts) @ unique
        exceedances += int(np.count_nonzero(np.abs(signed_sums / len(values)) >= observed))
    return float((exceedances + 1) / (repetitions + 1))


def analyze_dataset(
    group: pd.DataFrame,
    bootstrap_repetitions: int,
    permutation_repetitions: int,
    rng: np.random.Generator,
) -> tuple[dict, np.ndarray]:
    rr_stage1 = group["true_official_rr_stage1"].to_numpy(dtype=float)
    rr_final = group["true_official_rr_final"].to_numpy(dtype=float)
    if not np.isfinite(rr_stage1).all() or not np.isfinite(rr_final).all():
        raise ValueError("Official paired input contains missing or non-finite RR values")
    delta = rr_final - rr_stage1
    bootstrap = grouped_bootstrap_means(delta, bootstrap_repetitions, rng)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
    t_result = stats.ttest_1samp(delta, popmean=0.0)
    return (
        {
            "n_queries": int(len(group)),
            "mrr_stage1": float(rr_stage1.mean()),
            "mrr_final": float(rr_final.mean()),
            "delta_mrr": float(delta.mean()),
            "relative_gain": float(rr_final.mean() / rr_stage1.mean() - 1.0),
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
            "sign_flip_p": grouped_sign_flip_pvalue(delta, permutation_repetitions, rng),
            "paired_t_statistic": float(t_result.statistic),
            "paired_t_p": float(t_result.pvalue),
            "wins": int(np.count_nonzero(delta > 0)),
            "ties": int(np.count_nonzero(delta == 0)),
            "losses": int(np.count_nonzero(delta < 0)),
        },
        bootstrap,
    )


def format_p(value: float) -> str:
    return f"{value:.2e}" if value < 1e-4 else f"{value:.4f}"


def write_readme(results: pd.DataFrame, output_dir: Path, bootstrap: int, permutations: int) -> None:
    dataset_rows = results[results["scope"] == "dataset"]
    lines = [
        "# Official Filtered Query-Level Paired Test",
        "",
        "This analysis compares Stage 1 and the final selected RuleDep model on exactly the same",
        "official filtered test query-directions. The paired outcome is `RR_final - RR_stage1`;",
        "therefore, its sample mean is exactly the reported MRR difference.",
        "",
        f"The 95% confidence intervals use {bootstrap:,} paired bootstrap resamples. Two-sided",
        f"p-values use {permutations:,} paired sign-flip randomizations and are Holm-adjusted",
        "across the six datasets. No query is missing and no missing-rank imputation is used.",
        "",
        "| Dataset | Queries | Stage 1 MRR | Final MRR | Delta MRR | Relative gain | 95% CI | Holm p | Win/Tie/Loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in dataset_rows.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {int(row.n_queries):,} | {row.mrr_stage1:.6f} | "
            f"{row.mrr_final:.6f} | {row.delta_mrr:+.6f} | {row.relative_gain:+.2%} | "
            f"[{row.ci95_low:+.6f}, {row.ci95_high:+.6f}] | "
            f"{format_p(row.sign_flip_p_holm)} | "
            f"{int(row.wins):,}/{int(row.ties):,}/{int(row.losses):,} |"
        )
    macro = results[results["scope"] == "dataset_macro"].iloc[0]
    lines.extend(
        [
            "",
            f"Dataset-macro Delta MRR: `{macro.delta_mrr:+.6f}` "
            f"(95% bootstrap CI `[{macro.ci95_low:+.6f}, {macro.ci95_high:+.6f}]`).",
            "",
            "The paired t-test and unadjusted randomization p-values are retained in",
            "`paired_test_results.csv` as secondary diagnostics.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    required = {"dataset", "true_official_rr_stage1", "true_official_rr_final"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    bootstraps = []
    root_rng = np.random.default_rng(args.seed)
    for dataset, group in data.groupby("dataset", sort=True):
        rng = np.random.default_rng(root_rng.integers(0, np.iinfo(np.int64).max))
        result, bootstrap_samples = analyze_dataset(
            group, args.bootstrap, args.permutations, rng
        )
        rows.append({"scope": "dataset", "dataset": dataset, **result})
        bootstraps.append(bootstrap_samples)

    adjusted = holm_adjust(np.asarray([row["sign_flip_p"] for row in rows]))
    for row, value in zip(rows, adjusted):
        row["sign_flip_p_holm"] = float(value)

    macro_samples = np.mean(np.vstack(bootstraps), axis=0)
    macro_low, macro_high = np.quantile(macro_samples, [0.025, 0.975])
    rows.append(
        {
            "scope": "dataset_macro",
            "dataset": "MACRO",
            "delta_mrr": float(np.mean([row["delta_mrr"] for row in rows])),
            "ci95_low": float(macro_low),
            "ci95_high": float(macro_high),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "paired_test_results.csv", index=False)
    write_readme(results, args.output_dir, args.bootstrap, args.permutations)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
