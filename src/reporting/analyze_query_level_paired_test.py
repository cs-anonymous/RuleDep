#!/usr/bin/env python3
"""Run query-level paired tests on saved stage-1/stage-2 ranks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "reports" / "query_subset" / "all_queries_rank.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "query_level_paired_test"


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues, dtype=float)
    running = 0.0
    m = len(pvalues)
    for position, index in enumerate(order):
        value = min((m - position) * float(pvalues[index]), 1.0)
        running = max(running, value)
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
    output = np.empty(repetitions, dtype=float)
    n = len(values)
    for start in range(0, repetitions, batch_size):
        stop = min(start + batch_size, repetitions)
        sampled_counts = rng.multinomial(n, probabilities, size=stop - start)
        output[start:stop] = sampled_counts @ unique / n
    return output


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
    total_n = len(values)
    for start in range(0, repetitions, batch_size):
        size = min(batch_size, repetitions - start)
        positive_counts = rng.binomial(counts, 0.5, size=(size, len(counts)))
        signed_sums = (2 * positive_counts - counts) @ unique
        permuted = np.abs(signed_sums / total_n)
        exceedances += int(np.count_nonzero(permuted >= observed))
    return (exceedances + 1.0) / (repetitions + 1.0)


def analyze_group(
    group: pd.DataFrame,
    scenario: str,
    bootstrap_repetitions: int,
    permutation_repetitions: int,
    rng: np.random.Generator,
) -> tuple[dict, np.ndarray]:
    rank1 = group["GT Rank1"].to_numpy(dtype=float)
    rank2 = group["GT Rank2"].to_numpy(dtype=float)
    complete = np.isfinite(rank1) & np.isfinite(rank2)

    if scenario == "complete_case":
        rank1 = rank1[complete]
        rank2 = rank2[complete]
        rr1 = 1.0 / rank1
        rr2 = 1.0 / rank2
    elif scenario == "missing_as_tie":
        rr1 = np.zeros(len(group), dtype=float)
        rr2 = np.zeros(len(group), dtype=float)
        rr1[complete] = 1.0 / rank1[complete]
        rr2[complete] = 1.0 / rank2[complete]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    delta = rr2 - rr1
    bootstrap = grouped_bootstrap_means(delta, bootstrap_repetitions, rng)
    ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])
    t_result = stats.ttest_1samp(delta, popmean=0.0)
    permutation_p = grouped_sign_flip_pvalue(delta, permutation_repetitions, rng)

    mrr1 = float(rr1.mean())
    mrr2 = float(rr2.mean())
    mean_delta = float(delta.mean())
    return (
        {
            "n_total": int(len(group)),
            "n_paired": int(complete.sum()),
            "n_missing_both": int((~complete).sum()),
            "coverage": float(complete.mean()),
            "mrr_stage1": mrr1,
            "mrr_stage2": mrr2,
            "delta_mrr": mean_delta,
            "relative_gain": (mrr2 / mrr1 - 1.0) if mrr1 > 0 else np.nan,
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
            "paired_t_statistic": float(t_result.statistic),
            "paired_t_p": float(t_result.pvalue),
            "sign_flip_p": float(permutation_p),
            "wins": int(np.count_nonzero(delta > 0)),
            "ties": int(np.count_nonzero(delta == 0)),
            "losses": int(np.count_nonzero(delta < 0)),
        },
        bootstrap,
    )


def format_p(value: float) -> str:
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def write_readme(results: pd.DataFrame, output_dir: Path, bootstrap_repetitions: int, permutation_repetitions: int) -> None:
    lines = [
        "# Query-Level Paired Test",
        "",
        "This analysis compares the saved Stage 1 and Stage 2 ranks for the same test cases.",
        "For each paired case, the tested quantity is `d = 1/rank_stage2 - 1/rank_stage1`,",
        "so the mean paired difference is exactly the MRR difference on the analyzed cases.",
        "",
        "Two scenarios are reported:",
        "",
        "- `complete_case`: use only rows containing both saved ranks.",
        "- `missing_as_tie`: retain every row and assign zero RR to both stages when both ranks are missing; this treats all jointly missing rows as zero-gain ties.",
        "",
        f"Confidence intervals use {bootstrap_repetitions:,} paired bootstrap samples. The randomization check uses {permutation_repetitions:,} sign-flip samples.",
        "Dataset-level paired t-test p-values are Holm-adjusted across the seven datasets.",
        "",
    ]

    for scenario in ["complete_case", "missing_as_tie"]:
        subset = results[(results["scenario"] == scenario) & (results["scope"] == "dataset")].copy()
        lines.extend(
            [
                f"## {scenario}",
                "",
                "| Dataset | N paired/total | Coverage | Candidate-list MRR1 | Candidate-list MRR2 | Delta MRR | Relative gain | 95% CI | Holm p | Sign-flip p | Win/Tie/Loss |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in subset.itertuples(index=False):
            lines.append(
                f"| {row.dataset} | {int(row.n_paired):,}/{int(row.n_total):,} | {row.coverage:.2%} | "
                f"{row.mrr_stage1:.6f} | {row.mrr_stage2:.6f} | {row.delta_mrr:+.6f} | "
                f"{row.relative_gain:+.2%} | [{row.ci95_low:+.6f}, {row.ci95_high:+.6f}] | "
                f"{format_p(row.paired_t_p_holm)} | {format_p(row.sign_flip_p)} | "
                f"{int(row.wins):,}/{int(row.ties):,}/{int(row.losses):,} |"
            )
        macro = results[(results["scenario"] == scenario) & (results["scope"] == "macro")].iloc[0]
        lines.extend(
            [
                "",
                f"Dataset-macro Delta MRR: `{macro.delta_mrr:+.6f}` "
                f"(95% bootstrap CI `[{macro.ci95_low:+.6f}, {macro.ci95_high:+.6f}]`).",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation Constraint",
            "",
            "The source CSV is a legacy candidate-list export. Jointly missing ranks indicate that the",
            "gold target was not assigned a saved rank in either stage. The `missing_as_tie` result is",
            "therefore a sensitivity analysis, not a reconstruction of the full official filtered rank.",
            "",
            "Detailed machine-readable results are in `paired_test_results.csv`.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    required = {"Dataset", "GT Rank1", "GT Rank2"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    macro_bootstraps: dict[str, list[np.ndarray]] = {}
    root_rng = np.random.default_rng(args.seed)
    for scenario in ["complete_case", "missing_as_tie"]:
        macro_bootstraps[scenario] = []
        scenario_start = len(rows)
        for dataset, group in data.groupby("Dataset", sort=True):
            child_rng = np.random.default_rng(root_rng.integers(0, np.iinfo(np.int64).max))
            result, bootstrap = analyze_group(
                group,
                scenario,
                args.bootstrap,
                args.permutations,
                child_rng,
            )
            rows.append({"scenario": scenario, "scope": "dataset", "dataset": dataset, **result})
            macro_bootstraps[scenario].append(bootstrap)

        scenario_rows = rows[scenario_start:]
        pvalues = np.asarray([row["paired_t_p"] for row in scenario_rows], dtype=float)
        adjusted = holm_adjust(pvalues)
        for row, value in zip(scenario_rows, adjusted):
            row["paired_t_p_holm"] = float(value)

        macro_samples = np.mean(np.vstack(macro_bootstraps[scenario]), axis=0)
        macro_delta = float(np.mean([row["delta_mrr"] for row in scenario_rows]))
        macro_low, macro_high = np.quantile(macro_samples, [0.025, 0.975])
        rows.append(
            {
                "scenario": scenario,
                "scope": "macro",
                "dataset": "MACRO",
                "delta_mrr": macro_delta,
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
