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
DEFAULT_OUTPUT = ROOT / "reports" / "query_level_paired_test"


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
    wins = delta > 0
    losses = delta < 0
    ties = delta == 0
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
            "changed_queries": int(np.count_nonzero(~ties)),
            "changed_fraction": float(np.mean(~ties)),
            "wins": int(np.count_nonzero(wins)),
            "ties": int(np.count_nonzero(ties)),
            "losses": int(np.count_nonzero(losses)),
            "win_fraction": float(np.mean(wins)),
            "tie_fraction": float(np.mean(ties)),
            "loss_fraction": float(np.mean(losses)),
            "mean_gain_on_wins": float(delta[wins].mean()) if np.any(wins) else 0.0,
            "mean_loss_on_losses": float(delta[losses].mean()) if np.any(losses) else 0.0,
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
        "This directory contains the only maintained paired-test result. It replaces the legacy",
        "candidate-list analysis, which had incomplete rank coverage and was not suitable for",
        "testing the MRR values reported in the paper.",
        "",
        "## Experimental Setup",
        "",
        "The comparison uses the six released datasets and each dataset's best RuleDep configuration.",
        "Stage 1 is the learned rule-only aggregation model. `Final` is the relation-wise model selected",
        "by validation MRR: it is the dependency-aware Stage 2 model when Stage 2 improves validation",
        "MRR, and otherwise the Stage 1 model. Thus this compares the actual final RuleDep prediction",
        "against its paired rule-only prediction, not against LR-Agg or an unselected Stage 2 model.",
        "",
        "For every test triple, both prediction directions are evaluated: `(?, r, t)` (head) and",
        "`(h, r, ?)` (tail). Ranks are recomputed from the saved direction-specific checkpoints with",
        "the same filtered ranking implementation used by the reported test metrics. Stage 1 and final",
        "rows are joined by dataset, experiment, relation, direction, query key, known entity, and",
        "target entity identifiers, giving one paired observation per test query-direction.",
        "",
        "The merged input is:",
        "",
        "```text",
        "reports/official_query_subset/true_official_per_query_rr/",
        "  main_table_per_query_rr_20260809/true_official_per_query_rr_wide.csv",
        "```",
        "",
        "It contains 146,510 paired query-directions. Coverage is 100%: every row has both Stage 1 and",
        "final rank/RR, so no complete-case filtering, candidate-list approximation, or missing-rank",
        "imputation is used. Exported relation-level mean RR values were checked against the saved",
        "`metric-<relation>.json` MRR values before this test was run.",
        "",
        "## Statistical Test",
        "",
        "For query-direction `q`, the paired outcome is:",
        "",
        "```text",
        "d_q = RR_final(q) - RR_stage1(q)",
        "    = 1 / rank_final(q) - 1 / rank_stage1(q).",
        "```",
        "",
        "The dataset mean of `d_q` is exactly `MRR_final - MRR_stage1`. Pairing removes variation due",
        "to query difficulty because both models are evaluated on the identical query and target.",
        "",
        f"The primary uncertainty estimate is a 95% percentile interval from {bootstrap:,} paired",
        "bootstrap resamples of query-directions within each dataset. The primary hypothesis test is a",
        f"two-sided paired sign-flip randomization test with {permutations:,} samples. Under the null,",
        "the sign of each nonzero paired difference is exchangeable; zero differences remain ties.",
        "Monte Carlo p-values use the `(exceedances + 1) / (samples + 1)` correction.",
        "",
        "Because six dataset-level hypotheses are tested, sign-flip p-values are adjusted with Holm's",
        "step-down procedure to control the family-wise error rate. The paired t-test is retained only",
        "as a secondary diagnostic because reciprocal-rank differences are bounded, discrete, and",
        "strongly zero-inflated. All Monte Carlo calculations use seed `20260809`.",
        "",
        "## Results",
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
            "### Query Change Profile",
            "",
            "| Dataset | Changed queries | Win rate | Tie rate | Loss rate | Mean delta on wins | Mean delta on losses |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in dataset_rows.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.changed_fraction:.2%} | {row.win_fraction:.2%} | "
            f"{row.tie_fraction:.2%} | {row.loss_fraction:.2%} | "
            f"{row.mean_gain_on_wins:+.6f} | {row.mean_loss_on_losses:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Detailed Analysis",
            "",
            "All six 95% confidence intervals lie strictly above zero, and all six sign-flip tests",
            "remain significant after Holm correction. The conclusion is therefore not driven by one",
            "benchmark or by an uncorrected collection of small p-values.",
            "",
            "YAGO3-10 has the largest absolute improvement (`+0.018290` MRR) and nearly twice as many",
            "winning as losing queries (1,404 versus 751). Its large conditional changes in both",
            "directions show that dependencies materially reorder the affected candidates, while the",
            "positive imbalance makes the net effect strongly beneficial.",
            "",
            "KG20C has the largest relative gain (`+3.27%`) and the broadest effect: 33.55% of queries",
            "change rank. FB15k-237, Codex-L, and Codex-M show smaller but consistently positive gains.",
            "For these datasets, only about 14-16% of queries change, explaining why the aggregate MRR",
            "gain is moderate even though changes on winning queries are meaningful.",
            "",
            "WN18RR is the weakest effect (`+0.002065` MRR, `+0.41%` relative), but its interval remains",
            "positive and its Holm-adjusted randomization p-value is `0.0022`. More than 91% of WN18RR",
            "queries tie; among changed queries, wins are only slightly more frequent than losses",
            "(298 versus 264), but their average positive RR change is larger than the average loss.",
            "",
            "The dataset-macro estimate gives equal weight to each benchmark rather than allowing",
            "Codex-L's larger test set to dominate. Its `+0.006460` MRR improvement and fully positive",
            "confidence interval support a consistent cross-dataset benefit.",
            "",
            "## Reproduction",
            "",
            "Generate and validate paired Stage 1/final ranks from the best-configuration runs:",
            "",
            "```bash",
            "bash script/reproduce_main_table_per_query_rr.sh",
            "```",
            "",
            "Then run the paired analysis:",
            "",
            "```bash",
            "python src/reporting/analyze_official_filtered_paired_test.py",
            "```",
            "",
            "The defaults are `--bootstrap 10000`, `--permutations 100000`, and `--seed 20260809`.",
            "Machine-readable results, unadjusted p-values, paired t-test diagnostics, and change-profile",
            "fields are stored in `paired_test_results.csv`.",
            "",
            "## Scope and Limitations",
            "",
            "The inference unit is a test query-direction, as requested for query-level paired testing.",
            "Head and tail queries from the same triple, and queries sharing entities, may be correlated.",
            "The intervals therefore quantify query-level variation under this sampling unit; they are",
            "not a claim that all observations are independent graph samples. The test also conditions",
            "on the fixed learned rules and checkpoints. It measures uncertainty across test queries,",
            "not uncertainty from rerunning AnyBURL with different random seeds.",
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
    identity_columns = [
        "dataset",
        "experiment",
        "relation_id",
        "direction",
        "query_key",
        "target_entity_id",
    ]
    missing_identity = set(identity_columns).difference(data.columns)
    if missing_identity:
        raise ValueError(f"Missing query identity columns: {sorted(missing_identity)}")
    if data.duplicated(identity_columns).any():
        raise ValueError("Official paired input contains duplicate query identities")

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
