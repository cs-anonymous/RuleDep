# Filtered Query-Level Paired Test

This directory contains the only maintained paired-test result.

## Experimental Setup

The comparison uses the six released datasets and each dataset's best RuleDep configuration.
Stage 1 is the learned rule-only aggregation model. `Final` is the relation-wise model selected
by validation MRR: it is the dependency-aware Stage 2 model when Stage 2 improves validation
MRR, and otherwise the Stage 1 model. Thus this compares the actual final RuleDep prediction
against its paired rule-only prediction.

For every test triple, both prediction directions are evaluated: `(?, r, t)` (head) and
`(h, r, ?)` (tail). Ranks are recomputed from the saved direction-specific checkpoints with
the same filtered ranking implementation used by the reported test metrics. Stage 1 and final
rows are joined by dataset, experiment, relation, direction, query key, known entity, and
target entity identifiers, giving one paired observation per test query-direction.

The input contains 146,510 paired query-directions. Coverage is 100%: every row has both Stage 1 and
final rank/RR, so no complete-case filtering, candidate-list approximation, or missing-rank
imputation is used. Exported relation-level mean RR values were checked against the saved
`metric-<relation>.json` MRR values before this test was run.

## Statistical Test

For query-direction `q`, the paired outcome is:

```text
d_q = RR_final(q) - RR_stage1(q)
    = 1 / rank_final(q) - 1 / rank_stage1(q).
```

The dataset mean of `d_q` is exactly `MRR_final - MRR_stage1`. Pairing removes variation due
to query difficulty because both models are evaluated on the identical query and target.

The primary uncertainty estimate is a 95% percentile interval from 10,000 paired
bootstrap resamples of query-directions within each dataset. The primary hypothesis test is a
two-sided paired sign-flip randomization test with 100,000 samples. Under the null,
the sign of each nonzero paired difference is exchangeable; zero differences remain ties.
Monte Carlo p-values use the `(exceedances + 1) / (samples + 1)` correction.

Because six dataset-level hypotheses are tested, sign-flip p-values are adjusted with Holm's
step-down procedure to control the family-wise error rate. The paired t-test is retained only
as a secondary diagnostic because reciprocal-rank differences are bounded, discrete, and
strongly zero-inflated. All Monte Carlo calculations use seed `20260809`.

## Results

| Dataset | Queries | Stage 1 MRR | Final MRR | Delta MRR | Relative gain | 95% CI | Holm p | Win/Tie/Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FB15k-237 | 40,932 | 0.351835 | 0.355352 | +0.003517 | +1.00% | [+0.002759, +0.004286] | 6.00e-05 | 3,558/34,687/2,687 |
| KG20C | 7,448 | 0.226381 | 0.233776 | +0.007395 | +3.27% | [+0.005284, +0.009468] | 6.00e-05 | 1,484/4,949/1,015 |
| WN18RR | 6,268 | 0.499477 | 0.501542 | +0.002065 | +0.41% | [+0.000730, +0.003424] | 0.0022 | 298/5,706/264 |
| YAGO3-10 | 10,000 | 0.560634 | 0.578924 | +0.018290 | +3.26% | [+0.015230, +0.021473] | 6.00e-05 | 1,404/7,845/751 |
| codex-l | 61,240 | 0.329304 | 0.334173 | +0.004869 | +1.48% | [+0.004182, +0.005542] | 6.00e-05 | 5,666/51,461/4,113 |
| codex-m | 20,622 | 0.341877 | 0.344500 | +0.002624 | +0.77% | [+0.001814, +0.003466] | 6.00e-05 | 1,603/17,633/1,386 |

Dataset-macro Delta MRR: `+0.006460` (95% bootstrap CI `[+0.005761, +0.007156]`).

### Query Change Profile

| Dataset | Changed queries | Win rate | Tie rate | Loss rate | Mean delta on wins | Mean delta on losses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FB15k-237 | 15.26% | 8.69% | 84.74% | 6.56% | +0.109567 | -0.091505 |
| KG20C | 33.55% | 19.92% | 66.45% | 13.63% | +0.081281 | -0.064573 |
| WN18RR | 8.97% | 4.75% | 91.03% | 4.21% | +0.100894 | -0.064865 |
| YAGO3-10 | 21.55% | 14.04% | 78.45% | 7.51% | +0.243149 | -0.211028 |
| codex-l | 15.97% | 9.25% | 84.03% | 6.72% | +0.122988 | -0.096924 |
| codex-m | 14.49% | 7.77% | 85.51% | 6.72% | +0.082327 | -0.056180 |

## Detailed Analysis

All six 95% confidence intervals lie strictly above zero, and all six sign-flip tests
remain significant after Holm correction. The conclusion is therefore not driven by one
benchmark or by an uncorrected collection of small p-values.

YAGO3-10 has the largest absolute improvement (`+0.018290` MRR) and nearly twice as many
winning as losing queries (1,404 versus 751). Its large conditional changes in both
directions show that dependencies materially reorder the affected candidates, while the
positive imbalance makes the net effect strongly beneficial.

KG20C has the largest relative gain (`+3.27%`) and the broadest effect: 33.55% of queries
change rank. FB15k-237, Codex-L, and Codex-M show smaller but consistently positive gains.
For these datasets, only about 14-16% of queries change, explaining why the aggregate MRR
gain is moderate even though changes on winning queries are meaningful.

WN18RR is the weakest effect (`+0.002065` MRR, `+0.41%` relative), but its interval remains
positive and its Holm-adjusted randomization p-value is `0.0022`. More than 91% of WN18RR
queries tie; among changed queries, wins are only slightly more frequent than losses
(298 versus 264), but their average positive RR change is larger than the average loss.

The dataset-macro estimate gives equal weight to each benchmark rather than allowing
Codex-L's larger test set to dominate. Its `+0.006460` MRR improvement and fully positive
confidence interval support a consistent cross-dataset benefit.

## Reproduction

Generate and validate paired Stage 1/final ranks from the best-configuration runs:

```bash
bash script/reproduce_main_table_per_query_rr.sh
```

Then run the paired analysis:

```bash
python src/reporting/analyze_official_filtered_paired_test.py
```

The defaults are `--bootstrap 10000`, `--permutations 100000`, and `--seed 20260809`.
Machine-readable results, unadjusted p-values, paired t-test diagnostics, and change-profile
fields are stored in `paired_test_results.csv`.

## Scope and Limitations

The inference unit is a test query-direction, as requested for query-level paired testing.
Head and tail queries from the same triple, and queries sharing entities, may be correlated.
The intervals therefore quantify query-level variation under this sampling unit; they are
not a claim that all observations are independent graph samples. The test also conditions
on the fixed learned rules and checkpoints. It measures uncertainty across test queries,
not uncertainty from rerunning AnyBURL with different random seeds.
