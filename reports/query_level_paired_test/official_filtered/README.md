# Official Filtered Query-Level Paired Test

This analysis compares Stage 1 and the final selected RuleDep model on exactly the same
official filtered test query-directions. The paired outcome is `RR_final - RR_stage1`;
therefore, its sample mean is exactly the reported MRR difference.

The 95% confidence intervals use 10,000 paired bootstrap resamples. Two-sided
p-values use 100,000 paired sign-flip randomizations and are Holm-adjusted
across the six datasets. No query is missing and no missing-rank imputation is used.

| Dataset | Queries | Stage 1 MRR | Final MRR | Delta MRR | Relative gain | 95% CI | Holm p | Win/Tie/Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FB15k-237 | 40,932 | 0.351835 | 0.355352 | +0.003517 | +1.00% | [+0.002759, +0.004286] | 6.00e-05 | 3,558/34,687/2,687 |
| KG20C | 7,448 | 0.226381 | 0.233776 | +0.007395 | +3.27% | [+0.005284, +0.009468] | 6.00e-05 | 1,484/4,949/1,015 |
| WN18RR | 6,268 | 0.499477 | 0.501542 | +0.002065 | +0.41% | [+0.000730, +0.003424] | 0.0022 | 298/5,706/264 |
| YAGO3-10 | 10,000 | 0.560634 | 0.578924 | +0.018290 | +3.26% | [+0.015230, +0.021473] | 6.00e-05 | 1,404/7,845/751 |
| codex-l | 61,240 | 0.329304 | 0.334173 | +0.004869 | +1.48% | [+0.004182, +0.005542] | 6.00e-05 | 5,666/51,461/4,113 |
| codex-m | 20,622 | 0.341877 | 0.344500 | +0.002624 | +0.77% | [+0.001814, +0.003466] | 6.00e-05 | 1,603/17,633/1,386 |

Dataset-macro Delta MRR: `+0.006460` (95% bootstrap CI `[+0.005761, +0.007156]`).

The paired t-test and unadjusted randomization p-values are retained in
`paired_test_results.csv` as secondary diagnostics.
