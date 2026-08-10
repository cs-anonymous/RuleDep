# Query-Level Paired Test

This analysis compares the saved Stage 1 and Stage 2 ranks for the same test cases.
For each paired case, the tested quantity is `d = 1/rank_stage2 - 1/rank_stage1`,
so the mean paired difference is exactly the MRR difference on the analyzed cases.

Two scenarios are reported:

- `complete_case`: use only rows containing both saved ranks.
- `missing_as_tie`: retain every row and assign zero RR to both stages when both ranks are missing; this treats all jointly missing rows as zero-gain ties.

Confidence intervals use 10,000 paired bootstrap samples. The randomization check uses 20,000 sign-flip samples.
Dataset-level paired t-test p-values are Holm-adjusted across the seven datasets.

## complete_case

| Dataset | N paired/total | Coverage | Candidate-list MRR1 | Candidate-list MRR2 | Delta MRR | Relative gain | 95% CI | Holm p | Sign-flip p | Win/Tie/Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FB15k-237 | 36,455/40,819 | 89.31% | 0.610293 | 0.615570 | +0.005276 | +0.86% | [+0.004398, +0.006185] | 6.58e-30 | 5.00e-05 | 3,308/31,145/2,002 |
| KG20C | 5,617/7,448 | 75.42% | 0.415953 | 0.416031 | +0.000077 | +0.02% | [-0.000232, +0.000375] | 1.0000 | 0.6393 | 144/5,413/60 |
| WN18RR | 4,598/6,055 | 75.94% | 0.336440 | 0.336700 | +0.000260 | +0.08% | [-0.000820, +0.001329] | 1.0000 | 0.6436 | 109/4,435/54 |
| YAGO3-10 | 8,621/9,975 | 86.43% | 0.619186 | 0.676607 | +0.057421 | +9.27% | [+0.053007, +0.061785] | 5.23e-141 | 5.00e-05 | 1,855/6,092/674 |
| codex-l | 50,785/61,129 | 83.08% | 0.543904 | 0.544645 | +0.000741 | +0.14% | [+0.000550, +0.000928] | 6.01e-14 | 5.00e-05 | 1,209/48,285/1,291 |
| codex-m | 17,881/20,594 | 86.83% | 0.531025 | 0.532370 | +0.001345 | +0.25% | [+0.000905, +0.001796] | 1.06e-08 | 5.00e-05 | 321/17,275/285 |
| hetionet | 362,130/378,668 | 95.63% | 0.772039 | 0.772129 | +0.000090 | +0.01% | [-0.000183, +0.000359] | 1.0000 | 0.5111 | 12,113/337,865/12,152 |

Dataset-macro Delta MRR: `+0.009316` (95% bootstrap CI `[+0.008651, +0.009964]`).

## missing_as_tie

| Dataset | N paired/total | Coverage | Candidate-list MRR1 | Candidate-list MRR2 | Delta MRR | Relative gain | 95% CI | Holm p | Sign-flip p | Win/Tie/Loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FB15k-237 | 36,455/40,819 | 89.31% | 0.545046 | 0.549758 | +0.004712 | +0.86% | [+0.003915, +0.005520] | 6.67e-30 | 5.00e-05 | 3,308/35,509/2,002 |
| KG20C | 5,617/7,448 | 75.42% | 0.313696 | 0.313755 | +0.000058 | +0.02% | [-0.000176, +0.000298] | 1.0000 | 0.6396 | 144/7,244/60 |
| WN18RR | 4,598/6,055 | 75.94% | 0.255483 | 0.255681 | +0.000197 | +0.08% | [-0.000632, +0.001017] | 1.0000 | 0.6357 | 109/5,892/54 |
| YAGO3-10 | 8,621/9,975 | 86.43% | 0.535138 | 0.584765 | +0.049627 | +9.27% | [+0.045951, +0.053349] | 2.73e-140 | 5.00e-05 | 1,855/7,446/674 |
| codex-l | 50,785/61,129 | 83.08% | 0.451866 | 0.452482 | +0.000616 | +0.14% | [+0.000458, +0.000771] | 6.02e-14 | 5.00e-05 | 1,209/58,629/1,291 |
| codex-m | 17,881/20,594 | 86.83% | 0.461069 | 0.462237 | +0.001168 | +0.25% | [+0.000799, +0.001562] | 1.06e-08 | 5.00e-05 | 321/19,988/285 |
| hetionet | 362,130/378,668 | 95.63% | 0.738321 | 0.738407 | +0.000086 | +0.01% | [-0.000167, +0.000343] | 1.0000 | 0.5129 | 12,113/354,403/12,152 |

Dataset-macro Delta MRR: `+0.008066` (95% bootstrap CI `[+0.007509, +0.008629]`).

## Interpretation Constraint

The source CSV is a legacy candidate-list export. Jointly missing ranks indicate that the
gold target was not assigned a saved rank in either stage. The `missing_as_tie` result is
therefore a sensitivity analysis, not a reconstruction of the full official filtered rank.

Detailed machine-readable results are in `paired_test_results.csv`.
