# Run-to-Run Stability

This report evaluates the run-to-run stability of RuleDep after fixing the rule set. Dependency mining and dependency-aware aggregation training were repeated three times using the same best aggregation configuration for each dataset.

## Results

| Dataset | Mean final MRR | Sample std. |
| --- | ---: | ---: |
| KG20C | 0.2341 | 0.0001 |
| Codex-M | 0.3446 | 0.0002 |
| WN18RR | 0.5008 | 0.0013 |
| FB15k-237 | 0.3537 | 0.0015 |
| Codex-L | 0.3335 | 0.0006 |
| YAGO3-10 | 0.5770 | 0.0017 |

The overall dataset-macro final MRR is 0.3906 +/- 0.0008, compared with 0.3848 for the Stage-1/LR-Agg baseline. All three repeats outperform the baseline on all six datasets.

## Artifacts

- [`repeat_mrr_summary.csv`](repeat_mrr_summary.csv): per-run MRR values, dataset means, and sample standard deviations.
