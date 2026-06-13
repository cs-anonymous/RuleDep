# Official-Aligned Query Subset Analysis

Metric: `gain_pt = MRR_stage2 / MRR_stage1 - 1`, computed from per-query RR
after per-relation multiplicative scaling. At 100% coverage this aligns with
the official filtered MRR from `metric-*.json`.

## Active Files

- `official_query_triple_features.csv`: query-level feature table.
- `feature_threshold_curves.csv`: raw-attribute coverage curves at 2% increments.
- `feature_rankings_at_coverage.csv`: raw-attribute macro rankings at 10% and 20% coverage.
- `feature_rankings_at_coverage.md`: readable top rankings.
- `best_feature_threshold_summary.csv`: best per-dataset raw-attribute thresholds with coverage >=20%.
- `high_gain_formula_report.md`: compact formula and fixed-coverage results.
- `ml_selector_diverse/`: cleaned diverse ML selector results and the curated Balanced5 selector. Non-diverse runs and fitted configs with more than 10 features are no longer kept in the active report tree.
- `feature_plots/`: per-feature coverage-gain plots.

## Data Coverage

- Datasets: FB15k-237, KG20C, WN18RR, YAGO3-10, codex-l, codex-m, hetionet.
- Samples: 596,060 per-GT cases.
- Raw attributes ranked: 84.

## Diverse Selector Summary

The active ML selector subset keeps diverse feature sets with at most 10
features, plus the curated 5-feature Balanced5 selector. See
`ml_selector_diverse/` for per-dataset details.

| feature_set | features | 10% | 20% | 30% | 50% | 100% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `balanced5_rf` | 5 | 0.1519 | 0.0925 | 0.0650 | 0.0477 | 0.0216 |
| `balanced5_global_rf` | 5 | 0.0978 | 0.0713 | 0.0528 | 0.0401 | 0.0216 |
| `same_top8_diverse_a` | 8 | 0.1520 | 0.0900 | 0.0681 | 0.0474 | 0.0216 |
| `same_top10_diverse` | 10 | 0.1502 | 0.0893 | 0.0675 | 0.0470 | 0.0216 |
| `same_top8_diverse_b` | 8 | 0.1412 | 0.0877 | 0.0661 | 0.0463 | 0.0216 |
| `diverse_prefix5` | 5 | 0.1285 | 0.0830 | 0.0635 | 0.0443 | 0.0216 |
| `diverse_prefix4` | 4 | 0.1273 | 0.0818 | 0.0624 | 0.0433 | 0.0216 |
| `diverse_prefix3` | 3 | 0.1096 | 0.0740 | 0.0560 | 0.0409 | 0.0216 |

## Notes

- The cleaned selector report excludes non-diverse same-feature sweeps, generic
  ML sweeps, older RF selector outputs other than Balanced5, and
  dataset-specific formula search outputs.
- `same_top12_diverse` was excluded because it uses 12 fitted features.
- In-sample selector numbers are diagnostic upper bounds unless the selector is
  tuned on validation queries and reported once on held-out test queries.
