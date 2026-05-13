# Diverse ML Query Selectors

This directory keeps the official-query subset selector runs that use diverse
feature sets, plus the curated 5-feature Balanced5 selector. Older non-diverse
sweeps and fitted configurations with more than 10 features were removed from
the active report tree.

Metric: official-scaled `gain_pt = MRR_stage2 / MRR_stage1 - 1`.

## Files

- `diverse_selector_summary.csv`: per-dataset results for each retained diverse selector.
- `diverse_selector_macro_summary.csv`: macro average over datasets.
- `diverse_selector_feature_sets.csv`: feature definitions for each retained selector.
- `balanced5_report/`: curated 5-feature Balanced5 selector report, definitions, curves, and figures.
- `recommended_subset_criterion/`: paper-facing two-feature Global RF subset criterion
  (`synergy_weight_top5_mean` + `topk_rule_weight`) with top-10% ranges.

## Retained Configurations

| feature_set | features | 10% | 20% | 30% | 50% | 100% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `same_top8_diverse_a` | 8 | 0.1520 | 0.0900 | 0.0681 | 0.0474 | 0.0216 |
| `same_top10_diverse` | 10 | 0.1502 | 0.0893 | 0.0675 | 0.0470 | 0.0216 |
| `same_top8_diverse_b` | 8 | 0.1412 | 0.0877 | 0.0661 | 0.0463 | 0.0216 |
| `diverse_prefix5` | 5 | 0.1285 | 0.0830 | 0.0635 | 0.0443 | 0.0216 |
| `diverse_prefix4` | 4 | 0.1273 | 0.0818 | 0.0624 | 0.0433 | 0.0216 |
| `diverse_prefix3` | 3 | 0.1096 | 0.0740 | 0.0560 | 0.0409 | 0.0216 |

## Balanced5

Balanced5 is retained as a curated 5-feature selector rather than as part of the
old generic RF sweep. `balanced5_rf` trains one RF per dataset;
`balanced5_global_rf` trains one pooled RF over all datasets.

| selector | features | 10% | 20% | 30% | 50% | 100% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `balanced5_rf` | 5 | 0.1519 | 0.0925 | 0.0650 | 0.0477 | 0.0216 |
| `balanced5_global_rf` | 5 | 0.0978 | 0.0713 | 0.0528 | 0.0401 | 0.0216 |

See `balanced5_report/README.md` for feature definitions, per-dataset gains,
feature importances, selected ranges, and plots.

Excluded from this cleaned subset:

- Non-diverse selector outputs.
- `same_top8_synergy`, because it is a same-family synergy-only set.
- `same_top12_diverse`, because it uses 12 fitted features.
