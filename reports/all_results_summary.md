# 0407 Overall Results

This table summarizes all completed experiments in the current workspace. Metrics and times are reported in seconds where applicable.

Related files:

- `all_results_summary.csv`
- `best_config_by_dataset.csv`
- `overall_time_comparison.csv`
- `all_results_ensemble_debug.json`

Description:

- `eval-maxplus` / `eval-noisyor`: from application logs.
- `canonical`: parsed from the old-format directory (`head_mrr_*.p + tail_mrr_*.p + canonical.log`).
- `best_combination*`: excluded from this report as a group.
- `ensemble_best_valid`: selects per-relation best config by validation MRR, then reports test aggregate.
- `ensemble_best_test`: selects per-relation best config by test MRR, then reports test aggregate (oracle upper bound).
- `ensemble_safe_valid`: a stable variant that prefers validation-based selection, with fallback to the best single dataset-level model for unstable relations.
- `structural_rd__stage1 / structural_r2d3__stage1 / structural_r3d6__stage1`: stage1 test results from the same experiment.
- In the time comparison table, RuleDep stage 1 and 2 times are estimated proportionally from per-relation `epochs_trained` and divided by 2 (for `multiprocess=2`). Canonical times come from the serial old process and are not divided.

## Best Non-canonical Config Per Dataset

| Dataset | Best config | Best MRR | Ensemble-valid | Ensemble-safe | Ensemble-test | Canonical | Best app |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KG20C | tg_r2d3__pos_auto_ratio__ri_conf__dn_per_rule_degree__dl1_1e-5 | 0.233952 | 0.235725 | - | 0.239769 | 0.239401 | 0.229772 |
| codex-m | tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5 | 0.344803 | 0.346569 | - | 0.349667 | 0.345537 | 0.319487 |
| WN18RR | structural_rd | 0.502189 | 0.500511 | - | 0.503990 | 0.499500 | 0.496968 |
| FB15k-237 | tg_r2d3__pos_auto_ratio__ri_conf__dn_none__dl1_1e-5 | 0.355348 | 0.357476 | - | 0.362664 | 0.353919 | 0.337685 |
| codex-l | tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5 | 0.334178 | 0.334810 | - | 0.336508 | 0.331778 | 0.311458 |
| YAGO3-10 | tg_r3d6__pos_auto_sqrt__ri_surprisal__dn_none__dl1_1e-5 | 0.578974 | 0.578353 | - | 0.582268 | 0.573990 | 0.554384 |
| hetionet | tg_rd__pos_auto_ratio__ri_conf__dn_per_rule_degree__dl1_1e-5 | 0.390256 | 0.395001 | - | 0.395569 | - | 0.350685 |

## Estimated Runtime Breakdown

| Dataset | RuleDep config | Canonical time (s) | Canonical source | RuleDep stage1 est. (s) | RuleDep stage2 est. (s) | RuleDep total est. (s) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| KG20C | tg_r2d3__pos_auto_ratio__ri_conf__dn_per_rule_degree__dl1_1e-5 | 2448.914000 | actual | 246.780672 | 294.087057 | 540.867729 |
| codex-m | tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5 | 7890.144000 | actual | 1063.709421 | 678.443825 | 1742.153246 |
| WN18RR | structural_rd | 6391.193000 | actual | 894.579796 | 361.305494 | 1255.885290 |
| FB15k-237 | tg_r2d3__pos_auto_ratio__ri_conf__dn_none__dl1_1e-5 | 15532.391000 | actual | 3891.108337 | 2305.502197 | 6196.610535 |
| codex-l | tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5 | 33603.206000 | actual | 12567.181074 | 10070.174733 | 22637.355807 |
| YAGO3-10 | tg_r3d6__pos_auto_sqrt__ri_surprisal__dn_none__dl1_1e-5 | 18013.941000 | actual | 2526.499372 | 2228.888708 | 4755.388079 |
| hetionet | tg_rd__pos_auto_ratio__ri_conf__dn_per_rule_degree__dl1_1e-5 | - | - | 2147.438905 | 2035.149371 | 4182.588276 |

## Notes

- Datasets covered: `7`
- Datasets with completed canonical runs: `6`
- Datasets with ensemble-valid: `7`
- Datasets with ensemble-safe: `0`
- Datasets with ensemble-test: `7`
