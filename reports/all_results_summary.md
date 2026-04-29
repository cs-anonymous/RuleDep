# 0407 Overall Results

本表汇总当前仓库里所有已完成实验的 `test` 指标，时间统一尽量用秒表示。

相关表格：

- `all_results_summary.csv`
- `best_config_by_dataset.csv`
- `overall_time_comparison.csv`
- `all_results_ensemble_debug.json`

说明：

- `eval-maxplus` / `eval-noisyor` 来自 application 日志。
- `canonical` 按老格式目录解析：`head_mrr_*.p + tail_mrr_*.p + canonical.log 中的 Done`。
- `best_combination*` 配置已整体排除，不参与本报告。
- `ensemble_best_valid` 是逐 relation 在剩余配置中按 selected valid MRR 选模后的整体 test 汇总。
- `ensemble_best_test` 是逐 relation 在剩余配置中按 test MRR 选模后的整体 test 汇总（oracle 上界）。
- `ensemble_safe_valid` 是稳健版：优先 valid 选择，并在不稳定 relation 上回退到数据集级最佳单模型。
- `structural_rd__stage1 / structural_r2d3__stage1 / structural_r3d6__stage1` 是同一实验里的 stage1 test。
- 时间对比表中，RuleDep stage1/stage2 时间用 per-relation `epochs_trained` 比例估算，并按当前 `multiprocess=2` 除以 2；canonical 是串行老流程，不除以 2。

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

- 覆盖数据集数：`7`
- 有已完成 canonical 的数据集：`6`
- 有 ensemble-valid 的数据集：`7`
- 有 ensemble-safe 的数据集：`0`
- 有 ensemble-test 的数据集：`7`
