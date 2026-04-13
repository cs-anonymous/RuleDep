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
- `ensemble_best_valid` 是逐 relation 在所有非 canonical aggregation 中按 selected valid MRR 选模后的整体 test 汇总。
- `structural_rd__stage1 / structural_r2d3__stage1 / structural_r3d6__stage1` 是同一实验里的 stage1 test。
- 时间对比表中，RuleDep stage1/stage2 时间用 per-relation `epochs_trained` 比例估算，并按当前 `multiprocess=2` 除以 2；canonical 是串行老流程，不除以 2。

## Best Non-canonical Config Per Dataset

| Dataset | Best config | Best MRR | Ensemble | Canonical | Best app |
| --- | --- | ---: | ---: | ---: | ---: |
| KG20C | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 0.233342 | 0.235229 | 0.239401 | 0.229772 |
| codex-m | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 0.343797 | 0.346341 | 0.345537 | 0.319487 |
| WN18RR | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 0.502195 | 0.500155 | 0.499500 | 0.496968 |
| FB15k-237 | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 0.353831 | 0.356932 | 0.353919 | 0.337685 |
| codex-l | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 0.334126 | 0.335194 | 0.331778 | 0.311458 |
| YAGO3-10 | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 0.576319 | 0.577210 | 0.573990 | 0.554384 |
| hetionet | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 0.374393 | 0.380345 | - | 0.350685 |

## Estimated Runtime Breakdown

| Dataset | RuleDep config | Canonical time (s) | Canonical source | RuleDep stage1 est. (s) | RuleDep stage2 est. (s) | RuleDep total est. (s) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| KG20C | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 2448.914000 | actual | 351.945981 | 185.490058 | 537.436039 |
| codex-m | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 7890.144000 | actual | 1210.189747 | 756.584342 | 1966.774089 |
| WN18RR | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 6391.193000 | actual | 1404.501697 | 610.851930 | 2015.353627 |
| FB15k-237 | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 15532.391000 | actual | 3855.068284 | 2193.966475 | 6049.034759 |
| codex-l | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 33603.206000 | actual | 2960.063111 | 2396.180951 | 5356.244062 |
| YAGO3-10 | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 18013.941000 | actual | 3232.138001 | 1297.850487 | 4529.988488 |
| hetionet | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | - | - | 5680.921000 | 3677.189873 | 9358.110872 |

## Notes

- 覆盖数据集数：`8`
- 有已完成 canonical 的数据集：`6`
- 有 ensemble 的数据集：`7`
