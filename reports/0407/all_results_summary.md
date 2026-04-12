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
| KG20C | init_dep_with_lift | 0.233655 | 0.235229 | 0.239401 | 0.229772 |
| codex-m | structural_r3d6 | 0.343299 | 0.346357 | 0.345537 | 0.319487 |
| WN18RR | structural_rd | 0.502189 | 0.500230 | 0.499500 | 0.496968 |
| FB15k-237 | best_combination | 0.353336 | 0.356980 | 0.353919 | 0.337685 |
| codex-l | structural_dep_scale | 0.333772 | 0.334876 | - | 0.311458 |
| YAGO3-10 | structural_dep_scale | 0.577961 | 0.576375 | 0.573990 | 0.554384 |
| hetionet | dep_scale_surprisal_init | 0.378467 | 0.379310 | - | 0.350685 |

## Estimated Runtime Breakdown

| Dataset | RuleDep config | Canonical time (s) | Canonical source | RuleDep stage1 est. (s) | RuleDep stage2 est. (s) | RuleDep total est. (s) |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| KG20C | init_dep_with_lift | 2448.914000 | actual | 225.424290 | 208.297789 | 433.722079 |
| codex-m | structural_r3d6 | 7890.144000 | actual | 1484.923102 | 801.030448 | 2285.953550 |
| WN18RR | structural_rd | 6391.193000 | actual | 894.579796 | 361.305494 | 1255.885290 |
| FB15k-237 | best_combination | 15532.391000 | actual | 3466.393995 | 1878.079785 | 5344.473781 |
| codex-l | structural_dep_scale | 33685.558747 | estimated | 2953.679018 | 2677.159422 | 5630.838440 |
| YAGO3-10 | structural_dep_scale | 18013.941000 | actual | 2294.825899 | 1439.794369 | 3734.620268 |
| hetionet | dep_scale_surprisal_init | - | - | 3676.193010 | 2731.743648 | 6407.936658 |

## Notes

- 覆盖数据集数：`8`
- 有已完成 canonical 的数据集：`5`
- 有 ensemble 的数据集：`7`
