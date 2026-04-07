# 0407 Overall Results

本表汇总当前仓库里所有已完成实验的 `test` 指标，时间统一尽量用秒表示。

相关表格：

- `all_results_summary.csv`
- `best_config_by_dataset.csv`
- `all_results_ensemble_debug.json`

说明：

- `eval-maxplus` / `eval-noisyor` 来自 application 日志。
- `canonical` 按老格式目录解析：`head_mrr_*.p + tail_mrr_*.p + canonical.log 中的 Done`。
- `ensemble_best_valid` 是逐 relation 在所有非 canonical aggregation 中按 selected valid MRR 选模后的整体 test 汇总。
- `structural_rd__stage1 / structural_r2d3__stage1 / structural_r3d6__stage1` 是同一实验里的 stage1 test。

## Best Non-canonical Config Per Dataset

| Dataset | Best config | Best MRR | Ensemble | Canonical | Best app |
| --- | --- | ---: | ---: | ---: | ---: |
| KG20C | init_dep_with_lift | 0.233655 | 0.235229 | 0.239401 | 0.229772 |
| codex-m | structural_r3d6 | 0.343299 | 0.346220 | 0.345537 | 0.319487 |
| WN18RR | structural_rd | 0.502189 | 0.499851 | 0.499500 | 0.496968 |
| FB15k-237 | structural_r2d3 | 0.351555 | 0.356535 | 0.353919 | 0.337685 |
| codex-l | structural_dep_scale | 0.333772 | 0.335050 | - | 0.311458 |
| YAGO3-10 | structural_dep_scale | 0.577961 | 0.576309 | 0.573990 | 0.554384 |
| hetionet | structural_r2d3 | 0.369847 | 0.371678 | - | 0.230313 |

## Notes

- 覆盖数据集数：`8`
- 有已完成 canonical 的数据集：`5`
- 有 ensemble 的数据集：`7`
