# Coverage>30% And Gain>5% Status

当前目标：在某个 `relation-wise` 或 `query-wise` 子集上，满足：

- `test coverage > 30%`
- `stage2` 相对 `stage1` 的 `relative gain > 5%`

## 已满足

| Dataset | Mode | Experiment | Subset | Coverage | Relative Gain |
| --- | --- | --- | --- | ---: | ---: |
| KG20C | relation-wise | `r3_mt3_mul1_sr_pair` | relation `3` | 0.3883 | 7.71% |
| codex-m | relation-wise | `exp_codexm_synergy_pair_and_type_lift_oldcfg` | relation `2` | 0.3563 | 10.36% |
| codex-l | relation-wise | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1` | relation `2` | 0.3078 | 6.76% |
| FB15k-237 | relation-wise | `exp-1_LinearAggregator_1_1_1_auto_sqrt_1_1_0` | top-gain relations cumulative to 30% coverage | 0.3026 | 5.50% |

## 仍未满足

| Dataset | Best confirmed mode so far | Coverage | Relative Gain | Notes |
| --- | --- | ---: | ---: | --- |
| WN18RR | query-wise (`active_candidate_count >= 1`) | 0.3705 | 0.06% | exact replay 已完成，现有 config 不够 |
| YAGO3-10 | query-wise (`active_candidate_count >= 1`) | 0.9055 | 0.13% | exact replay 已完成，现有 config 不够，正在跑 `global_max_count` |

## WN18RR Exact Query-Wise Result

文件：

- [query_subset_metrics_exact.json](/home/sy/2026/RuleDep/script/RuleDep/data/WN18RR/aggregation/exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0/query_subset_metrics_exact.json)

关键结果：

- 全部 query：`0.495024 -> 0.496478`，相对 `+0.294%`
- `active_candidate_count >= 1`：coverage `0.3705`，`0.847889 -> 0.848414`，相对 `+0.062%`
- `active_candidate_count >= 2`：coverage `0.2932`，`0.865339 -> 0.865739`，相对 `+0.046%`
- 对脚本内扫描的 4 类结构阈值（`total_active_pairs / active_candidate_count / max_active_pairs / hub_degree_sum`），当前没有找到 `>5%` 的 query-wise 子集

## In Progress

- `WN18RR`: 正在跑 `global_max_count` 新配置
  - [config.json](/home/sy/2026/RuleDep/script/RuleDep/data/WN18RR/aggregation/exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0_globalmax/config.json)
- `YAGO3-10`: `global_max_count` 新配置正在跑
  - [config.json](/home/sy/2026/RuleDep/script/RuleDep/data/YAGO3-10/aggregation/exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_0_globalmax/config.json)
- `YAGO3-10`: 当前主配置 exact query-wise 结果已完成
  - [query_subset_metrics_exact.json](/home/sy/2026/RuleDep/script/RuleDep/data/YAGO3-10/aggregation/exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_0/query_subset_metrics_exact.json)
- `FB15k-237`: 正在生成 matched `rule-only` baseline，后续会接 exact query-wise replay
  - [config.json](/home/sy/2026/RuleDep/script/RuleDep/data/FB15k-237/aggregation/exp_stage1_match_auto_sqrt_ruleonly/config.json)
