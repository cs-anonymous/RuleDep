# Coverage>30% Gain>5% Summary

判定标准：

- `coverage > 30%`
- `stage2` 相对 `stage1` 的 `relative gain > 5%`

当前按“每个数据集已确认的最佳子集结果”汇总如下。

| Dataset | Status | Best Confirmed Mode | Coverage | Relative Gain | Evidence |
| --- | --- | --- | ---: | ---: | --- |
| KG20C | pass | relation-wise, relation `3` | 0.3883 | 7.71% | [all_dataset_relation_hits_5pct.md](/home/sy/2026/RuleDep/script/RuleDep/reports/stage2_gain_subsets/all_dataset_relation_hits_5pct.md) |
| codex-m | pass | relation-wise, relation `2` | 0.3563 | 10.36% | [all_dataset_relation_hits_5pct.md](/home/sy/2026/RuleDep/script/RuleDep/reports/stage2_gain_subsets/all_dataset_relation_hits_5pct.md) |
| codex-l | pass | relation-wise, relation `2` | 0.3078 | 6.76% | [metric-2.json](/home/sy/2026/RuleDep/script/RuleDep/data/codex-l/aggregation/exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1/metric-2.json) |
| FB15k-237 | pass | relation-wise, top-gain relations cumulative to 30% coverage | 0.3026 | 5.50% | [coverage30_gain5_status.md](/home/sy/2026/RuleDep/script/RuleDep/reports/stage2_gain_subsets/coverage30_gain5_status.md) |
| WN18RR | fail | query-wise, `active_candidate_count >= 1` | 0.3705 | 0.06% | [query_subset_metrics_exact.json](/home/sy/2026/RuleDep/script/RuleDep/data/WN18RR/aggregation/exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0/query_subset_metrics_exact.json) |
| YAGO3-10 | fail | query-wise, `active_candidate_count >= 1` | 0.9055 | 0.13% | [query_subset_metrics_exact.json](/home/sy/2026/RuleDep/script/RuleDep/data/YAGO3-10/aggregation/exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_0/query_subset_metrics_exact.json) |
| wikidata5m | not_started | waiting for YAGO aggregation to finish | N/A | N/A | [coverage30_gain5_status.md](/home/sy/2026/RuleDep/script/RuleDep/reports/stage2_gain_subsets/coverage30_gain5_status.md) |

## Notes

- `FB15k-237` 不是单 relation 命中，而是 relation-wise 的累计子集命中。
- `WN18RR` 和 `YAGO3-10` 当前在大 coverage 子集上都明显达不到 `+5%`，说明仅靠换 split 不够，还需要新的 dependency 训练/归一化策略。
- `wikidata5m` 的 `tmux` 会话已经创建，但目前仍在等待 `YAGO3-10` 的新聚合实验完成后才会自动开始：
  - session: `wikidata5m_full`
