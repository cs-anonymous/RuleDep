# Stage2 Gain Summary

这份目录用于汇总“在 rule aggregation 上加入 dependency 后，`stage2` 相对 `stage1` 的收益”。

## 口径修正

- 本目录现在统一使用：
  - `stage1 MRR = test_after_stage1`
  - `stage2 MRR = test_after_stage2`
- 不再把 `test_before_stage2` 当作 `stage1` 指标。
- 因此，部分早期文档里曾出现的 subset gain 结论被高估了，尤其是 `codex-m` 和 `codex-l` 的一些旧表。

## 到目前为止做过的尝试

1. 直接复用现有 `step5` 配置：
   - `synergy`
   - `redundancy`
   - `synergy + redundancy`
   - `sign_constraint_dependency`
   - `init_dep_with_lift`
   - `pos = auto_ratio / auto_sqrt`
2. relation-wise 子集分析：
   - 单 relation 命中
   - top-gain relations 累计到 `coverage > 30%`
3. query-wise 子集分析：
   - 精确 bidirectional replay
   - `active_candidate_count`
   - `total_active_pairs`
   - `hub_degree_sum`
4. dependency 重打分 / 归一化：
   - `global_max_count`
   - `global_sqrt_count`
   - `batch_max_count`
   - `batch_sqrt_count`
5. 更激进的 stage2：
   - relation-local candidate variants
   - nonzero dependency training
   - joint / freeze / pair / full 候选搜索
6. step4 方向：
   - 重新过滤 dependency
   - `min_train`
   - `dep_per_rule_multiplier`

## 主要结论

1. 用正确口径 `test_after_stage1 -> test_after_stage2` 重算后，当前已经确认完成的实验里，真正满足 `coverage > 30%` 且 `relative gain > 5%` 的数据集只有：
   - `KG20C`
   - `FB15k-237`
2. 之前一些看起来已经“过线”的结果，实质上是因为误用了 `test_before_stage2` 作为 stage1。
3. `codex-m`、`codex-l`、`WN18RR`、`YAGO3-10` 在修正口径后，当前都还没有达到 `coverage > 30%` 且 `gain > 5%`。
4. `YAGO3-10` 目前仍是最难的数据集：
   - 已完成的 relation-wise cumulative 最好结果只有 `44.22% coverage, +1.21%`
   - 已完成的 exact query-wise `global_max_count` 最好大覆盖结果是 `90.55% coverage, +0.31%`
   - 更激进的 `multivariant + global_max_count + joint training` 仍在继续跑
5. `valid` 提升但 `test` 下降不是偶发噪声，而是系统性现象。当前最重要的原因有：
   - stage2 接受条件使用的是 `best_valid_mrr`
   - 但 checkpoint 保存用的是 `best_valid_combined_raw`
   - `best_valid_mrr` 还会拼接 head / tail 各自的历史最优，不一定对应同一个 epoch
   - 训练目标是 pointwise BCE，而评估目标是 query-level MRR

## 当前推荐的读法

- 如果目的是写论文中的“当前已确认结果”，请先看：
  - [all_datasets_stage1_stage2_tables.md](/home/sy/2026/RuleDep/script/RuleDep/reports/stage2_gain_subsets/all_datasets_stage1_stage2_tables.md)
- 如果目的是看旧过程记录和中间推导，请再看：
  - [coverage30_gain5_status.md](/home/sy/2026/RuleDep/script/RuleDep/reports/stage2_gain_subsets/coverage30_gain5_status.md)
  - [paper_tables_coverage30_gain5.md](/home/sy/2026/RuleDep/script/RuleDep/reports/stage2_gain_subsets/paper_tables_coverage30_gain5.md)

## 当前最重要的未解决问题

- `YAGO3-10` 仍未达到 `coverage > 30%` 且 `gain > 5%`
- `codex-m` 和 `codex-l` 在修正后也没有过线
- 下一步最值得继续推进的是：
  - 修正 stage2 的 valid 接受逻辑
  - 避免 `epoch 0` / `combined_raw` 不涨却被接受
  - 继续做 `global_max_count + candidate variants + stronger step4 filtering`
