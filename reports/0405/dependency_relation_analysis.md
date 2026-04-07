# Dependency Relation-wise Analysis

本文档基于当前最优 structural 配置，对 7 个数据集做 relation 级分析，目标是回答：

1. dependency 在什么情况下成立
2. 哪些 relation 会显著受益
3. 这些 relation 有什么共同结构
4. 哪些 relation 不适合 dependency
5. 后续论文分析应该如何继续推进

## Data And Method

本次分析使用每个数据集全局最优的主 structural 配置：

- `FB15k-237` -> `structural_r2d3`
- `KG20C` -> `structural_none`
- `WN18RR` -> `structural_none`
- `YAGO3-10` -> `structural_r2d3`
- `codex-l` -> `structural_none`
- `codex-m` -> `structural_r3d6`
- `hetionet` -> `structural_r2d3`

对每个 `dataset, relation`，定义：

- `baseline = test_after_stage1.mrr`
- `final = test.mrr`
- `abs_gain = final - baseline`
- `rel_gain_pct = 100 * (final - baseline) / baseline`

relation 分三组：

- `positive`: `rel_gain_pct > 3`
- `neutral`: `-3 <= rel_gain_pct <= 3`
- `negative`: `rel_gain_pct < -3`

分析表还额外收集了：

- `train / valid / test triple count`
- `num_relation_rules`
- `num_relation_dependencies`
- `dep_per_rule`
- `B / Uc / Ud` 规则数与比例
- `filtered synergy / redundancy` 数量
- `best_valid_stage1 / best_valid_stage2`
- `selected_stage`
- learned type weights

核心输出文件：

- [relation_dependency_analysis.csv](/home/sy/RuleDep/reports/0405/relation_dependency_analysis.csv)
- [relation_gain_group_summary.csv](/home/sy/RuleDep/reports/0405/relation_gain_group_summary.csv)
- [relation_gain_dataset_summary.csv](/home/sy/RuleDep/reports/0405/relation_gain_dataset_summary.csv)
- [relation_gain_stage1_bucket_summary.csv](/home/sy/RuleDep/reports/0405/relation_gain_stage1_bucket_summary.csv)
- [relation_gain_dep_density_bucket_summary.csv](/home/sy/RuleDep/reports/0405/relation_gain_dep_density_bucket_summary.csv)
- [relation_type_weight_summary.csv](/home/sy/RuleDep/reports/0405/relation_type_weight_summary.csv)
- [relation_relative_gain_gt_3pct_best_structural.csv](/home/sy/RuleDep/reports/0405/relation_relative_gain_gt_3pct_best_structural.csv)
- [relation_relative_gain_lt_minus_3pct_best_structural.csv](/home/sy/RuleDep/reports/0405/relation_relative_gain_lt_minus_3pct_best_structural.csv)

图：

- [plot_gain_vs_stage1.png](/home/sy/RuleDep/reports/0405/plot_gain_vs_stage1.png)
- [plot_gain_vs_dep_density.png](/home/sy/RuleDep/reports/0405/plot_gain_vs_dep_density.png)
- [plot_stage1_bucket_summary.png](/home/sy/RuleDep/reports/0405/plot_stage1_bucket_summary.png)
- [plot_dep_density_bucket_summary.png](/home/sy/RuleDep/reports/0405/plot_dep_density_bucket_summary.png)
- [plot_dataset_gain_mix.png](/home/sy/RuleDep/reports/0405/plot_dataset_gain_mix.png)
- [plot_type_weight_summary.png](/home/sy/RuleDep/reports/0405/plot_type_weight_summary.png)

## High-level Findings

总共分析了 `425` 个 relation：

- `positive`: `36`
- `neutral`: `378`
- `negative`: `11`

这说明 dependency 的作用不是“几乎所有 relation 都涨”，而是非常明显的 relation-heterogeneous 现象：

- 多数 relation 最终变化不大
- 少量 relation 显著受益
- 更少量 relation 显著受损

这支持一个重要结论：

- dependency 不是 uniform feature
- dependency 更像 selective structural evidence

## Q1: Dependency 在什么情况下成立？

从 [relation_gain_group_summary.csv](/home/sy/RuleDep/reports/0405/relation_gain_group_summary.csv) 看：

- `positive` relation 的平均 `stage1_mrr = 0.32935`
- `neutral` relation 的平均 `stage1_mrr = 0.36342`
- `negative` relation 的平均 `stage1_mrr = 0.39544`

这说明 dependency 更容易在 `stage1` 还不算太强的 relation 上成立。

更直观看 [relation_gain_stage1_bucket_summary.csv](/home/sy/RuleDep/reports/0405/relation_gain_stage1_bucket_summary.csv)：

- `[0.2, 0.4)` 这个 bucket 的 `positive_ratio = 13.669%`，是四个桶里最高的
- 同时这个桶的 `avg_rel_gain_pct = 1.76412%` 也是最高的
- `[0.6, 1.0]` 这个强 baseline 区间，`avg_rel_gain_pct = -0.27492%`

这支持一个很强的 hypothesis：

- dependency 最适合“stage1 有一定能力但还未饱和”的 relation
- 如果 stage1 已经很强，dependency 更容易过调或重复放大已有证据

第二个条件是 dependency density。

从 [relation_gain_dep_density_bucket_summary.csv](/home/sy/RuleDep/reports/0405/relation_gain_dep_density_bucket_summary.csv) 看：

- `Q1` 依赖最稀疏时，`positive_ratio = 4.673%`
- `Q2` 时上升到 `11.321%`
- `Q3` 时仍然较高，为 `9.434%`
- `Q4` 最稠密时，`negative_ratio = 4.717%`，风险变高

这说明：

- dependency 太少时，信号不够
- dependency 适中时，最容易带来收益
- dependency 过多时，冗余和误导的风险会上升

## Q2: 哪些 relation 会显著受益？

显著受益 relation 的典型特征：

- `selected_stage = dependency`
- `stage1_mrr` 通常不在最高区间
- `dep_per_rule` 往往不低，但通常不是极端最大
- 往往存在较高比例的 unary-like rule

代表性正例见 [relation_relative_gain_gt_3pct_best_structural.csv](/home/sy/RuleDep/reports/0405/relation_relative_gain_gt_3pct_best_structural.csv)，例如：

- `FB15k-237`
  - `/award/award_winner/.../award_winner`
  - `/organization/organization/headquarters.../country`
  - `/tv/tv_program/languages`
- `YAGO3-10`
  - `influences`
- `hetionet`
  - `DrD`

几个最典型的例子：

- `/award/award_winner/.../award_winner`
  - `stage1_mrr = 0.23443`
  - `final = 0.37227`
  - `rel_gain = +58.80%`
- `DrD`
  - `stage1_mrr = 0.22244`
  - `final = 0.30250`
  - `rel_gain = +35.99%`

这些 relation 的共同点更像：

- 不是 stage1 已经解决得很好的 relation
- 关系语义本身容易由多条互补证据共同支持
- relation 常带有 compositional / multi-hop / many-to-many 味道

## Q3: 这些 relation 有什么共同结构？

从 group summary 看，正增益 relation 的结构特征是：

- `avg_dep_per_rule = 1.56103`
- `negative` relation 的 `avg_dep_per_rule = 2.12907`
- `neutral` relation 的 `avg_dep_per_rule = 1.24354`

这再次说明：

- 正增益 relation 常有一定 dependency 密度
- 但不是最密的那一批

再看 rule type ratio：

- `positive`
  - `B_ratio = 0.20979`
  - `Uc_ratio = 0.51922`
  - `Ud_ratio = 0.27046`
- `negative`
  - `B_ratio = 0.16507`
  - `Uc_ratio = 0.54270`
  - `Ud_ratio = 0.29157`

这说明：

- 单纯的 `B/Uc/Ud` 占比本身，不足以单独解释正负收益
- type composition 有信号，但不是唯一决定因素

更合理的理解是：

- rule type 提供了“结构风格”
- 真正决定 dependency 是否成立的，还要结合：
  - stage1 强弱
  - dependency density
  - relation 样本规模

## Q4: 哪些 relation 不适合 dependency？为什么？

当前 `negative` relation 一共 `11` 个，而且全部都是：

- `selected_stage = dependency`

也就是说，它们不是“未接受 stage2 的伪负例”，而是真正被接受但最终 test 更差的 relation。

代表性负例见 [relation_relative_gain_lt_minus_3pct_best_structural.csv](/home/sy/RuleDep/reports/0405/relation_relative_gain_lt_minus_3pct_best_structural.csv)，例如：

- `/media_common/netflix_genre/titles`
- `/base/locations/continents/countries_within`
- `/film/film/written_by`
- `participatedIn`

这些 relation 的共同点通常是：

- stage1 已经不弱，headroom 小
- dependency 密度偏高，容易出现过量累加
- 样本数较小，validation 选出来的 dependency stage 更容易在 test 上失配

## Q5: 这种方法更擅长完成哪些任务？

目前看更擅长：

- 需要多条规则共同支持的 relation
- 单条 rule 不够，但局部结构组合能补充信息的 relation
- stage1 还没饱和、但也不是完全无信号的 relation

不太擅长：

- stage1 已经很强的 relation
- dependency 极稠密、冗余明显的 relation
- 小样本且 validation-test mismatch 风险高的 relation

## Next Steps

如果继续推进这条线，最值得做的是：

1. 提高 positive relation 的覆盖率，而不是只追 overall 的小数点提升
2. 优先验证 `dependency density control`
3. 分析 `selected_stage = dependency` 但 test 下降的 relation，找出更稳的 acceptance rule
4. 继续比较 `structural_none / structural_rd / structural_r2d3 / structural_r3d6` 与新 RD 变体的作用边界
