# Dependency Relation-wise Analysis

本文档基于当前最优 structural 配置，对 7 个数据集做 relation 级分析，目标是回答：

1. dependency 在什么情况下成立
2. 哪些 relation 会显著受益
3. 这些 relation 有什么共同结构
4. 哪些 relation 不适合 dependency
5. 后续论文分析该如何继续推进

## Data And Method

本次分析使用每个数据集全局最优的 structural 配置：

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

本次 analysis table 还额外收集了：

- `train/valid/test triple count`
- `num_relation_rules`
- `num_relation_dependencies`
- `dep_per_rule`
- `B/Uc/Ud` 规则数与比例
- `filtered synergy / redundancy` 的 relation-local 数量
- `best_valid_stage1 / best_valid_stage2`
- `selected_stage`
- learned type weights

核心输出文件：

- [relation_dependency_analysis.csv](/home/sy/RuleDep/reports/relation_dependency_analysis.csv)
- [relation_gain_group_summary.csv](/home/sy/RuleDep/reports/relation_gain_group_summary.csv)
- [relation_gain_dataset_summary.csv](/home/sy/RuleDep/reports/relation_gain_dataset_summary.csv)
- [relation_gain_stage1_bucket_summary.csv](/home/sy/RuleDep/reports/relation_gain_stage1_bucket_summary.csv)
- [relation_gain_dep_density_bucket_summary.csv](/home/sy/RuleDep/reports/relation_gain_dep_density_bucket_summary.csv)
- [relation_type_weight_summary.csv](/home/sy/RuleDep/reports/relation_type_weight_summary.csv)

图：

- [plot_gain_vs_stage1.png](/home/sy/RuleDep/reports/plot_gain_vs_stage1.png)
- [plot_gain_vs_dep_density.png](/home/sy/RuleDep/reports/plot_gain_vs_dep_density.png)
- [plot_stage1_bucket_summary.png](/home/sy/RuleDep/reports/plot_stage1_bucket_summary.png)
- [plot_dep_density_bucket_summary.png](/home/sy/RuleDep/reports/plot_dep_density_bucket_summary.png)
- [plot_dataset_gain_mix.png](/home/sy/RuleDep/reports/plot_dataset_gain_mix.png)
- [plot_type_weight_summary.png](/home/sy/RuleDep/reports/plot_type_weight_summary.png)

## High-level Findings

总共分析了 `425` 个 relation。

- `positive`: `36`
- `neutral`: `378`
- `negative`: `11`

这说明 dependency 的作用不是“几乎所有 relation 都涨”，而是明显的 relation-heterogeneous 现象：

- 多数 relation 最终变化不大
- 少量 relation 显著受益
- 更少量 relation 显著受损

这是一个非常重要的结论，因为它支持：

- dependency 不是 uniform feature
- dependency 更像 selective structural evidence

## Q1: Dependency 在什么情况下成立？

从 [relation_gain_group_summary.csv](/home/sy/RuleDep/reports/relation_gain_group_summary.csv) 看：

- `positive` relation 的平均 `stage1_mrr = 0.32935`
- `neutral` relation 的平均 `stage1_mrr = 0.36342`
- `negative` relation 的平均 `stage1_mrr = 0.39544`

这说明 dependency 更容易在 `stage1` 还不算太强的 relation 上成立。

更直观看 [relation_gain_stage1_bucket_summary.csv](/home/sy/RuleDep/reports/relation_gain_stage1_bucket_summary.csv)：

- `[0.2,0.4)` 这个 bucket 的 `positive_ratio = 13.669%`，是四个桶里最高的
- 同时这个桶的 `avg_rel_gain_pct = 1.76412%` 也是最高的
- `[0.6,1.0]` 这个强 baseline 区间，`avg_rel_gain_pct = -0.27492%`

这支持一个很强的 hypothesis：

- dependency 最适合“stage1 有一定能力但还未饱和”的 relation
- 如果 stage1 已经很强，dependency 更容易过调或重复放大已有证据

第二个条件是 dependency density。

从 [relation_gain_dep_density_bucket_summary.csv](/home/sy/RuleDep/reports/relation_gain_dep_density_bucket_summary.csv) 看：

- `Q1` 依赖最稀疏时，`positive_ratio = 4.673%`
- `Q2` 时上升到 `11.321%`
- `Q3` 时仍然较高，为 `9.434%`
- `Q4` 最稠密时，`negative_ratio = 4.717%`，风险变高

这说明：

- dependency 太少时，信号不够
- dependency 适中时，最容易带来收益
- dependency 过多时，冗余和误导的风险会上升

所以当前最合理的结论是：

- dependency 成立的条件，不是“越多越好”
- 而是“中等难度 + 中等 dependency density”的 relation 上最容易成立

## Q2: 哪些 relation 会显著受益？

显著受益 relation 的典型特征：

- `selected_stage = dependency`
- `stage1_mrr` 通常不在最高区间
- `dep_per_rule` 往往不低，但通常不是极端最大
- 往往存在较高比例的 unary-like rule

top positive 例子见 [relation_dependency_analysis.csv](/home/sy/RuleDep/reports/relation_dependency_analysis.csv)，代表性 relation 包括：

- `FB15k-237`
  - `/award/award_winner/.../award_winner`
  - `/olympics/olympic_games/sports`
  - `/location/location/partially_contains`
  - `/tv/tv_program/languages`
  - `/film/film/distributors.../region`
- `YAGO3-10`
  - `influences`
- `hetionet`
  - `DrD`

其中几个最典型的正例：

- `/award/award_winner/.../award_winner`
  - `stage1_mrr = 0.23443`
  - `final = 0.37227`
  - `rel_gain = +58.80%`
- `/olympics/olympic_games/sports`
  - `stage1_mrr = 0.28666`
  - `final = 0.46603`
  - `rel_gain = +62.57%`
- `DrD`
  - `stage1_mrr = 0.22244`
  - `final = 0.30250`
  - `rel_gain = +35.99%`

这些 relation 的共同点更像：

- 不是 stage1 已经解决得很好的 relation
- 关系语义本身容易由多条互补证据共同支持
- relation 常带有 compositional / multi-hop / many-to-many 味道

换句话说，这类 relation 更像“需要证据组合”，而不是“单条强规则就能决定”。

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

这个全局对比说明：

- 单纯的 `B/Uc/Ud` 占比本身，不足以单独解释正负收益
- type composition 是有信号，但不是唯一决定因素

更合理的理解是：

- rule type 提供了“结构风格”
- 真正决定 dependency 是否成立的，还要结合：
  - stage1 强弱
  - dependency density
  - relation 样本规模

也就是说，共同结构不是单一类型标签，而是一个组合条件：

- 中等 baseline 难度
- 中等 dependency density
- unary rule 占比不低
- 关系语义允许互补证据发挥作用

## Q4: 哪些 relation 不适合 dependency？为什么？

当前 `negative` relation 只剩 `11` 个，而且全部都是：

- `selected_stage = dependency`

这说明它们是真正被最终系统选中了 dependency，却仍然比 stage1 差，不是伪负例。

典型负例：

- `FB15k-237`
  - `/base/locations/continents/countries_within`
  - `/film/film_set_designer/film_sets_designed`
  - `/film/film/written_by`
  - `/organization/non_profit_organization/registered_with...`
- `YAGO3-10`
  - `participatedIn`
  - `wroteMusicFor`
- `codex-m`
  - `P30`

这些 relation 的特征：

- 平均 `stage1_mrr = 0.39544`，高于 positive 组
- 平均 `dep_per_rule = 2.12907`，高于 positive 组
- 平均 `test_triple_count = 31.55`，显著小于 neutral 组的 `774.54`

所以最合理的解释是三条：

1. stage1 已经偏强，dependency 容易过度修正  
2. dependency 太密，容易引入冗余和误导  
3. test relation 太小，selection noise 更容易放大

最典型例子是：

- `/base/locations/continents/countries_within`
  - `stage1_mrr = 0.82011`
  - `test_triple_count = 5`
  - `dep_per_rule = 4.0`
  - 最终 `rel_gain = -35.45%`

这几乎就是一个 textbook 式失败模式：

- baseline 已经很高
- relation 太小
- dependency 又很密

这种 relation 很不适合再用 dependency 做强修正。

## Q5: type weight 有什么信息？

从 [relation_type_weight_summary.csv](/home/sy/RuleDep/reports/relation_type_weight_summary.csv) 看，数据集间差异非常明显。

### FB15k-237

- `rule_weight_B = 0.79286`
- `rule_weight_U = 0.93864`
- `dep_weight_BB = 1.06104`
- `dep_weight_BU = 1.06570`
- `dep_weight_UU = 1.06930`

解读：

- rule 整体被轻微压低
- dependency 整体被放大
- 说明该数据集更像是“dependency 有用，但只是温和增强”

### YAGO3-10

- `rule_weight_B = 1.04441`
- `rule_weight_U = 1.19994`
- `dep_weight_BB = 1.27414`
- `dep_weight_BU = 1.33274`
- `dep_weight_UU = 1.30529`

解读：

- rule 和 dependency 都被显著放大
- 尤其 unary rule 和 dependency 被明显重视
- 说明该数据集对结构组合更敏感

### codex-m

- `rule_weight_B = 0.83903`
- `rule_weight_Uc = 1.51961`
- `rule_weight_Ud = 1.09631`
- `dep_weight_B_Uc = 1.08676`
- `dep_weight_Uc_Uc = 1.11937`
- `dep_weight_Uc_Ud = 1.10711`

解读：

- `B` 被压低
- `Uc` 被强烈放大
- dependency 里偏 unary 的组合更被偏好

这说明 `codex-m` 的有效结构证据非常偏向 unary-like pattern，这与它在 `R3D6` 上最好是一致的。

### hetionet

- `rule_weight_B = 0.74138`
- `rule_weight_U = 0.78570`
- dependency type 权重几乎在 `1.0` 附近

解读：

- type model 在这里主要不是放大 dependency，而更像在整体压缩 rule contribution
- 说明 type bias 的作用方式也可以因数据集不同而完全不同

### 一个重要结论

`type weight` 的价值不是“稳定提高所有数据集表现”，而是：

- 暴露不同数据集的结构偏好
- 让我们观察 dependency 为什么在某些数据集有效，在另一些数据集不明显

这也解释了为什么：

- `codex-m` 最好的是 `R3D6`
- `FB15k-237 / YAGO3-10 / hetionet` 最好的是 `R2D3`
- `KG20C / WN18RR / codex-l` 最好的是 `RD`

## Which Datasets Show Stronger Dependency Heterogeneity?

从 [relation_gain_dataset_summary.csv](/home/sy/RuleDep/reports/relation_gain_dataset_summary.csv) 看：

### heterogeneity 最强

- `FB15k-237`
  - `positive = 24`
  - `negative = 7`
- `YAGO3-10`
  - `positive = 2`
  - `negative = 2`
- `hetionet`
  - `positive = 3`
  - `negative = 1`

这些数据集更像“dependency 很有用，但也更容易错”。

### dependency 效果较弱但更稳定

- `KG20C`
  - `positive = 1`
  - `negative = 0`
- `WN18RR`
  - `positive = 1`
  - `negative = 0`
- `codex-l`
  - `positive = 5`
  - `negative = 0`

这些数据集更像“dependency 不是完全没用，但提升更温和，也更稳定”。

### 一个特别有意思的数据集

- `codex-m`
  - `positive = 0`
  - `negative = 1`
  - 但整体最好 structural 仍是 `R3D6`

这说明 `codex-m` 的 gain 更像：

- 不是靠少量 relation 的巨大跳升
- 而是靠很多 relation 的小幅稳定修正

这也是为什么它会偏好更细的 type model。

## Main Hypotheses Supported By Data

当前数据最支持下面 4 个 hypothesis。

### H1. Dependency is conditionally useful rather than uniformly useful

支持。

因为：

- 只有 `36/425` relation 显著正增益
- `11/425` 显著负增益
- 绝大多数 relation 的变化接近 0

### H2. Dependency works best on medium-difficulty relations

支持。

最明显证据就是 stage1 bucket：

- `[0.2,0.4)` 最好
- `[0.6,1.0]` 平均反而转负

### H3. Dependency density has a non-monotonic effect

支持。

- 太稀疏时不够有用
- 中等时最有效
- 太稠密时负面风险上升

### H4. Type-aware modeling captures dataset-specific structural preference

支持。

因为不同数据集学到的 type weight 模式非常不同，而且与最优配置一致。

## What This Means For The Paper

当前结果不适合支撑这样一种 claim：

- “dependency universally gives a large global boost”

但非常适合支撑另一种更强、也更精确的 claim：

- “dependency is relation-conditional structural evidence”
- “global averaging hides when dependency truly works”
- “relation-wise and type-aware modeling are necessary to expose this heterogeneity”

这类 claim 的关键不是 overall 平均提升有多大，而是：

- 你能否解释 dependency 什么时候成立
- 什么时候不成立
- 为什么

当前分析已经为这个故事提供了比较强的实证支撑。

## Recommended Next Steps

如果后面要把这块往论文推进，我建议按这个顺序补强：

1. 加入 old canonical / global model 的最终对照
2. 对 positive / negative relation 做更细的 case study
3. 检查 relation 语义类别与 gain 的对应关系
4. 加一个简单的 predictor

可以做一个很小的 binary classifier 或 rule-based predictor，判断：

- 某个 relation 是否适合开启 dependency

输入只用：

- stage1_mrr
- dep_per_rule
- test_triple_count
- B/Uc/Ud ratio

如果这个 predictor 已经能预测出大部分正负 relation，那论文故事会更完整：

- 不只是“我们观察到了异质性”
- 而是“这种异质性是可刻画、可预测、可利用的”

## Summary

一句话总结：

- dependency 不是普适增益项，而是 relation-conditional 的结构证据
- 它最适合中等难度、dependency 密度适中的 relation
- 它不适合 stage1 已经很强、样本很小、dependency 很密的 relation
- type weight 的主要价值不是一致提高平均分，而是暴露不同数据集的结构偏好

这套结果已经足够支撑“深入 relation-wise analysis 是必要的”这一点，也说明论文主线应该从“平均提升有多大”转向“dependency 在什么条件下成立”。 
