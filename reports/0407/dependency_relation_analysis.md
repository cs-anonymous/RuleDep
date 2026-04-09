# 0407 Relation-wise Analysis

本节关注一个更细粒度的问题：当我们在数据集级别选定最优 aggregation 配置之后，dependency 究竟改善了哪些 relation，又在哪些 relation 上带来了负迁移。

相关表格：

- `relation_dependency_analysis.csv`
- `relation_gain_group_summary.csv`
- `relation_gain_dataset_summary.csv`
- `relation_gain_stage1_bucket_summary.csv`
- `relation_gain_dep_density_bucket_summary.csv`
- `relation_relative_gain_gt_3pct_best_config.csv`
- `relation_relative_gain_lt_0_best_config.csv`
- `relation_positive_examples_gt3_dependency.csv`
- `relation_positive_examples_gt3_dependency.md`

实验口径如下：

- `baseline = structural_none stage1 test_mrr`
- `final = best_config final test_mrr`
- `rel_gain_pct = 100 * (final - baseline) / baseline`

各数据集使用的数据集级最优配置如下：

- `KG20C` -> `init_dep_with_lift`
- `codex-m` -> `structural_r3d6`
- `WN18RR` -> `structural_rd`
- `FB15k-237` -> `best_combination`
- `codex-l` -> `structural_dep_scale`
- `YAGO3-10` -> `structural_dep_scale`
- `hetionet` -> `best_combination`

## Main Findings

在全部 `425` 个 relation 中，`83` 个 relation 的相对增益超过 `3%`，`217` 个 relation 落在 `0%-3%` 的稳定提升区间，另有 `125` 个 relation 出现负迁移。整体上，dependency 的收益并不是均匀分布的，而更像是集中出现在一批“baseline 尚未饱和但结构信号较强”的 relation 上。

<p align="center"><img src="plot_gain_vs_stage1.png" alt="Gain vs Stage1" width="60%"></p>

<p align="center"><em>Figure 1: relation-level relative gain versus stage1 baseline MRR.</em></p>

<p align="center"><img src="plot_gain_vs_dep_density.png" alt="Gain vs Dependency Density" width="60%"></p>

<p align="center"><em>Figure 2: relation-level relative gain versus dependency density.</em></p>

## Positive vs Negative Relations

正增益 relation 的平均 stage1 MRR 为 `0.27965`，低于负增益 relation 的 `0.42066`；而其平均 dependency density 为 `1.23303`，高于负增益 relation 的 `1.59494`。 这说明 dependency 更容易帮助那些仍有提升空间、且规则交互相对更密的 relation。

进一步看中位数统计，正增益 relation 的典型规模特征如下：

- `train triples` 中位数：`797.00000`，负增益为 `664.00000`
- `test triples` 中位数：`41.00000`，负增益为 `36.00000`
- `#rules` 中位数：`1396.00000`，负增益为 `1211.00000`
- `#dependencies` 中位数：`1283.00000`，负增益为 `1961.00000`
- `dep_per_rule` 中位数：`0.77006`，负增益为 `1.18426`

按最终被选中的 stage 看，正增益 relation 更常落在 dependency stage：

- 正增益 relation：`dependency = 56`，`rule_only = 27`
- 负增益 relation：`dependency = 81`，`rule_only = 44`

<p align="center"><img src="plot_stage1_bucket_summary.png" alt="Stage1 Bucket Summary" width="60%"></p>

<p align="center"><em>Figure 3: average gain across stage1 baseline buckets.</em></p>

<p align="center"><img src="plot_dep_density_bucket_summary.png" alt="Dependency Density Bucket Summary" width="60%"></p>

<p align="center"><em>Figure 4: average gain across dependency-density buckets.</em></p>

## Dataset-level Pattern

不同数据集上的 relation-level 增益分布差异明显，说明 dependency 的收益不仅取决于单个 relation 的局部结构，也取决于整个数据集的规则池与候选依赖边的质量。

| Dataset | Config | Positive | Neutral | Negative | Avg gain pct |
| --- | --- | ---: | ---: | ---: | ---: |
| KG20C | init_dep_with_lift | 1 | 3 | 1 | 2.18316 |
| codex-m | structural_r3d6 | 4 | 30 | 12 | 0.55821 |
| WN18RR | structural_rd | 1 | 8 | 2 | 0.90663 |
| FB15k-237 | best_combination | 62 | 92 | 83 | 1.66240 |
| codex-l | structural_dep_scale | 4 | 48 | 13 | 0.57201 |
| YAGO3-10 | structural_dep_scale | 4 | 28 | 5 | 1.41338 |
| hetionet | best_combination | 7 | 8 | 9 | 2.84077 |

<p align="center"><img src="plot_dataset_gain_mix.png" alt="Dataset Gain Mix" width="60%"></p>

<p align="center"><em>Figure 5: positive, neutral, and negative relation counts across datasets.</em></p>

<p align="center"><img src="plot_type_weight_summary.png" alt="Type Weight Summary" width="60%"></p>

<p align="center"><em>Figure 6: average learned type weights associated with relation-level gain.</em></p>

## Representative Positive Relations

代表性正例并不一定是训练样本最多的 relation。更常见的模式是：baseline 已经有可用规则信号，但尚未饱和；同时规则数与 dependency 数达到一定规模，从而允许模型通过 rule interaction 做进一步修正。

- baseline stage1 还没有饱和，但已经有一定 rule 信号
- 往往拥有中等到偏高的 rule 数和 dependency 数
- 很多例子最终还是选择了 `dependency` stage，而不是只靠更强的 stage1
- 更适合被多条互补规则共同支持

更细的一张表已经单独放到 `relation_positive_examples_gt3_dependency.{csv,md}` 中。那张表只保留真正满足以下条件的 relation：

- 使用该数据集当前最佳已完成配置
- `selected_stage = dependency`
- 相对 `test_after_stage1` 的最终 `test.mrr` 提升 `> 3%`

也就是说，那张表排除了所有 “最终还是回退到 rule_only” 的伪正例，只保留真正因为 dependency stage 被选中、且带来显著 test 提升的 relation。

从这批严格正例可以看到几条更稳定的模式：

- `FB15k-237` 的正例大量出现在 mediator/CVT 风格关系上，例如 distributor、award、team roster、organization role。这类关系通常需要多个局部槽位同时命中，dependency 更像是在做“多规则共识”。
- `YAGO3-10` 的强正例更偏事件参与、地点定位与语义关联，如 `dealsWith / participatedIn / isLocatedIn`。它们通常需要来自不同 path 的证据共同成立。
- `WN18RR` 的正例集中在 `has_part / member_meronym` 这类层级或部件关系上，说明 dependency 对层级一致性较强的 one-to-many 关系更自然。
- `hetionet` 的强正例是 `DrD` 和 `DpS`，都属于多跳生物医学关系；这里 dependency 的作用更像是要求多个生物学信号同时支持同一个候选。
- `codex-l` 虽然只有少量正例，但都出现在 baseline 较低、dependency-per-rule 也不高的 relation 上，说明少量高精度 dependency 就足以改变排序。

## Representative Negative Relations

负例通常对应两种风险：其一是 baseline 本身已经较强，额外 dependency 容易过修正；其二是 dependency 虽多，但质量不稳定，valid 侧的偏好无法稳定迁移到 test。

- `FB15k-237` / `/base/locations/continents/countries_within`: baseline `0.81961`, final `0.51488`, rel_gain `-37.17981%`, selected_stage `dependency`
- `FB15k-237` / `/music/artist/origin`: baseline `0.15966`, final `0.11940`, rel_gain `-25.22008%`, selected_stage `rule_only`
- `FB15k-237` / `/sports/sports_team/roster./baseball/baseball_roster_position/position`: baseline `0.64444`, final `0.55833`, rel_gain `-13.36207%`, selected_stage `dependency`
- `FB15k-237` / `/film/film/film_art_direction_by`: baseline `0.75000`, final `0.66667`, rel_gain `-11.11111%`, selected_stage `rule_only`
- `FB15k-237` / `/music/group_member/membership./music/group_membership/role`: baseline `0.25564`, final `0.22805`, rel_gain `-10.78991%`, selected_stage `dependency`
- `FB15k-237` / `/organization/non_profit_organization/registered_with./organization/non_profit_registration/registering_agency`: baseline `0.77494`, final `0.69759`, rel_gain `-9.98077%`, selected_stage `dependency`
- `FB15k-237` / `/location/location/partially_contains`: baseline `0.32828`, final `0.29660`, rel_gain `-9.65049%`, selected_stage `rule_only`
- `FB15k-237` / `/celebrities/celebrity/celebrity_friends./celebrities/friendship/friend`: baseline `0.06144`, final `0.05588`, rel_gain `-9.04255%`, selected_stage `rule_only`
- `FB15k-237` / `/olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/olympics`: baseline `0.37638`, final `0.34256`, rel_gain `-8.98724%`, selected_stage `rule_only`
- `FB15k-237` / `/education/educational_institution/students_graduates./education/education/major_field_of_study`: baseline `0.23666`, final `0.21592`, rel_gain `-8.76611%`, selected_stage `rule_only`
