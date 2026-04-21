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

实验口径如下：

- `baseline = structural_none stage1 test_mrr`
- `final = best_config final test_mrr`
- `rel_gain_pct = 100 * (final - baseline) / baseline`

各数据集使用的数据集级最优配置如下：

- `KG20C` -> `tg_r2d3__pos_auto_ratio__ri_conf__dn_per_rule_degree__dl1_1e-5`
- `codex-m` -> `tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5`
- `WN18RR` -> `structural_rd`
- `FB15k-237` -> `tg_r2d3__pos_auto_ratio__ri_conf__dn_none__dl1_1e-5`
- `codex-l` -> `tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5`
- `YAGO3-10` -> `structural_dep_scale`
- `hetionet` -> `dep_scale_surprisal_init`

## Main Findings

在全部 `425` 个 relation 中，`89` 个 relation 的相对增益超过 `3%`，`234` 个 relation 落在 `0%-3%` 的稳定提升区间，另有 `102` 个 relation 出现负迁移。整体上，dependency 的收益并不是均匀分布的，而更像是集中出现在一批“baseline 尚未饱和但结构信号较强”的 relation 上。

<p align="center"><img src="plot_gain_vs_stage1.png" alt="Gain vs Stage1" width="60%"></p>

<p align="center"><em>Figure 1: relation-level relative gain versus stage1 baseline MRR.</em></p>

<p align="center"><img src="plot_gain_vs_dep_density.png" alt="Gain vs Dependency Density" width="60%"></p>

<p align="center"><em>Figure 2: relation-level relative gain versus dependency density.</em></p>

## Positive vs Negative Relations

正增益 relation 的平均 stage1 MRR 为 `0.30652`，低于负增益 relation 的 `0.39046`；而其平均 dependency density 为 `1.46799`，高于负增益 relation 的 `1.38492`。 这说明 dependency 更容易帮助那些仍有提升空间、且规则交互相对更密的 relation。

进一步看中位数统计，正增益 relation 的典型规模特征如下：

- `train triples` 中位数：`756.00000`，负增益为 `859.00000`
- `test triples` 中位数：`49.00000`，负增益为 `42.00000`
- `#rules` 中位数：`1466.00000`，负增益为 `1692.00000`
- `#dependencies` 中位数：`1587.00000`，负增益为 `1583.00000`
- `dep_per_rule` 中位数：`1.16929`，负增益为 `1.08299`

按最终被选中的 stage 看，正增益 relation 更常落在 dependency stage：

- 正增益 relation：`dependency = 66`，`rule_only = 23`
- 负增益 relation：`dependency = 73`，`rule_only = 29`

<p align="center"><img src="plot_stage1_bucket_summary.png" alt="Stage1 Bucket Summary" width="60%"></p>

<p align="center"><em>Figure 3: average gain across stage1 baseline buckets.</em></p>

<p align="center"><img src="plot_dep_density_bucket_summary.png" alt="Dependency Density Bucket Summary" width="60%"></p>

<p align="center"><em>Figure 4: average gain across dependency-density buckets.</em></p>

## Dataset-level Pattern

不同数据集上的 relation-level 增益分布差异明显，说明 dependency 的收益不仅取决于单个 relation 的局部结构，也取决于整个数据集的规则池与候选依赖边的质量。

| Dataset | Config | Positive | Neutral | Negative | Avg gain pct |
| --- | --- | ---: | ---: | ---: | ---: |
| KG20C | tg_r2d3__pos_auto_ratio__ri_conf__dn_per_rule_degree__dl1_1e-5 | 3 | 0 | 2 | 3.09453 |
| codex-m | tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5 | 7 | 29 | 10 | 0.93932 |
| WN18RR | structural_rd | 1 | 8 | 2 | 0.90663 |
| FB15k-237 | tg_r2d3__pos_auto_ratio__ri_conf__dn_none__dl1_1e-5 | 67 | 107 | 63 | 2.43870 |
| codex-l | tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5 | 6 | 46 | 13 | 0.77517 |
| YAGO3-10 | structural_dep_scale | 4 | 28 | 5 | 1.41338 |
| hetionet | dep_scale_surprisal_init | 1 | 16 | 7 | 0.98626 |

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

表格文件 `relation_relative_gain_gt_3pct_best_config.csv` 给出了完整列表，下面列出最有代表性的正例及其规模信息：

- `FB15k-237` / `/olympics/olympic_games/sports`: baseline `0.26323`, final `0.41538`, rel_gain `57.80471%`, `train=664`, `test=13`, `rules=9650`, `deps=21010`, `dep_per_rule=2.17720`, `selected_stage=dependency`
- `FB15k-237` / `/award/award_winner/awards_won./award/award_honor/award_winner`: baseline `0.26534`, final `0.37804`, rel_gain `42.47328%`, `train=8423`, `test=41`, `rules=79116`, `deps=316464`, `dep_per_rule=4.00000`, `selected_stage=dependency`
- `FB15k-237` / `/film/film/distributors./film/film_film_distributor_relationship/region`: baseline `0.56497`, final `0.76833`, rel_gain `35.99515%`, `train=157`, `test=10`, `rules=916`, `deps=1814`, `dep_per_rule=1.98035`, `selected_stage=dependency`
- `FB15k-237` / `/film/film/film_festivals`: baseline `0.19895`, final `0.26012`, rel_gain `30.74384%`, `train=264`, `test=18`, `rules=827`, `deps=157`, `dep_per_rule=0.18984`, `selected_stage=rule_only`
- `FB15k-237` / `/music/genre/parent_genre`: baseline `0.16270`, final `0.20479`, rel_gain `25.87013%`, `train=678`, `test=97`, `rules=454`, `deps=188`, `dep_per_rule=0.41410`, `selected_stage=rule_only`
- `FB15k-237` / `/location/country/official_language`: baseline `0.27862`, final `0.35018`, rel_gain `25.68450%`, `train=225`, `test=16`, `rules=991`, `deps=3964`, `dep_per_rule=4.00000`, `selected_stage=rule_only`
- `FB15k-237` / `/language/human_language/countries_spoken_in`: baseline `0.30643`, final `0.38072`, rel_gain `24.24287%`, `train=335`, `test=25`, `rules=1883`, `deps=7532`, `dep_per_rule=4.00000`, `selected_stage=dependency`
- `FB15k-237` / `/award/award_nominee/award_nominations./award/award_nomination/award_nominee`: baseline `0.27075`, final `0.33360`, rel_gain `23.21116%`, `train=15989`, `test=214`, `rules=126142`, `deps=379530`, `dep_per_rule=3.00875`, `selected_stage=dependency`
- `YAGO3-10` / `dealsWith`: baseline `0.26473`, final `0.32516`, rel_gain `22.82627%`, `train=1302`, `test=7`, `rules=5173`, `deps=15381`, `dep_per_rule=2.97332`, `selected_stage=dependency`
- `FB15k-237` / `/base/popstra/celebrity/friendship./base/popstra/friendship/participant`: baseline `0.05312`, final `0.06226`, rel_gain `17.21281%`, `train=1511`, `test=32`, `rules=2911`, `deps=464`, `dep_per_rule=0.15940`, `selected_stage=rule_only`

从这些代表性正例可以看到两类模式：

- 一类是 `FB15k-237` 上那种高规则数、高 dependency 数的 dense relation，dependency 更像是在已有 rule pool 上做强组合。
- 另一类是 `hetionet: DrD` 这种规模并不大、但结构很明确的 relation，少量高质量 dependency 也能带来明显收益。

## Representative Negative Relations

负例通常对应两种风险：其一是 baseline 本身已经较强，额外 dependency 容易过修正；其二是 dependency 虽多，但质量不稳定，valid 侧的偏好无法稳定迁移到 test。

- `FB15k-237` / `/base/locations/continents/countries_within`: baseline `0.81961`, final `0.51394`, rel_gain `-37.29496%`, selected_stage `dependency`
- `FB15k-237` / `/music/artist/origin`: baseline `0.15966`, final `0.11940`, rel_gain `-25.22063%`, selected_stage `rule_only`
- `FB15k-237` / `/film/film/film_art_direction_by`: baseline `0.75000`, final `0.66667`, rel_gain `-11.11111%`, selected_stage `rule_only`
- `FB15k-237` / `/olympics/olympic_participating_country/medals_won./olympics/olympic_medal_honor/olympics`: baseline `0.37638`, final `0.34443`, rel_gain `-8.48860%`, selected_stage `rule_only`
- `FB15k-237` / `/base/popstra/celebrity/dated./base/popstra/dated/participant`: baseline `0.01377`, final `0.01275`, rel_gain `-7.40305%`, selected_stage `dependency`
- `FB15k-237` / `/film/director/film`: baseline `0.34902`, final `0.32469`, rel_gain `-6.97012%`, selected_stage `dependency`
- `FB15k-237` / `/award/award_category/disciplines_or_subjects`: baseline `0.67226`, final `0.62958`, rel_gain `-6.34906%`, selected_stage `rule_only`
- `FB15k-237` / `/base/biblioness/bibs_location/country`: baseline `0.55069`, final `0.51692`, rel_gain `-6.13321%`, selected_stage `dependency`
- `FB15k-237` / `/tv/tv_program/languages`: baseline `0.63164`, final `0.59499`, rel_gain `-5.80249%`, selected_stage `dependency`
- `FB15k-237` / `/organization/non_profit_organization/registered_with./organization/non_profit_registration/registering_agency`: baseline `0.77494`, final `0.73577`, rel_gain `-5.05415%`, selected_stage `dependency`
