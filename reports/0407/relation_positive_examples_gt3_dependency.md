# 0407 Positive Dependency Cases (>3%)

本表只保留满足以下条件的 relation：

- 使用该数据集当前最佳已完成配置
- `selected_stage = dependency`
- 相对 `test_after_stage1` 的最终 `test.mrr` 提升 `> 3%`

相关文件：

- `relation_positive_examples_gt3_dependency.csv`
- `relation_case_study_examples.csv`

## Best Config By Dataset

- `FB15k-237` -> `best_combination`
- `KG20C` -> `init_dep_with_lift`
- `WN18RR` -> `structural_rd`
- `YAGO3-10` -> `structural_dep_scale`
- `codex-l` -> `structural_dep_scale`
- `codex-m` -> `structural_r3d6`
- `hetionet` -> `best_combination`

## High-level Pattern

当前一共有 `33` 个 relation 满足 `selected_stage=dependency` 且相对 stage1 提升超过 `3%`。这些正例并不均匀分布：`FB15k-237` 最多，随后是 `YAGO3-10 / codex-l / hetionet / WN18RR / KG20C`；`codex-m` 在当前最佳配置下没有满足这个阈值的 relation。

从映射方向上看，正例更多集中在 `many-to-many` 和 `many-to-one` 关系上。这类关系往往有多条部分正确的规则同时激活，dependency 可以把“共同成立”的结构信号从简单加分提升为更强的排序证据。

## Full Table

| dataset | best_config | relation_name | relation_gloss | relation_gloss_zh | mapping_direction | baseline_mrr | baseline_h1 | final_mrr | final_h1 | rel_mrr_gain_pct | test_triple_count | num_relation_rules | num_relation_dependencies | dependency_per_rule | why_dependency_helps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FB15k-237 | best_combination | /military/military_combatant/military_conflicts./military/military_combatant_group/combatants | military -> military_combatant -> military_conflicts. -> military -> military_combatant_group -> combatants | 军事战斗方参与的军事冲突中的作战方/参战者 | many-to-many | 0.25477 | 0.02941 | 0.32488 | 0.05882 | 27.51959 | 17 | 28658 | 80136 | 2.79629 | Mediator/CVT-style relations have multiple complementary subpaths; dependency can reward co-firing rules that jointly identify the same role slot. |
| FB15k-237 | best_combination | /film/film/distributors./film/film_film_distributor_relationship/region | film -> film -> distributors. -> film -> film_film_distributor_relationship -> region | 电影发行对应的地区 | many-to-one | 0.57264 | 0.5 | 0.65717 | 0.55 | 14.76032 | 10 | 916 | 1814 | 1.98035 | Mediator/CVT-style relations have multiple complementary subpaths; dependency can reward co-firing rules that jointly identify the same role slot. |
| FB15k-237 | best_combination | /film/film/distributors./film/film_film_distributor_relationship/film_distribution_medium | film -> film -> distributors. -> film -> film_film_distributor_relationship -> film_distribution_medium | 电影发行对应的发行介质 | many-to-one | 0.54608 | 0.35526 | 0.61232 | 0.48684 | 12.13082 | 38 | 894 | 2178 | 2.43624 | Mediator/CVT-style relations have multiple complementary subpaths; dependency can reward co-firing rules that jointly identify the same role slot. |
| FB15k-237 | best_combination | /sports/pro_athlete/teams./sports/sports_team_roster/team | sports -> pro_athlete -> teams. -> sports -> sports_team_roster -> team | 职业运动员所属球队 | many-to-many | 0.16194 | 0.03125 | 0.18028 | 0.03125 | 11.32512 | 16 | 1504 | 181 | 0.12035 | Mediator/CVT-style relations have multiple complementary subpaths; dependency can reward co-firing rules that jointly identify the same role slot. |
| FB15k-237 | best_combination | /language/human_language/countries_spoken_in | language -> human_language -> countries_spoken_in | 该语言使用/通行的国家 | many-to-many | 0.33496 | 0.2 | 0.35911 | 0.24 | 7.2091 | 25 | 1883 | 7532 | 4.0 | Dependency helps when multiple partially informative rules fire together and their joint pattern is more reliable than either rule alone. |
| FB15k-237 | best_combination | /location/location/adjoin_s./location/adjoining_relationship/adjoins | location -> location -> adjoin_s. -> location -> adjoining_relationship -> adjoins | 地点相邻的地点 | many-to-many | 0.40048 | 0.18182 | 0.42897 | 0.21818 | 7.11557 | 55 | 5919 | 9103 | 1.53793 | Mediator/CVT-style relations have multiple complementary subpaths; dependency can reward co-firing rules that jointly identify the same role slot. |
| FB15k-237 | best_combination | /tv/tv_network/programs./tv/tv_network_duration/program | tv -> tv_network -> programs. -> tv -> tv_network_duration -> program | 电视网络播出的节目 | one-to-many | 0.18587 | 0.05769 | 0.19856 | 0.07692 | 6.82793 | 26 | 369 | 564 | 1.52846 | Mediator/CVT-style relations have multiple complementary subpaths; dependency can reward co-firing rules that jointly identify the same role slot. |
| FB15k-237 | best_combination | /award/award_winning_work/awards_won./award/award_honor/award | award -> award_winning_work -> awards_won. -> award -> award_honor -> award | 获奖作品所获得的奖项 | many-to-many | 0.19642 | 0.11017 | 0.20912 | 0.12712 | 6.46671 | 118 | 19741 | 40669 | 2.06013 | Mediator/CVT-style relations have multiple complementary subpaths; dependency can reward co-firing rules that jointly identify the same role slot. |
| FB15k-237 | best_combination | /people/person/gender | people -> person -> gender | 人物性别 | many-to-one | 0.53436 | 0.48394 | 0.56825 | 0.53555 | 6.34245 | 436 | 7376 | 3062 | 0.41513 | This is close to an attribute-selection task; dependency helps enforce type compatibility and suppress conflicting candidates. |
| FB15k-237 | best_combination | /sports/sports_position/players./sports/sports_team_roster/team | sports -> sports_position -> players. -> sports -> sports_team_roster -> team | 某体育位置对应球员所在球队 | many-to-many | 0.37122 | 0.25 | 0.39189 | 0.27273 | 5.56679 | 22 | 19359 | 39691 | 2.05026 | Mediator/CVT-style relations have multiple complementary subpaths; dependency can reward co-firing rules that jointly identify the same role slot. |
| FB15k-237 | best_combination | /organization/organization/headquarters./location/mailing_address/country | organization -> organization -> headquarters. -> location -> mailing_address -> country | 组织总部所在国家 | many-to-one | 0.45383 | 0.42857 | 0.47862 | 0.42857 | 5.46238 | 7 | 580 | 2033 | 3.50517 | Mediator/CVT-style relations have multiple complementary subpaths; dependency can reward co-firing rules that jointly identify the same role slot. |
| FB15k-237 | best_combination | /olympics/olympic_games/participating_countries | olympics -> olympic_games -> participating_countries | 奥运会参赛国家 | many-to-many | 0.42698 | 0.30435 | 0.44992 | 0.32609 | 5.37442 | 23 | 8960 | 35840 | 4.0 | Dependency helps when multiple partially informative rules fire together and their joint pattern is more reliable than either rule alone. |
| FB15k-237 | best_combination | /film/film/music | film -> film -> music | 电影配乐/音乐 | many-to-one | 0.20497 | 0.13043 | 0.21385 | 0.14674 | 4.33582 | 92 | 2463 | 1964 | 0.7974 | This is close to an attribute-selection task; dependency helps enforce type compatibility and suppress conflicting candidates. |
| FB15k-237 | best_combination | /location/statistical_region/places_exported_to./location/imports_and_exports/exported_to | location -> statistical_region -> places_exported_to. -> location -> imports_and_exports -> exported_to | 统计区域出口到的地区/国家 | many-to-many | 0.18217 | 0.07692 | 0.19005 | 0.07692 | 4.32147 | 13 | 2606 | 4795 | 1.83998 | Mediator/CVT-style relations have multiple complementary subpaths; dependency can reward co-firing rules that jointly identify the same role slot. |
| FB15k-237 | best_combination | /film/film/produced_by | film -> film -> produced_by | 电影制片方/制作公司 | many-to-one | 0.38025 | 0.26667 | 0.3956 | 0.30833 | 4.035 | 60 | 2937 | 6035 | 2.05482 | This is close to an attribute-selection task; dependency helps enforce type compatibility and suppress conflicting candidates. |
| FB15k-237 | best_combination | /award/award_nominee/award_nominations./award/award_nomination/award | award -> award_nominee -> award_nominations. -> award -> award_nomination -> award | 被提名者所提名的奖项 | many-to-many | 0.20622 | 0.09653 | 0.21444 | 0.10825 | 3.98238 | 1067 | 50398 | 97814 | 1.94083 | Mediator/CVT-style relations have multiple complementary subpaths; dependency can reward co-firing rules that jointly identify the same role slot. |
| FB15k-237 | best_combination | /award/award_ceremony/awards_presented./award/award_honor/honored_for | award -> award_ceremony -> awards_presented. -> award -> award_honor -> honored_for | 颁奖典礼表彰的作品/事项 | many-to-many | 0.39314 | 0.2686 | 0.40724 | 0.2686 | 3.5846 | 121 | 2200 | 1367 | 0.62136 | Mediator/CVT-style relations have multiple complementary subpaths; dependency can reward co-firing rules that jointly identify the same role slot. |
| FB15k-237 | best_combination | /government/legislative_session/members./government/government_position_held/legislative_sessions | government -> legislative_session -> members. -> government -> government_position_held -> legislative_sessions | 立法会期的成员/议员所属会期 | many-to-many | 0.39182 | 0.19231 | 0.40504 | 0.19231 | 3.37349 | 13 | 8311 | 17090 | 2.05631 | Mediator/CVT-style relations have multiple complementary subpaths; dependency can reward co-firing rules that jointly identify the same role slot. |
| FB15k-237 | best_combination | /education/educational_institution/colors | education -> educational_institution -> colors | 教育机构代表颜色 | many-to-many | 0.22118 | 0.10556 | 0.22834 | 0.12222 | 3.23847 | 90 | 1481 | 4237 | 2.8609 | Dependency helps when multiple partially informative rules fire together and their joint pattern is more reliable than either rule alone. |
| FB15k-237 | best_combination | /organization/role/leaders./organization/leadership/organization | organization -> role -> leaders. -> organization -> leadership -> organization | 某角色对应的领导所属组织 | one-to-many | 0.45599 | 0.37143 | 0.47029 | 0.37857 | 3.13635 | 70 | 971 | 3884 | 4.0 | Mediator/CVT-style relations have multiple complementary subpaths; dependency can reward co-firing rules that jointly identify the same role slot. |
| KG20C | init_dep_with_lift | author_write_paper | author write paper | 作者撰写论文 | many-to-many | 0.23051 | 0.14277 | 0.24791 | 0.15422 | 7.5477 | 830 | 16644 | 3659 | 0.21984 | Only a small set of dependencies is active, suggesting the gain comes from a few high-precision interactions rather than dense accumulation. |
| WN18RR | structural_rd | _has_part | has part | 具有组成部分 | one-to-many | 0.20516 | 0.13372 | 0.21926 | 0.14535 | 6.87496 | 172 | 5002 | 1372 | 0.27429 | Hierarchical or roster-style relations benefit when several structural cues agree on the same expansion. |
| WN18RR | structural_rd | _member_meronym | member meronym | 成员-整体关系 | one-to-many | 0.31793 | 0.2253 | 0.32862 | 0.23715 | 3.36086 | 253 | 2129 | 591 | 0.2776 | Hierarchical or roster-style relations benefit when several structural cues agree on the same expansion. |
| YAGO3-10 | structural_dep_scale | dealsWith | dealsWith | 涉及/处理 | many-to-many | 0.26473 | 0.14286 | 0.32516 | 0.14286 | 22.82627 | 7 | 5173 | 15381 | 2.97332 | Dependency helps when multiple partially informative rules fire together and their joint pattern is more reliable than either rule alone. |
| YAGO3-10 | structural_dep_scale | diedIn | diedIn | 逝世于 | many-to-one | 0.18 | 0.11224 | 0.20078 | 0.13265 | 11.54464 | 49 | 1492 | 960 | 0.64343 | This is close to an attribute-selection task; dependency helps enforce type compatibility and suppress conflicting candidates. |
| YAGO3-10 | structural_dep_scale | participatedIn | participatedIn | 参与于 | many-to-many | 0.33396 | 0.23684 | 0.36673 | 0.26316 | 9.8124 | 19 | 8616 | 15765 | 1.82974 | Dependency helps when multiple partially informative rules fire together and their joint pattern is more reliable than either rule alone. |
| YAGO3-10 | structural_dep_scale | isLocatedIn | isLocatedIn | 位于 | many-to-many | 0.38799 | 0.31022 | 0.40666 | 0.32847 | 4.81417 | 411 | 20377 | 16752 | 0.8221 | Dependency helps when multiple partially informative rules fire together and their joint pattern is more reliable than either rule alone. |
| codex-l | structural_dep_scale | P749 | parent organization or unit | 上级组织/母组织 | many-to-one | 0.27503 | 0.16667 | 0.29319 | 0.22222 | 6.60632 | 18 | 85 | 14 | 0.16471 | This is close to an attribute-selection task; dependency helps enforce type compatibility and suppress conflicting candidates. |
| codex-l | structural_dep_scale | P737 | influenced by | 受……影响 | many-to-many | 0.11394 | 0.05 | 0.11945 | 0.05 | 4.83645 | 90 | 622 | 13 | 0.0209 | Baseline is moderate/low, so there is headroom; a few corroborating dependencies can produce a noticeable ranking jump. |
| codex-l | structural_dep_scale | P161 | cast member | 演员/出演成员 | many-to-many | 0.08128 | 0.05004 | 0.08433 | 0.05367 | 3.75152 | 1239 | 7641 | 475 | 0.06216 | Baseline is moderate/low, so there is headroom; a few corroborating dependencies can produce a noticeable ranking jump. |
| codex-l | structural_dep_scale | P159 | headquarters location | 总部所在地 | many-to-one | 0.34518 | 0.25 | 0.35593 | 0.26562 | 3.11437 | 32 | 204 | 85 | 0.41667 | This is close to an attribute-selection task; dependency helps enforce type compatibility and suppress conflicting candidates. |
| hetionet | best_combination | DrD | Disease resembles Disease | 疾病与疾病相似 | many-to-many | 0.26286 | 0.14516 | 0.30949 | 0.17742 | 17.74063 | 62 | 583 | 182 | 0.31218 | Biomedical similarity depends on multiple corroborating gene/symptom paths; dependency helps require co-support instead of a single noisy rule. |
| hetionet | best_combination | DpS | Disease presents Symptom | 疾病表现出症状 | many-to-many | 0.23266 | 0.11337 | 0.24077 | 0.11773 | 3.48682 | 344 | 3490 | 354 | 0.10143 | Disease-symptom links are multi-causal; dependency helps combine complementary symptom-generation rules. |

## Dataset-level Interpretation

### FB15k-237

- best config: `best_combination`
- positive dependency cases: `20`
- average relative gain: `7.30544%`
- average stage1 baseline MRR: `0.34376`
- average dependency-per-rule: `2.13010`
- mapping mix: `{'many-to-many': 12, 'many-to-one': 6, 'one-to-many': 2}`

这些正例大量出现在带 mediator path 的 Freebase 关系上，例如 distributor、award、team roster、organization role 等。它们的共同点是：单条规则通常只能命中某个局部槽位，而 dependency 能把几个互补的槽位约束同时纳入排序。

### KG20C

- best config: `init_dep_with_lift`
- positive dependency cases: `1`
- average relative gain: `7.54770%`
- average stage1 baseline MRR: `0.23051`
- average dependency-per-rule: `0.21984`
- mapping mix: `{'many-to-many': 1}`

KG20C 只有一个强正例，说明 dependency 的收益更集中在个别 schema 明确、规则互补性强的 relation 上，而不是普遍现象。

### WN18RR

- best config: `structural_rd`
- positive dependency cases: `2`
- average relative gain: `5.11791%`
- average stage1 baseline MRR: `0.26155`
- average dependency-per-rule: `0.27594`
- mapping mix: `{'one-to-many': 2}`

WN18RR 的正例集中在层级/部件关系，这类 one-to-many 关系本来就适合依靠多条语义相近的规则做一致性加强。

### YAGO3-10

- best config: `structural_dep_scale`
- positive dependency cases: `4`
- average relative gain: `12.24937%`
- average stage1 baseline MRR: `0.29167`
- average dependency-per-rule: `1.56715`
- mapping mix: `{'many-to-many': 3, 'many-to-one': 1}`

YAGO3-10 的正例更像事件参与、地理定位和语义关联任务。它们往往需要来自不同 path 的佐证共同成立，dependency 在这里像一个“共识增强器”。

### codex-l

- best config: `structural_dep_scale`
- positive dependency cases: `4`
- average relative gain: `4.57716%`
- average stage1 baseline MRR: `0.20386`
- average dependency-per-rule: `0.16611`
- mapping mix: `{'many-to-one': 2, 'many-to-many': 2}`

codex-l 当前正例不多，但都来自 baseline 较低且依赖密度不高的 relation，说明少量高精度 dependency 就足以推动排序。由于本地只有 `Pxx` 代码，没有更细标签，这里更像是一个“结构有效但语义标签缺失”的案例。

### hetionet

- best config: `best_combination`
- positive dependency cases: `2`
- average relative gain: `10.61372%`
- average stage1 baseline MRR: `0.24776`
- average dependency-per-rule: `0.20681`
- mapping mix: `{'many-to-many': 2}`

Hetionet 的强正例是 `DrD` 和 `DpS`。它们都属于多跳生物医学关系，正确候选通常需要多个生物学信号一起支持，因此 dependency 比单条规则更自然。

## Case Study: Query-level Flips

我们进一步对三个代表关系做了更严格的 query 级检查：

- `FB15k-237 / /film/film/distributors./film/film_film_distributor_relationship/region`
- `YAGO3-10 / dealsWith`
- `hetionet / DrD`

标准是：

- stage1 下该 query 预测失败（gold rank `> 1`）
- 引入 dependency 后最终预测成功（gold rank `= 1`）

结果见：

- `relation_case_study_examples.csv`

在这个最严格的标准下，当前只发现了 `hetionet / DrD` 的明确翻转样例；`FB15k-237 region` 和 `YAGO3-10 dealsWith` 虽然 relation-level 指标显著上升，但没有出现同样清晰的 top-1 flip。这说明：

- `hetionet / DrD` 的 dependency 收益更像“把错误 top-1 直接纠正成正确答案”
- `FB15k-237 region` 与 `YAGO3-10 dealsWith` 的收益更像“整体 rank 改善”而非少数 query 的 dramatic flip

最典型的样例是：

- query: `(Disease::DOID:11615, DrD, ?)`
- gold: `Disease::DOID:11054`
- stage1 top-1: `Disease::DOID:4045`
- final top-1: `Disease::DOID:11054`
- stage1 rank: `5`
- final rank: `1`

这类案例很符合 `DrD` 的语义特征：疾病相似性通常需要多条生物医学证据共同支持，dependency 能把单条规则的“弱证据”提升为更可靠的共支持信号。
