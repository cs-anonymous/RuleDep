# Case Study Notes

## Summary Table

Below is a consolidated comparison table of all three cases. Each row shows how the dependency mechanism corrects the ranking of the gold answer and its main distractor(s).

| | **Case 1: Iraq Language** | **Case 2: South Governorate Location** | **Case 3: Jasper County Location** |
|---|---|---|---|
| **Dataset** | FB15k-237 | YAGO3-10 | YAGO3-10 |
| **Query** | `? --countries_spoken_in--> Iraq` | `? --isLocatedIn--> South_Governorate` | `Jasper_County,_Missouri --isLocatedIn--> ?` |
| **Gold Answer** | Persian_Language | Tyre | Missouri |
| **Main Distractor(s)** | English_Language | Lebanon | Joplin / Webb City / Carthage, MO |
| **Demonstrated Effect** | Redundancy correction | Redundancy correction | Synergy promotion |
| **Gold rank (before→after)** | 2 → **1** | 2 → **1** | 4 → **1** |
| **Distractor rank (before→after)** | 1 → 2 | 1 → **19** | 2 → 3 (tied, all three) |
| **Gold score (before→after)** | 0.2898 → 0.3018 | 0.0167 → 0.0259 | 0.0103 → 0.0689 |
| **Distractor score (before→after)** | 0.3022 → 0.2581 | 0.1119 → 0.0022 | 0.0877 → 0.0655 (all three) |
| **Gold rule_total** | 3.072 | 2.202 | 2.886 |
| **Gold dep_total** | **+0.409** | **+0.126** | **+0.616** |
| **Distractor rule_total** | 3.001 | 7.246 | 3.664 (all three) |
| **Distractor dep_total** | **−0.904** | **−7.372** | **0.0** (all three) |
| **Key Dependency** | D202 (B_B neg), many small neg terms | D3821 (redundancy, −5.77 raw) | D3815 (synergy, +0.62 raw) |
| **Interpretation** | English has strong rule support but its rule combination is generic/redundant; distributed negative corrections suppress it | Lebanon is semantically related but at wrong granularity (`South_Gov → Lebanon` exists, but query asks for `? → South_Gov`); massive redundancy penalty | Nearby cities are locally plausible but lack pairwise synergy; Missouri gets a single strong synergy boost absent from distractors |

### Case Mechanism Breakdown

| Metric | Case 1 Gold (Persian) | Case 1 Distractor (English) | Case 2 Gold (Tyre) | Case 2 Distractor (Lebanon) | Case 3 Gold (Missouri) | Case 3 Distractors (Cities) |
|---|---|---|---|---|---|---|
| stage2 rule_total | 3.072 | 3.001 | 2.202 | 7.246 | 2.886 | 3.664 |
| dep B_B | +0.242 | −0.314 | — | — | — | — |
| dep U_U | +0.167 | −0.590 | — | — | — | — |
| dep redundancy | — | — | — | −6.477 | — | — |
| dep synergy | — | — | +0.126 | −0.895 | +0.616 | 0.0 |
| **dep_total** | **+0.409** | **−0.904** | **+0.126** | **−7.372** | **+0.616** | **0.0** |
| **final score** | **0.3018** | **0.2581** | **0.0259** | **0.0022** | **0.0689** | **0.0655** |

### Key Observations

1. **Case 1 (Redundancy — distributed)**: English_Language 的规则打分 (3.001) 与 Persian_Language (3.072) 接近，但依赖项发现 English 的规则组合在此查询上下文中冗余（高频通用语言），分布式地施加了 −0.904 的负修正，多个小的 B_B 和 U_U 负依赖项累加。Persian 获得 +0.409 的正修正。
2. **Case 2 (Redundancy — concentrated)**: Lebanon 的 rule_total (7.246) 远高于 Tyre (2.202)，但依赖项施加了 −7.372 的巨大负修正。最大的贡献来自 D3821（原始权重 −5.77），多个 redundancy 依赖项共同惩罚了 Lebanon。KG 中存在 `South_Governorate → Lebanon` 的边，但查询方向是 `? → South_Governorate`，Lebanon 不是正确粒度的答案。
3. **Case 3 (Synergy — clean)**: Missouri 的 rule_total (2.886) 低于三个城市级干扰项 (3.664)，但 D3815 给予 Missouri 唯一的 +0.616 synergy 正修正。三个城市级候选没有任何活跃依赖项 (dep_total = 0)，最终被超越。这是最干净的 synergy 案例。

---

This folder keeps three final case-study examples for the paper. Each case has two files:

- `*.yml`: the full per-example explanation generated from the trained aggregation model.
- `*_graph.json`: a compact drawable local graph with observed KG edges, candidate prediction edges, and the rule/dependency annotation layer.

Important caveat: `kg_edges` in the graph JSON are real triples from `train` / `valid` / `test`. The `rule_dependency_layer` records the top active rule and dependency IDs and learned weights, but the current application files do not preserve full grounding witnesses for every rule body. Therefore the rule/dependency layer should be used as an annotation over candidates, not as a claim that every rule text edge is directly present in the displayed KG subgraph.

## Case 1: Iraq Language Prediction

Files:

- `FB15k-237_countries_spoken_in_m_0d05q4_0112.yml`
- `FB15k-237_countries_spoken_in_m_0d05q4_0112_graph.json`

Purpose: show how dependency suppresses a high-frequency generic language candidate.

Query: `? -- human languages are spoken in various countries --> Iraq`

Gold answer: `Persian_Language`

Main distractor: `English_Language`

Before dependency, `English_Language` is ranked first and `Persian_Language` is ranked second. After dependency, `Persian_Language` becomes rank 1 and `English_Language` drops to rank 2.

Key candidate scores from `candidate_explanations`:

- `Persian_Language`: rank `2 -> 1`, score `0.289759 -> 0.301753`, stage2 `rule_total = 3.071647`, `dependency_total = +0.408520`.
- `English_Language`: rank `1 -> 2`, score `0.302235 -> 0.258086`, stage2 `rule_total = 3.000873`, `dependency_total = -0.903968`.

Largest dependency terms:

- For `English_Language`, the largest negative term is `D202`, a `B_B` negative dependency with raw stage2 weight `-0.015985`. It connects `R19792` and `R965904`. `R19792` is an award/TV-cast pattern, `/award/award_category/winners... <= /tv/tv_program/regular_cast...`, and `R965904` is a music performance-role pattern, `/music/performance_role/track_performances... <= /music/performance_role/track_performances..., /music/artist/track_contributions..., /music/instrument/instrumentalists...`.
- Other large negative terms for `English_Language` include `D199 = -0.015294`, `D209 = -0.015121`, and `D210 = -0.015081`. Together, many small negative pairwise terms sum to `dependency_total = -0.903968`, so the effect is distributed rather than dominated by one single dependency.
- For `Persian_Language`, the largest positive term is `D4345`, a `U_U` positive dependency with raw stage2 weight `+0.014950`. It connects `R686199`, `/music/record_label/artist... <= /music/genre/artists...`, and `R935363`, a sports draft pattern. The individual rule texts are not themselves a human-readable Iraq/Persian rule; they are learned statistical rule patterns. The important evidence is that the same aggregation layer assigns positive pairwise support to Persian but a large net negative correction to English.

Interpretation: `English_Language` still has strong rule evidence after stage2, so the failure is not caused by lack of rule support. Instead, dependency identifies that the active rule combination for English is redundant or misleading in this query context and applies a large negative correction. Persian receives a positive dependency correction and becomes top-1.

Useful local graph evidence:

- The graph includes the predicted query edge `Persian_Language -- countries_spoken_in --> Iraq`.
- The graph also includes many generic language context edges around `English_Language`, which helps explain why a rule-only scorer can over-rank English.

Suggested paper use: this is a strong redundancy/general-frequency case. It supports the claim that dependency can correct a candidate that is individually well supported by many rules but whose combined evidence is not query-specific enough.

## Case 2: South Governorate Location Prediction

Files:

- `YAGO3-10_isLocatedIn_South_Governorate_0086.yml`
- `YAGO3-10_isLocatedIn_South_Governorate_0086_graph.json`

Purpose: show a clean redundancy case where the wrong answer is semantically related but has the wrong granularity or direction.

Query: `? -- isLocatedIn --> South_Governorate`

Gold answer: `Tyre`

Main distractor: `Lebanon`

Before dependency, `Lebanon` is ranked first and `Tyre` is ranked second. After dependency, `Tyre` becomes rank 1 and `Lebanon` drops to rank 19.

Key candidate scores from `candidate_explanations`:

- `Tyre`: rank `2 -> 1`, score `0.016719 -> 0.025903`, stage2 `rule_total = 2.201745`, `dependency_total = +0.126009`.
- `Lebanon`: rank `1 -> 19`, score `0.111941 -> 0.002197`, stage2 `rule_total = 7.245650`, `dependency_total = -7.372026`.

Largest dependency terms:

- For `Lebanon`, the largest term is `D3821`, a negative `redundancy` dependency. Its raw stage2 weight is `-5.773287`, and after active-dependency scaling its estimated contribution to the candidate is about `-1.154657`, roughly `15.66%` of Lebanon's total dependency correction.
- `D3821` connects `R124457` and `R553731`. `R124457` is `isAffiliatedTo(X,Leicester_City_F.C.) <=` with stage2 effective rule weight `0.500951`; `R553731` is `playsFor(Samir_Muratović,Y) <= isAffiliatedTo(Samir_Muratović,Y)` with stage2 effective rule weight `4.108878`.
- The next largest negative redundancy terms for `Lebanon` share the same high-weight rule `R553731`: `D12020 = -5.730206`, `D10019 = -5.291723`, and `D2463 = -4.221309`. Their estimated scaled contributions are about `-1.146041`, `-1.058345`, and `-0.844262`.
- For `Tyre`, the only active dependency in the selected explanation is `D11674`, a positive `synergy` dependency with raw stage2 weight `+0.126009`. It connects two airport-connectivity rule patterns, `R455584` and `R903078`; its full contribution accounts for Tyre's `dependency_total = +0.126009`.

Interpretation: `Lebanon` is related to `South_Governorate`, but it is not the correct head entity for this query. The local KG shows `South_Governorate -- isLocatedIn --> Lebanon`, while the query asks for an entity located in South Governorate. Dependency strongly penalizes the misleading rule combination for Lebanon, especially through redundancy dependencies.

Mechanism caveat: the largest rule-pair IDs are learned statistical patterns, not a direct human-readable geospatial proof. The local KG subgraph provides the semantic story, while the dependency weights show what actually drove the model's score correction.

Useful local graph evidence:

- `Tyre -- isLocatedIn --> South_Governorate` appears as the gold query edge.
- `South_Governorate -- isLocatedIn --> Lebanon` appears as an observed context edge.
- `Tyre -- isLocatedIn --> Lebanon` also appears as context, explaining why Lebanon is highly related but still wrong for the query direction.
- The top dependency layer contains large negative redundancy terms for the Lebanon candidate, for example `D3821`, `D12020`, `D10019`, and `D2463`.

Suggested paper use: this is the best redundancy case. It shows that dependency can distinguish “strongly related” from “correct under the query direction and granularity.”

## Case 3: Jasper County Location Prediction

Files:

- `YAGO3-10_isLocatedIn_Jasper_County_Missouri_0080.yml`
- `YAGO3-10_isLocatedIn_Jasper_County_Missouri_0080_graph.json`

Purpose: show a compact synergy case where the correct candidate receives a positive dependency signal while nearby-city distractors do not.

Query: `Jasper_County,_Missouri -- isLocatedIn --> ?`

Gold answer: `Missouri`

Main distractors: `Joplin,_Missouri`, `Webb_City,_Missouri`, and `Carthage,_Missouri`

Before dependency, the nearby city candidates are tied ahead of the gold answer. After dependency, `Missouri` becomes rank 1.

Key candidate scores from `candidate_explanations`:

- `Missouri`: rank `4 -> 1`, score `0.010289 -> 0.068869`, stage2 `rule_total = 2.886462`, `dependency_total = +0.616094`.
- `Joplin,_Missouri`: rank `2 -> 3`, score `0.087715 -> 0.065496`, stage2 `rule_total = 3.664291`, `dependency_total = 0`.
- `Webb_City,_Missouri`: rank `2 -> 3`, score `0.087715 -> 0.065496`, stage2 `rule_total = 3.664291`, `dependency_total = 0`.
- `Carthage,_Missouri`: rank `2 -> 3`, score `0.087715 -> 0.065496`, stage2 `rule_total = 3.664291`, `dependency_total = 0`.

Largest dependency terms:

- For `Missouri`, the dependency signal is concentrated in one term: `D3815`, a positive `synergy` dependency with raw stage2 weight `+0.616094`. Because it is the only active dependency for this selected gold candidate, it accounts for `100%` of `Missouri`'s `dependency_total`.
- `D3815` connects `R124457` and `R538213`. `R124457` is `isAffiliatedTo(X,Leicester_City_F.C.) <=` with stage2 effective rule weight `0.109755`; `R538213` is `isAffiliatedTo(X,FC_SKVO_Rostov-on-Don) <= isAffiliatedTo(X,A)` with stage2 effective rule weight `1.173180`.
- The nearby-city distractors `Joplin,_Missouri`, `Webb_City,_Missouri`, and `Carthage,_Missouri` have no active dependency in this selected explanation, so their stage2 `dependency_total` is `0`.

Interpretation: the nearby cities are locally plausible, but the correct answer is the state-level location. The dependency term gives `Missouri` a positive synergy contribution, while the nearby-city distractors do not receive active dependency support.

Mechanism caveat: as in Case 2, the named rule texts are statistical patterns rather than directly readable Missouri-location rules. This case is still valuable because the local KG gives a clean county-city-state subgraph, and the trained dependency layer gives the correct state candidate an explicit pairwise boost absent from the nearby-city distractors.

Useful local graph evidence:

- `Jasper_County,_Missouri -- isLocatedIn --> Missouri` appears as the gold query edge.
- `Joplin,_Missouri -- isLocatedIn --> Jasper_County,_Missouri` and `Joplin,_Missouri -- isLocatedIn --> Missouri` are observed context edges.
- `Carthage,_Missouri -- isLocatedIn --> Jasper_County,_Missouri` and `Webb_City,_Missouri -- isLocatedIn --> Jasper_County,_Missouri` are observed context edges.
- The top dependency layer contains one positive synergy dependency, `D3815`, with stage2 effective weight `+0.616094`.

Suggested paper use: this is the cleanest synergy example. It shows that dependency can promote the correct hierarchical answer over plausible local-neighbor distractors.

## Recommended Narrative

Use Case 2 as the main redundancy figure: it is visually intuitive and the negative dependency effect is very large. Use Case 3 as the main synergy figure: it has a compact local graph and a single positive dependency, making the story easy to explain. Use Case 1 as a complementary robustness example for FB15k-237: it shows that the same mechanism also works in a noisier, high-frequency Freebase relation where generic language evidence can dominate.
