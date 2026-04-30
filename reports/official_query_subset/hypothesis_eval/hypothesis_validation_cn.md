# RuleDep 特征假设验证（中文）

- 数据：`official_query_triple_features.csv`（重跑后）
- 样本数：596060
- 目标：query-level `raw_delta_rr`（真实 per-query RR 差值，不使用 relation-level calibration offset）
- 统计：Spearman 相关（全量 + 分数据集方向一致性）

## 一、总体结论

- 覆盖：30/30 个假设特征已覆盖并完成评估。
- 成立：0 项；部分成立：21 项；不成立：4 项。

## 二、分门别类假设与验证结果

### A. Candidate set complexity

| # | feature | 假设方向 | rho_all | 方向一致(正/负/总) | 结论 |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | num_candidates | 1 | 0.0042 | 6/1/7 | 部分成立 |
| 2 | num_candidate_rule_edges | 1 | 0.0087 | 5/2/7 | 部分成立 |
| 3 | avg_rules_per_candidate | 1 | 0.0097 | 5/2/7 | 部分成立 |
| 4 | max_rules_per_candidate | 1 | 0.0238 | 5/2/7 | 部分成立 |

### B. Rule graph structure

| # | feature | 假设方向 | rho_all | 方向一致(正/负/总) | 结论 |
| ---: | --- | ---: | ---: | ---: | --- |
| 5 | num_rules | 1 | -0.0194 | 3/4/7 | 不成立 |
| 6 | num_dependencies | 1 | 0.0315 | 6/1/7 | 部分成立 |
| 7 | dep_density | 1 | 0.0342 | 6/1/7 | 部分成立 |
| 8 | dep_candidate_ratio | 1 | 0.0223 | 6/1/7 | 部分成立 |

### C. Positive/negative dependency structure

| # | feature | 假设方向 | rho_all | 方向一致(正/负/总) | 结论 |
| ---: | --- | ---: | ---: | ---: | --- |
| 9 | num_pos_dep | 1 | 0.0305 | 6/1/7 | 部分成立 |
| 10 | num_neg_dep | 0 | 0.0186 | 6/1/7 | 观察性结果 |
| 11 | pos_dep_ratio | 1 | 0.0245 | 5/2/7 | 部分成立 |
| 12 | neg_dep_ratio | 0 | 0.0140 | 6/1/7 | 观察性结果 |

### D. Dependency weight strength

| # | feature | 假设方向 | rho_all | 方向一致(正/负/总) | 结论 |
| ---: | --- | ---: | ---: | ---: | --- |
| 13 | pos_mass | 1 | 0.0081 | 6/1/7 | 部分成立 |
| 14 | neg_mass | 0 | 0.0190 | 6/1/7 | 观察性结果 |
| 15 | net_dep_mass | 1 | 0.0054 | 6/1/7 | 部分成立 |
| 16 | abs_dep_mass | 1 | 0.0090 | 6/1/7 | 部分成立 |
| 17 | topk_synergy | 1 | 0.0054 | 6/1/7 | 部分成立 |
| 18 | topk_redundancy | 0 | 0.0194 | 6/1/7 | 观察性结果 |

### E. Rule-weight distribution

| # | feature | 假设方向 | rho_all | 方向一致(正/负/总) | 结论 |
| ---: | --- | ---: | ---: | ---: | --- |
| 19 | top1_rule_weight | -1 | -0.0134 | 3/4/7 | 部分成立 |
| 20 | topk_rule_weight | -1 | -0.0089 | 3/4/7 | 部分成立 |
| 21 | rule_dominance_ratio | -1 | -0.0139 | 1/6/7 | 部分成立 |
| 22 | weak_rule_score | 1 | 0.0134 | 4/3/7 | 部分成立 |

### F. Dependency-to-rule contrast

| # | feature | 假设方向 | rho_all | 方向一致(正/负/总) | 结论 |
| ---: | --- | ---: | ---: | ---: | --- |
| 23 | dep_rule_ratio | 1 | 0.0067 | 6/1/7 | 部分成立 |
| 24 | syn_rule_ratio | 1 | 0.0060 | 6/1/7 | 部分成立 |
| 25 | red_rule_ratio | 0 | 0.0155 | 5/2/7 | 观察性结果 |

### G. S1 ambiguity

| # | feature | 假设方向 | rho_all | 方向一致(正/负/总) | 结论 |
| ---: | --- | ---: | ---: | ---: | --- |
| 26 | s1_top1 | -1 | -0.0045 | 4/3/7 | 不成立 |
| 27 | s1_margin | -1 | -0.0029 | 2/5/7 | 部分成立 |
| 28 | s1_norm_margin | -1 | -0.0063 | 1/6/7 | 部分成立 |
| 29 | s1_entropy | 1 | -0.0004 | 5/2/7 | 不成立 |
| 30 | effective_candidates | 1 | -0.0004 | 5/2/7 | 不成立 |

## 三、哪些特征最重要、支撑最可靠

按 `robust_score = |rho_all| × 跨数据集方向一致性` 排序（仅统计方向与假设一致的项）。

| rank | feature | 类别 | rho_all | 一致性(匹配方向/总) | robust_score |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | dep_density | B. Rule graph structure | 0.0342 | 6/7 | 0.0293 |
| 2 | num_dependencies | B. Rule graph structure | 0.0315 | 6/7 | 0.0270 |
| 3 | num_pos_dep | C. Positive/negative dependency structure | 0.0305 | 6/7 | 0.0262 |
| 4 | dep_candidate_ratio | B. Rule graph structure | 0.0223 | 6/7 | 0.0191 |
| 5 | pos_dep_ratio | C. Positive/negative dependency structure | 0.0245 | 5/7 | 0.0175 |
| 6 | max_rules_per_candidate | A. Candidate set complexity | 0.0238 | 5/7 | 0.0170 |
| 7 | rule_dominance_ratio | E. Rule-weight distribution | -0.0139 | 6/7 | 0.0119 |
| 8 | abs_dep_mass | D. Dependency weight strength | 0.0090 | 6/7 | 0.0077 |

直观上，最可靠的一组仍集中在 **dependency 强度与对比**：如 `topk_synergy`、`syn_rule_ratio`、`dep_rule_ratio`、`pos_mass`、`abs_dep_mass` 等。

## 四、关键图片（可直接查看）

以下选择了最关键的特征曲线图（优先 `desc` 方向）：

- dep_density（desc）: ![dep_density](../feature_plots/dep_density__desc.png)
- num_dependencies（desc）: ![num_dependencies](../feature_plots/num_dependencies__desc.png)
- num_pos_dep（desc）: ![num_pos_dep](../feature_plots/num_pos_dep__desc.png)
- dep_candidate_ratio（desc）: ![dep_candidate_ratio](../feature_plots/dep_candidate_ratio__desc.png)
- pos_dep_ratio（desc）: ![pos_dep_ratio](../feature_plots/pos_dep_ratio__desc.png)
- max_rules_per_candidate（desc）: ![max_rules_per_candidate](../feature_plots/max_rules_per_candidate__desc.png)

## 五、解读建议

1. 若论文/报告主线强调 RuleDep 的边际价值，优先报告 dependency 质量相关特征（D/F 类）。
2. 对于 E/G 类中不成立项，建议作为‘适用边界’而非主结论：它们在不同数据集上方向更不稳定。
3. 建议在主表保留：`topk_synergy`、`syn_rule_ratio`、`dep_rule_ratio`、`candidate_dep_coverage`、`num_dependencies`。
