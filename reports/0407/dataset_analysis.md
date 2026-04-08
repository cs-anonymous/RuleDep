# 0407 Dataset Analysis

本表统计各数据集规模、rule 数量、dependency 数量，以及按 type 聚合后的 rule/dependency 结构。

相关表格：

- `dataset_size_rule_dependency_stats.csv`

## Headline Table

| Dataset | #entity | #relation | #train | #valid | #test | #rule | #filtered_dep | filtered_dep_per_rule |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KG20C | 16362 | 5 | 48213 | 3670 | 3724 | 154790 | 168315 | 1.08738 |
| codex-m | 17050 | 51 | 185584 | 10310 | 10311 | 518279 | 883207 | 1.70411 |
| WN18RR | 40943 | 11 | 86835 | 3034 | 3134 | 76909 | 45212 | 0.58786 |
| FB15k-237 | 14541 | 237 | 272115 | 17535 | 20466 | 1737378 | 3946389 | 2.27146 |
| codex-l | 77951 | 69 | 551193 | 30622 | 30622 | 273472 | 455330 | 1.66500 |
| YAGO3-10 | 123182 | 37 | 1079040 | 5000 | 5000 | 990481 | 1707159 | 1.72357 |
| hetionet | 45158 | 24 | 1800157 | 225020 | 225020 | 6103910 | 7154604 | 1.17213 |
| wikidata5m | 4818679 | 828 | 21343681 | 5357 | 5321 | 1395755 | 0 | 0.00000 |

## Highlights

- 规则最多的数据集：`hetionet`，共有 `6103910` 条 rule。
- filtered dependency 最密的数据集：`FB15k-237`，`filtered_dep_per_rule = 2.27146`。
- `B` rule 占比最高：`WN18RR`，`B_rule_ratio = 0.11417`。
- `Ud` rule 占比最高：`codex-l`，`Ud_rule_ratio = 0.18214`。

## Interpretation

- `FB15k-237 / codex-m / codex-l / YAGO3-10` 这类数据集更像大规模 relation-wise rule aggregation 场景。
- `KG20C / WN18RR` 关系数较少，但 rule 和 dependency 结构更紧凑，适合做受控比较。
- `hetionet` 的 graph 规模和语义复杂度都高，通常更依赖更强的 structural bias。
- `wikidata5m` 当前只有 rule application 结果，dependency / aggregation 统计仍为空，这一点在表格里会直接体现。
