# Dataset Analysis

This table shows the size of each dataset, the number of rules and dependency pairs, and how rule and dependency types break down across the data.

Related files:

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

## Highlights

- **Most rules**: `hetionet` with `6,103,910` rules.
- **Densest dependencies**: `FB15k-237` with `filtered_dep_per_rule = 2.27`.
- **Highest B-rule proportion**: `WN18RR` at `B_rule_ratio = 0.11417`.
- **Highest Ud-rule proportion**: `codex-l` at `Ud_rule_ratio = 0.18214`.

## Interpretation

- `FB15k-237`, `codex-m`, `codex-l`, and `YAGO3-10` fit the large-scale relation-wise rule aggregation regime.
- `KG20C` and `WN18RR` have fewer relations but more compact rule and dependency structures, making them suitable for controlled comparisons.
- `hetionet` is large in both scale and semantic complexity, and tends to rely more on strong structural bias.
- `wikidata5m` currently only has rule application results; dependency and aggregation statistics are still empty, which is directly reflected in the table.
