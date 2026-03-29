# All-Dataset 5% Hit Notes

## Previous Version

规则：不强求统一的结构阈值；对每个数据集，扫描现有已完成配置，选择一个能达到 `stage2 相对 stage1 > 5%` 的 relation 子集，并尽量让该 relation 的 `num_test` 最大。

| Dataset | Experiment | Relation | Num Test | Relative Gain |
| --- | --- | ---: | ---: | ---: |
| KG20C | r3_mt3_mul1_sr_pair | 3 | 1446 | 7.71% |
| codex-m | exp_codexm_synergy_pair_and_type_lift_oldcfg | 2 | 3635 | 10.36% |
| WN18RR | exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0 | 4 | 253 | 5.75% |
| FB15k-237 | exp-1_LinearAggregator_1_1_1_auto_sqrt_1_1_0 | 19 | 1067 | 15.83% |
| codex-l | exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1 | 2 | 9426 | 6.76% |
| YAGO3-10 | exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_0 | 15 | 47 | 11.38% |

结论：这版已经让 6/6 数据集都命中 `>5%`。

## Unified-Rule Version

我尝试了统一的 relation-wise 结构规则搜索，使用同一套静态特征和同一种选择器，在所有数据集上自动挑 relation。

当前找到的最好粗粒度统一规则是：在所有 relation 中，选择 dep_per_rule 最大的 relation；限制 avg_tails_per_sp <= 6.0，不加其他阈值。

| Dataset | Picked Relation | Num Test | Relative Gain | Hit >5% |
| --- | ---: | ---: | ---: | --- |
| KG20C | 3 | 1446 | 5.19% | yes |
| codex-m | 35 | 13 | 5.75% | yes |
| WN18RR | 5 | 114 | 0.00% | no |
| FB15k-237 | 3 | 41 | 56.07% | yes |
| codex-l | 28 | 111 | 0.24% | no |
| YAGO3-10 | 28 | 7 | 30.15% | yes |

结论：这条统一规则只能命中 4/6 个数据集，不足以替代前一版。

## Takeaway

- 如果目标是“所有数据集都要有某个子集 >5%”，前一版已经完成。
- 如果目标升级为“尽量统一的静态规则”，当前还没有找到一条能覆盖 6/6 数据集的简单规则。
- 目前最强的统一性，只能做到“统一使用 relation-wise 的单 relation 子集”，但具体 relation 和 config 仍需按数据集选择。
