# 0405 Report Summary

本目录汇总截至 `2026-04-05` 的最新统计分析与阶段性结论。核心输入来自：

- [all_results_summary.csv](/home/sy/RuleDep/reports/0405/all_results_summary.csv)
- [structural_filtered_comparison.csv](/home/sy/RuleDep/reports/0405/structural_filtered_comparison.csv)
- [relation_dependency_analysis.csv](/home/sy/RuleDep/reports/0405/relation_dependency_analysis.csv)
- [dependency_relation_analysis.md](/home/sy/RuleDep/reports/0405/dependency_relation_analysis.md)
- [dataset_size_rule_dependency_stats.csv](/home/sy/RuleDep/reports/0405/dataset_size_rule_dependency_stats.csv)
- [analysis.md](/home/sy/RuleDep/reports/0405/analysis.md)

## Files

- `all_results_summary.csv`
  统一汇总所有当前保留实验，默认指标使用最终 `test`。
- `all_results_ensemble_debug.json`
  relation-wise ensemble 的选模明细。
- `structural_filtered_comparison.csv`
  `structural_none / structural_r2d3 / structural_r3d6` 三个主 structural 配置的对比。
- `relation_dependency_analysis.csv`
  每个 `dataset + relation` 的主分析表。
- `relation_relative_gain_gt_3pct_best_structural.csv`
  每个数据集选最优 structural 配置后，最终 `test` 相对 stage1 提升 `> 3%` 的 relation。
- `relation_relative_gain_lt_minus_3pct_best_structural.csv`
  同口径下最终 `test` 下降 `< -3%` 的 relation。
- `dataset_size_rule_dependency_stats.csv`
  7 个数据集的规模、rule、dependency 统计。
- `dependency_relation_analysis.md`
  关系级 dependency 适用性分析。
- `analysis.md`
  最新的 global-vs-relational 与新 RD 变体结论。

## Main Conclusions

### 1. Dependency 有用，但覆盖率仍然不高

按每个数据集最优 structural 配置统计，relation 级最终 `test` 收益分布是：

- `positive (> 3%)`: `36 / 425`
- `neutral [-3%, 3%]`: `378 / 425`
- `negative (< -3%)`: `11 / 425`

这说明 dependency 不是 uniformly useful，而是明显的 relation-conditional 信号。  
它能在一部分 relation 上带来真实增益，但覆盖率还不足以支撑“普适提升”的强表述。

### 2. 三个主 structural 配置仍然是数据集相关的

当前 7 个数据集的最优主 structural 配置为：

- `FB15k-237` -> `structural_r2d3`
- `KG20C` -> `structural_none`
- `WN18RR` -> `structural_none`
- `YAGO3-10` -> `structural_r2d3`
- `codex-l` -> `structural_none`
- `codex-m` -> `structural_r3d6`
- `hetionet` -> `structural_r2d3`

结论仍然是：

- 更细的 type grouping 不是越细越强。
- 最优 aggregation bias 依赖数据集。

### 3. 新增的 4 个 RD 变体说明“dependency 总作用量控制”是值得继续挖的

相对 `structural_none`，4 个新增变体的最好结果如下：

- `FB15k-237`: `structural_dep_scale`，`+0.001240`
- `KG20C`: `structural_rule_mask`，`+0.000063`
- `WN18RR`: `structural_rd`，`+0.001655`
- `YAGO3-10`: `structural_dep_scale`，`+0.002585`
- `codex-l`: `structural_dep_scale`，`+0.000237`
- `codex-m`: `structural_dep_scale`，`+0.000501`
- `hetionet`: `structural_surprisal_init`，`+0.003036`

阶段性判断：

- `dep_scale` 最稳，在 6 个数据集里赢了 4 个。
- `structural_rd` 在 `WN18RR` 和 `hetionet` 上也很有竞争力。
- `rule_mask` 基本只在 `KG20C` 持平略优。
- `surprisal_init` 不是普遍最优，但在 `hetionet` 上非常有效。

### 4. Global canonical 的优势大概率不是简单由优化配置造成的

在 matched control 里，我们把 relation-wise stage1 尽量对齐到 old canonical 的训练 recipe，结果：

- `KG20C`: `0.22913` vs canonical `0.23940`
- `codex-m`: `0.34197` vs canonical `0.34554`
- `WN18RR`: `0.50010` vs canonical `0.49950`

这说明：

- 在 `KG20C / codex-m` 上，gap 只靠把 `pos/lr/epoch/evaluate_every` 对齐并不能消掉。
- `WN18RR` 依然是例外，relation-wise 更适合。

### 5. Global canonical 的代价明显更高

对比 canonical old 与 matched relation-wise stage1 的墙钟时间：

- `KG20C`: `5.18x`
- `WN18RR`: `7.48x`
- `codex-m`: `7.49x`
- `FB15k-237`: `3.89x`

所以论文里更合理的定位是：

- `global canonical` 作为 strong baseline 保留
- `relation-wise` 作为更便宜、也更适合 dependency 分析的平台

## Recommended Reading Order

1. 先看 [all_results_summary.csv](/home/sy/RuleDep/reports/0405/all_results_summary.csv)
2. 再看 [structural_filtered_comparison.csv](/home/sy/RuleDep/reports/0405/structural_filtered_comparison.csv)
3. 然后看 [dependency_relation_analysis.md](/home/sy/RuleDep/reports/0405/dependency_relation_analysis.md)
4. 最后看 [analysis.md](/home/sy/RuleDep/reports/0405/analysis.md)

## Note

`0405` 下的 PNG 图是基于当前同口径结果复制自 `0403`。  
本次环境里没有 `matplotlib`，所以没有重新绘制，但 relation 分析主表与结论已重新生成并对齐到最新结果。
