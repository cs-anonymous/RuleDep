# 0407 Rule / Dependency Weight Analysis

本节沿用第 2 部分的 best config，仅使用 `dependency_final`，考察 rule 与 dependency 的参数变化，以及最终被保留的 dependency 权重分布。

相关表格：

- `best_config_weight_summary.csv`
- `dependency_sign_by_type.csv`

<p align="center"><img src="plot_weight_near_zero_ratio.png" alt="Weight Near-zero Ratio" width="60%"></p>

<p align="center"><em>Figure 1: near-zero ratio of learned rule and dependency weights.</em></p>

<p align="center"><img src="plot_weight_max_abs.png" alt="Weight Max Abs" width="60%"></p>

<p align="center"><em>Figure 2: maximum absolute value of learned rule and dependency weights.</em></p>

<p align="center"><img src="plot_dependency_sign_by_type.png" alt="Dependency Sign by Type" width="60%"></p>

<p align="center"><em>Figure 3: positive-weight ratio of synergy and redundancy dependencies in dependency_final.</em></p>

## Definitions (gold-effective metrics)

下面新增 4 个统计，统一基于 `dependency_final` 与阈值 `delta=0.01`，并且都在 gold `(q,c)` 粒度上计算：

- gold 平均有效 Rule 数量：对每个 gold `(q,c)`，统计该候选上 `rule_weight > delta` 的规则条目数，再对所有 gold `(q,c)` 求平均。
- gold 平均有效 Dep 数量：对每个 gold `(q,c)`，统计由激活规则诱导、且 `dep_weight > delta` 的 dependency 边数，再对所有 gold `(q,c)` 求平均。
- gold top3 Rule 占比：对每个 gold `(q,c)`，`sum(top3(rule_weight>delta)) / sum(rule_weight>delta)`，再求平均。
- gold top3 Dep 占比：对每个 gold `(q,c)`，`sum(top3(dep_weight>delta)) / sum(dep_weight>delta)`，再求平均。

## Dataset Summary

| Dataset | Config | Rule near-zero | Dep final near-zero | Rule max abs | Dep final max abs | Gold avg eff Rule per (q,c) | Gold avg eff Dep per (q,c) | Gold top1 Rule share | Gold top1 Dep share | Gold top3 Rule share | Gold top3 Dep share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KG20C | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 15.81498 | 61.95652 | 3.06073 | 0.03305 | 4.28220 | 0.00109 | 56.15831% | 100.00000% | 87.04001% | 100.00000% |
| codex-m | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 11.37283 | 96.44370 | 2.99124 | 0.83764 | 26.69186 | 0.80814 | 34.32042% | 63.22469% | 61.74383% | 87.23223% |
| WN18RR | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 10.68406 | 83.93285 | 4.24963 | 0.47246 | 13.34295 | 0.13434 | 52.12713% | 73.62292% | 75.42712% | 92.27528% |
| FB15k-237 | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 15.90109 | 75.55495 | 5.06752 | 0.56410 | 72.03324 | 2.25969 | 14.97162% | 41.50786% | 31.26329% | 68.02608% |
| codex-l | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 31.99267 | 83.09374 | 4.32778 | 2.94214 | 13.46621 | 2.46496 | 37.88959% | 60.45793% | 68.69940% | 84.26927% |
| YAGO3-10 | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 43.77469 | 98.86781 | 5.43007 | 1.53437 | 8.81508 | 1.41706 | 40.45558% | 68.84824% | 81.25051% | 99.66125% |
| hetionet | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 34.11212 | 97.23578 | 5.62772 | 0.32629 | 57.74164 | 0.68645 | 13.19220% | 47.07051% | 29.22229% | 86.80424% |
| Average | - | 23.37892 | 85.29791 | 4.39353 | 0.95858 | 28.05331 | 1.11025 | 35.58783% | 64.96174% | 62.09235% | 88.32405% |

## Dependency Sign vs Type

在 `dependency_final` 中，`synergy` 的平均正权重比例为 `48.03646%`，`redundancy` 为 `35.16616%`。

## Global View

rule 权重的平均绝对变化为 `0.31900`，dependency(final) 为 `0.00588`。
近零比例方面，rule 权重均值为 `23.37892%`，dependency(final) 为 `85.29791%`。
gold 口径下，平均有效 Rule 数量为 `28.05331`，平均有效 Dep 数量为 `1.11025`。
用于统计的 gold `(q,c)` 数量均值为 `48800.85714`。
gold top1 占比方面，Rule 为 `35.58783%`，Dep 为 `64.96174%`。
gold top3 占比方面，Rule 为 `62.09235%`，Dep 为 `88.32405%`。

## Interpretation

这些结果共同说明，模型确实会主动稀疏化大量 dependency 边；在最终被保留的 `dependency_final` 中，依然只有少数边具有显著权重，因此它们更像是稀疏但强烈的修正项，而不是均匀分布在所有规则对上的微弱偏置。

从符号分布看，`synergy` 更容易获得正权重，而 `redundancy` 通常更保守，这与依赖类型本身的语义方向基本一致，但并不是绝对的一一对应关系。
