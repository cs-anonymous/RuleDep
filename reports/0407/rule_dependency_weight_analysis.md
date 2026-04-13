# 0407 Rule / Dependency Weight Analysis

本节沿用第 2 部分的 best config，考察 rule 与 dependency 的参数在训练前后如何变化，以及模型最终是否会把大量 dependency 权重压回到接近零的区域。

相关表格：

- `best_config_weight_summary.csv`
- `dependency_sign_by_type.csv`

<p align="center"><img src="plot_weight_near_zero_ratio.png" alt="Weight Near-zero Ratio" width="60%"></p>

<p align="center"><em>Figure 1: near-zero ratio of learned rule and dependency weights.</em></p>

<p align="center"><img src="plot_weight_max_abs.png" alt="Weight Max Abs" width="60%"></p>

<p align="center"><em>Figure 2: maximum absolute value of learned rule and dependency weights.</em></p>

<p align="center"><img src="plot_dependency_sign_by_type.png" alt="Dependency Sign by Type" width="60%"></p>

<p align="center"><em>Figure 3: positive-weight ratio of synergy and redundancy dependencies before and after selection.</em></p>

## Definition of `dependency_trial` and `dependency_final`

为避免歧义，这里明确第 5 部分的两个 dependency 统计对象：

- `dependency_trial`：来自 `dependency-trial-<relation>.csv`。它表示 relation 上一旦实际训练了 dependency stage，就把该 stage 学到的 dependency 权重记下来，不要求这个 stage 最终被接受。换句话说，`trial` 反映的是“候选 dependency stage 训练后会学成什么样”。
- `dependency_final`：来自 `dependency-final-<relation>.csv`。它只在 dependency stage 的 best valid 表现超过 rule-only stage、并且最终被模型选择时才会出现。也就是说，`final` 反映的是“真正进入最终测试输出的 dependency 权重”。

因此，`trial` 和 `final` 的差别不只是训练前后两个时间点，而是“尝试过的 dependency 模型”与“最终被接受的 dependency 模型”的区别。通常 `final` 会比 `trial` 更稀疏，因为它已经经过了一次 relation-level model selection。

## Dataset Summary

| Dataset | Config | Rule near-zero | Dep trial near-zero | Dep final near-zero | Rule max abs | Dep trial max abs | Dep final max abs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KG20C | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 15.81498 | 99.89603 | 61.95652 | 3.06073 | 0.03305 | 0.03305 |
| codex-m | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 11.37283 | 98.36131 | 96.44370 | 2.99124 | 0.83764 | 0.83764 |
| WN18RR | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 10.68406 | 97.12718 | 83.93285 | 4.24963 | 0.47246 | 0.47246 |
| FB15k-237 | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 15.90109 | 90.85123 | 75.55495 | 5.06752 | 0.56410 | 0.56410 |
| codex-l | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 31.99267 | 94.29173 | 83.09374 | 4.32778 | 2.94214 | 2.94214 |
| YAGO3-10 | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 43.77469 | 98.88571 | 98.86781 | 5.43007 | 1.53437 | 1.53437 |
| hetionet | best_combination_dep_l1_regularization_dep_fix_topk8_0412 | 34.11212 | 98.34087 | 97.23578 | 5.62772 | 0.32629 | 0.32629 |

## Dependency Sign vs Type

在 trial 阶段，`synergy` 的平均正权重比例为 `18.23198%`，`redundancy` 为 `16.25816%`。
经过最终选择后，`synergy` 的平均正权重比例上升到 `48.03646%`，`redundancy` 也上升到 `35.16616%`。

## Global View

rule 权重的平均绝对变化为 `0.31900`，dependency 在 trial 与 final 阶段分别为 `0.00157` 和 `0.00588`。
近零比例方面，rule 权重均值为 `23.37892%`，dependency 在 trial 阶段为 `96.82201%`，final 阶段为 `85.29791%`。

## Interpretation

这些结果共同说明，模型确实会主动稀疏化大量 dependency 边，尤其是在 trial 阶段，许多候选边最终被压回到零附近。与此同时，少数被保留下来的 dependency 仍可能具有较大的绝对权重，因此它们更像是稀疏但强烈的修正项，而不是均匀分布在所有规则对上的微弱偏置。

从符号分布看，`synergy` 更容易获得正权重，而 `redundancy` 通常更保守，这与依赖类型本身的语义方向基本一致，但并不是绝对的一一对应关系。最终被选择进入 final 阶段的 dependency，往往是那些既能在 valid 上稳定受益、又没有明显过拟合迹象的边。
