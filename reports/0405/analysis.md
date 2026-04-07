# 0405 Analysis

## Global Rule-Only vs Relational Stage1

本节比较两条尽量可比的 rule-only 路线：

- `global canonical old`
- `matched relational stage1 oldlike`

其中 `matched relational stage1 oldlike` 已尽量对齐 old canonical 的训练超参：

- `pos = 5`
- `lr = 0.005`
- `max_epoch = 40`
- `evaluate_every = 1`
- `early_stopping = -1`
- `sign_constraint = true`
- `shuffle_train = false`
- `batch_size = 4096`
- `type_grouping = none`

### Accuracy

当前已经确认的结果：

| dataset | canonical old | matched relational stage1 | delta |
| --- | ---: | ---: | ---: |
| KG20C | 0.23940 | 0.22913 | -0.01027 |
| WN18RR | 0.49950 | 0.50010 | +0.00060 |
| codex-m | 0.34554 | 0.34197 | -0.00356 |

含义很清楚：

- 在 `KG20C` 和 `codex-m` 上，即使把 relation-wise stage1 的训练 recipe 尽量对齐到 old canonical，差距仍然明显。
- 在 `WN18RR` 上，relation-wise 反而略好。
- 所以 `global > relational` 这件事大概率不是简单由 `pos/lr/epoch/evaluate_every` 的不一致造成的。

### 对默认 Stage1 的提升

matched relational stage1 相对当前默认 `structural_none__stage1` 的提升只有约 `+0.001`：

| dataset | matched stage1 | current stage1 | delta |
| --- | ---: | ---: | ---: |
| KG20C | 0.22913 | 0.22830 | +0.00083 |
| WN18RR | 0.50010 | 0.49916 | +0.00095 |
| codex-m | 0.34197 | 0.34084 | +0.00114 |

这说明：

- old 风格优化配置确实有一点帮助；
- 但帮助很小，不足以解释 `KG20C` 和 `codex-m` 上 canonical 的优势。

## Why The Gap Is Probably Not Just Optimization

目前可以基本排除的因素：

1. 不是 `pos/lr/epoch/evaluate_every` 的简单问题。  
   matched control 只恢复了约 `+0.001`。

2. 不是 hidden type-weight / score-scale 参数造成的差异。  
   在 `type_grouping = none` 时，当前 `aggregation.py` 不会启用 `rule_type`、`dependency_type` 或 `global_scale`。

3. 不是训练集中混入了跨 relation 的 rule。  
   之前检查过 `KG20C` 和 `codex-m`，没有跨 relation rule 混入的问题。

4. 不像是单纯过拟合。  
   在 `KG20C / codex-m / FB15k-237` 上，global 在 valid 上也往往更高。

因此当前更合理的解释是：

- `global canonical` 和当前 relation-wise 路线在 forward 形式上接近，但训练范式仍不同；
- 差距更像是 `global vs relation-wise` 本身的 inductive bias 差异；
- 这种差异强烈依赖数据集。

## Cost Comparison

`global canonical old` 的计算代价明显更高。用 canonical log 的墙钟时间，对比 matched relation-wise stage1 的 `sweep` 时间：

| dataset | canonical wall time (s) | matched relational sweep (s) | ratio |
| --- | ---: | ---: | ---: |
| KG20C | 2448.9 | 472.5 | 5.18x |
| WN18RR | 6391.2 | 854.1 | 7.48x |
| codex-m | 7890.1 | 1053.0 | 7.49x |
| FB15k-237 | 15532.4 | 3992.4 | 3.89x |

结论：

- global rule-only mode 在 wall-clock 上不是“略慢”，而是大约 `4x` 到 `7.5x` 更慢。

### Why Global Is More Expensive

1. old canonical 是单一全局模型，按 epoch 扫全部 relation。
2. 每扫完一个 relation 就做 valid 检查。
3. head/tail 都 separately 保存 best checkpoint。
4. 训练期间会为多个 relation 保留 model copy，CPU 内存和存储开销更大。

## New RD-Ablation Findings

新加的 4 个设置分别是：

- `structural_surprisal_init`
- `structural_dep_scale`
- `structural_rd`
- `structural_rule_mask`

它们都以 `structural_none` 为 baseline。

### Per-dataset Best Improvements

| dataset | baseline (`structural_none`) | best new variant | best mrr | delta |
| --- | ---: | --- | ---: | ---: |
| FB15k-237 | 0.348228 | structural_dep_scale | 0.349468 | +0.001240 |
| KG20C | 0.232779 | structural_rule_mask | 0.232842 | +0.000063 |
| WN18RR | 0.500535 | structural_rd | 0.502189 | +0.001655 |
| YAGO3-10 | 0.575377 | structural_dep_scale | 0.577961 | +0.002585 |
| codex-l | 0.333535 | structural_dep_scale | 0.333772 | +0.000237 |
| codex-m | 0.342474 | structural_dep_scale | 0.342975 | +0.000501 |
| hetionet | 0.362111 | structural_surprisal_init | 0.365147 | +0.003036 |

### Interpretation

1. `dep_scale` 最稳。  
   它在 `FB15k-237 / YAGO3-10 / codex-l / codex-m` 上都是最好的新变体，说明“dependency 在 dense query 上容易累加过量”这件事大概率是真的。

2. `structural_rd` 在部分数据集上有效。  
   `WN18RR` 上它最好，`hetionet` 上也很强，说明显式学习 `rule : dependency` 的整体比例是有意义的。

3. `surprisal_init` 有 dataset-specific 收益。  
   它没有普遍最优，但在 `hetionet` 上最好，说明“更稳的 rule initialization”在大而复杂的数据集上可能更重要。

4. `rule_mask` 当前收益最弱。  
   这说明“按低权重 rule 直接 mask dependency”这个想法可能方向没错，但当前固定阈值形式还不够好。

## Interim Takeaway

当前最稳的结论是：

1. `global canonical` 在若干数据集上确实更强，而且这件事大概率不是简单由优化配置造成的。
2. 但 `global canonical` 的代价显著更高，因此仍应把它作为 strong baseline，而不是默认主线。
3. 对 relation-wise dependency aggregation 来说，当前最有希望的改进方向是：
   - `dependency density scaling`
   - `explicit global rule/dependency ratio`
   - dataset-specific 的 `surprisal initialization`
