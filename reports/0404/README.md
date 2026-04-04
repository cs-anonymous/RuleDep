# 0404 Notes

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
- `sign_constraint = True`
- `shuffle_train = False`
- `batch_size = 4096`
- `type_grouping = none`

### Accuracy

当前已经完成并确认的结果：

| dataset | canonical old | matched relational stage1 | delta |
| --- | ---: | ---: | ---: |
| KG20C | 0.23940 | 0.22913 | -0.01027 |
| WN18RR | 0.49950 | 0.50010 | +0.00060 |
| codex-m | 0.34554 | 0.34197 | -0.00356 |

说明：

- 在 `KG20C` 和 `codex-m` 上，即使把 relational stage1 的训练 recipe 对齐到 old canonical，仍然存在明显差距。
- 在 `WN18RR` 上，relation-wise 反而略好。
- 因此，`global > relational` 这件事大概率不是简单由 `pos/lr/epoch/evaluate_every` 这类优化配置造成的。

### 对默认 RD Stage1 的提升

matched relational stage1 相对当前默认 `structural_rd_filtered__stage1` 的提升只有大约 `+0.001`：

| dataset | matched stage1 | current RD stage1 | delta |
| --- | ---: | ---: | ---: |
| KG20C | 0.22913 | 0.22830 | +0.00083 |
| WN18RR | 0.50010 | 0.49916 | +0.00095 |
| codex-m | 0.34197 | 0.34084 | +0.00114 |

这进一步说明：

- 调整成 old 风格的优化配置确实有一点帮助；
- 但帮助很小，不足以解释 `KG20C` 和 `codex-m` 上 global/canonical 的优势。

## Why The Gap Is Probably Not Just Optimization

已经排除或基本排除的因素：

1. 不是 `pos/lr/epoch/evaluate_every` 的简单问题。
   原因：matched control 只带来约 `+0.001` 的提升。

2. 不是 hidden type-weight / score-scale 参数导致的差异。
   在 `type_grouping = none` 时，当前 `aggregation.py` 不会启用 `rule_type`、`dependency_type` 或 `global_scale`。

3. 不是训练集中混入了跨 relation 的 rule。
   已检查：
   - `KG20C`: 5 个 relation 全部 `0` 个 cross-relation rule mention
   - `codex-m`: `affected_count = 0`

4. 不像是单纯过拟合。
   之前的 relation-level valid/test 对比显示，在 `KG20C`、`codex-m`、`FB15k-237` 上，global 在 valid 上也往往更高。

因此，目前更合理的解释是：

- `global canonical` 和当前 `relation-wise` 在 forward 形式上很接近，但训练范式仍然不同；
- 差距更像是 `global vs relation-wise` 本身的 inductive bias 差异；
- 这种差异具有强烈的数据集相关性：
  - `KG20C / codex-m / FB15k-237` 更偏向 global
  - `WN18RR` 更偏向 relation-wise specialization

## Cost Comparison

`global canonical old` 的计算代价明显更高。下面用 canonical log 的墙钟时间，对比 matched relational stage1 的 `sweep` 时间：

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

### Memory

纯参数显存上，global 通常也更大，因为它要把全数据集所有 rule embedding 一次性放到模型里，而 relation-wise 只放当前 relation 的局部规则。

rule 数量统计：

| dataset | total rules | avg rules / relation | max rules / relation |
| --- | ---: | ---: | ---: |
| KG20C | 154790 | 30958.0 | 101322 |
| WN18RR | 76909 | 6991.7 | 46349 |
| codex-m | 518279 | 11266.9 | 268476 |
| FB15k-237 | 1737378 | 7330.7 | 299952 |

因此：

- 相比“单 relation 的 local model”，global 在参数规模上通常更大；
- 不过由于 embedding 维度只有 `1`，纯参数显存并不是最大瓶颈；
- 更突出的代价其实是 wall-clock、CPU memory、checkpoint copy 和整体训练流程。

## Interim Takeaway

当前最稳的结论是：

1. `global canonical` 在若干数据集上确实更强。
2. 这种优势大概率不是简单由优化超参不一致造成的。
3. `global canonical` 的代价显著更高，尤其是运行时间。
4. 因此在论文里更合理的定位是：
   - 把 global canonical 作为强 baseline 保留；
   - 但 relation-wise 仍然有重要意义，因为它便宜得多，也更适合作为 dependency 分析平台。

