# Report Notes

本目录用于汇总当前实验现象与阶段性结论。这里的结论都基于当前已有文件，尤其是：

- [all_results_summary.csv](/home/sy/RuleDep/reports/all_results_summary.csv)
- [structural_filtered_comparison.csv](/home/sy/RuleDep/reports/structural_filtered_comparison.csv)
- [relation_relative_gain_gt_3pct_best_structural.csv](/home/sy/RuleDep/reports/relation_relative_gain_gt_3pct_best_structural.csv)
- [relation_relative_gain_lt_minus_3pct_best_structural.csv](/home/sy/RuleDep/reports/relation_relative_gain_lt_minus_3pct_best_structural.csv)
- [dataset_size_rule_dependency_stats.csv](/home/sy/RuleDep/reports/dataset_size_rule_dependency_stats.csv)

## 1. Dependency 到底有没有用？

结论：有用，而且通常是有稳定增益的；但增益幅度通常不大，且依赖数据集与 relation。

支持证据：

- 从总表看，很多带 dependency 的配置都优于 `eval-maxplus` / `eval-noisyor`。
- 在 7 个数据集上，最佳结果大多来自包含 dependency 的配置，而不是纯 rule baseline。
- 对 structural 系列，`stage2` 相对 `stage1` 的最终 `test` 往往仍然是正增益。
- relation 级别上，存在不少 `> 3%` 的真实增益 relation，见 [relation_relative_gain_gt_3pct_best_structural.csv](/home/sy/RuleDep/reports/relation_relative_gain_gt_3pct_best_structural.csv)。

但需要注意：

- dependency 的收益通常不是“质变级”大幅提升，而是 `+0.001` 到 `+0.01` 量级的累计改善。
- 也存在明显负增益 relation，见 [relation_relative_gain_lt_minus_3pct_best_structural.csv](/home/sy/RuleDep/reports/relation_relative_gain_lt_minus_3pct_best_structural.csv)。
- dependency 的收益具有很强的 relation 选择性，不适合简单理解为“加了就一定更好”。

### 用处大吗？

中等偏小，但真实存在。

- `FB15k-237`：最佳 structural 配置为 `structural_r2d3 = 0.351555`，明显高于 `eval-noisyor = 0.337685`。
- `codex-m`：最佳 structural 配置为 `structural_r3d6 = 0.343299`，高于 `eval-noisyor = 0.341133` 量级附近的 baseline 线。
- `YAGO3-10`：最佳 structural 配置 `0.576659`，也高于纯 rule 型配置。
- `WN18RR`：dependency 有用，但收益更小，很多配置都非常接近。

### 计算复杂度大吗？

大。

直接现象：

- `#synergy` 和 `#redundancy` 数量非常大，见 [dataset_size_rule_dependency_stats.csv](/home/sy/RuleDep/reports/dataset_size_rule_dependency_stats.csv)。
- 例如：
  - `FB15k-237`: `#synergy = 3021796`, `#redundancy = 9908013`
  - `hetionet`: `#synergy = 24410345`, `#redundancy = 18865199`
  - `codex-m`: `#synergy = 1055428`, `#redundancy = 1578683`
- filtered 后虽然会显著下降，但规模仍然很大。

所以 dependency 的问题不是“有没有信号”，而是：

- 信号有
- 但计算开销很高
- 而且高复杂度未必总能换来稳定大增益

这意味着 dependency 更适合：

- 做 filtered / relation-local 的选择性使用
- 做更强的 relation 选择或候选筛选
- 而不是无条件全量堆上去

## 2. Relation-wise model vs Global model

当前可先写的结论：

- 从实现上看，两者不是同一个训练问题。
- relation-wise 与 global 不能仅按“公式一样”视为同一模型。

关键差异：

- global old model 在所有 relation 上共享同一套 rule weight，仅 relation bias 分开。
- relation-wise model 为每个 relation 单独建模，只使用该 relation 的局部规则。
- old global canonical 还存在不同的 checkpoint 选择逻辑。

当前是否更好：

- 这部分暂不下最终结论。
- 当前已有一个很强的信号：`KG20C` 上 old canonical 曾经跑到 `0.23957`，高于当前 relation-wise ensemble `0.23537`。
- 这说明 relation-wise 并不天然支配 global。

运行时间：

- 暂留空。
- 需要等 old canonical 新一轮统一跑完，再补充 relation-wise vs global 的实际 wall-clock 对比。

待补：

- `TODO`: old canonical 新设定全部跑完后，补一张 runtime 对照表。

## 3. synergy 和 redundancy 都有用吗？

结论：两者都不是“完全没用”，但 usefulness 很明显依赖数据集；目前看 synergy 更稳定，redundancy 更 dataset-dependent。

从总表看：

- `FB15k-237`
  - `synergy = 0.348182`
  - `redundancy = 0.347210`
  - 两者都有用，synergy 略强
- `KG20C`
  - `synergy = 0.232885`
  - `redundancy = 0.229664`
  - synergy 明显更强
- `WN18RR`
  - `synergy = 0.499568`
  - `redundancy = 0.499462`
  - 二者都很弱，且非常接近
- `codex-m`
  - `redundancy = 0.342124`
  - `synergy` 不在最佳前列
  - redundancy 在该数据集上更有竞争力
- `codex-l`
  - synergy 和 redundancy 都接近，但 synergy 略强

阶段性判断：

- synergy 不是在所有数据集都最好，但通常更稳定。
- redundancy 不是没用，但更像“在某些数据集或某些 relation 上起作用”。
- 它们的作用方式可能不同：
  - synergy 更像补充互补证据
  - redundancy 更像抑制重复证据或做结构去重

所以更合理的结论是：

- 不是“只有一个有用”
- 而是“二者都可能有用，但数据集依赖很强”

## 4. type 权重有没有用？它学到了什么？

结论：type 权重并不一致优于不使用 type，但在部分数据集上是有明显信号的；效果高度依赖数据集。

从 [structural_filtered_comparison.csv](/home/sy/RuleDep/reports/structural_filtered_comparison.csv) 看：

- `FB15k-237`: `R2D3` 最好
- `KG20C`: `RD` 最好
- `WN18RR`: `RD` 最好
- `YAGO3-10`: `R2D3` 最好
- `codex-l`: `RD` 最好
- `codex-m`: `R3D6` 最好
- `hetionet`: `R2D3` 最好

这说明：

- type 共享权重不是 universally better
- 更细的类型划分也不是永远更强
- 最适合的粒度与数据集有关

### 学到的 type weight 有什么特点？

从若干代表性 relation 的 `metric-*.json` 看，主要特征是：

1. 很多 relation 上，`Uc` 或 `U` 类型会被放大得更多

例如 `codex-m / structural_r3d6`：

- relation `16`
  - `B: 1.0 -> 0.946452`
  - `Uc: 1.0 -> 1.0848672`
  - `Ud: 1.0 -> 0.9312151`
- relation `27`
  - `B: 1.0 -> 0.9372675`
  - `Uc: 1.0 -> 2.5911515`
  - `Ud: 1.0 -> 1.782889`

这说明在 `codex-m` 上，模型倾向于：

- 压低 `B`
- 放大 `Uc`
- 部分 relation 也放大 `Ud`

2. dependency type 权重通常只做温和缩放，而不是剧烈翻转

例如 `FB15k-237 / structural_r2d3`：

- relation `147`
  - `["B","B"]`: `1.0 -> 1.033335`
  - `["B","U"]`: `1.0 -> 1.0353349`
  - `["U","U"]`: `1.0 -> 1.0423061`

这类数据集上，type 权重更像轻微重加权，而不是完全重写原模型。

3. 某些 relation 上 type 权重几乎不学

例如一些 `selected_stage = rule_only` 的 relation，type 权重会维持在 `1.0` 附近。

这说明：

- 并不是所有 relation 都需要 type bias
- 数据集内部也存在强烈异质性

### 从这些权重可以反推什么数据集特点？

阶段性推测：

- `codex-m` 更适合细粒度类型建模，因为 `R3D6` 最优，且很多 relation 的 type 权重明显偏离 `1`
- `FB15k-237` 和 `hetionet` 更适合较粗的 `R2D3`
- `KG20C`、`WN18RR`、`codex-l` 上，额外 type bias 的收益较小，说明这些数据集可能更依赖“是否有 dependency”，而不是“dependency 再怎么细分”

## 5. dep 在不同 relation 上是否表现不同？

结论：非常不同，而且这种差异是 dependency 现象里最明显的部分。

从 [relation_relative_gain_gt_3pct_best_structural.csv](/home/sy/RuleDep/reports/relation_relative_gain_gt_3pct_best_structural.csv) 和 [relation_relative_gain_lt_minus_3pct_best_structural.csv](/home/sy/RuleDep/reports/relation_relative_gain_lt_minus_3pct_best_structural.csv) 可以直接看到：

- 有些 relation 真正受益很大
- 有些 relation 即便最终接受了 dependency，仍会明显退化

### 哪些 relation 上表现更好？

当前表里，提升明显的 relation 往往有这些特点：

- `selected_stage = dependency`
- test triple count 通常不算特别大，很多是中小 relation
- 关系语义更像可由多条证据互补支持的关系

代表例子：

- `FB15k-237`
  - `/award/award_winner/...`
  - `/location/location/adjoin_s...`
  - `/film/film/prequel`
  - `/tv/tv_program/languages`
- `YAGO3-10`
  - 一部分事件参与、地理包含、作品关系类 relation

这些 relation 的共同点通常是：

- 可由多条规则形成互补证据
- stage1 不是已经接近饱和
- relation 结构不止一个单一强规则

### 哪些 relation 上表现不好？

负向 relation 里，目前都是真正被选中了 dependency 的 relation，而不是伪负例。

典型例子：

- `FB15k-237`
  - `/film/film/written_by`
  - `/media_common/netflix_genre/titles`
  - `/organization/non_profit_organization/registered_with...`
  - `/base/locations/continents/countries_within`
- `YAGO3-10`
  - `participatedIn`
  - `wroteMusicFor`
- `codex-m`
  - `P30`

这些 relation 可能的共同点：

- stage1 本身已经很强，dependency 容易过调
- relation 的有效规则可能高度冗余，额外 dependency 只是把已有信号重复放大
- 部分 relation test triple count 很小，容易受 validation 选择噪声影响

### 这种方法擅长完成哪些任务？

目前看更擅长：

- 需要多条规则共同支持的 relation
- 单条 rule 不够、但局部结构组合能补充信息的 relation
- stage1 仍有一定错误空间的 relation

不太擅长：

- 已有单规则非常强、stage1 已接近饱和的 relation
- 重复证据过多、dependency 容易把偏差再放大的 relation
- 极小样本 relation

## 当前最重要的阶段性结论

1. dependency 是有用的，但不是“无条件大幅有用”。
2. dependency 的收益高度 relation-dependent。
3. synergy 和 redundancy 都可能有效，但 synergy 更稳定。
4. type 权重不是 universally better，它明显依赖数据集。
5. `codex-m` 更支持细粒度 type；`KG20C/WN18RR/codex-l` 更像不太需要复杂 type bias。
6. 当前最值得继续深入的是 relation-level 分析，而不是只看全局平均分。

## 后续建议

- 补齐 old canonical 新版结果后，再写 `global vs relation-wise` 的正式结论。
- 对 `KG20C` 做 old canonical vs 当前 relation-wise 的 relation 级对比。
- 进一步分析：
  - 正增益 relation 的 rule count / dependency count / B/Uc/Ud 构成
  - 负增益 relation 是否更容易出现 stage1 高饱和或高冗余
