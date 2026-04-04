# Aggregation Strategy Audit

本文档总结当前 relation-wise `aggregation.py` 路径下的 aggregation 策略，目标是回答两个问题：

1. 现在的 dependency 到底是怎么被用进去的
2. 从当前实现出发，哪些改动最可能提升 dependency 的效用

本说明只基于当前已有代码和结果文件，不新增实验、不重算结果，也不覆盖旧文档 [`reports/0403/README.md`](/home/sy/RuleDep/reports/0403/README.md)。

当前主要参考：

- [`aggregation.py`](/home/sy/RuleDep/aggregation.py)
- [`structural_filtered_comparison.csv`](/home/sy/RuleDep/reports/0403/structural_filtered_comparison.csv)
- [`relation_dependency_analysis.csv`](/home/sy/RuleDep/reports/0403/relation_dependency_analysis.csv)
- [`dependency_relation_analysis.md`](/home/sy/RuleDep/reports/0403/dependency_relation_analysis.md)
- [`structural_rd_filtered/config.json`](/home/sy/RuleDep/data/codex-m/aggregation/structural_rd_filtered/config.json)
- [`synergy_filtered/config.json`](/home/sy/RuleDep/data/codex-m/aggregation/synergy_filtered/config.json)
- [`structural_r2d3_filtered/config.json`](/home/sy/RuleDep/data/FB15k-237/aggregation/structural_r2d3_filtered/config.json)

## Current Default Strategy

当前主实验默认是 `LinearAggregator`。这一点可以直接从代表性配置看到：

- [`structural_rd_filtered/config.json`](/home/sy/RuleDep/data/codex-m/aggregation/structural_rd_filtered/config.json)
- [`synergy_filtered/config.json`](/home/sy/RuleDep/data/codex-m/aggregation/synergy_filtered/config.json)
- [`structural_r2d3_filtered/config.json`](/home/sy/RuleDep/data/FB15k-237/aggregation/structural_r2d3_filtered/config.json)

这些当前主线配置有几个共同点：

- `model = LinearAggregator`
- `pos = auto_sqrt`
- `sign_constraint = true`
- `sign_constraint_dependency = false`
- `train_rule_in_dependency_stage = true`
- stage1 用 `0.01,0.005,0.001`
- stage2 用更保守的单阶段小学习率，默认只取 stage1 schedule 中最小的那个

当前整体训练流程是：

1. stage1 先训练 rule-only 模型
2. stage2 从 stage1 最优 checkpoint 初始化出带 dependency 的模型
3. 用 validation MRR 比较 `stage2` 和 `stage1`
4. 只有 `best_valid_stage2.mrr > best_valid_stage1.mrr` 时才接受 stage2

这意味着当前系统不是 joint-from-scratch，也不是简单把 rule 和 dep 一起从零端到端训练。

## Answer To 13 Questions

### 1. BCE pos 用的是 sqrt 吗？

是的，当前主实验默认是 `auto_sqrt`。  
实现位置在 [`aggregation.py`](/home/sy/RuleDep/aggregation.py) 的 `resolve_pos_weight()`。逻辑是：

- `ratio = num_negative / num_positive`
- `pos_weight = sqrt(ratio)`

代表配置也都在用这个默认：

- [`structural_rd_filtered/config.json`](/home/sy/RuleDep/data/codex-m/aggregation/structural_rd_filtered/config.json)
- [`synergy_filtered/config.json`](/home/sy/RuleDep/data/codex-m/aggregation/synergy_filtered/config.json)

结论：当前主线确实是 `BCE + sqrt(pos ratio)`。

### 2. 有没有限制 rule weight positive？

有，当前主实验默认限制 rule weight 非负。

原因是：

- `sign_constraint = true`
- 在 `LinearAggregator.forward()` 中，rule raw parameter 会先取 embedding，再做平方

因此真正生效的是：

- `effective_rule_weight = raw_rule_weight^2`

所以虽然存储的 raw parameter 可以是任意实数，但生效权重是非负的。

### 3. 有没有使用 surprisal 而不是 conf 来初始化 rule weight？

当前主实验没有。

当前 `LinearAggregator.init_weights()` 使用的是 rule confidence：

- 先从规则文件读 `conf = num_true / (num_preds + 5)`
- 如果启用 sign constraint，则 raw init 设成 `sqrt(conf)`
- forward 时再平方，恢复到接近 `conf`

判断：  
这是一个很值得补的对照。即使继续保留 `LinearAggregator`，也完全可以只把 rule init 改成 `surprisal = -log(1 - conf)`。理论上它会比直接用 `conf` 更 robust，尤其当高 confidence 规则需要更大区分度时。

### 4. 有没有限制 dependency weight positive？

当前主实验默认没有。

因为：

- `sign_constraint_dependency = false`

这意味着 dependency 的 raw embedding 直接作为 effective weight，用正负号共同表达：

- positive dependency contribution
- negative dependency contribution

从代表性文件也能直接看到正负混合，例如：

- [`dependency-final-10.csv`](/home/sy/RuleDep/data/codex-m/aggregation/structural_rd_filtered/dependency-final-10.csv)
- [`dependency-final-100.csv`](/home/sy/RuleDep/data/FB15k-237/aggregation/structural_r2d3_filtered/dependency-final-100.csv)
- [`dependency-final-0.csv`](/home/sy/RuleDep/data/YAGO3-10/aggregation/structural_r3d6_filtered/dependency-final-0.csv)

结论：当前 dep 默认是可正可负的，不受正约束。

### 5. 有没有使用 surprisal 初始化 dep weight？

没有。

当前 dependency 初始化有两种：

- 默认：全 0
- 如果 `init_dep_with_lift = true`：用 `0.1 * lift`

也就是说，当前 dep 初始化不是 conf，也不是 surprisal，而是：

- 零初始化
- 或 lift 初始化

结论：当前没有 surprisal-style dependency initialization。

### 6. 有没有对 dep 进行缩放，以避免过于稠密的 dep 作用于一个 query？

当前没有显式缩放。

需要区分两件事：

- 有 `dependency_chunk_size`
- 但这只是计算分块，不是 score normalization

当前 forward 里，如果某个 query 激活了很多 dependency pair，这些 pair 的贡献会直接线性求和，没有额外：

- `/ n`
- `/ sqrt(n)`
- `/ log(1+n)`
- capped sum

结论：当前没有显式 density control，这正是我认为最值得优先修的点之一。

### 7. RD 配置中，会不会有一个整体的 rule weight ratio 和 dep weight ratio？

当前 RD 没有这种显式全局 ratio。

`RD` 中只有：

- per-rule weight
- per-dependency weight
- bias

没有单独的：

- `gamma_rule`
- `gamma_dep`

也没有 query-level 或 relation-level 的全局 mixing coefficient。

结论：当前 rule 和 dep 的相对强度完全交给单条参数去学，没有一个显式总开关。

### 8. rule weight 和 dependency 应该一起训练还是分 stage 训练？

当前实现是分 stage 训练。

现状：

1. stage1 训练 rule-only
2. stage2 在 stage1 checkpoint 基础上继续训练带 dep 的模型

我对当前策略的判断是：

- 分 stage 是合理的，比 joint-from-scratch 稳
- 尤其在 dependency 稀疏、relation 差异大时，先把 rule marginals 学稳更合适

所以当前我不建议直接改成 joint-from-scratch 作为主线。  
更值得比较的是：stage2 到底该 freeze 多少 rule 参数。

### 9. dep 应该作为加项，还是作为已有规则的乘法项？

当前实现是加项，不是乘法项。

也就是：

- `score = rule_score + dep_score + bias`

而不是：

- `score = rule_score * f(dep)`
- 或每条 rule 再被 dependency gate

我目前的判断是：

- 乘法项从表达上更强
- 但当前 active pair 已经是 hot path
- 如果把 dep 做成乘法 interaction，计算量和实现复杂度都会明显上升

所以现在不建议优先走乘法项，除非先把“线性 dep 为什么不够”这件事验证得更明确。

### 10. 如果分 stage 训练，在 stage2 训练 dependency 时，是否有必要让 rule weight 也能训练？

当前默认是“有必要”，因为主实验里：

- `train_rule_in_dependency_stage = true`

并且 optimizer 做了分组：

- rule / rule-type / bias 用 `0.1x lr`
- dependency / dependency-type 用主学习率

也就是说，当前实现已经不是完全 freeze，而是保守 joint finetune。

我的判断：

- 这不是明显错误
- 但它很可能是覆盖率不足的一个来源

因为 stage2 一旦继续改 rule，就可能把本来已经不错的 stage1 拖偏。  
因此这里非常值得做三组对照：

- freeze rule
- 当前 0.1x jointly finetune
- 只训练 bias/type/dep

### 11. rule dependency 是否有可能作用过量？

有可能，而且当前没有显式机制阻止它。

当前没有约束：

- dep 总绝对值不能超过 rule 总绝对值
- dep 对同一 query 的累积值不能过大
- 某类 relation 的 dep 不得主导最终分数

所以理论上完全可能出现：

- `|dep_score| > |rule_score|`

从代码上看，这种情况不会被显式截断。  
从结果上看，relation-level 负收益 relation 的存在，也和这个风险一致。

### 12. 训练 rule weight 已经学到了边际效应，那么 dep 的意义是什么？

dep 的意义在于它表达的是 rule marginals 之外的 pairwise interaction。

rule weight 学到的是：

- 单条规则命中时，平均而言值不值得信

dep 学到的是：

- 两条规则同时命中时，是否出现额外的 synergy / redundancy

也就是说：

- rule = marginal effect
- dep = interaction effect

这在理论上当然是有新增信息的。  
问题不在于 dep 没意义，而在于当前实现下：

- 这些 interaction 是否被约束得足够好
- 是否在合适的 relation 上被激活
- 是否没有因为过量累加而伤害强 baseline relation

### 13. 如果某条 rule 的权重训练后接近 0，那么相关 dep 是否还有意义？

当前实现下，相关 dep 仍然可能有意义，也仍然可能生效。

原因是当前 dep 是否触发，看的是：

- 这两条 rule 是否在该 query 中被激活

而不是看：

- 这两条 rule 的训练后边际权重是否还足够大

所以即便某条 rule 的边际权重已经接近 0，只要它还命中，相关 dep 仍会参与打分。  
这就是当前实现的一个潜在问题：dep 与弱 rule 脱钩。

我的判断是：  
这很可能会制造噪声，因此值得加一个 `rule-strength-aware dep gating/pruning`。

## Current Risks

当前 dependency 不是“完全没信号”，真正的问题更像下面四个。

### 1. Coverage 不足

从 relation-level 分析看，dependency 的显著提升覆盖率还不够高。  
这意味着当前主问题不是“有没有信号”，而是“有效 relation 太少”。

### 2. dep 无约束累加

当前没有按 active dep 数做显式归一化。  
在 dependency 很密的 relation 或 query 上，dep 很容易线性累加过量。

### 3. stage2 易漂移

当前 stage2 默认继续训练 rule / bias / type。  
这会让 dependency stage 不只是“补 interaction”，而是也在重写 stage1 的 marginal structure。

### 4. dep 与弱 rule 脱钩

当前 rule 边际权重很弱时，相关 dep 仍可能触发。  
这说明 dep 有可能把本来已被 stage1 判为不重要的 rule pair 又重新放大。

## Most Promising Improvements

下面是我认为最值得按顺序尝试的改动。

### 1. dep scaling

优先级最高。

核心目标是避免 dense dependency query 的 dep 直接线性爆掉。  
最小版本就够：

- `dep_score / sqrt(num_active_dep)`
- 或 `dep_score / log(1 + num_active_dep)`

我更推荐先试：

- `1 / sqrt(n)`

因为它最简单，也比较稳。

### 2. explicit dep ratio

在 RD 中加入一个显式总开关：

- `score = rule_score + gamma_dep * dep_score`

这样至少能让模型先学“这个 relation 整体该信 dep 到什么程度”，而不是把一切都压到单个 dependency weight 上。

### 3. stage2 freeze-vs-finetune

我认为这会直接影响覆盖率。

最值得对比的是三种：

- freeze rule
- 当前 `0.1x` jointly finetune
- 只训练 bias/type/dep

如果 dependency 的真正价值是 interaction，那么 freeze 或半 freeze 往往更容易保住 stage1。

### 4. surprisal-based rule init

这是一个很自然、也很干净的对照。

当前 rule init 仍然基于 confidence。  
我认为值得至少试：

- 继续保留 `LinearAggregator`
- 只把初始 rule weight 改成 surprisal 风格

理论上这会让高 confidence 区间的区分更 robust，同时避免引入新的 aggregation 形式。

### 5. rule-strength-aware dep gating / pruning

如果某条 rule 的边际权重已经接近 0，就不该让它继续通过 dep 产生很大作用。  
最小可行版本可以是：

- 训练后或 stage2 中，对低权重 rule 相关的 dep 直接 mask
- 或 dep score 乘上 endpoint rule strength 的函数

这一步非常契合当前实现的已知弱点。

### 6. 乘法型 dep

这是可研究方向，但我不建议优先。

原因很简单：

- 当前 active pair 已经是热路径
- 乘法 interaction 会明显增加 forward 复杂度
- 在覆盖率问题没先搞清楚之前，先把 dep 从“无约束加项”变成“更复杂乘法项”，风险较大

我更建议先把：

- scaling
- ratio
- stage2 freeze
- surprisal init

这些低风险改完，再考虑乘法结构。

## Takeaway

当前 dependency 的问题不是“没有信息”，而是“有信号但覆盖率不足”。  
从实现上看，造成这个现象的最可能原因不是 dep 特征本身，而是当前 aggregation 策略还偏弱：

- dep 线性累加但没有显式缩放
- 没有总量 ratio
- stage2 默认会改 rule
- dep 对弱 rule 没有抑制

所以如果目标是提高 dependency 的实际效用，我认为最值得优先推进的顺序是：

1. `dep scaling`
2. `explicit dep ratio`
3. `freeze-vs-finetune in stage2`
4. `surprisal-based rule init`
5. `rule-strength-aware dep gating`

如果这些改动之后 coverage 仍然上不去，再考虑更重的结构改造，例如乘法型 dependency。
