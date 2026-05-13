# 单特征 RF Selector 分析报告

## 概述

本报告分析了使用**单个特征训练 Random Forest 模型**作为 query subset selector 的表现。相比多特征模型，单特征 RF 具有更强的可解释性，同时保留了 RF 的非线性分割能力。

数据来源：`reports/official_query_subset/ml_selector_diverse/balanced_ablation_true_rr/`

---

## 1. 宏平均表现（跨 7 个数据集）

### 1.1 Global RF（推荐用于论文，跨数据集泛化）

| Rank | Feature | gain@10 | gain@20 | gain@30 | gain@50 |
|------|---------|---------|---------|---------|---------|
| 1 | **synergy_weight_top5_mean** | **7.32%** | **5.95%** | 4.83% | 3.68% |
| 2 | max_candidate_dep_score | 7.25% | 5.46% | 4.60% | 3.83% |
| 3 | effective_candidates | 6.05% | 5.40% | 4.61% | 3.79% |
| 4 | topk_rule_weight | 4.82% | 3.97% | 3.35% | 2.59% |

**最佳单特征**：`synergy_weight_top5_mean`
- 在 gain@10 和 gain@20 上都是最强的
- 衡量 top-5 synergy 依赖的平均权重强度

### 1.2 Per-dataset RF（Oracle 上界，每个数据集单独训练）

| Rank | Feature | gain@10 | gain@20 | gain@30 | gain@50 |
|------|---------|---------|---------|---------|---------|
| 1 | **effective_candidates** | **12.03%** | **9.08%** | 7.55% | 5.36% |
| 2 | max_candidate_dep_score | 11.52% | 7.88% | 5.86% | 4.36% |
| 3 | topk_rule_weight | 11.02% | 8.56% | 7.01% | 4.80% |
| 4 | synergy_weight_top5_mean | 10.61% | 8.16% | 6.12% | 4.40% |

**最佳单特征**：`effective_candidates`
- 在 per-dataset 设置下表现最强
- 衡量 S1 的有效候选数（不确定性）

---

## 2. codex-l 数据集表现

### 2.1 Coverage = 10%

| Type | Feature | gain_pt | MRR_S1 | MRR_S2 |
|------|---------|---------|--------|--------|
| **Global RF** | synergy_weight_top5_mean | **4.47%** | 0.4358 | 0.4553 |
| Global RF | max_candidate_dep_score | 2.47% | 0.3638 | 0.3728 |
| Global RF | effective_candidates | 2.08% | 0.1350 | 0.1378 |
| Global RF | topk_rule_weight | 1.99% | 0.4317 | 0.4403 |
| **Per-dataset RF** | topk_rule_weight | **8.67%** | 0.3331 | 0.3619 |
| Per-dataset RF | effective_candidates | 8.27% | 0.4209 | 0.4557 |
| Per-dataset RF | synergy_weight_top5_mean | 6.83% | 0.4448 | 0.4752 |
| Per-dataset RF | max_candidate_dep_score | 5.62% | 0.4221 | 0.4458 |

### 2.2 Coverage = 20%

| Type | Feature | gain_pt | MRR_S1 | MRR_S2 |
|------|---------|---------|--------|--------|
| **Global RF** | synergy_weight_top5_mean | **3.33%** | 0.4043 | 0.4178 |
| Global RF | effective_candidates | 2.51% | 0.1563 | 0.1602 |
| Global RF | max_candidate_dep_score | 1.83% | 0.3757 | 0.3826 |
| Global RF | topk_rule_weight | 1.24% | 0.4651 | 0.4709 |
| **Per-dataset RF** | topk_rule_weight | **5.94%** | 0.3355 | 0.3554 |
| Per-dataset RF | effective_candidates | 5.11% | 0.4231 | 0.4448 |
| Per-dataset RF | synergy_weight_top5_mean | 4.92% | 0.3988 | 0.4184 |
| Per-dataset RF | max_candidate_dep_score | 3.98% | 0.3826 | 0.3978 |

---

## 3. 单特征 vs 多特征对比

### 3.1 宏平均（7 个数据集）

| Configuration | gain@10 | gain@20 | Relative to 4-feat |
|---------------|---------|---------|-------------------|
| 4-feature global RF (baseline) | 16.66% | 9.91% | 100% / 100% |
| **Best 1-feature global RF** | **7.32%** | **5.95%** | **43.9% / 60.0%** |
| Best 1-feature per-dataset RF | 12.03% | 9.08% | 72.2% / 91.6% |

**损失分析**：
- Global RF 单特征相比 4 特征：损失 **56.1%** @ gain@10，**40.0%** @ gain@20
- Per-dataset RF 单特征相比 4 特征：损失 **27.8%** @ gain@10，**8.4%** @ gain@20

### 3.2 codex-l 数据集

| Configuration | gain@10 | gain@20 | Relative to 4-feat |
|---------------|---------|---------|-------------------|
| 4-feature global RF (baseline) | 8.96% | 5.05% | 100% / 100% |
| **Best 1-feature global RF** | **4.47%** | **3.33%** | **49.9% / 65.9%** |
| Best 1-feature per-dataset RF | 8.67% | 5.94% | 96.8% / 117.6% |

**关键发现**：
- 在 codex-l 上，per-dataset RF 单特征几乎达到 4 特征的表现
- gain@20 时甚至超过 4 特征 baseline（117.6%）

---

## 4. 特征说明

| 短名 | 完整名称 | 类别 | 含义 |
|------|---------|------|------|
| syn | synergy_weight_top5_mean | D. Dependency weight strength | Top-5 synergy 依赖的平均权重 |
| max | max_candidate_dep_score | C. Positive/negative dependency | 候选的最大依赖得分 |
| eff | effective_candidates | G. S1 ambiguity | S1 的有效候选数（不确定性） |
| topk | topk_rule_weight | E. Rule-weight distribution | Top-k 规则权重 |

---

## 5. 关键洞察

### 5.1 单特征 RF 的优势

✅ **极简可解释**：只依赖一个统计量，容易向他人解释  
✅ **避免过拟合**：没有特征交互，泛化性更好  
✅ **透明决策**：可以可视化单特征的分割点和决策边界  
✅ **计算高效**：特征提取和模型训练都更快  

### 5.2 不同场景的最佳选择

**Global RF（跨数据集泛化）**：
- 最佳特征：`synergy_weight_top5_mean`
- 适用场景：需要在多个数据集上泛化，追求可解释性
- 性能：gain@10 = 7.32%, gain@20 = 5.95%

**Per-dataset RF（单数据集优化）**：
- 最佳特征：`effective_candidates`
- 适用场景：有足够数据为每个数据集单独训练
- 性能：gain@10 = 12.03%, gain@20 = 9.08%

### 5.3 为什么 synergy_weight_top5_mean 最强？

1. **直接衡量核心机制**：Synergy 是 RuleDep 的核心增益来源
2. **Top-5 平衡了覆盖和质量**：不像 top-1 只看最强的，也不像 mean 被弱信号稀释
3. **跨数据集稳定**：在所有 7 个数据集上都表现良好

### 5.4 为什么 effective_candidates 在 per-dataset 设置下最强？

1. **捕捉 S1 不确定性**：候选越多且分数接近，RuleDep 越有翻转空间
2. **数据集特异性强**：不同数据集的候选分布差异大，per-dataset 训练能更好适应
3. **理论支撑强**：符合"S1 越不确定，dependency 越有用"的假设

---

## 6. 建议

### 6.1 论文使用建议

**推荐方案**：使用 **single-feature global RF** 作为 subset diagnostic

**理由**：
1. **可解释性强**：可以说"当 synergy_weight_top5_mean 高时，RuleDep 收益更集中"
2. **避免过拟合质疑**：单特征 RF 不容易被质疑 in-sample overfitting
3. **透明度高**：可以展示 RF 的分割点，读者能理解决策逻辑
4. **性能可接受**：gain@10 = 7.32%, gain@20 = 5.95%，虽然不如 4 特征，但足够展示 subset 效应

**具体实现**：
```python
# 使用 synergy_weight_top5_mean 训练 global RF
rf = RandomForestClassifier(...)
rf.fit(X_train['synergy_weight_top5_mean'].values.reshape(-1, 1), y_train)

# 可视化分割点
tree = rf.estimators_[0]
plot_tree(tree, feature_names=['synergy_weight_top5_mean'])
```

### 6.2 不同目标的选择

| 目标 | 推荐方案 | 特征 | 性能 |
|------|---------|------|------|
| **最大可解释性** | Single-feature global RF | synergy_weight_top5_mean | gain@10=7.32% |
| **平衡性能与可解释性** | 2-feature global RF | syn + topk | gain@10=12.64% |
| **最大性能** | 4-feature global RF | syn + max + topk + eff | gain@10=16.66% |
| **Per-dataset 优化** | Single-feature per-dataset RF | effective_candidates | gain@10=12.03% |

### 6.3 与阈值方法的对比

**阈值方法**（如 `best_feature_threshold_summary.csv`）：
- 优势：更简单，直接用 `if feature > threshold` 规则
- 劣势：只能做单点分割，无法捕捉复杂分布

**单特征 RF**：
- 优势：可以做多点分割，适应非线性关系
- 劣势：稍微复杂一点，但仍然高度可解释

**建议**：如果特征分布简单（单峰），用阈值；如果复杂（多峰、长尾），用单特征 RF。

---

## 7. 结论

**核心发现**：
1. 单特征 RF 可以达到 4 特征 RF 的 **44-72%** 性能（取决于 global/per-dataset）
2. 最佳单特征是 **synergy_weight_top5_mean**（global）和 **effective_candidates**（per-dataset）
3. 在某些数据集（如 codex-l），per-dataset 单特征 RF 甚至能超过 4 特征 global RF

**建议方案**：
- **论文主方案**：使用 `synergy_weight_top5_mean` 单特征 global RF
- **补充分析**：展示 per-dataset 单特征 RF 的 oracle 上界
- **消融实验**：对比 1/2/3/4 特征的性能-可解释性权衡

这个方案比"训练一个复杂 selector 模型"更透明、更容易被接受，同时仍然能展示显著的 subset 效应。
