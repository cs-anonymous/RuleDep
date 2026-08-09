# Single Feature RF Selector Analysis Report

## Overview

This report analyzes how well **Random Forest models trained on a single feature** perform as query subset selectors. Single-feature RF models offer stronger interpretability than multi-feature models while retaining RF's nonlinear segmentation capability.

Data source: `reports/official_query_subset/ml_selector_diverse/balanced_ablation_true_rr/`

---

## 1. Macro Average Performance (Across 7 Datasets)

### 1.1 Global RF (Recommended for papers, generalizes across datasets)

| Rank | Feature | gain@10 | gain@20 | gain@30 | gain@50 |
|------|---------|---------|---------|---------|---------|
| 1 | **synergy_weight_top5_mean** | **7.32%** | **5.95%** | 4.83% | 3.68% |
| 2 | max_candidate_dep_score | 7.25% | 5.46% | 4.60% | 3.83% |
| 3 | effective_candidates | 6.05% | 5.40% | 4.61% | 3.79% |
| 4 | topk_rule_weight | 4.82% | 3.97% | 3.35% | 2.59% |

**Best single feature**: `synergy_weight_top5_mean`
- Strongest at gain@10 and gain@20
- Measures the average weight strength of top-5 synergy dependencies

### 1.2 Per-dataset RF (Oracle upper bound, each dataset trained separately)

| Rank | Feature | gain@10 | gain@20 | gain@30 | gain@50 |
|------|---------|---------|---------|---------|---------|
| 1 | **effective_candidates** | **12.03%** | **9.08%** | 7.55% | 5.36% |
| 2 | max_candidate_dep_score | 11.52% | 7.88% | 5.86% | 4.36% |
| 3 | topk_rule_weight | 11.02% | 8.56% | 7.01% | 4.80% |
| 4 | synergy_weight_top5_mean | 10.61% | 8.16% | 6.12% | 4.40% |

**Best single feature**: `effective_candidates`
- Best performance under per-dataset settings
- Measures the number of valid candidates in stage 1 (uncertainty)

---

## 2. Codex-L Dataset Performance

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

## 3. Single Feature vs Multi-Feature Comparison

### 3.1 Macro Average (7 Datasets)

| Configuration | gain@10 | gain@20 | Relative to 4-feat |
|---------------|---------|---------|-------------------|
| 4-feature global RF (baseline) | 16.66% | 9.91% | 100% / 100% |
| **Best 1-feature global RF** | **7.32%** | **5.95%** | **43.9% / 60.0%** |
| Best 1-feature per-dataset RF | 12.03% | 9.08% | 72.2% / 91.6% |

**Loss analysis**:
- Global RF single feature vs 4 features: Loss **56.1%** at gain@10, **40.0%** at gain@20
- Per-dataset RF single feature vs 4 features: Loss **27.8%** at gain@10, **8.4%** at gain@20

### 3.2 Codex-L Dataset

| Configuration | gain@10 | gain@20 | Relative to 4-feat |
|---------------|---------|---------|-------------------|
| 4-feature global RF (baseline) | 8.96% | 5.05% | 100% / 100% |
| **Best 1-feature global RF** | **4.47%** | **3.33%** | **49.9% / 65.9%** |
| Best 1-feature per-dataset RF | 8.67% | 5.94% | 96.8% / 117.6% |

**Key findings**:
- On codex-l, per-dataset RF with a single feature nearly matches the 4-feature performance
- gain@20 even exceeds the 4-feature baseline (117.6%)

---

## 4. Feature Descriptions

| Short name | Full name | Category | Meaning |
|------|---------|------|------|
| syn | synergy_weight_top5_mean | D: Dependency weight strength | Average weight of top-5 synergy dependencies |
| max | max_candidate_dep_score | C: Positive/negative dependency | Maximum dependency score across candidates |
| eff | effective_candidates | G: S1 ambiguity | Number of valid candidates in stage 1 (uncertainty) |
| topk | topk_rule_weight | E: Rule-weight distribution | Top-k rule weight aggregated value |

---

## 5. Key Insights

### 5.1 Single Feature RF Advantages

- **Minimal and interpretable**: Depends on only one statistic, easy to explain
- **Avoids overfitting**: No feature interaction, better generalization
- **Transparent decision making**: Split points and decision boundaries are directly visible
- **Computationally efficient**: Both feature extraction and model training are faster

### 5.2 Best Choice by Scenario

**Global RF (generalizes across datasets)**:
- Best feature: `synergy_weight_top5_mean`
- Use when you need to generalize across multiple datasets with good interpretability
- Performance: gain@10 = 7.32%, gain@20 = 5.95%

**Per-dataset RF (single dataset optimization)**:
- Best feature: `effective_candidates`
- Use when enough data is available to train per-dataset models
- Performance: gain@10 = 12.03%, gain@20 = 9.08%

### 5.3 Why synergy_weight_top5_mean Performs Best

1. **Directly measures core mechanism**: Synergy is the primary source of RuleDep's gains
2. **Top-5 balances coverage and quality**: Unlike top-1 (too narrow) or mean (diluted by weak signals)
3. **Stable across datasets**: Performs well on all 7 datasets

### 5.4 Why effective_candidates Excels in Per-dataset Settings

1. **Captures S1 uncertainty**: More candidates with closer scores means more room for RuleDep to flip rankings
2. **Dataset specificity is high**: Candidate distributions vary significantly across datasets
3. **Strong theoretical support**: Matches the "more S1 uncertainty means dependency is more useful" hypothesis

---

## 6. Recommendations

### 6.1 Paper Usage Suggestion

**Recommended approach**: Use a **single-feature global RF** as a subset diagnostic.
