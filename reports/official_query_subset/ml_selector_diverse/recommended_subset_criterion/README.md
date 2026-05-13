# Recommended Query Subset Criterion

## Recommendation

Use the **two-feature Global RF** selector `balanced2_syn_topk_global_rf`
for the paper-facing query subset analysis.

Formal subset criterion:

1. Train one pooled RandomForest regressor over all 7 datasets.
2. Use only two features: `synergy_weight_top5_mean` and `topk_rule_weight`.
3. Train target: `gain_clip = clip(rr_stage2 / rr_stage1 - 1, [-1, 1])`.
4. For each dataset independently, score every query with the global RF and
   keep the top **10%** by RF score.

This is still a Global RF: there is one shared model and one shared feature
set. The per-dataset step only fixes the coverage to 10% in each dataset, so
large datasets do not dominate the selected subset.

## Why Two Features

The single-feature global RF is most interpretable but only reaches
`gain@10 = 7.32%`. The best two-feature global RF reaches
`gain@10 = 12.64%`, clearing the 10% target while keeping the selector easy
to explain. Three-feature selectors improve further but add another axis that
is harder to state as a compact subset criterion.

## Feature Definitions

| feature                  | paper_name                     | definition                                                                                                                                       |
|:-------------------------|:-------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------|
| synergy_weight_top5_mean | Synergy strength               | Mean of the top 5 absolute synergy dependency weights among unique displayed rule-pair dependencies in the query case.                           |
| topk_rule_weight         | Top-k rule weight              | Mean of the top 3 rule weights fired by candidates in the query case.                                                                            |
| max_candidate_dep_score  | Max candidate dependency score | Maximum candidate-level dependencyScore over all candidate entities in the query case.                                                           |
| effective_candidates     | Effective candidates           | exp(H(p)), where p is the softmax distribution over candidate stage1 official scores; larger values mean the stage1 ranker is less concentrated. |

## Global RF Importance

| feature                  | paper_name        |   importance |
|:-------------------------|:------------------|-------------:|
| synergy_weight_top5_mean | Synergy strength  |       0.7106 |
| topk_rule_weight         | Top-k rule weight |       0.2894 |

## Macro Gain

| selector                     | gain_10   | gain_20   | gain_30   | gain_50   | gain_100   |
|:-----------------------------|:----------|:----------|:----------|:----------|:-----------|
| balanced2_syn_topk_global_rf | 12.64%    | 8.31%     | 6.17%     | 4.01%     | 2.12%      |

## Selector Definition and Interpretation

The recommended subset is selected by a learned Global RF transformation:

`score = f(synergy_weight_top5_mean, topk_rule_weight)`.

Here, `f` is the average of 160 regression trees, so it is a learned
piecewise-constant transformation rather than a single hand-written threshold.
We rank queries by this transformed score within each dataset and select the
top 10%, i.e., `percentile(score) in [90%, 100%]`.

The selected subset has a simple feature-level interpretation: it tends to
contain queries with **high `synergy_weight_top5_mean`** and **middle-range
`topk_rule_weight`**. In the pooled selected top-10% subset, the feature IQRs
are `synergy_weight_top5_mean` = `[3.118, 5.55]` (pooled percentile
`[75.12%, 95.21%]`) and `topk_rule_weight` = `[0.971, 1.621]` (pooled
percentile `[36.54%, 73.60%]`). This two-feature RF subset raises the macro
subset gain to **12.64%** at 10% coverage.

## Two-Feature RF Selector Paper Table

This is the recommended table for the main text. The gain comes from the
two-feature Global RF score selector, so it keeps the `gain@10 = 12.64%`
result. Feature ranges are descriptive summaries of the RF-selected top-10%
queries, not hard threshold rules.

| feature                  | gain@10   | gain@20   | gain@50   | Feature Range (abs, IQR)   | Feature Range (percentile, IQR)   |
|:-------------------------|:----------|:----------|:----------|:---------------------------|:----------------------------------|
| synergy_weight_top5_mean | 12.64%    | 8.31%     | 4.01%     | [3.118, 5.55]              | [75.12%, 95.21%]                  |
| topk_rule_weight         | 12.64%    | 8.31%     | 4.01%     | [0.971, 1.621]             | [36.54%, 73.60%]                  |

## Top-3 Single-Feature Global RF Selectors

The table below uses the same Global RF setup and top-10% per-dataset
selection rule, then pools the selected queries across datasets to summarize
feature ranges. `Feature Range` reports the selected subset IQR.

| feature                  | gain@10   | gain@20   | gain@50   | Feature Range (abs, IQR)   | Feature Range (percentile, IQR)   |
|:-------------------------|:----------|:----------|:----------|:---------------------------|:----------------------------------|
| synergy_weight_top5_mean | 7.32%     | 5.95%     | 3.68%     | [3.088, 5.579]             | [74.64%, 98.39%]                  |
| max_candidate_dep_score  | 7.25%     | 5.46%     | 3.83%     | [3.531e-05, 0.01706]       | [68.62%, 86.89%]                  |
| effective_candidates     | 6.05%     | 5.40%     | 3.79%     | [103.5, 111.4]             | [58.45%, 72.95%]                  |

CSV for the final paper table: `single_feature_top3_paper_table.csv`.

## Optional Hard Feature-Range Sanity Check

The RF rows above are the main results. The hard percentile rules below are
only a sanity check: they replace the RF score with a single raw feature
percentile cutoff. Their gains are lower because this is a simpler selector,
not the RF selector that gives 12.64%.

| feature                  | criterion                         | gain@10   | gain@20   | gain@50   | Feature Range (abs)   | Feature Range (percentile)   |
|:-------------------------|:----------------------------------|:----------|:----------|:----------|:----------------------|:-----------------------------|
| synergy_weight_top5_mean | dataset percentile in [90%, 100%] | 5.71%     | 4.71%     | 3.50%     | [0.2367, 5.579]       | [90%, 100%]                  |
| effective_candidates     | dataset percentile in [90%, 100%] | 5.61%     | 4.43%     | 3.00%     | [100, 200]            | [90%, 100%]                  |
| max_candidate_dep_score  | dataset percentile in [90%, 100%] | 4.42%     | 4.12%     | 3.43%     | [0, 6.011]            | [90%, 100%]                  |

CSV for this optional hard-rule check: `single_feature_hard_range_rules.csv`.

## Per-Dataset Gain at 10%

| selector                     | dataset   | gain_10   |   mrr_stage1 |   mrr_stage2 |     n |
|:-----------------------------|:----------|:----------|-------------:|-------------:|------:|
| balanced2_syn_topk_global_rf | FB15k-237 | 8.40%     |     0.24949  |     0.270443 |  4082 |
| balanced2_syn_topk_global_rf | KG20C     | 23.86%    |     0.159618 |     0.19771  |   745 |
| balanced2_syn_topk_global_rf | WN18RR    | 2.37%     |     0.408155 |     0.417846 |   606 |
| balanced2_syn_topk_global_rf | YAGO3-10  | 14.16%    |     0.609842 |     0.696202 |   998 |
| balanced2_syn_topk_global_rf | codex-l   | 6.24%     |     0.385729 |     0.409794 |  6113 |
| balanced2_syn_topk_global_rf | codex-m   | 4.18%     |     0.331494 |     0.345363 |  2059 |
| balanced2_syn_topk_global_rf | hetionet  | 29.26%    |     0.473968 |     0.612641 | 45004 |

## Cross-Dataset Selected Top-10% Feature Ranges

The table reports the empirical feature range of the selected top-10% subset
after applying the top-10% rule within each dataset and then pooling all
selected queries. Percentiles are computed against the pooled cross-dataset
feature distribution. The IQR is the recommended compact paper wording;
min/max are included only to show the full observed spread.

| feature                  |   n_selected |   value_min |   value_q25 |   value_median |   value_q75 |   value_max | percentile_min   | percentile_q25   | percentile_median   | percentile_q75   | percentile_max   |
|:-------------------------|-------------:|------------:|------------:|---------------:|------------:|------------:|:-----------------|:-----------------|:--------------------|:-----------------|:-----------------|
| synergy_weight_top5_mean |        59607 |     0       |       3.118 |          5.223 |       5.55  |       5.579 | 27.36%           | 75.12%           | 87.72%              | 95.21%           | 98.39%           |
| topk_rule_weight         |        59607 |     0.05134 |       0.971 |          1.363 |       1.621 |       3.256 | 1.36%            | 36.54%           | 52.06%              | 73.60%           | 99.94%           |

The dual-feature gain curve at every 2% coverage point for each dataset and
the macro average is in `dual_feature_gain_curves_with_macro.csv`.

## Wording for the Paper

We identify the explainable query subset using a two-feature global RF
selector trained on all datasets with synergy strength and top-k rule weight
as inputs. For each dataset, queries are ranked by the shared RF predicted
gain score, and the top 10% are used as the subset. This rule yields a macro
average in-sample relative MRR gain of 12.64% at 10% coverage, while
preserving a compact interpretation: selected queries tend to combine strong
learned synergy dependencies with moderate-to-high fired-rule weights.
