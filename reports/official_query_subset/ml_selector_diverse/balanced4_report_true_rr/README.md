# Balanced4 Selector Report (true per-query RR)

Drops `num_neg_dep` from the original Balanced5 set. The remaining 4
attributes are:

- `synergy_weight_top5_mean`: Synergy strength
- `max_candidate_dep_score`: Max candidate dependency score
- `topk_rule_weight`: Top-k rule weight
- `effective_candidates`: Effective candidates

The dropped attribute had the lowest single-feature RF gain (6.76% @10%
coverage vs 10.6-11.8% for the others), the lowest mean per-dataset RF
importance (~0.116 vs 0.17-0.24), and a sign-flipping Ridge coefficient
across datasets.

## Macro Gain Table

| selector                 |   gain_10 |   gain_20 |   gain_30 |   gain_50 |   gain_100 |
|:-------------------------|----------:|----------:|----------:|----------:|-----------:|
| balanced4_rf             |    0.2239 |    0.1409 |    0.0994 |    0.0617 |     0.0212 |
| balanced4_global_rf      |    0.1666 |    0.0991 |    0.07   |    0.0425 |     0.0212 |
| balanced5_rf             |    0.2221 |    0.1376 |    0.0989 |    0.0622 |     0.0212 |
| balanced5_global_rf      |    0.1642 |    0.1042 |    0.0719 |    0.0437 |     0.0212 |
| synergy_weight_top5_mean |    0.106  |    0.0817 |    0.0612 |    0.0443 |     0.0212 |
| max_candidate_dep_score  |    0.1155 |    0.0791 |    0.0589 |    0.0437 |     0.0212 |
| topk_rule_weight         |    0.1109 |    0.087  |    0.0708 |    0.0478 |     0.0212 |
| effective_candidates     |    0.1181 |    0.0889 |    0.075  |    0.0535 |     0.0212 |

## Balanced4 vs Balanced5 head-to-head

Both are re-fit on the same data with stable seeds in this report so the
comparison is like-for-like.

| scope       | metric   | balanced5   | balanced4   | delta_pt   |
|:------------|:---------|:------------|:------------|:-----------|
| per-dataset | gain_10  | 22.21%      | 22.39%      | +0.18 pt   |
| per-dataset | gain_20  | 13.76%      | 14.09%      | +0.33 pt   |
| per-dataset | gain_30  | 9.89%       | 9.94%       | +0.05 pt   |
| per-dataset | gain_50  | 6.22%       | 6.17%       | -0.05 pt   |
| per-dataset | gain_100 | 2.12%       | 2.12%       | +0.00 pt   |
| global      | gain_10  | 16.42%      | 16.66%      | +0.24 pt   |
| global      | gain_20  | 10.42%      | 9.91%       | -0.50 pt   |
| global      | gain_30  | 7.19%       | 7.00%       | -0.19 pt   |
| global      | gain_50  | 4.37%       | 4.25%       | -0.11 pt   |
| global      | gain_100 | 2.12%       | 2.12%       | +0.00 pt   |

## Global RF Feature Importance (one ranking, all datasets pooled)

This is new in this report. The original Balanced5 report only emitted
per-dataset RF importance; here we expose the importance of the single
unified RF used as the selector.

| selector            |   synergy_weight_top5_mean |   max_candidate_dep_score |   topk_rule_weight |   effective_candidates | num_neg_dep   |
|:--------------------|---------------------------:|--------------------------:|-------------------:|-----------------------:|:--------------|
| balanced4_global_rf |                     0.4868 |                    0.1424 |             0.1898 |                 0.181  |               |
| balanced5_global_rf |                     0.4498 |                    0.1077 |             0.1633 |                 0.1567 | 0.1225        |

## Per-dataset RF Feature Importance (Balanced4)

| dataset   |   synergy_weight_top5_mean |   max_candidate_dep_score |   topk_rule_weight |   effective_candidates |
|:----------|---------------------------:|--------------------------:|-------------------:|-----------------------:|
| FB15k-237 |                     0.2739 |                    0.3021 |             0.2178 |                 0.2062 |
| KG20C     |                     0.2437 |                    0.2357 |             0.2567 |                 0.2639 |
| WN18RR    |                     0.1931 |                    0.295  |             0.2409 |                 0.2709 |
| YAGO3-10  |                     0.1622 |                    0.2954 |             0.273  |                 0.2695 |
| codex-l   |                     0.2558 |                    0.166  |             0.3063 |                 0.2719 |
| codex-m   |                     0.1476 |                    0.1134 |             0.3736 |                 0.3654 |
| hetionet  |                     0.4405 |                    0.1498 |             0.2185 |                 0.1911 |

Per-dataset importances are noisy across datasets; the global RF importance
above is the one we recommend citing in a paper.

## RR Source (true per-query vs scaled fallback)

Same data plumbing as `balanced5_report_true_rr`. Fallback rows by dataset:

| dataset   | rr_source_stage1   | rr_source_stage2   |   n_rows |
|:----------|:-------------------|:-------------------|---------:|
| WN18RR    | scaled_fallback    | scaled_fallback    |     4854 |
| hetionet  | true               | scaled_fallback    |   106954 |

Full breakdown is in `rr_source_summary.csv`.

## Figures

- `balanced4_rf`: [figures/balanced4_rf.png](figures/balanced4_rf.png)
- `balanced4_global_rf`: [figures/balanced4_global_rf.png](figures/balanced4_global_rf.png)
- `balanced5_rf`: [figures/balanced5_rf.png](figures/balanced5_rf.png)
- `balanced5_global_rf`: [figures/balanced5_global_rf.png](figures/balanced5_global_rf.png)
- `synergy_weight_top5_mean`: [figures/synergy_weight_top5_mean.png](figures/synergy_weight_top5_mean.png)
- `max_candidate_dep_score`: [figures/max_candidate_dep_score.png](figures/max_candidate_dep_score.png)
- `topk_rule_weight`: [figures/topk_rule_weight.png](figures/topk_rule_weight.png)
- `effective_candidates`: [figures/effective_candidates.png](figures/effective_candidates.png)
