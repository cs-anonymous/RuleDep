# Balanced5 Selector Report

Balanced5 shared attributes:

- `synergy_weight_top5_mean`: Synergy strength
- `max_candidate_dep_score`: Max candidate dependency score
- `topk_rule_weight`: Top-k rule weight
- `num_neg_dep`: Number of negative dependencies
- `effective_candidates`: Effective candidates

## Attribute Definitions

| feature                  | paper_name                      | definition                                                                                                                                       |
|:-------------------------|:--------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------|
| synergy_weight_top5_mean | Synergy strength                | Mean of the top 5 absolute synergy dependency weights among unique displayed rule-pair dependencies in the query case.                           |
| max_candidate_dep_score  | Max candidate dependency score  | Maximum candidate-level dependencyScore over all candidate entities in the query case.                                                           |
| topk_rule_weight         | Top-k rule weight               | Mean of the top 3 rule weights fired by candidates in the query case.                                                                            |
| num_neg_dep              | Number of negative dependencies | Number of unique displayed dependency rule-pairs whose learned dependency type is redundancy.                                                    |
| effective_candidates     | Effective candidates            | exp(H(p)), where p is the softmax distribution over candidate stage1 official scores; larger values mean the stage1 ranker is less concentrated. |

## Dataset-specific and Global RF Selectors

`balanced5_rf` and the plain feature-name rows train one RF per dataset. Rows ending in `_global_rf` train one pooled RF over all datasets, then evaluate coverage separately within each dataset.

## Macro Gain Table

| selector                           |   gain_10 |   gain_20 |   gain_30 |   gain_50 |   gain_100 |
|:-----------------------------------|----------:|----------:|----------:|----------:|-----------:|
| balanced5_rf                       |    0.2221 |    0.1376 |    0.0989 |    0.0622 |     0.0212 |
| balanced5_global_rf                |    0.1642 |    0.1042 |    0.0719 |    0.0437 |     0.0212 |
| synergy_weight_top5_mean           |    0.106  |    0.0817 |    0.0612 |    0.0443 |     0.0212 |
| synergy_weight_top5_mean_global_rf |    0.0732 |    0.0595 |    0.0483 |    0.0368 |     0.0212 |
| max_candidate_dep_score            |    0.1155 |    0.0791 |    0.0589 |    0.0437 |     0.0212 |
| max_candidate_dep_score_global_rf  |    0.0725 |    0.0546 |    0.046  |    0.0383 |     0.0212 |
| topk_rule_weight                   |    0.1109 |    0.087  |    0.0708 |    0.0478 |     0.0212 |
| topk_rule_weight_global_rf         |    0.0482 |    0.0397 |    0.0335 |    0.0259 |     0.0212 |
| num_neg_dep                        |    0.0676 |    0.0498 |    0.0378 |    0.0325 |     0.0212 |
| num_neg_dep_global_rf              |    0.0457 |    0.0365 |    0.0342 |    0.0287 |     0.0212 |
| effective_candidates               |    0.1181 |    0.0889 |    0.075  |    0.0535 |     0.0212 |
| effective_candidates_global_rf     |    0.0605 |    0.054  |    0.0461 |    0.0379 |     0.0212 |
| balanced5_global_ridge             |    0.0526 |    0.0439 |    0.034  |    0.026  |     0.0212 |
| balanced5_ridge                    |    0.0761 |    0.0615 |    0.0482 |    0.0363 |     0.0212 |

## Balanced5 Per-dataset Gain

| dataset   |   gain_10 |   gain_20 |   gain_30 |   gain_50 |   gain_100 |
|:----------|----------:|----------:|----------:|----------:|-----------:|
| FB15k-237 |    0.2415 |    0.1211 |    0.0696 |    0.0396 |     0.0092 |
| KG20C     |    0.3279 |    0.2405 |    0.1897 |    0.1345 |     0.0336 |
| WN18RR    |    0.0478 |    0.0347 |    0.0286 |    0.0235 |     0.0042 |
| YAGO3-10  |    0.3889 |    0.2446 |    0.1789 |    0.0917 |     0.033  |
| codex-l   |    0.1225 |    0.0767 |    0.0563 |    0.0399 |     0.015  |
| codex-m   |    0.1159 |    0.065  |    0.0473 |    0.0311 |     0.0076 |
| hetionet  |    0.31   |    0.1809 |    0.1218 |    0.0752 |     0.0459 |

## Balanced5 RF Feature Importance

Each dataset trains its own RF model with the same five attributes. Values are impurity-based RF feature importances and sum to 1 within each dataset.

| dataset   |   synergy_weight_top5_mean |   max_candidate_dep_score |   topk_rule_weight |   num_neg_dep |   effective_candidates |
|:----------|---------------------------:|--------------------------:|-------------------:|--------------:|-----------------------:|
| FB15k-237 |                     0.2432 |                    0.2531 |             0.1738 |        0.1708 |                 0.1593 |
| KG20C     |                     0.2372 |                    0.2345 |             0.2396 |        0.0443 |                 0.2443 |
| WN18RR    |                     0.1829 |                    0.2813 |             0.2165 |        0.0636 |                 0.2557 |
| YAGO3-10  |                     0.1301 |                    0.2277 |             0.2291 |        0.206  |                 0.2071 |
| codex-l   |                     0.2172 |                    0.1388 |             0.2721 |        0.1206 |                 0.2513 |
| codex-m   |                     0.1394 |                    0.1199 |             0.3392 |        0.0624 |                 0.3392 |
| hetionet  |                     0.3867 |                    0.1179 |             0.1862 |        0.1382 |                 0.1709 |

## Balanced5 Top-10% Attribute Ranges

Ranges are empirical IQRs among query cases selected by the dataset-specific RF score at 10% coverage.

| dataset   | synergy_weight_top5_mean      | max_candidate_dep_score            | topk_rule_weight               | num_neg_dep        | effective_candidates        |
|:----------|:------------------------------|:-----------------------------------|:-------------------------------|:-------------------|:----------------------------|
| FB15k-237 | [0.5128, 1.369] median=0.993  | [0.08912, 0.8341] median=0.469     | [0.5791, 1.3] median=0.918     | [3, 92] median=17  | [99.76, 107.9] median=100.8 |
| KG20C     | [0.5106, 1.072] median=0.7512 | [0.002101, 0.0154] median=0.003438 | [0.196, 0.4055] median=0.2839  | [0, 0] median=0    | [99.96, 101] median=99.99   |
| WN18RR    | [0, 0.6468] median=0.3448     | [0, 0.2705] median=0.1262          | [0.1628, 0.4041] median=0.2676 | [0, 0] median=0    | [99.95, 100] median=99.99   |
| YAGO3-10  | [2.127, 2.932] median=2.918   | [0.1561, 0.5393] median=0.4672     | [1.282, 2.235] median=1.961    | [34, 93] median=59 | [98.93, 100.3] median=99.84 |
| codex-l   | [2.018, 2.509] median=2.113   | [-0.01148, 0] median=-1.432e-05    | [0.4669, 0.9823] median=0.6139 | [6, 29] median=16  | [99.8, 100.3] median=99.95  |
| codex-m   | [0, 0.3329] median=0.05265    | [0, 0.01014] median=0.0004308      | [0.1208, 1.421] median=0.2234  | [0, 1] median=0    | [99.99, 102.7] median=100   |
| hetionet  | [5.255, 5.55] median=5.493    | [0, 0.01747] median=0.003741       | [1.145, 1.63] median=1.46      | [3, 37] median=19  | [101.1, 109.7] median=104.1 |

## Balanced5 Global-RF Top-10% Attribute Ranges

Ranges are empirical IQRs among query cases selected by the pooled global RF score at 10% coverage. The `global` row pools the selected cases before splitting by dataset.

| dataset   | synergy_weight_top5_mean       | max_candidate_dep_score             | topk_rule_weight               | num_neg_dep        | effective_candidates        |
|:----------|:-------------------------------|:------------------------------------|:-------------------------------|:-------------------|:----------------------------|
| global    | [5.207, 5.55] median=5.464     | [0, 0.0186] median=0.004663         | [1.245, 1.666] median=1.539    | [5, 46] median=26  | [101.2, 110] median=104.5   |
| FB15k-237 | [0.5747, 1.386] median=1.075   | [0.1943, 0.8811] median=0.5293      | [0.621, 1.386] median=1.12     | [6, 105] median=30 | [99.94, 110.4] median=102.6 |
| KG20C     | [0.7139, 1.072] median=0.8111  | [0.001893, 0.01119] median=0.002784 | [0.1318, 0.3269] median=0.2231 | [0, 0] median=0    | [99.98, 100.9] median=100   |
| WN18RR    | [0.3384, 0.7201] median=0.4587 | [0.05461, 0.3701] median=0.1844     | [0.2293, 0.5019] median=0.3323 | [0, 1] median=0    | [99.96, 100.2] median=100   |
| YAGO3-10  | [2.868, 2.932] median=2.92     | [0.1562, 0.5442] median=0.5384      | [1.961, 2.425] median=1.961    | [36, 90] median=59 | [98.94, 100.2] median=99.75 |
| codex-l   | [2.036, 2.573] median=2.173    | [-0.007252, 0] median=0             | [0.4635, 1.057] median=0.6157  | [8, 29] median=18  | [99.83, 100.6] median=99.96 |
| codex-m   | [0, 1.33] median=0.9985        | [0, 0.02401] median=0.004391        | [0.3719, 1.464] median=0.7472  | [0, 23] median=1   | [99.98, 104.7] median=100.8 |
| hetionet  | [5.223, 5.55] median=5.483     | [0, 0.01933] median=0.005244        | [1.139, 1.636] median=1.479    | [3, 39] median=21  | [101.1, 109.3] median=104   |

Full top-10/20/50 range statistics are in `balanced5_selected_feature_ranges.csv` and `balanced5_global_selected_feature_ranges.csv`.

## Figures

Plots start at 10% coverage and follow the same dataset-color style as `feature_plots/`; black is the macro average.

- `balanced5_rf`: [figures/balanced5_rf.png](figures/balanced5_rf.png)
- `balanced5_global_rf`: [figures/balanced5_global_rf.png](figures/balanced5_global_rf.png)
- `synergy_weight_top5_mean`: [figures/synergy_weight_top5_mean.png](figures/synergy_weight_top5_mean.png)
- `synergy_weight_top5_mean_global_rf`: [figures/synergy_weight_top5_mean_global_rf.png](figures/synergy_weight_top5_mean_global_rf.png)
- `max_candidate_dep_score`: [figures/max_candidate_dep_score.png](figures/max_candidate_dep_score.png)
- `max_candidate_dep_score_global_rf`: [figures/max_candidate_dep_score_global_rf.png](figures/max_candidate_dep_score_global_rf.png)
- `topk_rule_weight`: [figures/topk_rule_weight.png](figures/topk_rule_weight.png)
- `topk_rule_weight_global_rf`: [figures/topk_rule_weight_global_rf.png](figures/topk_rule_weight_global_rf.png)
- `num_neg_dep`: [figures/num_neg_dep.png](figures/num_neg_dep.png)
- `num_neg_dep_global_rf`: [figures/num_neg_dep_global_rf.png](figures/num_neg_dep_global_rf.png)
- `effective_candidates`: [figures/effective_candidates.png](figures/effective_candidates.png)
- `effective_candidates_global_rf`: [figures/effective_candidates_global_rf.png](figures/effective_candidates_global_rf.png)
- `balanced5_global_ridge`: [figures/balanced5_global_ridge.png](figures/balanced5_global_ridge.png)
- `balanced5_ridge`: [figures/balanced5_ridge.png](figures/balanced5_ridge.png)

## RR Source (true per-query vs scaled fallback)

This variant of the Balanced5 report uses the **true official filtered RR per query** (merged from `reports/official_query_subset/true_official_per_query_rr/true_official_per_query_rr_wide.csv`) as both the RF training target and the coverage-curve evaluation target. When a row has no matching true per-query RR (test triple absent from the rerun export), the value falls back to the legacy relation-level `official_scaled_rr` (`raw_rr * official_relation_MRR / mean(raw_rr_within_relation)`). Per-row provenance is in `rr_source_stage1` and `rr_source_stage2` of the joined data and aggregated below.

Known fallback sources:
- **hetionet `AeG` (relation 1) stage2**: stage2 dependency-stage rerun OOM'd in evaluation (`active_matrix = zeros((eval_batch, 1.75M rules))` -> 17.13 GiB single allocation). Stage1 RR is the true value; stage2 RR is the scaled fallback for these 106,954 rows.
- **WN18RR `_hypernym` tail (~4,854 rows)**: the legacy `official_query_triple_features.csv` was generated against a slightly different test sample set than the rerun export, so a subset of `_hypernym` tail queries has no matching true RR.

### Fallback rows by dataset and stage

| dataset   | rr_source_stage1   | rr_source_stage2   |   n_rows |
|:----------|:-------------------|:-------------------|---------:|
| WN18RR    | scaled_fallback    | scaled_fallback    |     4854 |
| hetionet  | true               | scaled_fallback    |   106954 |

Full breakdown (one row per `dataset x relation x stage1_source x stage2_source`) is in `rr_source_summary.csv`.

## Ridge Linear Selector (interpretable formula)

We additionally fit `Ridge(alpha=1.0)` on `SimpleImputer(median) -> StandardScaler -> Ridge` for two configurations:

- `balanced5_ridge`: one Ridge model **per dataset** (5 standardized coefficients per dataset).
- `balanced5_global_ridge`: one Ridge model trained on **all datasets pooled** (one set of 5 standardized coefficients), then evaluated within each dataset.

Coefficients are on **standardized features** (mean 0, std 1 per training distribution), so the magnitude is directly comparable across features within a model. The sign indicates whether higher feature value pushes the predicted `gain_clip = clip(rr_stage2 / rr_stage1 - 1, [-1, 1])` up or down.

### Global Ridge coefficients (one formula for all datasets)

| feature                  |   std_coef |   intercept |
|:-------------------------|-----------:|------------:|
| synergy_weight_top5_mean |     0.0579 |      0.0525 |
| max_candidate_dep_score  |     0.003  |      0.0525 |
| topk_rule_weight         |     0.0015 |      0.0525 |
| num_neg_dep              |    -0.0041 |      0.0525 |
| effective_candidates     |    -0.0082 |      0.0525 |

### Per-dataset Ridge coefficients

| dataset   |   synergy_weight_top5_mean |   max_candidate_dep_score |   topk_rule_weight |   num_neg_dep |   effective_candidates |
|:----------|---------------------------:|--------------------------:|-------------------:|--------------:|-----------------------:|
| FB15k-237 |                     0.0159 |                    0.0079 |            -0.0037 |        0.0102 |                 0.0016 |
| KG20C     |                     0.0118 |                    0.0018 |            -0.0103 |        0.0011 |                 0.0085 |
| WN18RR    |                     0.0014 |                    0.01   |            -0.005  |        0.0012 |                 0.0008 |
| YAGO3-10  |                     0.0397 |                    0.0176 |            -0.0189 |        0.0194 |                -0.0058 |
| codex-l   |                     0.0146 |                   -0.0022 |            -0.0083 |       -0.0031 |                 0.0009 |
| codex-m   |                     0.0018 |                   -0      |            -0.0075 |       -0.0004 |                 0.0108 |
| hetionet  |                     0.0647 |                   -0.0012 |             0.0052 |       -0.0073 |                -0.0137 |

Full coefficients (with intercepts) are in `balanced5_ridge_coefficients.csv`.
Coverage curves and per-dataset gains for the Ridge selectors are in `balanced5_gain_curves.csv` and figures `figures/balanced5_ridge.png` / `figures/balanced5_global_ridge.png`.
