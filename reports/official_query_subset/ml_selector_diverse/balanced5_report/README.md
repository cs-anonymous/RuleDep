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
| balanced5_rf                       |    0.1519 |    0.0925 |    0.065  |    0.0477 |     0.0216 |
| balanced5_global_rf                |    0.0978 |    0.0713 |    0.0528 |    0.0401 |     0.0216 |
| synergy_weight_top5_mean           |    0.0812 |    0.0632 |    0.0494 |    0.0369 |     0.0216 |
| synergy_weight_top5_mean_global_rf |    0.0469 |    0.0399 |    0.042  |    0.0319 |     0.0216 |
| max_candidate_dep_score            |    0.0851 |    0.0594 |    0.0461 |    0.0376 |     0.0216 |
| max_candidate_dep_score_global_rf  |    0.0662 |    0.0462 |    0.0409 |    0.0342 |     0.0216 |
| topk_rule_weight                   |    0.072  |    0.0539 |    0.0457 |    0.0367 |     0.0216 |
| topk_rule_weight_global_rf         |    0.0367 |    0.0297 |    0.0276 |    0.0267 |     0.0216 |
| num_neg_dep                        |    0.0557 |    0.041  |    0.0339 |    0.0301 |     0.0216 |
| num_neg_dep_global_rf              |    0.0388 |    0.031  |    0.0304 |    0.0267 |     0.0216 |
| effective_candidates               |    0.0835 |    0.0591 |    0.05   |    0.0387 |     0.0216 |
| effective_candidates_global_rf     |    0.0484 |    0.0401 |    0.0417 |    0.0337 |     0.0216 |

## Balanced5 Per-dataset Gain

| dataset   |   gain_10 |   gain_20 |   gain_30 |   gain_50 |   gain_100 |
|:----------|----------:|----------:|----------:|----------:|-----------:|
| FB15k-237 |    0.1474 |    0.0734 |    0.0452 |    0.0245 |     0.01   |
| KG20C     |    0.0738 |    0.0729 |    0.0705 |    0.0654 |     0.0334 |
| WN18RR    |    0.0476 |    0.0339 |    0.0303 |    0.0234 |     0.0053 |
| YAGO3-10  |    0.1827 |    0.1308 |    0.0979 |    0.0676 |     0.0334 |
| codex-l   |    0.4468 |    0.1945 |    0.0861 |    0.0513 |     0.0148 |
| codex-m   |    0.0489 |    0.026  |    0.0207 |    0.0169 |     0.0086 |
| hetionet  |    0.1161 |    0.1161 |    0.1044 |    0.0848 |     0.0457 |

## Balanced5 RF Feature Importance

Each dataset trains its own RF model with the same five attributes. Values are impurity-based RF feature importances and sum to 1 within each dataset.

| dataset   |   synergy_weight_top5_mean |   max_candidate_dep_score |   topk_rule_weight |   num_neg_dep |   effective_candidates |
|:----------|---------------------------:|--------------------------:|-------------------:|--------------:|-----------------------:|
| FB15k-237 |                     0.2841 |                    0.2623 |             0.1588 |        0.1457 |                 0.1491 |
| KG20C     |                     0.2476 |                    0.116  |             0.4732 |        0.0216 |                 0.1414 |
| WN18RR    |                     0.2245 |                    0.3061 |             0.1943 |        0.0814 |                 0.1938 |
| YAGO3-10  |                     0.1771 |                    0.2668 |             0.1666 |        0.1744 |                 0.215  |
| codex-l   |                     0.3113 |                    0.1616 |             0.1466 |        0.16   |                 0.2204 |
| codex-m   |                     0.288  |                    0.2031 |             0.1632 |        0.1491 |                 0.1965 |
| hetionet  |                     0.6251 |                    0.0922 |             0.0749 |        0.1819 |                 0.0259 |

## Balanced5 Top-10% Attribute Ranges

Ranges are empirical IQRs among query cases selected by the dataset-specific RF score at 10% coverage.

| dataset   | synergy_weight_top5_mean       | max_candidate_dep_score                 | topk_rule_weight                | num_neg_dep           | effective_candidates        |
|:----------|:-------------------------------|:----------------------------------------|:--------------------------------|:----------------------|:----------------------------|
| FB15k-237 | [0.8577, 1.358] median=1.134   | [0.2309, 0.9608] median=0.5366          | [0.6024, 1.2] median=0.8933     | [2, 45] median=14     | [99.49, 102.1] median=100.1 |
| KG20C     | [0.2354, 0.6895] median=0.4435 | [0.0009982, 0.003873] median=0.00313    | [0.08914, 0.2174] median=0.1485 | [0, 0] median=0       | [99.98, 100] median=99.99   |
| WN18RR    | [0, 0.6468] median=0.3448      | [0, 0.2533] median=0.1489               | [0.1897, 0.4064] median=0.274   | [0, 0] median=0       | [99.9, 100] median=99.99    |
| YAGO3-10  | [2.127, 2.932] median=2.92     | [0.148, 0.5384] median=0.1713           | [1.17, 2.378] median=1.961      | [21, 72.75] median=47 | [99.6, 100.6] median=99.99  |
| codex-l   | [2.374, 2.887] median=2.611    | [-0.02143, -1.432e-05] median=-0.006992 | [1.373, 2.219] median=1.897     | [15, 51] median=31    | [112.9, 150.7] median=130.8 |
| codex-m   | [0, 0.7598] median=0.1952      | [0, 0.05314] median=0.001261            | [0.08041, 1.569] median=0.4055  | [0, 8] median=0       | [99.96, 122.8] median=100   |
| hetionet  | [5.203, 5.55] median=5.421     | [-1.205e-05, 0.002713] median=0         | [1.511, 1.726] median=1.625     | [11, 54] median=28    | [102.5, 134.4] median=112.1 |

## Balanced5 Global-RF Top-10% Attribute Ranges

Ranges are empirical IQRs among query cases selected by the pooled global RF score at 10% coverage. The `global` row pools the selected cases before splitting by dataset.

| dataset   | synergy_weight_top5_mean       | max_candidate_dep_score                | topk_rule_weight               | num_neg_dep        | effective_candidates        |
|:----------|:-------------------------------|:---------------------------------------|:-------------------------------|:-------------------|:----------------------------|
| global    | [5.084, 5.55] median=5.421     | [-1.482e-05, 0.006796] median=0        | [1.435, 1.709] median=1.585    | [6, 40] median=19  | [104.2, 138.9] median=114.9 |
| FB15k-237 | [0.993, 1.407] median=1.218    | [0.3338, 1.06] median=0.6304           | [0.6501, 1.2] median=1.034     | [4, 43] median=17  | [99.57, 102.7] median=100.5 |
| KG20C     | [1.333, 1.664] median=1.502    | [0.0002879, 0.002374] median=0.001012  | [0.4406, 0.6113] median=0.5306 | [2, 9] median=5    | [99.91, 103.7] median=101.5 |
| WN18RR    | [0.3448, 0.7543] median=0.5347 | [0.08964, 0.3701] median=0.1844        | [0.2293, 0.4572] median=0.3302 | [0, 1] median=0    | [99.94, 100] median=99.99   |
| YAGO3-10  | [2.858, 2.932] median=2.92     | [0, 0.5385] median=0.1721              | [1.223, 2.443] median=2.037    | [19, 70] median=45 | [99.84, 100.6] median=99.99 |
| codex-l   | [2.462, 2.887] median=2.631    | [-0.03437, -0.001139] median=-0.006992 | [1.521, 2.219] median=1.91     | [15, 51] median=31 | [117.1, 150.7] median=135.8 |
| codex-m   | [0.9518, 1.872] median=1.167   | [0, 0.02481] median=0.005054           | [0.5257, 1.569] median=0.823   | [1, 48] median=11  | [99.86, 108.8] median=100.8 |
| hetionet  | [5.204, 5.55] median=5.464     | [-1.479e-05, 0.005615] median=0        | [1.424, 1.699] median=1.567    | [5, 35] median=15  | [103.5, 140.8] median=114.8 |

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
