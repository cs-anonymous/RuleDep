# Feature Rankings at Fixed Coverage

Metric: official-scaled `gain_pt`, with coverage sampled every 2%.

## Coverage 10%

|   rank | feature                      | sort_direction   |   macro_gain_pt |   positive_datasets |   min_dataset_gain |   max_dataset_gain |
|-------:|:-----------------------------|:-----------------|----------------:|--------------------:|-------------------:|-------------------:|
|      1 | dep_candidate_ratio          | desc             |          0.0418 |                   7 |             0.0154 |             0.1161 |
|      2 | candidate_dep_coverage       | desc             |          0.0418 |                   7 |             0.0154 |             0.1161 |
|      3 | synergy_weight_mean          | desc             |          0.0412 |                   7 |             0.0057 |             0.1062 |
|      4 | synergy_weight_top1_sum      | desc             |          0.0407 |                   7 |             0.0007 |             0.1161 |
|      5 | synergy_weight_top1_mean     | desc             |          0.0407 |                   7 |             0.0007 |             0.1161 |
|      6 | synergy_weight_max           | desc             |          0.0407 |                   7 |             0.0007 |             0.1161 |
|      7 | redundancy_weight_top10_mean | desc             |          0.0405 |                   7 |             0.005  |             0.0774 |
|      8 | topk_synergy                 | desc             |          0.0397 |                   6 |            -0.0135 |             0.1161 |
|      9 | synergy_weight_top3_mean     | desc             |          0.0397 |                   6 |            -0.0135 |             0.1161 |
|     10 | synergy_weight_top3_sum      | desc             |          0.0393 |                   6 |            -0.0135 |             0.1161 |
|     11 | topk_synergy_sum             | desc             |          0.0393 |                   6 |            -0.0135 |             0.1161 |
|     12 | redundancy_weight_top5_mean  | desc             |          0.0386 |                   7 |             0.0047 |             0.0768 |
|     13 | redundancy_weight_top1_sum   | desc             |          0.0386 |                   7 |             0.0038 |             0.0739 |
|     14 | redundancy_weight_max        | desc             |          0.0386 |                   7 |             0.0038 |             0.0739 |
|     15 | redundancy_weight_top1_mean  | desc             |          0.0386 |                   7 |             0.0038 |             0.0739 |
|     16 | redundancy_weight_top5_sum   | desc             |          0.0382 |                   7 |             0.006  |             0.077  |
|     17 | synergy_weight_top5_mean     | desc             |          0.0382 |                   6 |            -0.0256 |             0.1161 |
|     18 | redundancy_weight_top10_sum  | desc             |          0.038  |                   7 |             0.0063 |             0.0775 |
|     19 | redundancy_weight_top3_mean  | desc             |          0.0378 |                   7 |             0.0041 |             0.0761 |
|     20 | topk_redundancy              | desc             |          0.0378 |                   7 |             0.0041 |             0.0761 |
|     21 | synergy_weight_top5_sum      | desc             |          0.0377 |                   6 |            -0.0256 |             0.1161 |
|     22 | redundancy_weight_top3_sum   | desc             |          0.0368 |                   7 |             0.0037 |             0.076  |
|     23 | effective_candidates         | desc             |          0.0362 |                   7 |             0.0165 |             0.0672 |
|     24 | s1_entropy                   | desc             |          0.0362 |                   7 |             0.0165 |             0.0672 |
|     25 | redundancy_weight_mean       | desc             |          0.0362 |                   7 |             0.002  |             0.0781 |
|     26 | num_candidates               | desc             |          0.0359 |                   7 |             0.0094 |             0.0746 |
|     27 | max_rules_per_candidate      | desc             |          0.0357 |                   7 |             0.0002 |             0.1161 |
|     28 | neg_dep_ratio                | desc             |          0.0315 |                   7 |             0.0103 |             0.0849 |
|     29 | sum_negative_dep             | desc             |          0.0308 |                   6 |            -0.0109 |             0.0652 |
|     30 | max_candidate_dep_score      | desc             |          0.0304 |                   7 |             0.0036 |             0.0569 |

## Coverage 20%

|   rank | feature                     | sort_direction   |   macro_gain_pt |   positive_datasets |   min_dataset_gain |   max_dataset_gain |
|-------:|:----------------------------|:-----------------|----------------:|--------------------:|-------------------:|-------------------:|
|      1 | synergy_weight_mean         | desc             |          0.0389 |                   7 |             0.0052 |             0.095  |
|      2 | dep_candidate_ratio         | desc             |          0.037  |                   7 |             0.0059 |             0.0882 |
|      3 | candidate_dep_coverage      | desc             |          0.037  |                   7 |             0.0059 |             0.0882 |
|      4 | synergy_weight_top3_mean    | desc             |          0.0352 |                   6 |            -0.0267 |             0.1161 |
|      5 | topk_synergy                | desc             |          0.0352 |                   6 |            -0.0267 |             0.1161 |
|      6 | topk_synergy_sum            | desc             |          0.0352 |                   6 |            -0.0267 |             0.1161 |
|      7 | synergy_weight_top3_sum     | desc             |          0.0352 |                   6 |            -0.0267 |             0.1161 |
|      8 | synergy_weight_top1_sum     | desc             |          0.0349 |                   6 |            -0.0252 |             0.1161 |
|      9 | synergy_weight_top1_mean    | desc             |          0.0349 |                   6 |            -0.0252 |             0.1161 |
|     10 | synergy_weight_max          | desc             |          0.0349 |                   6 |            -0.0252 |             0.1161 |
|     11 | synergy_weight_top5_mean    | desc             |          0.0348 |                   6 |            -0.0303 |             0.1161 |
|     12 | synergy_weight_top5_sum     | desc             |          0.0347 |                   6 |            -0.0303 |             0.1161 |
|     13 | effective_candidates        | desc             |          0.0342 |                   7 |             0.0133 |             0.0729 |
|     14 | s1_entropy                  | desc             |          0.0342 |                   7 |             0.0133 |             0.0729 |
|     15 | rule_dominance_ratio        | asc              |          0.0339 |                   7 |             0.0128 |             0.0708 |
|     16 | max_rules_per_candidate     | desc             |          0.0337 |                   7 |             0.0001 |             0.1064 |
|     17 | syn_rule_ratio              | desc             |          0.0336 |                   7 |             0.0064 |             0.0645 |
|     18 | max_candidate_dep_score     | desc             |          0.0334 |                   7 |             0.0072 |             0.0626 |
|     19 | dep_rule_ratio              | desc             |          0.0332 |                   7 |             0.0063 |             0.0641 |
|     20 | num_candidates              | desc             |          0.0319 |                   7 |             0.0122 |             0.0737 |
|     21 | synergy_weight_top10_mean   | desc             |          0.0319 |                   6 |            -0.0372 |             0.099  |
|     22 | neg_dep_ratio               | desc             |          0.0315 |                   7 |             0.0027 |             0.0818 |
|     23 | synergy_weight_top10_sum    | desc             |          0.0314 |                   6 |            -0.0374 |             0.0946 |
|     24 | redundancy_weight_top1_sum  | desc             |          0.0305 |                   7 |             0.0028 |             0.0804 |
|     25 | redundancy_weight_max       | desc             |          0.0305 |                   7 |             0.0028 |             0.0804 |
|     26 | redundancy_weight_top1_mean | desc             |          0.0305 |                   7 |             0.0028 |             0.0804 |
|     27 | sum_negative_dep            | desc             |          0.0301 |                   6 |            -0.0428 |             0.0795 |
|     28 | num_candidate_rule_edges    | desc             |          0.0299 |                   7 |             0.0016 |             0.0749 |
|     29 | redundancy_weight_mean      | desc             |          0.0297 |                   6 |            -0.0027 |             0.0814 |
|     30 | rule_mass                   | desc             |          0.0295 |                   7 |             0.0061 |             0.0751 |
