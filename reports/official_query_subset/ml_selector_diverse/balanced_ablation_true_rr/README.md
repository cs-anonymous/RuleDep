# Balanced Subset Ablation (true per-query RR)

Greedy and exhaustive feature-subset ablation built on top of Balanced4
(`synergy_weight_top5_mean`, `max_candidate_dep_score`, `topk_rule_weight`,
`effective_candidates`). For each non-empty subset of these 4 features we
fit:

- A per-dataset RandomForest (`<tag>_rf`).
- A single global RandomForest pooled across all 7 datasets (`<tag>_global_rf`).

The training distribution and seeds match the rest of the report family
(`generate_balanced5_selector_report_true_rr.py`).

## Best subset at each size

Rows below show the top-`gain_10` subset at each cardinality, separately
for the global selector (the recommended paper method) and the per-dataset
oracle ceiling.

| scope       |   size | selector                             | tag                        | gain_10   | gain_20   | gain_50   |
|:------------|-------:|:-------------------------------------|:---------------------------|:----------|:----------|:----------|
| global      |      4 | balanced4_syn_max_topk_eff_global_rf | balanced4_syn_max_topk_eff | 16.66%    | 9.91%     | 4.25%     |
| global      |      3 | balanced3_max_topk_eff_global_rf     | balanced3_max_topk_eff     | 14.18%    | 8.91%     | 4.26%     |
| global      |      2 | balanced2_syn_topk_global_rf         | balanced2_syn_topk         | 12.64%    | 8.31%     | 4.01%     |
| global      |      1 | balanced1_syn_global_rf              | balanced1_syn              | 7.32%     | 5.95%     | 3.68%     |
| per_dataset |      4 | balanced4_syn_max_topk_eff_rf        | balanced4_syn_max_topk_eff | 22.28%    | 14.12%    | 6.21%     |
| per_dataset |      3 | balanced3_syn_topk_eff_rf            | balanced3_syn_topk_eff     | 19.88%    | 12.46%    | 5.93%     |
| per_dataset |      2 | balanced2_syn_topk_rf                | balanced2_syn_topk         | 17.42%    | 10.97%    | 5.50%     |
| per_dataset |      1 | balanced1_eff_rf                     | balanced1_eff              | 12.03%    | 9.08%     | 5.36%     |

## Full macro gain across all 15 subsets x 2 scopes

| scope       |   size | selector                             | gain_10   | gain_20   | gain_30   | gain_50   |
|:------------|-------:|:-------------------------------------|:----------|:----------|:----------|:----------|
| global      |      4 | balanced4_syn_max_topk_eff_global_rf | 16.66%    | 9.91%     | 7.00%     | 4.25%     |
| global      |      3 | balanced3_max_topk_eff_global_rf     | 14.18%    | 8.91%     | 6.56%     | 4.26%     |
| global      |      3 | balanced3_syn_max_eff_global_rf      | 14.09%    | 9.05%     | 6.28%     | 4.37%     |
| global      |      3 | balanced3_syn_max_topk_global_rf     | 14.07%    | 9.24%     | 6.54%     | 3.94%     |
| global      |      3 | balanced3_syn_topk_eff_global_rf     | 13.85%    | 8.69%     | 6.40%     | 4.40%     |
| global      |      2 | balanced2_syn_topk_global_rf         | 12.64%    | 8.31%     | 6.17%     | 4.01%     |
| global      |      2 | balanced2_max_eff_global_rf          | 11.94%    | 8.21%     | 6.25%     | 4.34%     |
| global      |      2 | balanced2_syn_eff_global_rf          | 11.49%    | 7.72%     | 5.54%     | 4.21%     |
| global      |      2 | balanced2_max_topk_global_rf         | 10.94%    | 7.93%     | 5.68%     | 3.73%     |
| global      |      2 | balanced2_syn_max_global_rf          | 10.57%    | 7.15%     | 5.60%     | 3.93%     |
| global      |      2 | balanced2_topk_eff_global_rf         | 7.20%     | 5.38%     | 4.72%     | 3.62%     |
| global      |      1 | balanced1_syn_global_rf              | 7.32%     | 5.95%     | 4.83%     | 3.68%     |
| global      |      1 | balanced1_max_global_rf              | 7.25%     | 5.46%     | 4.60%     | 3.83%     |
| global      |      1 | balanced1_eff_global_rf              | 6.05%     | 5.40%     | 4.61%     | 3.79%     |
| global      |      1 | balanced1_topk_global_rf             | 4.82%     | 3.97%     | 3.35%     | 2.59%     |
| per_dataset |      4 | balanced4_syn_max_topk_eff_rf        | 22.28%    | 14.12%    | 9.94%     | 6.21%     |
| per_dataset |      3 | balanced3_syn_topk_eff_rf            | 19.88%    | 12.46%    | 9.25%     | 5.93%     |
| per_dataset |      3 | balanced3_syn_max_topk_rf            | 19.06%    | 11.88%    | 8.55%     | 5.43%     |
| per_dataset |      3 | balanced3_max_topk_eff_rf            | 18.01%    | 12.18%    | 8.79%     | 5.93%     |
| per_dataset |      3 | balanced3_syn_max_eff_rf             | 17.98%    | 11.91%    | 8.58%     | 5.61%     |
| per_dataset |      2 | balanced2_syn_topk_rf                | 17.42%    | 10.97%    | 8.19%     | 5.50%     |
| per_dataset |      2 | balanced2_max_topk_rf                | 16.93%    | 10.60%    | 7.92%     | 5.30%     |
| per_dataset |      2 | balanced2_topk_eff_rf                | 16.84%    | 11.38%    | 8.61%     | 5.97%     |
| per_dataset |      2 | balanced2_syn_eff_rf                 | 16.76%    | 11.31%    | 8.36%     | 5.61%     |
| per_dataset |      2 | balanced2_max_eff_rf                 | 16.41%    | 11.37%    | 8.15%     | 5.42%     |
| per_dataset |      2 | balanced2_syn_max_rf                 | 14.41%    | 9.96%     | 7.19%     | 4.77%     |
| per_dataset |      1 | balanced1_eff_rf                     | 12.03%    | 9.08%     | 7.55%     | 5.36%     |
| per_dataset |      1 | balanced1_max_rf                     | 11.52%    | 7.88%     | 5.86%     | 4.36%     |
| per_dataset |      1 | balanced1_topk_rf                    | 11.02%    | 8.56%     | 7.01%     | 4.80%     |
| per_dataset |      1 | balanced1_syn_rf                     | 10.61%    | 8.16%     | 6.12%     | 4.40%     |

## Global RF feature importance per subset

| tag                        |   effective_candidates |   max_candidate_dep_score |   synergy_weight_top5_mean |   topk_rule_weight |
|:---------------------------|-----------------------:|--------------------------:|---------------------------:|-------------------:|
| balanced1_eff              |                 1      |                  nan      |                   nan      |           nan      |
| balanced1_max              |               nan      |                    1      |                   nan      |           nan      |
| balanced1_syn              |               nan      |                  nan      |                     1      |           nan      |
| balanced1_topk             |               nan      |                  nan      |                   nan      |             1      |
| balanced2_max_eff          |                 0.4623 |                    0.5377 |                   nan      |           nan      |
| balanced2_max_topk         |               nan      |                    0.596  |                   nan      |             0.404  |
| balanced2_syn_eff          |                 0.3074 |                  nan      |                     0.6926 |           nan      |
| balanced2_syn_max          |               nan      |                    0.2944 |                     0.7056 |           nan      |
| balanced2_syn_topk         |               nan      |                  nan      |                     0.7106 |             0.2894 |
| balanced2_topk_eff         |                 0.523  |                  nan      |                   nan      |             0.477  |
| balanced3_max_topk_eff     |                 0.3125 |                    0.4211 |                   nan      |             0.2663 |
| balanced3_syn_max_eff      |                 0.2338 |                    0.2029 |                     0.5632 |           nan      |
| balanced3_syn_max_topk     |               nan      |                    0.1998 |                     0.5742 |             0.2259 |
| balanced3_syn_topk_eff     |                 0.2001 |                  nan      |                     0.5954 |             0.2045 |
| balanced4_syn_max_topk_eff |                 0.181  |                    0.1424 |                     0.4868 |             0.1898 |

## Notes

- `balanced1_syn_global_rf` should equal the single-feature `synergy_weight_top5_mean_global_rf`
  number reported in `balanced5_report_true_rr` modulo seed-key noise.
- For the paper, look at the `global` rows of the winners table: those tell
  you whether shrinking from 4 to 3 features (or 2, or 1) costs more or
  less gain than you can defend, and which subset is best at each size.
