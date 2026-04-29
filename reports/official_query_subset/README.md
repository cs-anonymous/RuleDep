# Official-aligned Query Subset Feature Analysis

This directory collects the feature search, fixed-coverage ranking, and hypothesis validation results for the official-aligned query subset.

Metric: `gain_pt = MRR_stage2 / MRR_stage1 - 1`, computed from official-aligned per-test-triple RR.

Feature policy: all ranking and threshold features are query/candidate-set aggregates. Threshold selection does not use `target_gt_entity`, GT rank, GT score, or any GT-specific dependency value.

## Files

- [`official_query_triple_features.csv`](official_query_triple_features.csv): query-level feature table used by the analyses.
- [`feature_rankings_at_coverage.csv`](feature_rankings_at_coverage.csv): macro-average feature rankings at fixed coverage.
- [`best_feature_threshold_summary.csv`](best_feature_threshold_summary.csv): best per-dataset thresholds with coverage at least 20%.
- [`feature_threshold_curves.csv`](feature_threshold_curves.csv): threshold curve data behind the plots.
- [`hypothesis_eval/hypothesis_validation.csv`](hypothesis_eval/hypothesis_validation.csv): 30 hypothesis checks with Spearman correlation and per-dataset direction counts.
- [`hypothesis_eval/feature_importance_ranking.csv`](hypothesis_eval/feature_importance_ranking.csv): hypothesis features sorted by robust score.
- [`feature_plots/`](feature_plots/): generated feature curves. `feature_plots/<feature>__desc.png` keeps larger feature values first; `feature_plots/<feature>__asc.png` keeps smaller feature values first.

## Main Takeaways

1. The most reliable explanatory signals are dependency strength and dependency-to-rule contrast: `topk_synergy`, `pos_mass`, `syn_rule_ratio`, `dep_rule_ratio`, `net_dep_mass`, and `abs_dep_mass`.
2. Fixed-coverage subset selection is strongest at 10% coverage for dependency-activity features such as `sum_positive_dep`, `combo_dependency_activity`, and `combo_complex_dep_low_conf`.
3. At 20% coverage, the best macro feature is `combo_complex_dep_low_conf`, followed by `num_candidates` and `combo_dep_activity_x_uncertainty`.
4. Rule-weight and Stage1-ambiguity hypotheses are less stable across datasets; they are better treated as boundary conditions than as the central explanation.

## Fixed-coverage Rankings

Ranking score: macro-average `gain_pt` across datasets at the same coverage. `sort_direction=desc` keeps larger feature values first; `asc` keeps smaller feature values first.

### Coverage 10%

| rank | feature | order | macro gain_pt | positive datasets | min dataset gain | max dataset gain |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `sum_positive_dep` | desc | 0.1585 | 6 | 0.0230 | 0.7493 |
| 2 | `combo_dependency_activity` | desc | 0.1566 | 6 | 0.0162 | 0.7305 |
| 3 | `combo_complex_dep_low_conf` | desc | 0.1539 | 6 | 0.0263 | 0.6530 |
| 4 | `sum_negative_dep` | desc | 0.1389 | 6 | 0.0113 | 0.6655 |
| 5 | `num_dependency_edges` | desc | 0.1289 | 6 | 0.0162 | 0.5464 |
| 6 | `query_num_edges` | desc | 0.1289 | 6 | 0.0162 | 0.5464 |
| 7 | `unique_synergy_edges` | desc | 0.1276 | 6 | 0.0092 | 0.5274 |
| 8 | `avg_positive_dep` | desc | 0.1274 | 6 | 0.0232 | 0.5686 |
| 9 | `num_candidates` | desc | 0.1105 | 6 | 0.0116 | 0.4372 |
| 10 | `redundancy_weight_top10_sum` | desc | 0.0951 | 6 | 0.0065 | 0.4403 |

### Coverage 20%

| rank | feature | order | macro gain_pt | positive datasets | min dataset gain | max dataset gain |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `combo_complex_dep_low_conf` | desc | 0.0870 | 6 | 0.0163 | 0.2865 |
| 2 | `num_candidates` | desc | 0.0793 | 6 | 0.0032 | 0.3165 |
| 3 | `combo_dep_activity_x_uncertainty` | desc | 0.0677 | 6 | 0.0168 | 0.1806 |
| 4 | `combo_candidate_complexity` | desc | 0.0591 | 6 | 0.0042 | 0.2000 |
| 5 | `num_dependency_edges` | desc | 0.0545 | 6 | 0.0063 | 0.1404 |
| 6 | `query_num_edges` | desc | 0.0545 | 6 | 0.0063 | 0.1404 |
| 7 | `combo_dependency_activity` | desc | 0.0519 | 6 | 0.0053 | 0.1420 |
| 8 | `unique_synergy_edges` | desc | 0.0514 | 6 | 0.0061 | 0.1378 |
| 9 | `avg_stage1_score` | desc | 0.0511 | 6 | 0.0054 | 0.1802 |
| 10 | `sum_positive_dep` | desc | 0.0508 | 6 | 0.0068 | 0.1443 |

## Best Per-dataset Thresholds

The table below reports the best feature for each dataset at the fixed coverage targets used in the previous summary. See [`best_feature_threshold_summary.csv`](best_feature_threshold_summary.csv) for the full threshold search with coverage at least 20%.

| dataset | coverage | feature | order | gain_pt | threshold |
| --- | ---: | --- | --- | ---: | ---: |
| FB15k-237 | 20% | `sum_negative_dep` | desc | 0.0569 | 5388 |
| FB15k-237 | 10% | `unique_redundancy_edges` | desc | 0.0699 | 54 |
| FB15k-237 | 5% | `unique_redundancy_edges` | desc | 0.1210 | 111 |
| KG20C | 20% | `top1_rule_weight` | asc | 0.0813 | 0.207639 |
| KG20C | 10% | `num_dependency_edges` | desc | 0.0975 | 38 |
| KG20C | 5% | `combo_complex_dep_low_conf` | desc | 0.0988 | 0.694485 |
| WN18RR | 20% | `rule_dominance_ratio` | asc | 0.0441 | 0.00978218 |
| WN18RR | 10% | `combo_dep_activity_x_uncertainty` | desc | 0.0658 | 0.414996 |
| WN18RR | 5% | `combo_dep_activity_x_uncertainty` | desc | 0.0889 | 0.439567 |
| YAGO3-10 | 20% | `max_stage2_score` | asc | 0.0931 | 0.119311 |
| YAGO3-10 | 10% | `combo_dep_activity_x_uncertainty` | desc | 0.0931 | 0.408976 |
| YAGO3-10 | 5% | `num_candidates` | desc | 0.1171 | 103 |
| codex-l | 20% | `s1_entropy` | desc | 0.3370 | 4.66611 |
| codex-l | 10% | `sum_positive_dep` | desc | 0.7463 | 3175 |
| codex-l | 5% | `redundancy_weight_top10_sum` | desc | 1.6105 | 4.1562 |
| codex-m | 20% | `s1_entropy` | desc | 0.0501 | 4.64758 |
| codex-m | 10% | `rule_dominance_ratio` | asc | 0.0697 | 0.00161222 |
| codex-m | 5% | `combo_candidate_complexity` | desc | 0.0972 | 0.920151 |

## Hypothesis Validation

Data: [`official_query_triple_features.csv`](official_query_triple_features.csv), 146020 samples.

Target: query-level `delta_rr`.

Statistic: Spearman correlation, measured both globally and by per-dataset direction consistency.

Summary: 30/30 hypothesis features are covered; 13 are supported, 3 are partially supported, and 9 are not supported. The remaining negative-dependency features are best treated as observational because their expected direction is intentionally non-monotonic.

### Most Reliable Hypothesis Features

`robust_score = |rho_all| * dataset_direction_consistency`.

| rank | feature | category | rho_all | matched datasets | robust_score |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | `topk_synergy` | Dependency weight strength | 0.3921 | 5/6 | 0.3267 |
| 2 | `pos_mass` | Dependency weight strength | 0.3568 | 5/6 | 0.2973 |
| 3 | `syn_rule_ratio` | Dependency-to-rule contrast | 0.3560 | 5/6 | 0.2967 |
| 4 | `dep_rule_ratio` | Dependency-to-rule contrast | 0.3530 | 5/6 | 0.2941 |
| 5 | `net_dep_mass` | Dependency weight strength | 0.3509 | 5/6 | 0.2924 |
| 6 | `abs_dep_mass` | Dependency weight strength | 0.3503 | 5/6 | 0.2919 |
| 7 | `dep_candidate_ratio` | Rule graph structure | 0.3306 | 5/6 | 0.2755 |
| 8 | `num_pos_dep` | Positive/negative dependency structure | 0.2981 | 5/6 | 0.2484 |

### Verdict by Feature Group

| group | supported | partially supported | not supported / observational |
| --- | --- | --- | --- |
| Candidate set complexity | `num_candidates` | - | `num_candidate_rule_edges`, `avg_rules_per_candidate`, `max_rules_per_candidate` |
| Rule graph structure | `num_rules`, `num_dependencies`, `dep_density`, `dep_candidate_ratio` | - | - |
| Positive/negative dependency structure | `num_pos_dep`, `pos_dep_ratio` | - | `num_neg_dep`, `neg_dep_ratio` are observational |
| Dependency weight strength | `pos_mass`, `net_dep_mass`, `abs_dep_mass`, `topk_synergy` | - | `neg_mass`, `topk_redundancy` are observational |
| Rule-weight distribution | - | `rule_dominance_ratio` | `top1_rule_weight`, `topk_rule_weight`, `weak_rule_score` |
| Dependency-to-rule contrast | `dep_rule_ratio`, `syn_rule_ratio` | - | `red_rule_ratio` is observational |
| S1 ambiguity | - | `s1_entropy`, `effective_candidates` | `s1_top1`, `s1_margin`, `s1_norm_margin` |

## Key Plots

These plots are the most useful for the main report narrative:

- [`topk_synergy__desc.png`](feature_plots/topk_synergy__desc.png)
- [`pos_mass__desc.png`](feature_plots/pos_mass__desc.png)
- [`syn_rule_ratio__desc.png`](feature_plots/syn_rule_ratio__desc.png)
- [`dep_rule_ratio__desc.png`](feature_plots/dep_rule_ratio__desc.png)
- [`net_dep_mass__desc.png`](feature_plots/net_dep_mass__desc.png)
- [`abs_dep_mass__desc.png`](feature_plots/abs_dep_mass__desc.png)
- [`combo_complex_dep_low_conf__desc.png`](feature_plots/combo_complex_dep_low_conf__desc.png)
- [`sum_positive_dep__desc.png`](feature_plots/sum_positive_dep__desc.png)

Each plot fixes one feature and shows six dataset curves. The x-axis is data coverage in the selected subset. The y-axis is `gain_pt`. `ALL macro avg` is the average of dataset-level `gain_pt` values at the same coverage point.

## Feature Definitions

All features are computed from the query/candidate-set graph before looking at which candidate is the GT.

### Basic Query Size

- `num_candidates`: number of candidates retained for the test triple query.
- `num_rule_nodes`: number of unique rule IDs appearing in any candidate active rule list.
- `num_dependency_edges`: number of unique displayed dependency pairs among active candidate rules.
- `query_num_nodes`, `query_num_edges`: node and edge counts stored in `queries.json`.

### Candidate Rule Support

- `avg_rules_per_candidate`, `max_rules_per_candidate`: mean/max `scoredRuleCount` over candidates.
- `candidate_rule_coverage`: fraction of candidates with at least one active rule.

### Candidate Dependency Activity

- `candidate_dep_coverage`: fraction of candidates with `positiveDep + negativeDep > 0`.
- `sum_positive_dep`, `sum_negative_dep`: sums over all candidates.
- `avg_positive_dep`, `avg_negative_dep`: means over all candidates.
- `max_positive_dep`, `max_negative_dep`: maxima over all candidates.
- `avg_candidate_dep_score`, `max_candidate_dep_score`: mean/max candidate `dependencyScore`.

### Candidate Score Distribution

- `avg_stage1_score`, `max_stage1_score`: mean/max official Stage1 candidate score.
- `stage1_top_margin`: top-1 minus top-2 official Stage1 candidate score.
- `avg_stage2_score`, `max_stage2_score`: mean/max official Stage2 candidate score.
- `stage2_top_margin`: top-1 minus top-2 official Stage2 candidate score.

### Dependency Edge Types

`displayedDependencyPairs` gives active rule-rule pairs in candidates. Pair type and weight are looked up from `data/<dataset>/rules/synergy_filtered.txt` and `data/<dataset>/rules/redundancy_filtered.txt`.

- `unique_synergy_edges`: unique active dependency pairs found in `synergy_filtered.txt`.
- `unique_redundancy_edges`: unique active dependency pairs found in `redundancy_filtered.txt`.

### Rule Weights

For each candidate, `maxplus` contains rule contribution values shown by the demo. The `rule_weight_*` features pool these values across all candidates in the query.

- `rule_weight_topK_sum`, `rule_weight_topK_mean`: sum/mean of the top K pooled rule contribution values, for K in `{1,3,5,10}`.
- `rule_weight_max`, `rule_weight_mean`: max/mean pooled rule contribution value.

### Synergy and Redundancy Weights

For active dependency pairs, the absolute filtered dependency weight is used. Synergy and redundancy are computed separately.

- `synergy_weight_topK_sum`, `redundancy_weight_topK_sum`: sum of top K absolute active dependency weights.
- `synergy_weight_topK_mean`, `redundancy_weight_topK_mean`: mean of top K absolute active dependency weights.
- `synergy_weight_max`, `redundancy_weight_max`: max absolute active dependency weight.
- `synergy_weight_mean`, `redundancy_weight_mean`: mean absolute active dependency weight.

### Composite Features

Composite features use dataset-wise percentile ranks so that features with different scales can be combined. A high percentile means the query is high on that feature within the dataset.

- `combo_dependency_activity`: mean percentile rank of `num_dependency_edges`, `candidate_dep_coverage`, `sum_positive_dep`, `sum_negative_dep`, `unique_synergy_edges`, and `unique_redundancy_edges`.
- `combo_candidate_complexity`: mean percentile rank of `num_candidates`, `avg_rules_per_candidate`, and `rule_weight_mean`.
- `combo_stage1_uncertainty`: mean of `1 - percentile(max_stage1_score)` and `1 - percentile(stage1_top_margin)`.
- `combo_low_rule_confidence`: `1 - percentile(rule_weight_max)`.
- `combo_dep_activity_x_uncertainty`: `combo_dependency_activity * combo_stage1_uncertainty`.
- `combo_complex_dep_low_conf`: mean of `combo_candidate_complexity`, `combo_dependency_activity`, and `combo_low_rule_confidence`.

## Reporting Recommendation

For the paper/report, make dependency quality the main story: `topk_synergy`, `syn_rule_ratio`, `dep_rule_ratio`, `candidate_dep_coverage`, and `num_dependencies` are the most defensible features to keep in the main table. Use the less stable E/G groups as applicability boundaries rather than headline claims.
