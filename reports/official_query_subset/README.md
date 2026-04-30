# Official-aligned Query Subset Feature Analysis

Metric: `gain_pt = MRR_stage2 / MRR_stage1 - 1`, computed from per-query RR after per-relation multiplicative scaling. This preserves query-level subset variation while making 100% coverage match the official `metric-*.json` Stage1/Stage2 MRR.

Coverage grid: 2%, 4%, ..., 100%. Ranking features are raw query/candidate-set attributes; outcome fields, calibration fields, official-scale diagnostic fields, and `combo_*` features are excluded from the raw-attribute rankings.

## Files

- [`official_query_triple_features.csv`](official_query_triple_features.csv): query-level feature table used by the analyses.
- [`feature_threshold_curves.csv`](feature_threshold_curves.csv): raw-attribute coverage curves at 2% increments.
- [`feature_rankings_at_coverage.csv`](feature_rankings_at_coverage.csv): raw-attribute macro rankings at 10% and 20% coverage.
- [`feature_rankings_at_coverage.md`](feature_rankings_at_coverage.md): readable top rankings.
- [`best_feature_threshold_summary.csv`](best_feature_threshold_summary.csv): best per-dataset raw-attribute thresholds with coverage >=20%.
- [`high_gain_formula_report.md`](high_gain_formula_report.md): selected compact formula and fixed-coverage results.
- [`feature_plots/`](feature_plots/): per-feature coverage-gain plots.

## Data Coverage

- Datasets: FB15k-237, KG20C, WN18RR, YAGO3-10, codex-l, codex-m, hetionet.
- Samples: 596,060 per-GT cases.
- Raw attributes ranked: 84.

## 100% Official Alignment Check

Values are official filtered MRR from `metric-*.json` for the best config per dataset (see `reports/best_config_by_dataset.csv`).

| dataset   |   mrr_stage1 |   mrr_stage2 |   gain_pt | best config |
|:----------|-------------:|-------------:|----------:|:------------|
| FB15k-237 |       0.3542 |       0.3578 |    0.0100 | tg_r2d3__pos_auto_ratio__ri_conf__dn_none__dl1_1e-5 |
| KG20C     |       0.2264 |       0.2340 |    0.0334 | tg_r2d3__pos_auto_ratio__ri_conf__dn_per_rule_degree__dl1_1e-5 |
| WN18RR    |       0.4990 |       0.5017 |    0.0054 | structural_rd |
| YAGO3-10  |       0.5603 |       0.5790 |    0.0334 | tg_r3d6__pos_auto_sqrt__ri_surprisal__dn_none__dl1_1e-5 |
| codex-l   |       0.3290 |       0.3339 |    0.0148 | tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5 |
| codex-m   |       0.3415 |       0.3445 |    0.0086 | tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5 |
| hetionet  |       0.3741 |       0.3912 |    0.0458 | tg_rd__pos_auto_ratio__ri_conf__dn_per_rule_degree__dl1_1e-5 |
| **macro** |              |              | **0.0216** |             |

> The 100% coverage values in `feature_threshold_curves.csv` are per-query raw RR macro-averages computed from the feature CSV (which contains mixed experiments per dataset). They differ from the official filtered MRR above for WN18RR and hetionet where the feature CSV experiment differs from the best config.

## Main Takeaways

1. Per-dataset best **raw** attributes (excluding stage2 features) give 6.27% macro `gain_pt` at 10% coverage and 5.28% at 20% coverage. The compact formula gives 4.29% / 4.05%.
2. The 10% coverage target (10% gain) is not met: WN18RR (3.34%) and codex-m (3.17%) are hard bottlenecks — no single raw attribute reaches >4% at 10% for these datasets.
3. The 20% coverage target (5% gain) is approximately met at 5.28% macro with per-dataset best raw attributes.
4. Monotonicity is not enforced; features are selected for high 10%/20% fixed-coverage gain.

## Per-Dataset gain_pt Tables

Top attributes (union of macro top-10 at 10% and 20% coverage). Values are `gain_pt` at fixed coverage thresholds.

### FB15k-237

| feature                        | dir  |    10% |    20% |    30% |    50% |   100% |
|:-------------------------------|:-----|-------:|-------:|-------:|-------:|-------:|
| dep_candidate_ratio            | desc |  0.0244 |  0.0415 |  0.0364 |  0.0198 |  0.0100 |
| candidate_dep_coverage         | desc |  0.0244 |  0.0415 |  0.0364 |  0.0198 |  0.0100 |
| synergy_weight_mean            | desc |  0.0315 |  0.0323 |  0.0313 |  0.0169 |  0.0100 |
| synergy_weight_top1_sum        | desc |  0.0398 |  0.0330 |  0.0315 |  0.0169 |  0.0100 |
| synergy_weight_top3_mean       | desc |  0.0395 |  0.0335 |  0.0316 |  0.0169 |  0.0100 |
| topk_synergy                   | desc |  0.0395 |  0.0335 |  0.0316 |  0.0169 |  0.0100 |
| redundancy_weight_top10_mean   | desc |  0.0255 |  0.0224 |  0.0226 |  0.0159 |  0.0100 |

### KG20C

| feature                        | dir  |    10% |    20% |    30% |    50% |   100% |
|:-------------------------------|:-----|-------:|-------:|-------:|-------:|-------:|
| dep_candidate_ratio            | desc |  0.0548 |  0.0550 |  0.0548 |  0.0561 |  0.0334 |
| candidate_dep_coverage         | desc |  0.0548 |  0.0550 |  0.0548 |  0.0561 |  0.0334 |
| synergy_weight_mean            | desc |  0.0512 |  0.0546 |  0.0568 |  0.0597 |  0.0334 |
| synergy_weight_top1_sum        | desc |  0.0504 |  0.0497 |  0.0547 |  0.0577 |  0.0334 |
| synergy_weight_top3_mean       | desc |  0.0495 |  0.0516 |  0.0525 |  0.0585 |  0.0334 |
| topk_synergy                   | desc |  0.0495 |  0.0516 |  0.0525 |  0.0585 |  0.0334 |
| redundancy_weight_top10_mean   | desc |  0.0379 |  0.0432 |  0.0361 |  0.0258 |  0.0334 |

### WN18RR

| feature                        | dir  |    10% |    20% |    30% |    50% |   100% |
|:-------------------------------|:-----|-------:|-------:|-------:|-------:|-------:|
| dep_candidate_ratio            | desc |  0.0273 |  0.0185 |  0.0169 |  0.0069 |  0.0053 |
| candidate_dep_coverage         | desc |  0.0273 |  0.0185 |  0.0169 |  0.0069 |  0.0053 |
| synergy_weight_mean            | desc |  0.0314 |  0.0185 |  0.0169 |  0.0069 |  0.0053 |
| synergy_weight_top1_sum        | desc |  0.0307 |  0.0185 |  0.0169 |  0.0069 |  0.0053 |
| synergy_weight_top3_mean       | desc |  0.0290 |  0.0185 |  0.0169 |  0.0069 |  0.0053 |
| topk_synergy                   | desc |  0.0290 |  0.0185 |  0.0169 |  0.0069 |  0.0053 |
| redundancy_weight_top10_mean   | desc |  0.0334 |  0.0220 |  0.0169 |  0.0069 |  0.0053 |

### YAGO3-10

| feature                        | dir  |    10% |    20% |    30% |    50% |   100% |
|:-------------------------------|:-----|-------:|-------:|-------:|-------:|-------:|
| dep_candidate_ratio            | desc |  0.0375 |  0.0402 |  0.0425 |  0.0376 |  0.0334 |
| candidate_dep_coverage         | desc |  0.0375 |  0.0402 |  0.0425 |  0.0376 |  0.0334 |
| synergy_weight_mean            | desc |  0.0378 |  0.0378 |  0.0404 |  0.0389 |  0.0334 |
| synergy_weight_top1_sum        | desc |  0.0427 |  0.0474 |  0.0477 |  0.0428 |  0.0334 |
| synergy_weight_top3_mean       | desc |  0.0532 |  0.0488 |  0.0491 |  0.0423 |  0.0334 |
| topk_synergy                   | desc |  0.0532 |  0.0488 |  0.0491 |  0.0423 |  0.0334 |
| redundancy_weight_top10_mean   | desc |  0.0440 |  0.0432 |  0.0378 |  0.0320 |  0.0334 |

### codex-l

| feature                        | dir  |    10% |    20% |    30% |    50% |   100% |
|:-------------------------------|:-----|-------:|-------:|-------:|-------:|-------:|
| dep_candidate_ratio            | desc |  0.0154 |  0.0094 |  0.0182 |  0.0170 |  0.0148 |
| candidate_dep_coverage         | desc |  0.0154 |  0.0094 |  0.0182 |  0.0170 |  0.0148 |
| synergy_weight_mean            | desc |  0.0248 |  0.0290 |  0.0269 |  0.0169 |  0.0148 |
| synergy_weight_top1_sum        | desc |  0.0007 | -0.0252 |  0.0116 |  0.0166 |  0.0148 |
| synergy_weight_top3_mean       | desc | -0.0135 | -0.0267 |  0.0141 |  0.0166 |  0.0148 |
| topk_synergy                   | desc | -0.0135 | -0.0267 |  0.0141 |  0.0166 |  0.0148 |
| redundancy_weight_top10_mean   | desc |  0.0599 | -0.0219 | -0.0017 |  0.0182 |  0.0148 |

### codex-m

| feature                        | dir  |    10% |    20% |    30% |    50% |   100% |
|:-------------------------------|:-----|-------:|-------:|-------:|-------:|-------:|
| dep_candidate_ratio            | desc |  0.0169 |  0.0059 |  0.0055 |  0.0079 |  0.0086 |
| candidate_dep_coverage         | desc |  0.0169 |  0.0059 |  0.0055 |  0.0079 |  0.0086 |
| synergy_weight_mean            | desc |  0.0057 |  0.0052 |  0.0056 |  0.0079 |  0.0086 |
| synergy_weight_top1_sum        | desc |  0.0042 |  0.0049 |  0.0057 |  0.0079 |  0.0086 |
| synergy_weight_top3_mean       | desc |  0.0041 |  0.0049 |  0.0056 |  0.0079 |  0.0086 |
| topk_synergy                   | desc |  0.0041 |  0.0049 |  0.0056 |  0.0079 |  0.0086 |
| redundancy_weight_top10_mean   | desc |  0.0050 |  0.0028 |  0.0055 |  0.0079 |  0.0086 |

### hetionet

| feature                        | dir  |    10% |    20% |    30% |    50% |   100% |
|:-------------------------------|:-----|-------:|-------:|-------:|-------:|-------:|
| dep_candidate_ratio            | desc |  0.1161 |  0.0882 |  0.0737 |  0.0733 |  0.0457 |
| candidate_dep_coverage         | desc |  0.1161 |  0.0882 |  0.0737 |  0.0733 |  0.0457 |
| synergy_weight_mean            | desc |  0.1062 |  0.0950 |  0.0822 |  0.0734 |  0.0457 |
| synergy_weight_top1_sum        | desc |  0.1161 |  0.1161 |  0.0948 |  0.0734 |  0.0457 |
| synergy_weight_top3_mean       | desc |  0.1161 |  0.1161 |  0.0926 |  0.0734 |  0.0457 |
| topk_synergy                   | desc |  0.1161 |  0.1161 |  0.0926 |  0.0734 |  0.0457 |
| redundancy_weight_top10_mean   | desc |  0.0774 |  0.0814 |  0.0755 |  0.0745 |  0.0457 |

## Query Selection Strategy

Each dataset has different raw attributes that perform best. We compare three strategies:

### Strategy 1: Per-Dataset Best Raw Attribute (non-stage2)

Each dataset uses its own best **raw** attribute at 10% coverage (stage2 features excluded since they cannot be computed before running stage2):

| dataset   | best attribute (dir)         |  10%  |  20%  |  30%  |  50%  |
|:----------|:-----------------------------|------:|------:|------:|------:|
| FB15k-237 | avg_candidate_dep_score (asc)| 0.0625| 0.0459| 0.0361| 0.0205|
| KG20C     | topk_rule_weight (asc)       | 0.0673| 0.0680| 0.0664| 0.0503|
| WN18RR    | neg_dep_ratio (desc)         | 0.0334| 0.0220| 0.0169| 0.0069|
| YAGO3-10  | num_candidates (desc)        | 0.0746| 0.0487| 0.0382| 0.0338|
| codex-l   | num_rule_nodes (desc)        | 0.0533| 0.0439| 0.0317| 0.0232|
| codex-m   | s1_entropy (desc)            | 0.0317| 0.0248| 0.0169| 0.0112|
| hetionet  | candidate_rule_coverage (desc)|0.1161| 0.1161| 0.1082| 0.0751|
| **macro** |                              |**0.0627**|**0.0528**|**0.0440**|**0.0316**|

**Targets**: 10% coverage >= 10%, 20% coverage >= 5%. The 20% target is approximately met (5.28%). The 10% target is not met (6.27% vs 10% target). WN18RR and codex-m are the hard bottlenecks — no single raw attribute reaches >4% at 10% for these datasets.

### Strategy 2: Single Best Attribute (All Datasets)

`dep_candidate_ratio` (desc) is the best single attribute that works uniformly across all datasets:

| coverage | macro | FB15k-237 | KG20C | WN18RR | YAGO3-10 | codex-l | codex-m | hetionet |
|:---------|------:|----------:|------:|-------:|---------:|--------:|--------:|---------:|
| 10%      | 0.0418|    0.0244 | 0.0548|  0.0273|   0.0375 |  0.0154 |  0.0169 |   0.1161 |
| 20%      | 0.0370|    0.0415 | 0.0550|  0.0185|   0.0402 |  0.0094 |  0.0059 |   0.0882 |
| 30%      | 0.0354|    0.0364 | 0.0548|  0.0169|   0.0425 |  0.0182 |  0.0055 |   0.0737 |
| 50%      | 0.0312|    0.0198 | 0.0561|  0.0069|   0.0376 |  0.0170 |  0.0079 |   0.0733 |
| 100%     | 0.0216|    0.0100 | 0.0334|  0.0054|   0.0334 |  0.0148 |  0.0086 |   0.0457 |

### Strategy 3: Compact Formula

`compact_score = max(P_d(candidate_dep_coverage), P_d(synergy_weight_mean), P_d(synergy_weight_top3_mean))`:

| dataset   |  10%  |  20%  | 100%  |
|:----------|------:|------:|------:|
| FB15k-237 | 0.0380| 0.0351| 0.0100|
| KG20C     | 0.0513| 0.0523| 0.0334|
| WN18RR    | 0.0285| 0.0185| 0.0054|
| YAGO3-10  | 0.0443| 0.0468| 0.0334|
| codex-l   | 0.0170| 0.0171| 0.0148|
| codex-m   | 0.0055| 0.0057| 0.0086|
| hetionet  | 0.1158| 0.1082| 0.0457|
| **macro** | **0.0429** | **0.0405** | **0.0216** |

### Comparison Summary

| strategy                              | 10%   | 20%   |
|:--------------------------------------|------:|------:|
| Per-dataset best raw attribute        | 0.0627| 0.0528|
| Compact formula                       | 0.0429| 0.0405|
| Single best attribute                 | 0.0418| 0.0370|

> The 10% gap (6.27% vs 10% target) cannot be closed by per-dataset feature selection alone. WN18RR and codex-m have no single raw attribute that reaches >4% at 10% coverage. A formula or combo approach that captures cross-feature interactions would be needed.

## Paper-facing Compact Formula

```text
compact_score = max(
  P_d(candidate_dep_coverage),
  P_d(synergy_weight_mean),
  P_d(synergy_weight_top3_mean)
)
```

See [`high_gain_formula_report.md`](high_gain_formula_report.md) for per-dataset values and component ablations.

## Fixed-coverage Rankings

### Coverage 10%

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

### Coverage 20%

|   rank | feature                  | sort_direction   |   macro_gain_pt |   positive_datasets |   min_dataset_gain |   max_dataset_gain |
|-------:|:-------------------------|:-----------------|----------------:|--------------------:|-------------------:|-------------------:|
|      1 | synergy_weight_mean      | desc             |          0.0389 |                   7 |             0.0052 |             0.095  |
|      2 | dep_candidate_ratio      | desc             |          0.037  |                   7 |             0.0059 |             0.0882 |
|      3 | candidate_dep_coverage   | desc             |          0.037  |                   7 |             0.0059 |             0.0882 |
|      4 | synergy_weight_top3_mean | desc             |          0.0352 |                   6 |            -0.0267 |             0.1161 |
|      5 | topk_synergy             | desc             |          0.0352 |                   6 |            -0.0267 |             0.1161 |
|      6 | topk_synergy_sum         | desc             |          0.0352 |                   6 |            -0.0267 |             0.1161 |
|      7 | synergy_weight_top3_sum  | desc             |          0.0352 |                   6 |            -0.0267 |             0.1161 |
|      8 | synergy_weight_top1_sum  | desc             |          0.0349 |                   6 |            -0.0252 |             0.1161 |
|      9 | synergy_weight_top1_mean | desc             |          0.0349 |                   6 |            -0.0252 |             0.1161 |
|     10 | synergy_weight_max       | desc             |          0.0349 |                   6 |            -0.0252 |             0.1161 |
