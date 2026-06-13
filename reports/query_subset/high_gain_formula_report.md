# Compact High-gain Formula Report

Headline: 10% coverage gives 4.29% `gain_pt`; 20% coverage gives 4.05% `gain_pt`.

Formula:

```text
compact_score = max(
  P_d(candidate_dep_coverage),
  P_d(synergy_weight_mean),
  P_d(synergy_weight_top3_mean)
)
```

Metric: official-scaled per-query RR; 100% coverage is aligned to `metric-*.json` official Stage1/Stage2 MRR.

Plot: [`feature_plots/high_gain_formula__desc.png`](feature_plots/high_gain_formula__desc.png)

## Fixed Coverage Results

| dataset   |    10% |    20% |   100% |
|:----------|-------:|-------:|-------:|
| FB15k-237 | 0.038  | 0.0351 | 0.01   |
| KG20C     | 0.0513 | 0.0523 | 0.0334 |
| WN18RR    | 0.0285 | 0.0185 | 0.0053 |
| YAGO3-10  | 0.0443 | 0.0468 | 0.0334 |
| codex-l   | 0.017  | 0.0171 | 0.0148 |
| codex-m   | 0.0055 | 0.0057 | 0.0086 |
| hetionet  | 0.1158 | 0.1082 | 0.0457 |
| macro     | 0.0429 | 0.0405 | 0.0216 |

## Formula Variables

| feature_or_component     | sort_direction   |    10% |    20% |   100% |
|:-------------------------|:-----------------|-------:|-------:|-------:|
| compact_formula          | desc             | 0.0429 | 0.0405 | 0.0216 |
| candidate_dep_coverage   | desc             | 0.0418 | 0.037  | 0.0216 |
| synergy_weight_mean      | desc             | 0.0412 | 0.0389 | 0.0216 |
| synergy_weight_top3_mean | desc             | 0.0397 | 0.0352 | 0.0216 |
