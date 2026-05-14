# Query-level Case Analysis

This directory contains the query-level analysis for the 0421 report. Use the official-aligned table for report claims, and use `raw_delta_rr` only for mechanism debugging.

## Source Files

- `query_case_level_analysis.csv`: main per-GT official-aligned case table.
- `query_subset_feature_candidates.csv`: candidate single-feature subset thresholds.
- `case_studies.md`: interpretable RuleDepDemo cases used for mechanism explanation.
- `figures/fb237_israel_eurasia_africa_tikz.tex`: compact TikZ figure for the FB15k-237 Israel case.
- `delta_mrr_plots/`: generated delta-RR distribution plots.

Data source: `RuleDepDemo/frontend/public/example/**`, rebuilt from official `processed_*_test.pkl`, then relation-level calibrated against aggregation `metric-*.json`.

## FB15k-237 Israel Figure Symbols

Figure file: `figures/fb237_israel_eurasia_africa_tikz.tex`.

Abbreviations: `c` = `/location/location/contains`, `cw` = `/location/location/countries_within`, and `adj` = `/location/location/adjoins`; `adj'` denotes the reversed adjacency atom used by the corresponding rule body. Entity shorthands in the figure are `Eu` = Eurasia, `As` = Asia, `Af` = Africa, and `Eg` = Egypt. Other compact predicates are `film` = `/film/film/release_region` and `oly` = `/olympics/olympic_athlete_affiliation/country`. Rule weights in the figure are rounded to two decimals.

The figure reports official demo scores (`S_1` and `S_2`) for the rank movement, not the raw internal `stage1/stage2` logits. This matters because Eurasia has a high raw rule logit but a low `maxConf` cap (`0.647059`), while Africa has a higher `maxConf` (`0.753731`). Therefore Stage1 ranks Africa above Eurasia in the official score space, and Stage2 flips the order mainly by applying a large negative dependency adjustment to Africa.

Rule symbols:

- `phi_1`: `c(X,A) and adj(A,Y) => c(X,Y)`.
- `phi_2`: `c(X,A) and adj(Y,A) => c(X,Y)`.
- `phi_3`: `c(Asia,Y) => c(Eurasia,Y)`.
- `phi_4`: `/film/film/release_region(The_Tourist,Y) => c(Eurasia,Y)`.
- `phi_5`: `/olympics/olympic_athlete_affiliation/country(Archery,Y) => c(Eurasia,Y)`.
- `phi_6`: `/film/film/release_region(Eternal_Sunshine_of_the_Spotless_Mind,Y) => c(Eurasia,Y)`.
- `phi_7`: `cw(X,A) and adj(B,A) and adj(Y,B) => c(X,Y)`.
- `phi_8`: `cw(X,A) and adj(A,B) and adj(B,Y) => c(X,Y)`.
- `phi_9`: `cw(X,A) and adj(B,A) and adj(B,Y) => c(X,Y)`.

The GT side is dominated by many small positive dependencies (`N_+=1939`, `S_dep=+0.394`): its representative rules are mostly independent, with a few complementary pairs. The C1 side has repeated Africa/Egypt-neighborhood rules with mostly negative dependencies (`N_-=232`, `S_dep=-1.194`): most representative dependency pairs are redundant.

## Scope

| item | count |
| --- | ---: |
| Total per-GT cases scanned | 146,020 |
| Valid cases | 146,020 |
| Improved cases | 103,335 |
| Worse cases | 24,964 |
| Unchanged cases | 17,721 |

The case table is per-GT rather than one row per query with only the first GT. The plotted `delta_rr` is calibrated so dataset/relation means are consistent with official Stage2-vs-Stage1 MRR. Raw uncalibrated demo reconstruction remains available as `raw_delta_rr` for debugging local ranking behavior.

## Dataset Summary

### Official-aligned Per-GT Cases

| dataset | cases | mean ΔRR | improved | worse | unchanged | sum positive | sum negative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FB15k-237 | 40,819 | +0.003521 | 15,605 | 11,583 | 13,631 | +253.332 | -109.600 |
| KG20C | 7,448 | +0.007571 | 5,746 | 964 | 738 | +57.667 | -1.277 |
| WN18RR | 6,055 | +0.002725 | 3,597 | 226 | 2,232 | +20.230 | -3.730 |
| YAGO3-10 | 9,975 | +0.018744 | 8,851 | 1,011 | 113 | +302.020 | -115.046 |
| codex-l | 61,129 | +0.004880 | 52,532 | 7,917 | 680 | +1076.566 | -778.227 |
| codex-m | 20,594 | +0.002932 | 17,004 | 3,263 | 327 | +78.862 | -18.487 |

### Raw Demo-local View

| dataset | cases | mean ΔRR | improved | worse | unchanged | sum positive | sum negative |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FB15k-237 | 40,819 | +0.002587 | 2,890 | 1,369 | 36,560 | +170.194 | -64.597 |
| KG20C | 7,448 | +0.000004 | 44 | 23 | 7,381 | +0.286 | -0.254 |
| WN18RR | 6,055 | +0.000431 | 53 | 22 | 5,980 | +4.856 | -2.248 |
| YAGO3-10 | 9,975 | +0.002935 | 888 | 740 | 8,347 | +153.720 | -124.440 |
| codex-l | 61,129 | -0.007557 | 2,748 | 4,129 | 54,252 | +397.204 | -859.181 |
| codex-m | 20,594 | +0.000360 | 307 | 196 | 20,091 | +13.165 | -5.747 |

## Why Query-level Analysis Is Necessary

Relation-level grouping is too coarse. Most raw relation/query pairs do not move at all, and relation averages can be carried by a small active subset. For relations with at least 5 queries, the median active-query fraction is 0 for FB15k-237, YAGO3-10, codex-l, and codex-m; KG20C and WN18RR also have very low active fractions.

Implication: a relation may look effective because a few queries move strongly. It is better to present RuleDep as identifying a useful query subset inside a relation, not as uniformly improving the whole relation.

## Failure Modes

| failure mode | symptom | likely reason |
| --- | --- | --- |
| No-op query | `raw_delta_rr = 0`, usually low `dep_nonzero_ratio`, low `num_edges`, low dependency counts | Rule dependency has no active path for the candidate set, so Stage2 cannot change the ranking. |
| Wrong-direction query | `raw_delta_rr < 0`, dependency activity is high, but GT support is weak or negative | Dependencies affect competitors more than the GT, or over-correct an already usable Stage1 ranking. |

Raw feature means show the key pattern:

| feature | improved | unchanged | worse |
| --- | ---: | ---: | ---: |
| dep_nonzero_ratio | 0.883 | 0.203 | 0.920 |
| avg_pos_dep | 27.643 | 3.134 | 15.601 |
| avg_neg_dep | 38.455 | 5.445 | 22.870 |
| num_edges | 170.187 | 26.476 | 146.504 |
| gt_dep_score | -0.194 | -0.056 | -0.283 |
| gt_rules | 44.613 | 31.790 | 33.794 |
| raw_stage1_rr | 0.171 | 0.490 | 0.320 |
| top_margin_stage1 | 0.086 | 0.147 | 0.118 |

Dependency activity is necessary but not sufficient. Worse queries are also dependency-active. The discriminative signal is whether dependency support lands on the GT rather than on competing candidates.

## Useful Query Subsets

Simple thresholds with at least 20% query coverage already identify stronger subsets on several datasets.

### Raw Query-level Subsets

| dataset | feature rule | coverage | mean raw ΔRR | base RR | relative gain |
| --- | --- | ---: | ---: | ---: | ---: |
| FB15k-237 | `avg_pos_dep >= 7.475` | 20.0% | +0.010916 | 0.333436 | +3.3% |
| FB15k-237 | `num_edges >= 137` | 20.0% | +0.010772 | 0.297321 | +3.6% |
| WN18RR | `raw_stage1_rr <= 0.5` | 50.7% | +0.001106 | 0.073673 | +1.5% |
| YAGO3-10 | `raw_stage1_rr <= 0.5` | 47.4% | +0.021075 | 0.154321 | +13.7% |
| codex-l | `raw_stage1_rr <= 0.5` | 60.5% | +0.004361 | 0.121470 | +3.6% |
| codex-m | `avg_neg_dep >= 0.12` | 20.0% | +0.001427 | 0.528457 | +0.3% |

### Official-aligned Subsets

Using each dataset's global Stage1 MRR as denominator:

| dataset | feature rule | coverage | mean aligned ΔMRR | relative to global Stage1 |
| --- | --- | ---: | ---: | ---: |
| FB15k-237 | `avg_pos_dep >= 7.475` | 20.0% | +0.013268 | +3.8% |
| KG20C | `gt_dep_margin_vs_best_non_gt <= -0.001844` | 25.0% | +0.010715 | +4.7% |
| WN18RR | `gt_stage2_margin_vs_best_non_gt <= -0.010882` | 20.0% | +0.004893 | +1.0% |
| YAGO3-10 | `gt_gain_margin_vs_best_non_gt >= 0.001020` | 25.0% | +0.044162 | +7.7% |
| codex-l | `gt_stage2_margin_vs_best_non_gt >= -0.001043` | 40.0% | +0.021039 | +6.5% |
| codex-m | `gt_stage2_margin_vs_best_non_gt >= -0.004140` | 40.0% | +0.003509 | +1.0% |

The stricter denominator means current single-threshold selectors do not reach +10% relative gain on most datasets. They still produce much higher gains than relation-level averages and identify where a learned or multi-feature selector should focus.

Very large relative gains based on rules like `gt_stage1_official <= 0` should be treated carefully because the denominator is near zero. See `query_subset_feature_candidates.csv` for the full candidate table.

## Feature Families to Report

1. Stage1 uncertainty: low `raw_stage1_rr` or small `top_margin_stage1`.
2. Dependency activity: high `dep_nonzero_ratio`, `num_edges`, `avg_pos_dep`, `avg_neg_dep`.
3. GT support quality: higher `gt_dep_score`, enough `gt_rules`, and better GT-vs-competitor dependency / score-gain margins.

The main CSV includes direct contrast features:

- `gt_dep_margin_vs_best_non_gt`
- `gt_gain_margin_vs_best_non_gt`
- `gt_stage1_margin_vs_best_non_gt`
- `gt_stage2_margin_vs_best_non_gt`
- `gt_positive_dep_margin_vs_best_non_gt`
- `gt_negative_dep_margin_vs_best_non_gt`

Initial Spearman correlations with raw query ΔRR are modest but directionally useful:

| feature | Spearman corr. with raw ΔRR |
| --- | ---: |
| `gt_official_gain` | +0.147 |
| `gt_dep_score` | +0.128 |
| `best_non_gt_dep_score` | +0.088 |
| `best_non_gt_official_gain` | +0.086 |
| `gt_gain_margin_vs_best_non_gt` | +0.085 |
| `gt_dep_margin_vs_best_non_gt` | +0.073 |

This suggests that a single feature is not enough. A defensible query subset should combine Stage1 uncertainty, dependency activity, and GT-vs-competitor support.

## Metric Guidance

Use two metrics separately:

- `delta_rr`: relation-calibrated value aligned to official Stage2-vs-Stage1 relation MRR. This is best for reporting official aggregate improvement.
- `raw_delta_rr`: local ranking behavior reconstructed from demo examples. This is best for mechanism debugging.

Older one-row-per-query recheck output was removed because its GT handling and weighting did not match the official evaluator. Use `query_case_level_analysis.csv` as the single source of truth.

## Recommended Next Analysis

Build a combined selector rather than a relation selector:

1. Keep queries with enough dependency activity, for example high `dep_nonzero_ratio` or `num_edges`.
2. Remove no-room queries where Stage1 is already too confident.
3. Prefer queries where `gt_official_gain`, `gt_dep_score`, or GT-vs-nonGT margins are favorable.
4. Report coverage and relative MRR gain at fixed coverage targets: 20%, 25%, 30%, and 40%.

This directly targets the paper claim: a query subset can cover at least 20% of queries while producing a much larger Stage2-vs-Stage1 gain than relation-level selection.
