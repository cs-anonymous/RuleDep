# Higher-Order Rule-Dependency Analysis

This directory contains the empirical analysis used to assess whether restricting
RuleDep to pairwise rule dependencies discards substantial higher-order evidence.
The analysis deliberately uses **rule pair** and **three-rule combination** rather
than "pair" and "triple" alone, because "triple" is easily confused with a
knowledge-graph fact `(head, relation, tail)`.

## Research Question

RuleDep models a correction for two rules that co-fire for the same query and
candidate entity. A reviewer asked whether three or more rules may exhibit joint
effects that cannot be represented by pairwise terms.

The relevant question is not simply whether three rules co-fire. Co-firing only
shows that a combination can occur. A useful higher-order term must:

1. occur in the relation-local candidate space;
2. have enough positive joint support to be estimated reliably; and
3. have a joint gain that is significantly stronger than the gains already
   exhibited by its three constituent rule pairs.

This analysis therefore compares rule pairs and three-rule combinations under
the same candidate selection, support thresholds, evidence transformation, and
confidence-interval procedure.

## Connection to `DepLearn.kt`

The implementation follows the main statistical choices in
`src/main/java/tarmorn/DepLearn.kt` and
`src/main/java/tarmorn/structure/Metric.kt`:

- relation-local analysis;
- at most `TOP_K=500` eligible rules per relation;
- minimum marginal evidence `e_min=0.05`;
- minimum joint support `n_min=2` for KG20C and WN18RR;
- minimum joint support `n_min=5` for the other datasets;
- unseen-negative smoothing count `lambda=3`;
- log-failure evidence capped at 7;
- minimum gain margin `g_min=0.01`.

For a rule or rule combination with positive support `s` and body count `b`, the
smoothed confidence and evidence are

```text
conf = s / (b + 3)
e    = min(-log(1 - conf), 7).
```

The paper's unified evidence-gain definition is used throughout this analysis.
This avoids the legacy implementation branch in `DepLearn.kt`, which used a
different expression for negative dependencies.

## Candidate Space

For relation `r`, let `k_r` be the number of selected eligible rules. The
theoretical relation-local spaces are

```text
possible rule pairs              = C(k_r, 2)
possible three-rule combinations = C(k_r, 3).
```

Dataset totals sum these quantities over relations. Combinations are never
formed across different relations.

The statistics use the compact relation-local training datasets under
`data/<dataset>/datasets/dataset_<relation>.p`. A positive row represents a
correct query-candidate instance. Rules in the same row are rules that co-fire
for that candidate.

## Definitions

### Observed

A rule pair or three-rule combination is **observed** when it occurs on at least
one positive training row. In other words, its positive joint support is at
least one.

Observed combinations are counted once per unique unordered rule-ID set, not
once per occurrence.

### Support-Qualified

An observed item is **support-qualified** when its positive joint support meets
the dataset threshold:

```text
support >= 2  for KG20C and WN18RR
support >= 5  for Codex-M, FB15k-237, Codex-L, and YAGO3-10.
```

### Significant Rule Pair

For rules `i` and `j`, the pairwise gain is

```text
g_ij = e_ij - e_i - e_j.
```

A support-qualified rule pair is significant when its conservative 95% gain
interval is entirely above `+0.01` or entirely below `-0.01`.

### Significant Three-Rule Combination

For rules `i`, `j`, and `k`, define the three-rule joint gain

```text
g_ijk = e_ijk - e_i - e_j - e_k.
```

The constituent pairwise effect range is

```text
delta+ = max(g_ij, g_ik, g_jk)
delta- = min(g_ij, g_ik, g_jk).
```

A support-qualified three-rule combination is significant only when its
conservative 95% gain interval is entirely above the upper confidence bound of
all constituent pair gains plus `0.01`, or entirely below the lower confidence
bound of all constituent pair gains minus `0.01`.

Thus a combination is not counted merely because its point estimate is positive
or negative. It must provide reliable evidence beyond the strength already
observed in its constituent pairwise interactions.

## Uncertainty Estimation

Wilson intervals are calculated for the smoothed confidence of every required
singleton, rule pair, and three-rule combination. Conservative evidence and gain
intervals are then obtained through interval arithmetic.

For example, if the joint evidence interval is `[L_ij, U_ij]` and the marginal
intervals are `[L_i, U_i]` and `[L_j, U_j]`, the pair-gain interval is

```text
[L_ij - U_i - U_j, U_ij - L_i - L_j].
```

This procedure is intentionally conservative: uncertainty in every component
widens the final gain interval and makes an interaction harder to label
significant.

## Main Results

`O/Q/S` denotes Observed, Support-qualified, and Significant.

| Dataset | Rule pair O/Q/S | Three-rule O/Q/S | Pair O/P | Three-rule O/P | Pair S/Q | Three-rule S/Q |
|---|---:|---:|---:|---:|---:|---:|
| KG20C | 1,870 / 1,538 / 34 | 863 / 596 / 0 | 0.300% | 0.0008% | 2.21% | 0.000% |
| WN18RR | 43,874 / 42,397 / 4,405 | 467,552 / 457,533 / 130 | 3.845% | 0.253% | 10.39% | 0.028% |
| Codex-M | 274,242 / 154,210 / 17,085 | 4,384,527 / 2,043,217 / 573 | 7.631% | 0.750% | 11.08% | 0.028% |
| FB15k-237 | 4,666,422 / 2,827,144 / 834,635 | 191,619,558 / 89,918,377 / 188,611 | 21.730% | 5.615% | 29.52% | 0.210% |
| Codex-L | 199,515 / 105,603 / 14,087 | 2,117,485 / 977,919 / 1,063 | 5.102% | 0.337% | 13.34% | 0.109% |
| YAGO3-10 | 201,799 / 120,180 / 18,734 | 2,531,837 / 1,148,813 / 2,203 | 5.980% | 0.462% | 15.59% | 0.192% |

Here `O/P` is the fraction of the theoretical relation-local space that is
observed, and `S/Q` is the fraction of support-qualified items that remains
significant after uncertainty filtering.

### Weighted Totals

Across the six datasets:

| Order | Possible | Observed | Qualified | Significant | O/P | S/Q |
|---|---:|---:|---:|---:|---:|---:|
| Rule pair | 34,117,358 | 5,387,722 | 3,251,072 | 888,980 | 15.79% | 27.34% |
| Three-rule combination | 5,461,046,598 | 201,121,822 | 94,546,455 | 192,580 | 3.68% | 0.204% |

The weighted values are dominated by FB15k-237 because it contributes the
largest number of relation-local combinations. Dataset-level ranges are more
appropriate when describing robustness across benchmarks.

## Interpretation

The comparison supports the pairwise restriction in two separate ways.

First, three-rule combinations occupy a much smaller fraction of their
theoretical candidate space. The observed fraction is `0.0008%-5.62%` for
three-rule combinations, compared with `0.30%-21.73%` for rule pairs.

Second, sufficient support does not imply a distinct higher-order effect. Among
support-qualified rule pairs, `2.21%-29.52%` have significant signed gains.
Among support-qualified three-rule combinations, only `0%-0.21%` have gains
that significantly exceed the range of their constituent pairwise gains.

For the five datasets with nonzero significant higher-order counts, the
significant proportion is approximately 81 to 396 times smaller than the
corresponding pairwise proportion. KG20C contains no significant three-rule
combination under the conservative criterion.

The main conclusion is therefore not that three rules never co-fire. Many do,
especially in FB15k-237. Rather, reliable three-rule effects that add evidence
beyond the constituent pairwise interactions are extremely rare.

## Suggested Paper Text

The following compact text is suitable for a rebuttal or a short experimental
paragraph:

> We further evaluated whether the pairwise restriction omits reliable
> higher-order effects. For each relation, we considered up to 500 eligible
> rules and enumerated observed rule pairs and three-rule combinations. A
> combination was support-qualified when its joint support met the same minimum
> threshold used in dependency mining. We classified a three-rule combination
> as significant only when its conservative 95% gain interval lay completely
> outside the gain range of all three constituent rule pairs. Across six
> datasets, only 0-0.21% of support-qualified three-rule combinations satisfied
> this criterion, compared with 2.21-29.52% of support-qualified rule pairs.
> Moreover, observed three-rule combinations occupied only 0.0008-5.62% of the
> relation-local candidate space. These results indicate that reliably
> estimable higher-order effects beyond pairwise dependencies are extremely
> sparse.

## Reproduction

The analysis script is:

```text
src/reporting/analyze_three_rule_combinations.py
```

The six-dataset run used 36 relation-level workers:

```bash
python src/reporting/analyze_three_rule_combinations.py \
  KG20C WN18RR codex-m FB15k-237 codex-l YAGO3-10 \
  --top-k 500 \
  --jobs 36 \
  --output reports/high_order_analysis/pair_vs_three_rule_combination_stats_six_datasets.csv
```

Each worker handles one `(dataset, relation)` task. This gives 410 independent
relation tasks for the six datasets.

## Files

- `pair_vs_three_rule_combination_stats_six_datasets.csv`: final per-relation
  pair-versus-higher-order comparison, plus one `ALL` row per dataset.
- `order_comparison_summary.csv`: compact dataset-level and weighted summary.
- `three_rule_combination_stats_six_datasets.csv`: earlier six-dataset run that
  records only three-rule-combination statistics.
- `three_rule_combination_stats_kg20c.csv`: KG20C diagnostic run at `TOP_K=500`.
- `three_rule_combination_stats_wn18rr.csv`: WN18RR diagnostic run at
  `TOP_K=500`.
- `three_rule_combination_stats_codex_m.csv`: Codex-M diagnostic run at
  `TOP_K=500`.
- `three_rule_combination_stats_wn18rr_r0.csv`: WN18RR relation-0 diagnostic.
- `three_rule_combination_stats_wn18rr_top100.csv`: WN18RR sensitivity/debug run
  using `TOP_K=100`; this is not used in the main result.

## Scope and Limitations

This is a relation-local aggregation analysis, matching the model's operational
candidate space. It is not a direct extension of `DepLearn.kt` that materializes
every symbolic three-branch rule grounding. The compact aggregation datasets
record actual co-fired rule sets and positive query-candidate labels, making
them appropriate for estimating how often higher-order features could affect
the aggregation model.

The analysis also intentionally uses a conservative interval test. Therefore,
the reported significant counts should be interpreted as reliably detectable
higher-order effects under the available support, not as proof that every other
combination has exactly zero interaction.

Hetionet is not included because its relation-local application and dataset
artifacts are not present in the current workspace. The six datasets above are
the complete set for which the required training artifacts are available.
