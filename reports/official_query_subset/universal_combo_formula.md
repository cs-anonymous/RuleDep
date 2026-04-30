# Universal Combo Selector Proposal

The current paper-facing selector is the compact formula selected in [`high_gain_formula_report.md`](high_gain_formula_report.md). It is optimized for 10% and 20% official-scaled `gain_pt`; monotonicity is not enforced.

```text
compact_score = max(
  P_d(candidate_dep_coverage),
  P_d(synergy_weight_mean),
  P_d(synergy_weight_top3_mean)
)
```
