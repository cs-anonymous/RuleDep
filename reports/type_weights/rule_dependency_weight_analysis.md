# Rule / Dependency Weight Analysis

This section follows the two partial best configs, focuses on `dependency_final`, and examines how rule and dependency parameters change during training, along with the final distribution of retained dependency weights.

Related files:

- `best_config_weight_summary.csv`
- `dependency_sign_by_type.csv`

<p align="center"><img src="plot_weight_near_zero_ratio.png" alt="Weight Near-zero Ratio" width="60%"></p>

<p align="center"><em>Figure 1: near-zero ratio of learned rule and dependency weights.</em></p>

<p align="center"><img src="plot_weight_max_abs.png" alt="Weight Max Abs" width="60%"></p>

<p align="center"><em>Figure 2: maximum absolute value of learned rule and dependency weights.</em></p>

<p align="center"><img src="plot_dependency_sign_by_type.png" alt="Dependency Sign by Type" width="60%"></p>

<p align="center"><em>Figure 3: positive-weight ratio of synergy and redundancy dependencies in dependency_final.</em></p>

## Definitions (gold-effective metrics)

Four statistics computed on `dependency_final` with threshold `delta=0.01`, all calculated at gold `(q,c)` granularity:

- **Gold avg effective Rule count**: For each gold `(q,c)`, count rules with `rule_weight > delta`, then average across all gold `(q,c)`.
- **Gold avg effective Dep count**: For each gold `(q,c)`, count dependencies induced by activated rules with `dep_weight > delta`, then average.
- **Gold top3 Rule Proportion**: For each gold `(q,c)`, `sum(top3(rule_weight > delta)) / sum(rule_weight > delta)`, then averaged.
- **Gold top3 Dep Proportion**: For each gold `(q,c)`, `sum(top3(dep_weight > delta)) / sum(dep_weight > delta)`, then averaged.

## Dataset Summary

| Dataset | Config | Rule near-zero | Dep final near-zero | Rule max abs | Dep final max abs | Gold avg eff Rule per (q,c) | Gold avg eff Dep per (q,c) | Gold top1 Rule share | Gold top1 Dep share | Gold top3 Rule share | Gold top3 Dep share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KG20C | tg_r2d3__pos_auto_ratio__ri_conf__dn_per_rule_degree__dl1_1e-5 | 13.14426 | 94.26110 | 1.57562 | 0.80416 | 5.14635 | 0.35203 | 53.37717% | 77.42949% | 83.96654% | 97.58938% |
| codex-m | tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5 | 9.82656 | 91.04703 | 3.03739 | 1.07963 | 27.77172 | 1.98487 | 32.76844% | 45.64555% | 60.17912% | 70.64828% |
| WN18RR | structural_rd | 10.75427 | 57.69792 | 4.24963 | 0.30246 | 13.32379 | 0.14037 | 52.25905% | 73.36409% | 75.59521% | 91.52040% |
| FB15k-237 | tg_r2d3__pos_auto_ratio__ri_conf__dn_none__dl1_1e-5 | 15.94725 | 84.37088 | 3.64719 | 0.50098 | 72.31812 | 3.07989 | 14.53210% | 39.53305% | 30.66500% | 64.83535% |
| codex-l | tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5 | 27.38087 | 92.19635 | 5.51841 | 12.53516 | 14.87504 | 4.03513 | 34.99680% | 55.74672% | 65.68267% | 78.07510% |
| YAGO3-10 | structural_dep_scale | 44.85790 | 14.59044 | 4.20227 | 3.25249 | 8.88630 | 2.17171 | 43.15764% | 75.53709% | 82.45460% | 93.34939% |
| hetionet | dep_scale_surprisal_init | 26.33060 | 4.92766 | 3.02947 | 1.28416 | 94.21018 | 1.06207 | 12.34501% | 64.48913% | 26.00430% | 85.85045% |
| Average | - | 21.17739 | 62.72734 | 3.60857 | 2.82272 | 33.79021 | 1.83229 | 34.77660% | 61.67787% | 60.64963% | 83.12405% |

## Dependency Sign vs Type

In `dependency_final`, synergy dependencies have an average positive weight ratio of `36.33%`, while redundancy dependencies average `32.12%`.

## Global View

- Average absolute weight change: rules = `0.25797`, dependencies (final) = `0.05473`.
- Near-zero proportion: rules = `21.18%`, dependencies (final) = `62.73%`.
- Under the gold criterion, average effective Rule count = `33.79`, average effective Dep count = `1.83`.
- Average number of gold `(q,c)` instances used for statistics: `48,800.86`.
- Gold top1 proportion: rules = `34.78%`, dependencies = `61.68%`.
- Gold top3 proportion: rules = `60.65%`, dependencies = `83.12%`.

## Interpretation

The model actively sparsifies most dependency edges. After training, `dependency_final` retains only a few edges with significant weights -- sparse but strong correction terms, not weak biases evenly distributed across all rule pairs.

From the sign distribution, synergy dependencies are more likely to receive positive weights, while redundancy dependencies tend to be more conservative. This aligns with the semantic direction of each dependency type, though the correspondence is not absolute.
