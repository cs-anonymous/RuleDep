# Best-combination Variant Details

This section follows the current reporting scope: all `best_combination*` entries are excluded. The best remaining configuration is selected for each dataset, and two relation-level ensemble results are reported.

Related files:

- `best_combination_variant_details.csv`
- `plot_best_combination_variant_delta.png`

| Dataset | Best remaining config | Best MRR | Ensemble-valid | Ensemble-safe | Ensemble-test | Delta(valid-best) | Delta(safe-best) | Delta(test-best) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KG20C | tg_r2d3__pos_auto_ratio__ri_conf__dn_per_rule_degree__dl1_1e-5 | 0.233952 | 0.235725 | 0.235325 | 0.239769 | 0.001773 | 0.001373 | 0.005817 |
| codex-m | tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5 | 0.344803 | 0.346569 | 0.344357 | 0.349667 | 0.001766 | -0.000446 | 0.004864 |
| WN18RR | structural_rd | 0.502189 | 0.500511 | 0.499341 | 0.503990 | -0.001678 | -0.002848 | 0.001801 |
| FB15k-237 | tg_r2d3__pos_auto_ratio__ri_conf__dn_none__dl1_1e-5 | 0.355348 | 0.357476 | 0.355354 | 0.362664 | 0.002128 | 0.000006 | 0.007316 |
| codex-l | tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5 | 0.334178 | 0.334817 | 0.334735 | 0.336422 | 0.000639 | 0.000557 | 0.002244 |
| YAGO3-10 | structural_dep_scale | 0.577961 | 0.576236 | 0.575860 | 0.579491 | -0.001725 | -0.002101 | 0.001530 |
| hetionet | dep_scale_surprisal_init | 0.378467 | 0.377523 | 0.375250 | 0.377751 | -0.000944 | -0.003217 | -0.000716 |

<p align="center"><img src="plot_best_combination_variant_delta.png" alt="ensemble-valid minus best remaining" width="60%"></p>

<p align="center"><em>Figure: relation-level ensemble-valid gain relative to the best single remaining config.</em></p>
