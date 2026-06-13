# CODEX-L Calibration Diagnostic

This diagnostic checks whether the compact high-gain formula's CODEX-L peak is a real query-subset effect or an artifact of relation-level calibration offsets.

## Compact Formula on CODEX-L

| coverage | n | calibrated gain_pt | raw gain_pt |
| ---: | ---: | ---: | ---: |
| 5% | 3056 | 149.01% | -20.73% |
| 10% | 6113 | 180.91% | -1.41% |
| 20% | 12226 | 41.83% | -7.02% |
| 30% | 18339 | 24.04% | -5.68% |
| 50% | 30564 | 5.32% | -6.91% |
| 100% | 61129 | 1.56% | -2.41% |

## Top Relations in the 10% CODEX-L Subset

| relation | n | raw stage1 MRR | raw stage2 MRR | calibrated stage2 MRR | calibration offset | official relation delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P106 | 5712 | 0.015755 | 0.015348 | 0.048236 | 0.032889 | 0.008419 |
| P1303 | 401 | 0.034295 | 0.036452 | 0.039682 | 0.003230 | 0.004228 |

## CODEX-L Other Configs: Full-test Weighted Gain

These numbers are full-test weighted MRR gains from `metric-*.json`; they are not query-subset coverage gains.

| experiment | stage1 MRR | stage2 MRR | gain_pt |
| --- | ---: | ---: | ---: |
| `tg_rd__pos_auto_ratio__ri_conf__dn_per_rule_degree__dl1_1e-5` | 0.325290 | 0.331662 | 1.96% |
| `tg_rd__pos_auto_ratio__ri_conf__dn_none__dl1_1e-5` | 0.325290 | 0.331598 | 1.94% |
| `tg_rd__pos_auto_ratio__ri_surprisal__dn_per_rule_degree__dl1_1e-5` | 0.325650 | 0.331685 | 1.85% |
| `tg_rd__pos_auto_ratio__ri_surprisal__dn_none__dl1_1e-5` | 0.325650 | 0.331457 | 1.78% |
| `tg_rd__pos_auto_ratio__ri_conf__dn_per_rule_degree__dl1_0` | 0.325290 | 0.330564 | 1.62% |
| `tg_rd__pos_auto_ratio__ri_conf__dn_none__dl1_0` | 0.325290 | 0.330430 | 1.58% |
| `tg_rd__pos_auto_ratio__ri_surprisal__dn_per_rule_degree__dl1_0` | 0.325650 | 0.330708 | 1.55% |
| `tg_rd__pos_auto_ratio__ri_surprisal__dn_none__dl1_0` | 0.325650 | 0.330594 | 1.52% |
| `tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5` | 0.329303 | 0.334178 | 1.48% |
| `tg_r3d6__pos_auto_ratio__ri_conf__dn_none__dl1_1e-5` | 0.326197 | 0.330761 | 1.40% |
| `tg_rd__pos_auto_sqrt__ri_surprisal__dn_per_rule_degree__dl1_1e-5` | 0.329495 | 0.334087 | 1.39% |
| `tg_rd__pos_auto_sqrt__ri_conf__dn_none__dl1_1e-5` | 0.329304 | 0.333619 | 1.31% |
| `tg_r3d6__pos_auto_ratio__ri_surprisal__dn_none__dl1_1e-5` | 0.325949 | 0.330196 | 1.30% |
| `tg_r2d3__pos_auto_ratio__ri_conf__dn_per_rule_degree__dl1_1e-5` | 0.326368 | 0.330300 | 1.20% |
| `tg_r2d3__pos_auto_ratio__ri_surprisal__dn_per_rule_degree__dl1_1e-5` | 0.325890 | 0.329756 | 1.19% |
| `tg_rd__pos_auto_sqrt__ri_surprisal__dn_none__dl1_1e-5` | 0.329496 | 0.333386 | 1.18% |
