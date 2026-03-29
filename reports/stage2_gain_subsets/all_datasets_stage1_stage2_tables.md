# All Datasets Stage1 vs Stage2 Tables

说明：

- 本文档统一使用 `stage1 = test_after_stage1`，`stage2 = test_after_stage2`。
- `Subset MRR` 统一采用：对每个数据集，在当前已完成实验中选出“relation-wise gain 从高到低累计到 `coverage > 30%`”后，收益最高的那个代表实验。
- 因此，第二张表现在是统一规则，不再混用单 relation 和 query-wise。

## Table 1. Global MRR

说明：

- 这里的 `Global MRR` 指覆盖整个测试集的最好结果。
- 这里的 `best experiment` 按全测试集上的 `relative gain` 选取。
- 比较口径全部是 `test_after_stage1 -> test_after_stage2`。

| Dataset | Experiment | Test Size | Stage1 MRR | Stage2 MRR | Relative Gain |
| --- | --- | ---: | ---: | ---: | ---: |
| FB15k-237 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_0` | 20466 | 0.344251 | 0.347334 | +0.90% |
| KG20C | `kg20c_multi_variant_mt3_trim` | 3724 | 0.211141 | 0.216741 | +2.65% |
| WN18RR | `exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0` | 3134 | 0.470549 | 0.471935 | +0.29% |
| YAGO3-10 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_0_0` | 5000 | 0.569984 | 0.573592 | +0.63% |
| codex-l | `exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0` | 30620 | 0.326902 | 0.329876 | +0.91% |
| codex-m | `exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0` | 10311 | 0.339785 | 0.342314 | +0.74% |

## Table 2. Subset MRR (`coverage > 30%`)

| Dataset | Experiment | Subset Type | Subset Definition | Test Size | Coverage | Stage1 MRR | Stage2 MRR | Relative Gain | Status |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| KG20C | `kg20c_multi_variant_mt3_trim` | relation-wise cumulative | relation 3 | 1446 | 0.3883 | 0.115507 | 0.124413 | 7.71% | pass |
| codex-m | `exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0` | relation-wise cumulative | top-gain relations cumulative to >30% coverage | 4398 | 0.4265 | 0.317710 | 0.323409 | 1.79% | fail |
| WN18RR | `exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0` | relation-wise cumulative | top-gain relations cumulative to >30% coverage | 1943 | 0.6200 | 0.205598 | 0.207880 | 1.11% | fail |
| FB15k-237 | `exp-1_LinearAggregator_1_1_1_auto_sqrt_1_1_0` | relation-wise cumulative | top-gain relations cumulative to >30% coverage | 6327 | 0.3091 | 0.320883 | 0.336942 | 5.00% | pass |
| codex-l | `exp-1_LinearAggregator_1_1_1_auto_sqrt_1_1_0` | relation-wise cumulative | top-gain relations cumulative to >30% coverage | 10251 | 0.3348 | 0.301680 | 0.309242 | 2.51% | fail |
| YAGO3-10 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_0` | relation-wise cumulative | top-gain relations cumulative to >30% coverage | 2211 | 0.4422 | 0.645223 | 0.653021 | 1.21% | fail |

## Table 3. Relation MRR (`relative gain > 5%`)

说明：

- 每一行表示一个 `dataset + relation`
- 这里保留该 relation 当前已知的最好结果
- 比较口径统一是 `test_after_stage1 -> test_after_stage2`

| Dataset | Relation | Best Experiment | Num Test | Stage1 MRR | Stage2 MRR | Relative Gain |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| FB15k-237 | 65 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_0` | 13 | 0.267153 | 0.434807 | +62.76% |
| FB15k-237 | 3 | `exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0` | 41 | 0.207962 | 0.324561 | +56.07% |
| FB15k-237 | 223 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_0` | 3 | 0.666667 | 1.000000 | +50.00% |
| FB15k-237 | 75 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1` | 53 | 0.112309 | 0.167448 | +49.10% |
| FB15k-237 | 152 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_0_0` | 17 | 0.272656 | 0.384779 | +41.12% |
| FB15k-237 | 9 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1` | 40 | 0.098901 | 0.138230 | +39.76% |
| FB15k-237 | 188 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_0` | 11 | 0.206222 | 0.257912 | +25.07% |
| FB15k-237 | 235 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1` | 6 | 0.189230 | 0.229250 | +21.15% |
| FB15k-237 | 51 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_0` | 24 | 0.693050 | 0.830585 | +19.84% |
| FB15k-237 | 45 | `exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0` | 32 | 0.056777 | 0.067775 | +19.37% |
| FB15k-237 | 72 | `exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0` | 31 | 0.371275 | 0.438274 | +18.05% |
| FB15k-237 | 151 | `exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0` | 17 | 0.429333 | 0.502275 | +16.99% |
| FB15k-237 | 58 | `iter6_fb237_relallow_actonly_joint` | 55 | 0.389380 | 0.452414 | +16.19% |
| FB15k-237 | 62 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_0_1_0` | 25 | 0.523591 | 0.605988 | +15.74% |
| FB15k-237 | 30 | `iter7_fb237_pairrerank_top50` | 118 | 0.184704 | 0.213736 | +15.72% |
| FB15k-237 | 8 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_0_1_0` | 214 | 0.270967 | 0.312330 | +15.27% |
| FB15k-237 | 49 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_0` | 3 | 0.644444 | 0.727778 | +12.93% |
| FB15k-237 | 12 | `iter6_fb237_relallow_actonly_joint` | 24 | 0.237465 | 0.266873 | +12.38% |
| FB15k-237 | 7 | `exp-1_LinearAggregator_1_1_1_auto_sqrt_1_1_0` | 858 | 0.205583 | 0.229881 | +11.82% |
| FB15k-237 | 4 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1` | 12 | 0.361571 | 0.403238 | +11.52% |
| FB15k-237 | 105 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_0_1_0` | 13 | 0.337896 | 0.376357 | +11.38% |
| FB15k-237 | 38 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_0_1_0` | 132 | 0.158296 | 0.174920 | +10.50% |
| FB15k-237 | 19 | `iter9p1_fb237_sign_log` | 1067 | 0.203285 | 0.224528 | +10.45% |
| FB15k-237 | 87 | `iter6_fb237_relallow_actonly` | 46 | 0.571099 | 0.628676 | +10.08% |
| FB15k-237 | 203 | `iter7_fb237_pairrerank_top50` | 20 | 0.258393 | 0.284246 | +10.01% |
| FB15k-237 | 90 | `exp-1_LinearAggregator_1_1_1_auto_sqrt_1_1_0` | 38 | 0.596955 | 0.655891 | +9.87% |
| FB15k-237 | 16 | `exp-1_LinearAggregator_1_1_1_auto_sqrt_1_1_0` | 7 | 0.442608 | 0.485168 | +9.62% |
| FB15k-237 | 222 | `exp-1_LinearAggregator_1_1_1_auto_sqrt_1_1_0` | 13 | 0.350410 | 0.380218 | +8.51% |
| FB15k-237 | 1 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_0_1_0` | 91 | 0.080548 | 0.087212 | +8.27% |
| FB15k-237 | 5 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_0_0` | 10 | 0.410555 | 0.443937 | +8.13% |
| FB15k-237 | 84 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_0` | 60 | 0.368989 | 0.397692 | +7.78% |
| FB15k-237 | 174 | `exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0` | 11 | 0.211688 | 0.228090 | +7.75% |
| FB15k-237 | 27 | `iter9p1_fb237_sign_sqrt` | 317 | 0.364611 | 0.392011 | +7.51% |
| FB15k-237 | 221 | `exp-1_LinearAggregator_1_1_1_auto_sqrt_1_1_0` | 16 | 0.400719 | 0.430516 | +7.44% |
| FB15k-237 | 44 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_0_1_0` | 85 | 0.243471 | 0.261108 | +7.24% |
| FB15k-237 | 103 | `exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0` | 121 | 0.399022 | 0.427801 | +7.21% |
| FB15k-237 | 93 | `exp-1_LinearAggregator_1_1_1_auto_sqrt_1_1_0` | 30 | 0.744861 | 0.793929 | +6.59% |
| FB15k-237 | 14 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1` | 314 | 0.514245 | 0.547780 | +6.52% |
| FB15k-237 | 162 | `exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0` | 9 | 0.255014 | 0.270983 | +6.26% |
| FB15k-237 | 116 | `exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0` | 92 | 0.192213 | 0.203700 | +5.98% |
| FB15k-237 | 120 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1` | 84 | 0.600394 | 0.635471 | +5.84% |
| FB15k-237 | 78 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1` | 5 | 0.611864 | 0.646887 | +5.72% |
| FB15k-237 | 141 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_0_0` | 10 | 0.316562 | 0.332745 | +5.11% |
| FB15k-237 | 55 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1` | 11 | 0.448677 | 0.471493 | +5.09% |
| KG20C | 3 | `kg20c_multi_variant_mt3_trim` | 1446 | 0.115507 | 0.124413 | +7.71% |
| WN18RR | 4 | `exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0` | 253 | 0.261930 | 0.276987 | +5.75% |
| YAGO3-10 | 28 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1` | 7 | 0.215658 | 0.291738 | +35.28% |
| YAGO3-10 | 20 | `exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0` | 5 | 0.117662 | 0.132627 | +12.72% |
| YAGO3-10 | 15 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_0` | 47 | 0.135791 | 0.151242 | +11.38% |
| YAGO3-10 | 29 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1` | 11 | 0.204329 | 0.226921 | +11.06% |
| YAGO3-10 | 25 | `exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0` | 15 | 0.304522 | 0.333235 | +9.43% |
| YAGO3-10 | 8 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_0_0` | 21 | 0.393594 | 0.427125 | +8.52% |
| YAGO3-10 | 18 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1` | 22 | 0.268182 | 0.285447 | +6.44% |
| YAGO3-10 | 12 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_0_1_0` | 19 | 0.305801 | 0.324176 | +6.01% |
| YAGO3-10 | 5 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1` | 42 | 0.236367 | 0.250080 | +5.80% |
| YAGO3-10 | 3 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1` | 49 | 0.181833 | 0.191731 | +5.44% |
| codex-l | 40 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1` | 78 | 0.251750 | 0.269967 | +7.24% |
| codex-l | 30 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1` | 46 | 0.435222 | 0.466485 | +7.18% |
| codex-l | 43 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1` | 18 | 0.246879 | 0.260936 | +5.69% |
| codex-m | 35 | `iter2_main_codexm_joint_strengthscaled_top512` | 13 | 0.456599 | 0.541794 | +18.66% |
| codex-m | 6 | `codexm_multi_variant_mt3_trim` | 85 | 0.148447 | 0.166391 | +12.09% |
| codex-m | 12 | `iter2_main_codexm_joint_typeonly_autosqrt` | 8 | 0.324225 | 0.359932 | +11.01% |
| codex-m | 19 | `iter2_main_codexm_joint_paironly_autosqrt` | 18 | 0.656008 | 0.719078 | +9.61% |
| codex-m | 39 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1` | 12 | 0.433529 | 0.466416 | +7.59% |
| codex-m | 18 | `codexm_multi_variant_mt3_trim` | 115 | 0.488419 | 0.546054 | +11.80% |
| codex-m | 23 | `exp-1_LinearAggregator_1_0_1_auto_ratio_1_0_0` | 71 | 0.117437 | 0.124201 | +5.76% |
| codex-m | 22 | `iter2_main_codexm_joint_additive_autosqrt` | 260 | 0.317633 | 0.335792 | +5.72% |
| codex-m | 9 | `exp-1_LinearAggregator_1_0_1_auto_ratio_1_0_0` | 79 | 0.617796 | 0.652052 | +5.54% |
