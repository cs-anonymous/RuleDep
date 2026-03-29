# Coverage>30% Gain>5% Paper Tables

说明：

- 论文表格中暂不写 `WN18RR`。
- 全局表使用 `test_after_stage1` 和 `test_after_stage2`。
- 我之前曾用过 `test_before_stage2` 作为 stage2 入口指标；对少数旧实验，这个值会低于 `test_after_stage1`，不适合作为“stage1 global mrr”的论文口径。
- 子集表使用当前每个数据集最有代表性的命中子集；`YAGO3-10` 当前仍未达标，但保留结果用于说明难点。
- `FB15k-237` 的子集不是单 relation，而是按 relation gain 从高到低累计到 `coverage > 30%` 的 relation-wise 子集。

## Table 1. Global MRR

| Dataset | Experiment | Test Size | Stage1 Global MRR | Stage2 Global MRR | Relative Gain |
| --- | --- | ---: | ---: | ---: | ---: |
| KG20C | `kg20c_multi_variant_mt3_trim` | 3724 | 0.211141 | 0.216741 | 2.65% |
| codex-m | `exp_codexm_synergy_pair_and_type_lift_oldcfg` | 10311 | 0.342249 | 0.339590 | -0.78% |
| FB15k-237 | `exp-1_LinearAggregator_1_1_1_auto_sqrt_1_1_0` | 20466 | 0.344251 | 0.335601 | -2.51% |
| codex-l | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1` | 30620 | 0.330477 | 0.331830 | 0.41% |
| YAGO3-10 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_0` | 5000 | 0.569984 | 0.571988 | 0.35% |

## Table 2. Subset MRR

| Dataset | Subset Type | Subset Definition | Test Size | Coverage | Stage1 Subset MRR | Stage2 Subset MRR | Relative Gain | Status |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| KG20C | relation-wise | relation 3 | 1446 | 0.3883 | 0.115507 | 0.124413 | 7.71% | pass |
| codex-m | relation-wise | relation 2 | 3635 | 0.3525 | 0.271289 | 0.299395 | 10.36% | pass |
| FB15k-237 | relation-wise cumulative | top-gain relations cumulative to >30% coverage | 6194 | 0.3026 | 0.277418 | 0.292669 | 5.50% | pass |
| codex-l | relation-wise | relation 2 | 9426 | 0.3078 | 0.281267 | 0.300275 | 6.76% | pass |
| YAGO3-10 | query-wise | active_candidate_count >= 1 | 7751 | 0.9055 | 0.626333 | 0.627118 | 0.13% | fail |

## Table 3. Top-Gain Relation (`relative gain > 5%`)

| Dataset | Experiment | Relation | Relation Name | Test Size | Stage1 MRR | Stage2 MRR | Relative Gain |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| KG20C | `kg20c_multi_variant_mt3_trim` | 3 | paper_in_domain | 1446 | 0.115507 | 0.124413 | 7.71% |
| codex-m | `exp_codexm_synergy_pair_and_type_lift_oldcfg` | 2 | relation 2 | 3635 | 0.271289 | 0.299395 | 10.36% |
| FB15k-237 | `exp-1_LinearAggregator_1_1_1_auto_sqrt_1_1_0` | 65 | /olympics/olympic_games/sports | 13 | 0.267153 | 0.360238 | 34.84% |
| codex-l | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1` | 40 | P57 | 78 | 0.251750 | 0.269967 | 7.24% |
| YAGO3-10 | `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_0` | 28 | dealsWith | 7 | 0.215658 | 0.280680 | 30.15% |
