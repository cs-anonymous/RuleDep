# Relation Subset Top-N Tables

选择规则：优先在每个数据集里寻找一个多 relation 子集，满足“子集中每个 relation 的相对增益都不低于 `5% / 2% / 1%` 之一，且子集覆盖率至少 `5%`”；如果不存在这样的多 relation 子集，则回退到“所有正增益 relation”的最大覆盖子集。

汇总表：

| Dataset | Experiment | Rule | N | Coverage | Subset Stage1 MRR | Subset Stage2 MRR | Subset Relative Gain |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| KG20C | exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1 | fallback: all positive-gain relations | 4 | 0.8711 | 0.190264 | 0.191376 | 0.58% |
| codex-m | exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_0 | relative_gain>=2% and coverage>=5% | 5 | 0.0606 | 0.253480 | 0.262889 | 3.71% |
| WN18RR | exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0 | relative_gain>=1% and coverage>=5% | 2 | 0.1197 | 0.282805 | 0.294030 | 3.97% |
| FB15k-237 | exp-1_LinearAggregator_1_1_1_auto_sqrt_1_1_0 | relative_gain>=5% and coverage>=5% | 21 | 0.1477 | 0.239050 | 0.262880 | 9.97% |
| codex-l | exp-1_LinearAggregator_1_1_1_auto_sqrt_1_1_0 | relative_gain>=2% and coverage>=5% | 7 | 0.3291 | 0.293696 | 0.301178 | 2.55% |
| YAGO3-10 | exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_0 | relative_gain>=2% and coverage>=5% | 6 | 0.1076 | 0.358812 | 0.370596 | 3.28% |

## KG20C

- Experiment: `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_1`
- Selection rule: `fallback: all positive-gain relations`
- Top N: `4`
- Subset coverage: `0.8711` (`3244` test examples)
- Subset MRR: `0.190264 -> 0.191376`
- Subset relative gain: `0.58%`
- Note: 该数据集没有找到覆盖率至少 5% 且每个 relation 都达到 1% 以上相对增益的多 relation 子集，所以这里使用全正增益回退子集。

| Relation | Name | Num Test | Num Triples | Avg Tails/SP | Dep/Rule | Stage1 MRR | Stage2 MRR | Relative Gain |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | author_write_paper | 830 | 12465 | 1.535 | 0.020 | 0.225721 | 0.227637 | 0.85% |
| 3 | paper_in_domain | 1446 | 17776 | 3.612 | 0.067 | 0.115507 | 0.116237 | 0.63% |
| 4 | paper_in_venue | 369 | 4288 | 1.000 | 0.058 | 0.386609 | 0.388347 | 0.45% |
| 2 | paper_cite_paper | 599 | 7382 | 2.628 | 0.063 | 0.200642 | 0.201178 | 0.27% |

## codex-m

- Experiment: `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_0`
- Selection rule: `relative_gain>=2% and coverage>=5%`
- Top N: `5`
- Subset coverage: `0.0606` (`625` test examples)
- Subset MRR: `0.253480 -> 0.262889`
- Subset relative gain: `3.71%`

| Relation | Name | Num Test | Num Triples | Avg Tails/SP | Dep/Rule | Stage1 MRR | Stage2 MRR | Relative Gain |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | P840 | 85 | 1348 | 1.478 | 0.203 | 0.148447 | 0.164430 | 10.77% |
| 35 | P37 | 13 | 373 | 1.295 | 3.582 | 0.456599 | 0.482835 | 5.75% |
| 12 | P159 | 8 | 156 | 1.068 | 0.085 | 0.286957 | 0.298037 | 3.86% |
| 22 | P20 | 260 | 4872 | 1.004 | 0.513 | 0.317644 | 0.328686 | 3.48% |
| 15 | P108 | 259 | 4301 | 1.530 | 0.104 | 0.212310 | 0.217025 | 2.22% |

## WN18RR

- Experiment: `exp-1_LinearAggregator_1_0_1_auto_ratio_1_1_0`
- Selection rule: `relative_gain>=1% and coverage>=5%`
- Top N: `2`
- Subset coverage: `0.1197` (`375` test examples)
- Subset MRR: `0.282805 -> 0.294030`
- Subset relative gain: `3.97%`

| Relation | Name | Num Test | Num Triples | Avg Tails/SP | Dep/Rule | Stage1 MRR | Stage2 MRR | Relative Gain |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | _member_meronym | 253 | 7402 | 2.392 | 0.027 | 0.261930 | 0.276987 | 5.75% |
| 2 | _instance_hypernym | 122 | 2921 | 1.185 | 0.083 | 0.326096 | 0.329372 | 1.00% |

## FB15k-237

- Experiment: `exp-1_LinearAggregator_1_1_1_auto_sqrt_1_1_0`
- Selection rule: `relative_gain>=5% and coverage>=5%`
- Top N: `21`
- Subset coverage: `0.1477` (`3023` test examples)
- Subset MRR: `0.239050 -> 0.262880`
- Subset relative gain: `9.97%`

| Relation | Name | Num Test | Num Triples | Avg Tails/SP | Dep/Rule | Stage1 MRR | Stage2 MRR | Relative Gain |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 65 | /olympics/olympic_games/sports | 13 | 664 | 15.442 | 2.163 | 0.267153 | 0.360238 | 34.84% |
| 75 | /music/performance_role/track_performances./music/track_contribution/role | 53 | 3795 | 31.625 | 2.050 | 0.112309 | 0.141225 | 25.75% |
| 51 | /tv/tv_program/languages | 24 | 254 | 1.081 | 2.957 | 0.693050 | 0.822650 | 18.70% |
| 235 | /music/instrument/family | 6 | 100 | 1.235 | 1.392 | 0.189230 | 0.222878 | 17.78% |
| 9 | /music/performance_role/regular_performances./music/group_membership/role | 40 | 2655 | 22.311 | 2.113 | 0.098901 | 0.110904 | 12.14% |
| 8 | /award/award_nominee/award_nominations./award/award_nomination/award_nominee | 214 | 15989 | 6.850 | 3.008 | 0.270967 | 0.303527 | 12.02% |
| 7 | /award/award_category/nominees./award/award_nomination/nominated_for | 858 | 9465 | 39.603 | 2.172 | 0.205583 | 0.229881 | 11.82% |
| 30 | /award/award_winning_work/awards_won./award/award_honor/award | 118 | 3310 | 3.280 | 1.974 | 0.184768 | 0.203488 | 10.13% |
| 90 | /film/film/distributors./film/film_film_distributor_relationship/film_distribution_medium | 38 | 285 | 1.447 | 2.313 | 0.596955 | 0.655891 | 9.87% |
| 188 | /education/field_of_study/students_majoring./education/education/major_field_of_study | 11 | 343 | 3.811 | 1.305 | 0.206222 | 0.226207 | 9.69% |
| 16 | /organization/organization/headquarters./location/mailing_address/country | 7 | 182 | 1.011 | 2.738 | 0.442608 | 0.485168 | 9.62% |
| 19 | /award/award_nominee/award_nominations./award/award_nomination/award | 1067 | 12157 | 3.644 | 1.774 | 0.203285 | 0.222584 | 9.49% |
| 38 | /award/award_nominee/award_nominations./award/award_nomination/nominated_for | 132 | 6277 | 2.695 | 1.425 | 0.158296 | 0.172081 | 8.71% |
| 222 | /location/location/partially_contains | 13 | 133 | 1.400 | 1.161 | 0.350410 | 0.380218 | 8.51% |
| 203 | /film/film/release_date_s./film/film_regional_release_date/film_regional_debut_venue | 20 | 246 | 1.367 | 1.026 | 0.258393 | 0.278078 | 7.62% |
| 221 | /location/country/official_language | 16 | 225 | 1.230 | 4.000 | 0.400719 | 0.430516 | 7.44% |
| 151 | /people/ethnicity/geographic_distribution | 17 | 122 | 3.050 | 1.846 | 0.391249 | 0.419857 | 7.31% |
| 93 | /government/politician/government_positions_held./government/government_position_held/legislative_sessions | 30 | 215 | 7.679 | 3.108 | 0.744861 | 0.793929 | 6.59% |
| 78 | /base/locations/continents/countries_within | 5 | 125 | 31.250 | 4.000 | 0.611864 | 0.646887 | 5.72% |
| 27 | /award/award_ceremony/awards_presented./award/award_honor/award_winner | 317 | 2727 | 19.340 | 2.854 | 0.364594 | 0.385434 | 5.72% |
| 12 | /award/award_winning_work/awards_won./award/award_honor/award_winner | 24 | 2992 | 2.643 | 2.769 | 0.237465 | 0.250435 | 5.46% |

## codex-l

- Experiment: `exp-1_LinearAggregator_1_1_1_auto_sqrt_1_1_0`
- Selection rule: `relative_gain>=2% and coverage>=5%`
- Top N: `7`
- Subset coverage: `0.3291` (`10076` test examples)
- Subset MRR: `0.293696 -> 0.301178`
- Subset relative gain: `2.55%`

| Relation | Name | Num Test | Num Triples | Avg Tails/SP | Dep/Rule | Stage1 MRR | Stage2 MRR | Relative Gain |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | P57 | 78 | 1533 | 1.059 | 0.024 | 0.251750 | 0.269967 | 7.24% |
| 43 | P749 | 18 | 330 | 1.071 | 0.012 | 0.246879 | 0.260936 | 5.69% |
| 10 | P140 | 346 | 6006 | 1.048 | 0.149 | 0.297066 | 0.306014 | 3.01% |
| 30 | P36 | 46 | 621 | 1.244 | 1.886 | 0.435222 | 0.447836 | 2.90% |
| 2 | P106 | 9426 | 169091 | 2.989 | 0.593 | 0.295749 | 0.303120 | 2.49% |
| 21 | P737 | 90 | 1634 | 2.581 | 0.021 | 0.121073 | 0.123788 | 2.24% |
| 22 | P740 | 72 | 1140 | 1.004 | 0.225 | 0.191238 | 0.195480 | 2.22% |

## YAGO3-10

- Experiment: `exp-1_LinearAggregator_1_0_1_auto_sqrt_1_1_0`
- Selection rule: `relative_gain>=2% and coverage>=5%`
- Top N: `6`
- Subset coverage: `0.1076` (`538` test examples)
- Subset MRR: `0.358812 -> 0.370596`
- Subset relative gain: `3.28%`

| Relation | Name | Num Test | Num Triples | Avg Tails/SP | Dep/Rule | Stage1 MRR | Stage2 MRR | Relative Gain |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 28 | dealsWith | 7 | 1302 | 5.918 | 2.491 | 0.215658 | 0.280680 | 30.15% |
| 15 | influences | 47 | 10710 | 4.875 | 0.411 | 0.135791 | 0.151242 | 11.38% |
| 8 | happenedIn | 21 | 5056 | 5.805 | 1.021 | 0.393594 | 0.424665 | 7.89% |
| 22 | isPoliticianOf | 10 | 2163 | 1.188 | 2.078 | 0.748811 | 0.776093 | 3.64% |
| 5 | graduatedFrom | 42 | 7348 | 1.558 | 0.172 | 0.242661 | 0.250365 | 3.17% |
| 0 | isLocatedIn | 411 | 88672 | 2.234 | 0.763 | 0.387357 | 0.396869 | 2.46% |
