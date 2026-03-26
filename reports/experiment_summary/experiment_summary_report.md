# Experiment Summary Report

- Datasets covered: KG20C, codex-m, WN18RR, FB15k-237, codex-l, YAGO3-10
- Canonical aggregation configs: 6

## Best Completed Aggregation Config Per Dataset

| Dataset | Best config | Stage2 MRR | H@1 | H@10 | Best application | App MRR | Delta vs best app |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| KG20C | synergy + redundancy \| auto_ratio | 0.213856 | 0.133190 | 0.356337 | rules + noisyor | 0.229772 | -0.015916 |
| codex-m | synergy + redundancy \| auto_sqrt | 0.342837 | 0.273252 | 0.479003 | rules + maxplus | 0.319487 | +0.023350 |
| WN18RR | synergy + redundancy \| auto_ratio | 0.471935 | 0.430919 | 0.542757 | rules + maxplus | 0.482515 | -0.010580 |
| FB15k-237 | synergy + redundancy \| auto_ratio | 0.349374 | 0.262826 | 0.523869 | rules + noisyor | 0.337685 | +0.011689 |
| codex-l | synergy + redundancy \| auto_sqrt | 0.333180 | 0.273122 | 0.448024 | rules + maxplus | 0.311458 | +0.021722 |
| YAGO3-10 | synergy only \| auto_sqrt | 0.573592 | 0.499600 | 0.704900 | base ranker + maxplus | 0.550335 | +0.023257 |

## Average Stage2 Minus Stage1 Gain By Config

| Config | Completed datasets | Avg delta MRR | Avg delta H@1 | Avg delta H@10 |
| --- | ---: | ---: | ---: | ---: |
| synergy + redundancy \| auto_sqrt | 6 | +0.002045 | +0.001808 | +0.002019 |
| synergy only \| auto_sqrt | 6 | +0.002071 | +0.001894 | +0.001744 |
| redundancy only \| auto_sqrt | 5 | +0.001261 | +0.001364 | +0.001014 |
| synergy + redundancy + dependency sign \| auto_sqrt | 5 | -0.001880 | -0.001069 | -0.004069 |
| synergy + redundancy + lift init \| auto_sqrt | 5 | -0.003462 | -0.002895 | -0.006134 |
| synergy + redundancy \| auto_ratio | 5 | +0.002510 | +0.002836 | +0.001883 |

## Aggregation Status Overview

- done: 32
- not_started: 3
- partial: 1

## Incomplete Or Missing Aggregation Runs

| Dataset | Config | Status | Progress | Experiment dir |
| --- | --- | --- | --- | --- |
| YAGO3-10 | redundancy only \| auto_sqrt | partial | 22/37 | data/YAGO3-10/aggregation/exp-1_LinearAggregator_1_0_1_auto_sqrt_0_1_0 |
| YAGO3-10 | synergy + redundancy + dependency sign \| auto_sqrt | not_started | 0/37 | - |
| YAGO3-10 | synergy + redundancy + lift init \| auto_sqrt | not_started | 0/37 | - |
| YAGO3-10 | synergy + redundancy \| auto_ratio | not_started | 0/37 | - |
