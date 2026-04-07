# Experiment Summary Report

- Datasets covered: KG20C, codex-m, WN18RR, FB15k-237, codex-l, YAGO3-10
- Canonical aggregation configs: 6

## Best Completed Aggregation Config Per Dataset

| Dataset | Best config | Stage2 MRR | H@1 | H@10 | Best application | App MRR | Delta vs best app |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| KG20C | - | - | - | - | base ranker + noisyor | 0.234384 | - |
| codex-m | - | - | - | - | rules + maxplus | 0.319487 | - |
| WN18RR | - | - | - | - | rules + maxplus | 0.496968 | - |
| FB15k-237 | - | - | - | - | rules + noisyor | 0.337685 | - |
| codex-l | - | - | - | - | rules + maxplus | 0.311458 | - |
| YAGO3-10 | - | - | - | - | rules + maxplus | 0.554384 | - |

## Average Stage2 Minus Stage1 Gain By Config

| Config | Completed datasets | Avg delta MRR | Avg delta H@1 | Avg delta H@10 |
| --- | ---: | ---: | ---: | ---: |
| synergy + redundancy \| auto_sqrt | 0 | - | - | - |
| synergy only \| auto_sqrt | 0 | - | - | - |
| redundancy only \| auto_sqrt | 0 | - | - | - |
| synergy + redundancy + dependency sign \| auto_sqrt | 0 | - | - | - |
| synergy + redundancy + lift init \| auto_sqrt | 0 | - | - | - |
| synergy + redundancy \| auto_ratio | 0 | - | - | - |

## Aggregation Status Overview

- not_started: 36

## Incomplete Or Missing Aggregation Runs

| Dataset | Config | Status | Progress | Experiment dir |
| --- | --- | --- | --- | --- |
| KG20C | synergy + redundancy \| auto_sqrt | not_started | 0/5 | - |
| KG20C | synergy only \| auto_sqrt | not_started | 0/5 | - |
| KG20C | redundancy only \| auto_sqrt | not_started | 0/5 | - |
| KG20C | synergy + redundancy + dependency sign \| auto_sqrt | not_started | 0/5 | - |
| KG20C | synergy + redundancy + lift init \| auto_sqrt | not_started | 0/5 | - |
| KG20C | synergy + redundancy \| auto_ratio | not_started | 0/5 | - |
| codex-m | synergy + redundancy \| auto_sqrt | not_started | 0/51 | - |
| codex-m | synergy only \| auto_sqrt | not_started | 0/51 | - |
| codex-m | redundancy only \| auto_sqrt | not_started | 0/51 | - |
| codex-m | synergy + redundancy + dependency sign \| auto_sqrt | not_started | 0/51 | - |
| codex-m | synergy + redundancy + lift init \| auto_sqrt | not_started | 0/51 | - |
| codex-m | synergy + redundancy \| auto_ratio | not_started | 0/51 | - |
| WN18RR | synergy + redundancy \| auto_sqrt | not_started | 0/11 | - |
| WN18RR | synergy only \| auto_sqrt | not_started | 0/11 | - |
| WN18RR | redundancy only \| auto_sqrt | not_started | 0/11 | - |
| WN18RR | synergy + redundancy + dependency sign \| auto_sqrt | not_started | 0/11 | - |
| WN18RR | synergy + redundancy + lift init \| auto_sqrt | not_started | 0/11 | - |
| WN18RR | synergy + redundancy \| auto_ratio | not_started | 0/11 | - |
| FB15k-237 | synergy + redundancy \| auto_sqrt | not_started | 0/237 | - |
| FB15k-237 | synergy only \| auto_sqrt | not_started | 0/237 | - |
| FB15k-237 | redundancy only \| auto_sqrt | not_started | 0/237 | - |
| FB15k-237 | synergy + redundancy + dependency sign \| auto_sqrt | not_started | 0/237 | - |
| FB15k-237 | synergy + redundancy + lift init \| auto_sqrt | not_started | 0/237 | - |
| FB15k-237 | synergy + redundancy \| auto_ratio | not_started | 0/237 | - |
| codex-l | synergy + redundancy \| auto_sqrt | not_started | 0/69 | - |
| codex-l | synergy only \| auto_sqrt | not_started | 0/69 | - |
| codex-l | redundancy only \| auto_sqrt | not_started | 0/69 | - |
| codex-l | synergy + redundancy + dependency sign \| auto_sqrt | not_started | 0/69 | - |
| codex-l | synergy + redundancy + lift init \| auto_sqrt | not_started | 0/69 | - |
| codex-l | synergy + redundancy \| auto_ratio | not_started | 0/69 | - |
| YAGO3-10 | synergy + redundancy \| auto_sqrt | not_started | 0/37 | - |
| YAGO3-10 | synergy only \| auto_sqrt | not_started | 0/37 | - |
| YAGO3-10 | redundancy only \| auto_sqrt | not_started | 0/37 | - |
| YAGO3-10 | synergy + redundancy + dependency sign \| auto_sqrt | not_started | 0/37 | - |
| YAGO3-10 | synergy + redundancy + lift init \| auto_sqrt | not_started | 0/37 | - |
| YAGO3-10 | synergy + redundancy \| auto_ratio | not_started | 0/37 | - |
