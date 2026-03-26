# Stage2 Gain Density Analysis

结论基于每个数据集当前最优已完成配置，比较 `stage2 - stage1` 的关系级增益是否集中在更稠密的关系上。

## 汇总表

| Dataset | Best config | Overall gain | Oracle topN | Oracle ratio | Oracle coverage | Density topN | Density ratio | Density coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KG20C | synergy + redundancy \| auto_ratio | 0.002672 | 1 | 2.244x | 0.388292 | 1 | 2.244x | 0.388292 |
| codex-m | synergy + redundancy \| auto_sqrt | 0.002388 | 5 | 3.692x | 0.115896 | 5 | 1.768x | 0.060227 |
| WN18RR | synergy + redundancy \| auto_ratio | 0.001385 | 1 | 10.869x | 0.080728 | 4 | 6.193x | 0.151563 |
| FB15k-237 | synergy + redundancy \| auto_ratio | 0.002992 | 16 | 9.565x | 0.058390 | 5 | 2.774x | 0.177221 |
| codex-l | synergy + redundancy \| auto_sqrt | 0.002702 | 8 | 3.505x | 0.058165 | 1 | 1.993x | 0.307838 |
| YAGO3-10 | synergy only \| auto_sqrt | 0.003609 | 5 | 5.399x | 0.025400 | 1 | 1.613x | 0.334600 |

## 数据集逐项结论

### KG20C

- 最优配置是 `synergy + redundancy | auto_ratio`，整体 `stage2-stage1` MRR 增益为 `0.002672`。
- 关系级 Spearman 相关：`train=0.300000`，`avg_tails=0.600000`，`dep_per_rule=0.900000`。
- Oracle 子集：按真实关系增益排序取前 `1` 个关系，子集增益达到 `0.005995`，是整体的 `2.244x`，覆盖测试集 `0.388292`。
- Density 子集：按 `top_train` 排序取前 `1` 个关系，子集增益为 `0.005995`，是整体的 `2.244x`，覆盖测试集 `0.388292`。
- Oracle 子集与全体关系相比：`avg(train)` 17776.000000 vs 9642.600000，`avg(tails)` 3.611540 vs 1.975350，`avg(dep/rule)` 0.067172 vs 0.043223。

### codex-m

- 最优配置是 `synergy + redundancy | auto_sqrt`，整体 `stage2-stage1` MRR 增益为 `0.002388`。
- 关系级 Spearman 相关：`train=0.392964`，`avg_tails=0.142978`，`dep_per_rule=0.241114`。
- Oracle 子集：按真实关系增益排序取前 `5` 个关系，子集增益达到 `0.008817`，是整体的 `3.692x`，覆盖测试集 `0.115896`。
- Density 子集：按 `low_avg_tails` 排序取前 `5` 个关系，子集增益为 `0.004223`，是整体的 `1.768x`，覆盖测试集 `0.060227`。
- Oracle 子集与全体关系相比：`avg(train)` 4383.600000 vs 4034.260870，`avg(tails)` 1.222572 vs 2.164297，`avg(dep/rule)` 0.993678 vs 0.659062。

### WN18RR

- 最优配置是 `synergy + redundancy | auto_ratio`，整体 `stage2-stage1` MRR 增益为 `0.001385`。
- 关系级 Spearman 相关：`train=-0.143019`，`avg_tails=0.390920`，`dep_per_rule=-0.333712`。
- Oracle 子集：按真实关系增益排序取前 `1` 个关系，子集增益达到 `0.015057`，是整体的 `10.869x`，覆盖测试集 `0.080728`。
- Density 子集：按 `top_avg_tails` 排序取前 `4` 个关系，子集增益为 `0.008580`，是整体的 `6.193x`，覆盖测试集 `0.151563`。
- Oracle 子集与全体关系相比：`avg(train)` 7402.000000 vs 7894.090909，`avg(tails)` 2.391599 vs 4.293053，`avg(dep/rule)` 0.027091 vs 0.072312。

### FB15k-237

- 最优配置是 `synergy + redundancy | auto_ratio`，整体 `stage2-stage1` MRR 增益为 `0.002992`。
- 关系级 Spearman 相关：`train=0.133247`，`avg_tails=-0.070179`，`dep_per_rule=0.169156`。
- Oracle 子集：按真实关系增益排序取前 `16` 个关系，子集增益达到 `0.028615`，是整体的 `9.565x`，覆盖测试集 `0.058390`。
- Density 子集：按 `score_support_compact` 排序取前 `5` 个关系，子集增益为 `0.008300`，是整体的 `2.774x`，覆盖测试集 `0.177221`。
- Oracle 子集与全体关系相比：`avg(train)` 1381.125000 vs 1148.164557，`avg(tails)` 5.703904 vs 6.986944，`avg(dep/rule)` 2.204383 vs 1.637545。

### codex-l

- 最优配置是 `synergy + redundancy | auto_sqrt`，整体 `stage2-stage1` MRR 增益为 `0.002702`。
- 关系级 Spearman 相关：`train=0.308994`，`avg_tails=0.177999`，`dep_per_rule=0.013945`。
- Oracle 子集：按真实关系增益排序取前 `8` 个关系，子集增益达到 `0.009473`，是整体的 `3.505x`，覆盖测试集 `0.058165`。
- Density 子集：按 `top_train` 排序取前 `1` 个关系，子集增益为 `0.005385`，是整体的 `1.993x`，覆盖测试集 `0.307838`。
- Oracle 子集与全体关系相比：`avg(train)` 4045.375000 vs 8612.031250，`avg(tails)` 1.232379 vs 1.895372，`avg(dep/rule)` 0.586039 vs 0.515063。

### YAGO3-10

- 最优配置是 `synergy only | auto_sqrt`，整体 `stage2-stage1` MRR 增益为 `0.003609`。
- 关系级 Spearman 相关：`train=0.179115`，`avg_tails=0.223290`，`dep_per_rule=0.199983`。
- Oracle 子集：按真实关系增益排序取前 `5` 个关系，子集增益达到 `0.019482`，是整体的 `5.399x`，覆盖测试集 `0.025400`。
- Density 子集：按 `top_train` 排序取前 `1` 个关系，子集增益为 `0.005820`，是整体的 `1.613x`，覆盖测试集 `0.334600`。
- Oracle 子集与全体关系相比：`avg(train)` 5315.800000 vs 29163.243243，`avg(tails)` 3.868748 vs 3.469363，`avg(dep/rule)` 0.991621 vs 0.581178。
