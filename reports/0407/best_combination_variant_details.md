# Best-combination concrete strategy by dataset

> 说明：`best_combination*` 不是单一固定策略，而是按数据集自动选择底座（type/init/pos/dep_filter 等），再叠加变体开关。

| dataset | base strategy (expanded) | base MRR | +dep_l1 | +dep_score_clip | +static_norm | +topk_per_kind |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| KG20C | type=none, init=conf, pos=auto_ratio, dep_filter=default, synergy=True, redundancy=True, sign_dep=False | 0.233439 | 0.233342 | 0.233448 | 0.231797 | 0.232752 |
| codex-m | type=r3d6, init=conf, pos=auto_sqrt, dep_filter=ratio_k2, synergy=True, redundancy=True, sign_dep=False | 0.343197 | 0.343797 | 0.343660 | 0.343331 | 0.343425 |
| WN18RR | type=rd, init=conf, pos=auto_sqrt, dep_filter=ratio_k1, synergy=True, redundancy=True, sign_dep=False | 0.501890 | 0.502195 | 0.500810 | 0.501095 | 0.501590 |
| FB15k-237 | type=r2d3, init=surprisal, pos=auto_ratio, dep_filter=ratio_k2, synergy=True, redundancy=True, sign_dep=False | 0.353336 | 0.353831 | 0.353529 | 0.352155 | 0.352434 |
| codex-l | type=none, init=conf, pos=auto_sqrt, dep_filter=mix_k1, synergy=True, redundancy=True, sign_dep=False | 0.333575 | 0.334126 | 0.333453 | 0.332486 | 0.332766 |
| YAGO3-10 | type=r2d3, init=surprisal, pos=auto_sqrt, dep_filter=ratio_k1, synergy=True, redundancy=True, sign_dep=False | 0.576069 | 0.576319 | 0.575810 | 0.574927 | 0.574772 |
| hetionet | type=r2d3, init=surprisal, pos=auto_ratio, dep_filter=lift_k1, synergy=True, redundancy=True, sign_dep=False | 0.373229 | 0.374393 | 0.372298 | 0.373354 | 0.372125 |