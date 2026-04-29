# Query-level Delta MRR Distribution

Generated from `reports/query_analysis/query_case_level_analysis.csv`.

Notes:

- Rows are per-GT cases, not merged multi-GT queries.
- `delta_rr` is relation-level calibrated so each relation mean matches official `test_after_stage2.mrr - test_after_stage1.mrr`.
- Raw demo-derived values are retained in `raw_rr_stage1`, `raw_rr_stage2`, and `raw_delta_rr`.

## Combined

![all_datasets_delta_rr_violin](all_datasets_delta_rr_violin.png)

## Per Dataset

### FB15k-237

![FB15k-237_delta_rr_hist.png](FB15k-237_delta_rr_hist.png)

### KG20C

![KG20C_delta_rr_hist.png](KG20C_delta_rr_hist.png)

### WN18RR

![WN18RR_delta_rr_hist.png](WN18RR_delta_rr_hist.png)

### YAGO3-10

![YAGO3-10_delta_rr_hist.png](YAGO3-10_delta_rr_hist.png)

### codex-l

![codex-l_delta_rr_hist.png](codex-l_delta_rr_hist.png)

### codex-m

![codex-m_delta_rr_hist.png](codex-m_delta_rr_hist.png)
