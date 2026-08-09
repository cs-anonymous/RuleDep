# Report Archive

This directory keeps the 0421 report artifacts and removes older report snapshots.

## Root Files

Small result groups are kept directly under `reports/`:

- `all_results_summary.*`: overall experiment summary.
- `best_*`: best configuration and variant summaries.
- `dataset_*`: dataset-level statistics.
- `stage1_stage2_test_mrr_table.*`: Stage1/Stage2 MRR tables.
- `overall_time_comparison.csv`: runtime comparison.
- `all_results_ensemble_debug.json`: raw debug output.

## Subdirectories

- `paper/`: LaTeX source, compiled preview artifacts, and bibliography.
- `figures/`: generated top-level report figures.
- `relation/`: relation-level gain and dependency analyses.
- `type_weights/`: rule/dependency type and weight analyses.
- `query_analysis/`: query-level case analysis and delta-MRR plots.
- `official_query_subset/`: official-aligned query subset feature analysis, CSVs, and generated feature plots.
- `high_order_analysis/`: rule-pair versus three-rule-combination support and significance analysis.
