# 0407 Report Summary

本目录包含三部分分析：

1. 整体指标总表与 summary
2. 数据集逐关系增益分析
3. 数据集规模与 rule/dependency 统计

## Key Files

- `all_results_summary.csv`
- `all_results_summary.md`
- `best_config_by_dataset.csv`
- `all_results_ensemble_debug.json`
- `relation_dependency_analysis.csv`
- `relation_relative_gain_gt_3pct_best_config.csv`
- `relation_relative_gain_lt_0_best_config.csv`
- `dependency_relation_analysis.md`
- `relation_positive_examples_gt3_dependency.csv`
- `relation_positive_examples_gt3_dependency.md`
- `relation_case_study_examples.csv`
- `dataset_size_rule_dependency_stats.csv`
- `dataset_analysis.md`
- `plot_gain_vs_stage1.png`
- `plot_gain_vs_dep_density.png`
- `plot_stage1_bucket_summary.png`
- `plot_dep_density_bucket_summary.png`
- `plot_dataset_gain_mix.png`
- `plot_type_weight_summary.png`

生成时间目录：`/home/sy/RuleDep/reports/0407`

## Representative Case Study

我们额外做了一个更严格的 query-level case study：只保留那些在 stage1 下预测失败、但在引入 dependency 后 top-1 预测成功的样例。结果写在：

- `relation_case_study_examples.csv`

当前最有代表性的样例来自 `hetionet / DrD`：

- query: `(Disease::DOID:11615, DrD, ?)`
- gold: `Disease::DOID:11054`
- stage1 top-1: `Disease::DOID:4045`
- final top-1: `Disease::DOID:11054`
- stage1 rank: `5`
- final rank: `1`

这个例子很符合 `DrD` 的语义：疾病相似性通常不能靠单条 path 决定，而需要多个基因/症状相关规则共同支持。dependency 的作用在这里不是简单加分，而是把“多条弱证据同时成立”的结构信号转化为正确的 top-1 排序。
