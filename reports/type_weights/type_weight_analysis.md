# Type-weight Analysis

This section moves beyond dataset-level type-weight averages and examines type preferences at relation-level granularity. For each dataset's best typed experiment, we ask which rule types and dependency interactions actually matter per relation.

Importance of a type in a relation is defined as: `support x trained_weight`.

- `trained_weight` is the multiplicative coefficient the model learned. A larger weight means the model amplifies that type's contribution more.
- `support` reflects how many rules or rule pairs of that type appear in the relation.
- `support x trained_weight` captures the total "effective mass" of that type in the relation. If a type has weight below `1`, its relative importance naturally declines.

Related files:

- `best_typed_experiment_by_dataset.csv`
- `relation_type_weight_importance.csv`
- `dataset_type_weight_summary.csv`
- `global_type_weight_summary.csv`

<p align="center"><img src="plot_dataset_rule_type_dominance.png" alt="Dataset Rule Type Dominance" width="60%"></p>

<p align="center"><em>Figure 1: share of relations whose dominant rule type is B / U / Uc / Ud in each dataset.</em></p>

<p align="center"><img src="plot_dataset_dependency_type_dominance.png" alt="Dataset Dependency Type Dominance" width="60%"></p>

<p align="center"><em>Figure 2: share of relations whose dominant dependency interaction type differs across datasets.</em></p>

<p align="center"><img src="plot_dataset_rule_type_impact_heatmap.png" alt="Dataset Rule Type Impact Heatmap" width="60%"></p>

<p align="center"><em>Figure 3: median support-weighted importance of each rule type in each dataset.</em></p>

## Best Typed Experiment Per Dataset

To avoid conflating "whether to use type weights" with "how well type weights work," we select one best typed experiment per dataset -- picking between `r2d3` and `r3d6` when both are available.

| Dataset | Best typed experiment | Grouping | Test MRR | Top rule type | Top dependency type |
| --- | --- | --- | ---: | --- | --- |
| KG20C | tg_r2d3__pos_auto_ratio__ri_conf__dn_per_rule_degree__dl1_1e-5 | r2d3 | 0.23395 | U | UU |
| codex-m | tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5 | rd | 0.34480 | - | - |
| WN18RR | structural_rd | rd | 0.50219 | - | - |
| FB15k-237 | tg_r2d3__pos_auto_ratio__ri_conf__dn_none__dl1_1e-5 | r2d3 | 0.35535 | U | UU |
| codex-l | tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5 | rd | 0.33418 | - | - |
| YAGO3-10 | structural_dep_scale | mixed | 0.57796 | - | - |
| hetionet | dep_scale_surprisal_init | mixed | 0.37847 | - | - |

## Dataset-level Pattern

The best typed experiments split as `r2d3 = 2` and `r3d6 = 0`. But this only picks the analysis entry point. The real story is that within each experiment, different relations show consistent but distinct type preferences.

Across all relations, the most common dominant rule type is `U`, and the most common dominant dependency interaction is `UU`.

## Within-dataset Heterogeneity

The following statistics show whether relations within a dataset converge to the same type or diverge. Low entropy means all relations prefer the same type; higher entropy means preferences vary.

- `KG20C`: dominant rule type is `U` for `100.00%` of relations; rule-type entropy = `0.00000`, dependency-type entropy = `0.72193`.
- `codex-m`: dominant rule type is `-` for `0.00%` of relations; rule-type entropy = `-`, dependency-type entropy = `-`.
- `WN18RR`: dominant rule type is `-` for `0.00%` of relations; rule-type entropy = `-`, dependency-type entropy = `-`.
- `FB15k-237`: dominant rule type is `U` for `87.76%` of relations; rule-type entropy = `0.53611`, dependency-type entropy = `0.68889`.
- `codex-l`: dominant rule type is `-` for `0.00%` of relations; rule-type entropy = `-`, dependency-type entropy = `-`.
- `YAGO3-10`: dominant rule type is `-` for `0.00%` of relations; rule-type entropy = `-`, dependency-type entropy = `-`.
- `hetionet`: dominant rule type is `-` for `0.00%` of relations; rule-type entropy = `-`, dependency-type entropy = `-`.

## What Matters in Each Dataset

If "importance" is `support x trained_weight`, the dominant type differs across datasets.

- `KG20C`: most important rule type is `U`, median importance `35161.72`; most important dependency is `BU`, median importance `8791.79`.
- `codex-m`: most important rule type is `-`, median importance `-`; most important dependency is `-`, median importance `-`.
- `WN18RR`: most important rule type is `-`, median importance `-`; most important dependency is `-`, median importance `-`.
- `FB15k-237`: most important rule type is `U`, median importance `844.24`; most important dependency is `UU`, median importance `931.16`.
- `codex-l`: most important rule type is `-`, median importance `-`; most important dependency is `-`, median importance `-`.
- `YAGO3-10`: most important rule type is `-`, median importance `-`; most important dependency is `-`, median importance `-`.
- `hetionet`: most important rule type is `-`, median importance `-`; most important dependency is `-`, median importance `-`.

## Representative Relation-level Diversity

Dataset means hide within-dataset variation. Here are some representative relations showing different dominant patterns inside the same dataset.

- `KG20C` / `paper_in_domain`: dominant rule type = `U`, weight `1.86721`, support `101313.00000`, importance `189172.97093`; dominant dependency type = `UU`.
- `FB15k-237` / `/film/actor/film./film/performance/film`: dominant rule type = `U`, weight `2.57158`, support `27826.00000`, importance `71556.82404`; dominant dependency type = `BB`.
- `FB15k-237` / `/base/biblioness/bibs_location/country`: dominant rule type = `B`, weight `1.28215`, support `966.00000`, importance `1238.55535`; dominant dependency type = `BB`.

## Interpretation

Analyzed at relation granularity, type weights do not reflect a fixed set of global preferences learned across the whole experiment. Instead, the model adapts to each relation's local structure.

Key takeaways:

- Dominant rule types and dependency interactions genuinely differ across datasets.
- Within the same dataset, different relations show substantially different dominant types.
- To understand `Ud < B < Uc`, check proportions at the relation or dataset level -- averaged global numbers alone are misleading.
