# RuleDep

Source code and reproduction artifacts for **RuleDep: Modeling Complementarity and Redundancy in Rule Aggregation for KGC**, an ICDE 2027 submission.

RuleDep is a dependency-aware rule aggregation framework for knowledge graph completion (KGC). Rule-based KGC is attractive because each prediction can be traced to symbolic evidence, but ranking quality depends heavily on how multiple fired rules are aggregated for the same candidate entity. Existing aggregators such as Max+, Noisy-OR, Sparse, SAFRAN, and LR-Agg mostly model marginal rule evidence. RuleDep adds an explicit second-order layer for co-fired rules: complementary rule pairs can contribute more-than-additive evidence, while redundant rule pairs can be discounted when they overlap.

The paper source is in [`RuleDep-ICDE2027/`](RuleDep-ICDE2027/), and the benchmark data release is hosted at:

<https://huggingface.co/datasets/yesun/RuleDepData>

## Method Overview

RuleDep models rule aggregation in log-failure evidence space. For a rule with confidence `conf`, its evidence is:

```text
e(rule) = -log(1 - conf(rule))
```

Under the Noisy-OR independence assumption, evidence from independently failing rules is additive. RuleDep therefore measures a pairwise rule dependency as a signed evidence gain:

```text
gain(i, j) = e(rule_i and rule_j) - e(rule_i) - e(rule_j)
```

Positive gain indicates complementarity: two rules together provide stronger evidence than their marginal signals suggest. Negative gain indicates redundancy: the rules overlap and should not be counted as independent support. Near-zero pairs are treated as approximately independent and are not retained as dependency terms.

The pipeline has five main stages:

1. **Rule learning**: mine symbolic rules with AnyBURL.
2. **Rule application**: apply mined rules to train/valid/test triples and collect fired-rule evidence.
3. **Dataset construction**: convert fired rules into relation-local training data.
4. **Dependency mining**: discover co-fired rule pairs and estimate complementarity/redundancy gains.
5. **Dependency-aware aggregation**: train relation-wise aggregation models with rule weights plus sparse dependency corrections.

## Datasets

The released experiments cover six KGC benchmarks:

| Dataset | Entities | Relations | Train | Valid | Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| KG20C | 16,362 | 5 | 48,213 | 3,670 | 3,724 |
| WN18RR | 40,943 | 11 | 86,835 | 3,034 | 3,134 |
| Codex-M | 17,050 | 51 | 185,584 | 10,310 | 10,311 |
| FB15k-237 | 14,541 | 237 | 272,115 | 17,535 | 20,466 |
| Codex-L | 77,951 | 69 | 551,193 | 30,622 | 30,622 |
| YAGO3-10 | 123,182 | 37 | 1,079,040 | 5,000 | 5,000 |

Download the data release from Hugging Face and place the dataset directories under `data/`:

```bash
pip install -U huggingface_hub
huggingface-cli download yesun/RuleDepData \
  --repo-type dataset \
  --local-dir data
```

The release contains both the standard KGC split files and the derived Step 1-4 artifacts needed by the aggregation model:

```text
data/<dataset>/train.txt
data/<dataset>/valid.txt
data/<dataset>/test.txt
```

```text
data/<dataset>/rules/
data/<dataset>/application/
data/<dataset>/datasets/
```

Therefore, reproducing the main RuleDep results does not require rerunning rule learning, rule application, dataset construction, or dependency mining.

## Environment

The Python part of the pipeline was developed with Python 3.8. The dependency-mining implementation uses Kotlin/JVM through Maven, and AnyBURL requires Java.

```bash
conda create -n ruledep python=3.8
conda activate ruledep
pip install -r requirements.txt
```

The following external KGE and PyClause dependencies are only needed when rebuilding preprocessing or rule-application artifacts from scratch; they are not required for the minimal Step 5 reproduction:

```bash
git clone https://github.com/uma-pi1/kge.git
cd kge
git checkout a9ecd249ec2d205df59287f64553a1536add4a43
pip install -e .
mv data data.bac
ln -s ../data data
cd ..

git clone https://github.com/cs-anonymous/PyClause.git
cd PyClause
git checkout cc3ef7c0aee51825d7d741b7ec03a0974f7c1619
pip install -e .
cd ..
```

The repository already includes the AnyBURL jar used by the scripts at `script/AnyBURL-23-1x.jar`.

## Running RuleDep

After downloading RuleDepData, reproduce the paper's best RuleDep configuration for one dataset:

```bash
./step5_aggregation.sh FB15k-237
```

Run the best configuration for all six datasets sequentially:

```bash
./step5_aggregation.sh all
```

The optional second argument controls the number of relation workers, and `GPU_ID` selects the GPU:

```bash
GPU_ID=1 ./step5_aggregation.sh codex-l 2
```

Runs are resumable at relation level. Results are written to `data/<dataset>/aggregation/reproduction/`, with logs under `logs/aggregation_reproduction/`.

To reproduce the complete 48-configuration search space used by the relation-wise ensemble, use the research launcher under `script/`:

```bash
GPU_IDS=0,1,2,3 MAX_PARALLEL_CONFIGS=4 \
  ./script/run_full_aggregation_grid.sh FB15k-237 2
```

All scripts assume they are executed from the repository root.

## Pipeline Details

### Steps 1-4: Released Artifacts

Steps 1-4 learn AnyBURL rules, apply them to KGC queries, construct relation-local training datasets, and mine pairwise complementarity/redundancy dependencies. Their outputs are already included in RuleDepData under `rules/`, `application/`, and `datasets/`. The corresponding top-level scripts remain available for researchers who need to rebuild or modify these artifacts, but they are not part of the minimal reproduction workflow.

### Step 5: Aggregation

```bash
./step5_aggregation.sh <dataset> [multiprocess]
```

This is the only stage required after downloading the released data. It selects the paper's best configuration for the requested dataset, trains relation-wise aggregation models, and resumes completed relations when rerun. The model first learns rule weights and then introduces dependency corrections initialized from mined gain values. Results are written under:

```text
data/<dataset>/aggregation/reproduction/
```

Typical output files include per-relation metrics, learned weights, dependency diagnostics, and final aggregate metrics.

## Paper Experiment Scripts

The repository keeps only a small set of maintained shell launchers under `script/`:

```text
script/run_lragg_baseline.sh          LR-Agg/canonical baseline for one dataset
script/run_lragg_baseline_all.sh      LR-Agg/canonical baseline batch runner
script/run_full_aggregation_grid.sh   Complete 48-configuration RuleDep search
script/run_full_ensemble_sweep_tmux.sh  Four-GPU tmux launcher for the full search
script/resume_ruledep_grid.sh         Resume missing configurations from the 48-config RuleDep grid
script/run_dependency_budget_sweep.sh Dependency filtering and budget sweep
script/export_per_query_rr.sh         Export stage1/stage2 per-query RR for query-subset analysis
```

Historical queue, tmux, one-off rerun, and dataset-specific recovery scripts were removed from the working tree. They can be recovered from Git history if needed.

## Main Results

In the ICDE 2027 submission, RuleDep is evaluated on six released datasets against latent, hybrid, and interpretable KGC baselines. The main findings are:

- RuleDep improves average MRR by **3.7%** over LR-Agg, the strongest interpretable supervised rule aggregator in the comparison.
- The relation-wise ensemble variant, RuleDep-ens, obtains the best interpretable result across the reported dataset/metric settings.
- The gain is concentrated on dependency-rich queries. On the top 10% of queries selected by a complementarity-to-rule-evidence ratio, RuleDep obtains a **10.77%** dataset-macro relative MRR gain.
- Dependency corrections are sparse: most retained dependencies are suppressed during supervised training, while a small number of high-impact corrections explain the final ranking changes.
- The full RuleDep pipeline runs in about **24%** of LR-Agg's runtime on average over the completed LR-Agg datasets.

### Run-to-Run Stability

To quantify run-to-run variability, we fixed the rule set and repeated dependency mining and dependency-aware aggregation training three times using the same best configuration for each dataset. The table reports the final MRR mean and sample standard deviation across the three runs.

| Dataset | Mean final MRR | Sample std. |
| --- | ---: | ---: |
| KG20C | 0.2341 | 0.0001 |
| Codex-M | 0.3446 | 0.0002 |
| WN18RR | 0.5008 | 0.0013 |
| FB15k-237 | 0.3537 | 0.0015 |
| Codex-L | 0.3335 | 0.0006 |
| YAGO3-10 | 0.5770 | 0.0017 |

The overall dataset-macro final MRR is **0.3906 +/- 0.0008**, compared with **0.3848** for the Stage-1/LR-Agg baseline. All three repeats consistently outperform the baseline on all six datasets.

## Reports

The repository includes the following experiment reports and supporting artifacts:

- [Overall experiment results](reports/all_results_summary.md): best configurations, ensemble variants, baseline comparisons, and runtime summaries.
- [Dataset and rule-dependency statistics](reports/dataset_analysis.md): dataset sizes, rule counts, dependency counts, and structural statistics.
- [Best-configuration and ensemble variants](reports/best_combination_variant_details.md): selected configurations and their relation-level ensemble results.
- [Run-to-run stability](reports/run_to_run_stability/README.md): repeated dependency-mining and aggregation-training results with MRR mean and standard deviation.
- [Evidence-cap sensitivity](reports/evidence_cap_sensitivity/README.md): sensitivity of dependency mining to capped and uncapped log-failure evidence.
- [Higher-order rule-dependency analysis](reports/high_order_analysis/README.md): pairwise versus three-rule co-firing support and higher-order-effect statistics.
- [Dependency-rich query-subset analysis](reports/query_subset/README.md): performance on queries ranked by complementary dependency evidence.
- [Query-level paired significance tests](reports/query_level_paired_test/README.md): paired query-level confidence intervals and corrected significance tests.
- [Rule and dependency weight analysis](reports/type_weights/rule_dependency_weight_analysis.md): sparsity, effective contributions, and signed dependency-weight statistics.
- [Relation-level type-weight analysis](reports/type_weights/type_weight_analysis.md): variation in rule and dependency type importance across relations.

## Repository Layout

```text
RuleDep-ICDE2027/        ICDE 2027 paper source and figures
script/                  Shell launchers and bundled AnyBURL resources
src/ruledep/             Core Python pipeline implementation
src/rule_application/    Rule application and AnyBURL-style evaluation helpers
src/baselines/           LR-Agg/canonical baseline implementation
src/rule_tools/          Rule parsing, validation, and support-analysis helpers
src/query_analysis/      Query-subset and per-query RR analysis scripts
src/reporting/           Paper table, summary, and figure generation scripts
src/case_studies/        Case-study extraction and inspection scripts
src/data_tools/          Dataset preparation and inspection helpers
src/main/java/tarmorn/   Kotlin dependency-mining implementation
reports/                 Generated summaries, tables, and case-study artifacts
data/                    Dataset splits and generated pipeline outputs
PyClause/                External PyClause checkout, if installed locally
kge/                     External KGE checkout, if installed locally
```

Minimal reproduction entrypoint:

```text
step5_aggregation.sh
```

The Step 1-4 entrypoints are retained for rebuilding the released intermediate artifacts. Core implementation files include:

```text
src/ruledep/preprocess.py
src/ruledep/process_rules.py
src/ruledep/filter_dependency.py
src/ruledep/aggregation.py
```

The repository root intentionally keeps only shell entrypoints and project metadata. Python implementation files live under `src/`, while generated CSV artifacts live under `reports/` or dataset-specific output directories.

## Citation

This repository accompanies an ICDE 2027 submission. Citation information will be added after the paper metadata is finalized.
