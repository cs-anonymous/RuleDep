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

The experiments use seven KGC benchmarks:

| Dataset | Entities | Relations | Train | Valid | Test |
| --- | ---: | ---: | ---: | ---: | ---: |
| KG20C | 16,362 | 5 | 48,213 | 3,670 | 3,724 |
| WN18RR | 40,943 | 11 | 86,835 | 3,034 | 3,134 |
| Codex-M | 17,050 | 51 | 185,584 | 10,310 | 10,311 |
| FB15k-237 | 14,541 | 237 | 272,115 | 17,535 | 20,466 |
| Codex-L | 77,951 | 69 | 551,193 | 30,622 | 30,622 |
| YAGO3-10 | 123,182 | 37 | 1,079,040 | 5,000 | 5,000 |
| Hetionet | 45,158 | 24 | 1,800,157 | 225,020 | 225,020 |

Download the data release from Hugging Face and place the dataset directories under `data/`:

```bash
pip install -U huggingface_hub
huggingface-cli download yesun/RuleDepData \
  --repo-type dataset \
  --local-dir data
```

Each dataset directory is expected to contain the standard KGC split files:

```text
data/<dataset>/train.txt
data/<dataset>/valid.txt
data/<dataset>/test.txt
```

Some stages also create or consume derived files under `data/<dataset>/rules/`, `data/<dataset>/application/`, `data/<dataset>/datasets/`, and `data/<dataset>/aggregation/`.

## Environment

The Python part of the pipeline was developed with Python 3.8. The dependency-mining implementation uses Kotlin/JVM through Maven, and AnyBURL requires Java.

```bash
conda create -n ruledep python=3.8
conda activate ruledep
pip install -r requirements.txt
```

Install the external KGE and PyClause dependencies used by the preprocessing, baseline, and rule-application scripts:

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

Run the full pipeline for one dataset:

```bash
./run.sh FB15k-237
```

Run the default batch:

```bash
./run.sh
```

The default batch currently covers:

```text
KG20C
codex-m
WN18RR
FB15k-237
codex-l
YAGO3-10
```

Hetionet is supported by the stage scripts and by `step5_aggregation.sh all`, but it is not included in the default `run.sh` list because of its larger runtime and memory footprint.

You can also run the stages manually:

```bash
./step1_learning.sh FB15k-237
./step2_application.sh FB15k-237
./step3_dataset.sh FB15k-237
./step4_dependency.sh FB15k-237
./step5_aggregation.sh FB15k-237
```

All scripts assume they are executed from the repository root.

## Pipeline Details

### Step 0: Preprocessing

```bash
python src/ruledep/preprocess.py data/<dataset>
```

This converts the raw `train.txt`, `valid.txt`, and `test.txt` split files into the `.del`, ID mapping, index, and `dataset.yaml` files used by downstream stages.

### Step 1: Rule Learning

```bash
./step1_learning.sh <dataset> [support_threshold] [snapshots] [worker_threads]
```

This stage runs AnyBURL on `data/<dataset>/train.txt`, stores snapshots under `data/<dataset>/rules/`, filters rules by support, and writes the final rule file to:

```text
data/<dataset>/rules/rule.txt
```

The default support threshold is `2` for KG20C and WN18RR, and `5` for other datasets.

### Step 2: Rule Application

```bash
./step2_application.sh <dataset>
```

This stage applies the learned rules and generates fired-rule evidence for train, validation, and test queries. Outputs are written under:

```text
data/<dataset>/application/
```

### Step 3: Relation-Local Dataset Construction

```bash
./step3_dataset.sh <dataset>
```

This stage builds the relation-local training data consumed by the aggregation model:

```text
data/<dataset>/datasets/dataset_<relation>.p
```

### Step 4: Dependency Mining

```bash
./step4_dependency.sh <dataset>
```

This stage mines pairwise dependencies between co-fired rules, estimates signed evidence gain, and separates complementarity and redundancy candidates. Important outputs include:

```text
data/<dataset>/rules/dependency.txt
data/<dataset>/rules/synergy.txt
data/<dataset>/rules/redundancy.txt
data/<dataset>/rules/synergy_filtered.txt
data/<dataset>/rules/redundancy_filtered.txt
```

The dependency learner is implemented in Kotlin under `src/main/java/tarmorn/` and is invoked through Maven.

### Step 5: Aggregation

```bash
./step5_aggregation.sh <dataset> [multiprocess]
```

This stage trains relation-wise aggregation models. It first learns a rule-only baseline, then introduces dependency corrections initialized from mined gain values. Results are written under:

```text
data/<dataset>/aggregation/
```

Typical output files include per-relation metrics, learned weights, dependency diagnostics, and final aggregate metrics.

## Paper Experiment Scripts

The repository keeps only a small set of maintained shell launchers under `script/`:

```text
script/run_lragg_baseline.sh          LR-Agg/canonical baseline for one dataset
script/run_lragg_baseline_all.sh      LR-Agg/canonical baseline batch runner
script/resume_ruledep_grid.sh         Resume missing configurations from the 48-config RuleDep grid
script/run_dependency_budget_sweep.sh Dependency filtering and budget sweep
script/export_per_query_rr.sh         Export stage1/stage2 per-query RR for query-subset analysis
```

Historical queue, tmux, one-off rerun, and dataset-specific recovery scripts were removed from the working tree. They can be recovered from Git history if needed.

## Main Results

In the ICDE 2027 submission, RuleDep is evaluated on seven datasets against latent, hybrid, and interpretable KGC baselines. The main findings are:

- RuleDep improves average MRR by **3.7%** over LR-Agg, the strongest interpretable supervised rule aggregator in the comparison.
- The relation-wise ensemble variant, RuleDep-ens, obtains the best interpretable result on all 21 dataset/metric settings in the reported benchmark table.
- The gain is concentrated on dependency-rich queries. On the top 10% of queries selected by a complementarity-to-rule-evidence ratio, RuleDep obtains a **10.77%** dataset-macro relative MRR gain.
- Dependency corrections are sparse: most retained dependencies are suppressed during supervised training, while a small number of high-impact corrections explain the final ranking changes.
- The full RuleDep pipeline runs in about **24%** of LR-Agg's runtime on average over the completed LR-Agg datasets, while Hetionet finishes in about 2.4 hours for RuleDep and exceeds 24 hours for LR-Agg.

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

Core top-level entrypoints:

```text
run.sh
step1_learning.sh
step2_application.sh
step3_dataset.sh
step4_dependency.sh
step5_aggregation.sh
src/ruledep/preprocess.py
src/ruledep/process_rules.py
src/ruledep/filter_dependency.py
src/ruledep/aggregation.py
```

The repository root intentionally keeps only shell entrypoints and project metadata. Python implementation files live under `src/`, while generated CSV artifacts live under `reports/` or dataset-specific output directories.

## Citation

This repository accompanies an ICDE 2027 submission. Citation information will be added after the paper metadata is finalized.
