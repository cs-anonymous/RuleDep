# RuleDep

RuleDep is a rule-based KGC pipeline. In practice, the most common entrypoints are:

- `./run.sh` for the full Stage 1-5 pipeline
- `./step3_dataset.sh`
- `./step4_dependency.sh`
- `./step5_aggregation.sh`

The default dataset list in `run.sh` is:

- `KG20C`
- `codex-m`
- `codex-l`
- `FB15k-237`
- `WN18RR`
- `YAGO3-10`

## Common Usage

Run one dataset end to end:

```bash
python preprocess.py data/FB15k-237
./run.sh FB15k-237
```

Run only the current core stages:

```bash
./step3_dataset.sh FB15k-237
./step4_dependency.sh FB15k-237
./step5_aggregation.sh FB15k-237
```

All shell scripts assume you run them from the repository root.

## Pipeline

### Step 0: Preprocess

Command:

```bash
python preprocess.py data/<dataset>
```

Input:

- `data/<dataset>/train.txt`
- `data/<dataset>/valid.txt`
- `data/<dataset>/test.txt`

Output:

- `data/<dataset>/entity_ids.del`
- `data/<dataset>/relation_ids.del`
- `data/<dataset>/train.del`
- `data/<dataset>/valid.del`
- `data/<dataset>/test.del`
- `data/<dataset>/dataset.yaml`

### Step 1: Learning

Command:

```bash
./step1_learning.sh <dataset>
```

Input:

- `data/<dataset>/train.txt`

Output:

- `data/<dataset>/rules/rules-*`
- `data/<dataset>/rules/rules-*-5`

Notes:

- Uses `script/AnyBURL-23-1x.jar`
- Filters rules with `supp >= 5`

### Step 2: Application

Command:

```bash
./step2_application.sh <dataset>
```

Input:

- `data/<dataset>/rules/rules-1000-5`
- `data/<dataset>/train.txt`
- `data/<dataset>/valid.txt`
- `data/<dataset>/test.txt`

Output:

- `data/<dataset>/application/eval-noisyor.log`
- `data/<dataset>/application/eval-maxplus.log`
- `data/<dataset>/application/applied_rules_train.json`
- `data/<dataset>/application/applied_rules_valid.json`
- `data/<dataset>/application/applied_rules_test.json`

Notes:

- Uses `script/eval.py`
- Uses `script/apply_pyclause.py`

### Step 3: Dataset

Command:

```bash
./step3_dataset.sh <dataset>
```

Input:

- `data/<dataset>/application/applied_rules_train.json`
- `data/<dataset>/application/applied_rules_valid.json`
- `data/<dataset>/application/applied_rules_test.json`
- `data/<dataset>/rules/rules-1000-5`

Output:

- `data/<dataset>/application/processed_sp_*.pkl`
- `data/<dataset>/application/processed_po_*.pkl`
- `data/<dataset>/datasets/dataset_<relation>.p`

Notes:

- This is where `process_rules.py` is executed

### Step 4: Dependency

Command:

```bash
./step4_dependency.sh <dataset>
```

Input:

- `data/<dataset>/rules/rules-1000-5`
- `data/<dataset>/application/processed_sp_train.pkl`
- `data/<dataset>/application/processed_po_train.pkl`

Output:

- `data/<dataset>/rules/dependency.txt`
- `data/<dataset>/rules/synergy.txt`
- `data/<dataset>/rules/redundancy.txt`
- `data/<dataset>/rules/synergy_filtered.txt`
- `data/<dataset>/rules/redundancy_filtered.txt`

### Step 5: Aggregation

Command:

```bash
./step5_aggregation.sh <dataset>
```

Input:

- `data/<dataset>/datasets/dataset_<relation>.p`
- `data/<dataset>/application/processed_sp_valid.pkl`
- `data/<dataset>/application/processed_po_valid.pkl`
- `data/<dataset>/application/processed_sp_test.pkl`
- `data/<dataset>/application/processed_po_test.pkl`
- `data/<dataset>/rules/rules-1000-5`
- `data/<dataset>/rules/synergy_filtered.txt`
- `data/<dataset>/rules/redundancy_filtered.txt`

Output:

- `data/<dataset>/aggregation/exp.../`
- `metric-<relation>.json`
- `weight-<relation>.csv`
- `dependency-trial-<relation>.csv`
- `dependency-final-<relation>.csv`
- `metrics-final.json`

## File Placement

- Top-level shell entrypoints: `step1_learning.sh`, `step2_application.sh`, `step3_dataset.sh`, `step4_dependency.sh`, `step5_aggregation.sh`, `run.sh`
- Stage-1/2 support files: `script/AnyBURL-23-1x.jar`, `script/eval.py`, `script/evaltc.py`, `script/apply_pyclause.py`
