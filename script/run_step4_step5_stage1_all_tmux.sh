#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/sy/RuleDep"
cd "$ROOT_DIR"

if [ -f /home/sy/anaconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /home/sy/anaconda3/etc/profile.d/conda.sh
  conda activate ruledep
else
  export PATH="/home/sy/anaconda3/envs/ruledep/bin:${PATH}"
fi

DATASETS=(
  "KG20C"
  "codex-m"
  "WN18RR"
  "FB15k-237"
  "codex-l"
  "YAGO3-10"
  "hetionet"
)

GPU_COUNT="${AGGREGATION_GPUS:-4}"
N_PROC_PER_GPU="${AGGREGATION_N_PROC:-1}"
RUN_TAG="${RUN_TAG:-stage1cache_0414}"
LOG_ROOT="$ROOT_DIR/logs/stage1_cache/$RUN_TAG"
mkdir -p "$LOG_ROOT"

for ds in "${DATASETS[@]}"; do
  echo "[$(date '+%F %T')] ===== DATASET: $ds ====="

  echo "[$(date '+%F %T')] Step4 start: $ds"
  bash "$ROOT_DIR/step4_dependency.sh" "$ds" 2>&1 | tee "$LOG_ROOT/${ds}_step4.log"
  echo "[$(date '+%F %T')] Step4 done: $ds"

  exp_dir="$ROOT_DIR/data/$ds/aggregation/stage1"
  stage1_state_dir="$ROOT_DIR/data/$ds/rules/stage1_state"
  mkdir -p "$exp_dir" "$stage1_state_dir"

  echo "[$(date '+%F %T')] Step5(stage1-only) start: $ds"
  (
    export EXPERIMENT_DIR="$exp_dir"
    export PYTHONUNBUFFERED=1
    python -u "$ROOT_DIR/aggregation.py" \
      -d "$ds" \
      --rule_file "data/$ds/rules/rule.txt" \
      --relation -1 \
      --gpus "$GPU_COUNT" \
      --n_proc "$N_PROC_PER_GPU" \
      --train_rule_in_dependency_stage \
      --synergy \
      --redundancy \
      --type_grouping rd \
      --dependency_scale_mode sqrt_active \
      --rule_init_mode surprisal \
      --pos auto_ratio \
      --dep_l1_lambda 1e-5 \
      --stage1_only \
        --save_stage1_state_dir "$stage1_state_dir"
  ) 2>&1 | tee "$LOG_ROOT/${ds}_step5_stage1.log"
  echo "[$(date '+%F %T')] Step5(stage1-only) done: $ds"

done

echo "[$(date '+%F %T')] ALL DONE"
