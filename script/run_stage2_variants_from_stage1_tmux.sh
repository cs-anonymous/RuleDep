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

RUN_TAG="${RUN_TAG:-stage2_variants_0415}"
LOG_ROOT="$ROOT_DIR/logs/stage2_variants/$RUN_TAG"
mkdir -p "$LOG_ROOT"

# Base configuration (same family as current fixed/default experimental setup)
BASE_ARGS=(
  --train_rule_in_dependency_stage
  --type_grouping rd
  --dependency_scale_mode sqrt_active
  --rule_init_mode surprisal
  --pos auto_ratio
  --dep_l1_lambda 1e-5
)

# NOTE: random-init(0 init) is equivalent to baseline here because init_dep_with_lift defaults to false,
# so to keep exactly 12 configs / dataset as requested, we do not duplicate a separate random-init run.
VARIANTS=(
  "baseline::--synergy --redundancy"
  "no-synergy::--redundancy"
  "no-redundancy::--synergy"
  "fix-init::--synergy --redundancy --dependency_init_only --init_dep_with_lift"
  "sign-constraint::--synergy --redundancy --sign_constraint_dependency"
  "auto-sqrt::--synergy --redundancy --pos auto_sqrt"
  "R2D3::--synergy --redundancy --type_grouping r2d3"
  "R3D6::--synergy --redundancy --type_grouping r3d6"
  "no-dep-scale::--synergy --redundancy --dependency_scale_mode none"
  "dep_l1_1e-4::--synergy --redundancy --dep_l1_lambda 1e-4"
  "dep_l2_1e-5::--synergy --redundancy --dep_l1_lambda 0 --dep_l2_lambda 1e-5"
  "dep_l2_1e-4::--synergy --redundancy --dep_l1_lambda 0 --dep_l2_lambda 1e-4"
)

run_one_variant() {
  local ds="$1"
  local gpu="$2"
  local variant_name="$3"
  shift 3
  local extra_args=("$@")

  local exp_dir="$ROOT_DIR/data/$ds/aggregation/$variant_name"
  local stage1_dir="$ROOT_DIR/data/$ds/rules/stage1_state"
  local log_file="$LOG_ROOT/${ds}_${variant_name}.log"

  mkdir -p "$exp_dir"

  (
    cd "$ROOT_DIR"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export EXPERIMENT_DIR="$exp_dir"
    export PYTHONUNBUFFERED=1

    python -u "$ROOT_DIR/aggregation.py" \
      -d "$ds" \
      --rule_file "data/$ds/rules/rule.txt" \
      --relation -1 \
      --gpus 1 \
      --n_proc 2 \
      --load_stage1_state_dir "$stage1_dir" \
      "${BASE_ARGS[@]}" \
      "${extra_args[@]}"
  ) >"$log_file" 2>&1
}

run_batch_of_four() {
  local ds="$1"
  shift
  local entries=("$@")
  local gpus=(0 1 2 3)
  local pids=()
  local idx=0

  for entry in "${entries[@]}"; do
    local name="${entry%%::*}"
    local arg_string="${entry#*::}"
    local args=()

    if [ -n "$arg_string" ]; then
      read -r -a args <<< "$arg_string"
    fi

    echo "[$(date '+%F %T')] START ds=$ds variant=$name gpu=${gpus[$idx]}" | tee -a "$LOG_ROOT/master.log"
    run_one_variant "$ds" "${gpus[$idx]}" "$name" "${args[@]}" &
    pids[$idx]=$!
    idx=$((idx + 1))
  done

  local status=0
  idx=0
  for pid in "${pids[@]}"; do
    local name="${entries[$idx]%%::*}"
    if wait "$pid"; then
      echo "[$(date '+%F %T')] DONE  ds=$ds variant=$name" | tee -a "$LOG_ROOT/master.log"
    else
      echo "[$(date '+%F %T')] FAIL  ds=$ds variant=$name" | tee -a "$LOG_ROOT/master.log"
      status=1
    fi
    idx=$((idx + 1))
  done

  return "$status"
}

run_dataset() {
  local ds="$1"
  echo "[$(date '+%F %T')] ===== DATASET START: $ds =====" | tee -a "$LOG_ROOT/master.log"

  local total=${#VARIANTS[@]}
  local start=0
  local batch_id=1
  while [ "$start" -lt "$total" ]; do
    local end=$((start + 4))
    if [ "$end" -gt "$total" ]; then
      end=$total
    fi

    local current=()
    local i
    for ((i=start; i<end; i++)); do
      current+=("${VARIANTS[$i]}")
    done

    echo "[$(date '+%F %T')] BATCH $batch_id START ds=$ds (${#current[@]} variants)" | tee -a "$LOG_ROOT/master.log"
    run_batch_of_four "$ds" "${current[@]}"
    echo "[$(date '+%F %T')] BATCH $batch_id END   ds=$ds" | tee -a "$LOG_ROOT/master.log"

    start=$end
    batch_id=$((batch_id + 1))
  done

  echo "[$(date '+%F %T')] ===== DATASET END: $ds =====" | tee -a "$LOG_ROOT/master.log"
}

for ds in "${DATASETS[@]}"; do
  run_dataset "$ds"
done

echo "[$(date '+%F %T')] ALL DATASETS DONE" | tee -a "$LOG_ROOT/master.log"
