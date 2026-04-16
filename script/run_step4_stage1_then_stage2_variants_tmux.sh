#!/usr/bin/env bash
set -u -o pipefail

ROOT_DIR="/home/sy/RuleDep"
cd "$ROOT_DIR"

if [ -f /home/sy/anaconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /home/sy/anaconda3/etc/profile.d/conda.sh
  conda activate ruledep
else
  export PATH="/home/sy/anaconda3/envs/ruledep/bin:${PATH}"
fi

RUN_TAG="${RUN_TAG:-stage2_pipeline_0415}"
LOG_ROOT="$ROOT_DIR/logs/stage2_pipeline/$RUN_TAG"
mkdir -p "$LOG_ROOT"
MASTER_LOG="$LOG_ROOT/master.log"

# Stage4+Stage1 only for these two datasets first
PREP_DATASETS=(
  "YAGO3-10"
  "hetionet"
)

# Stage2 variants for all datasets afterwards (in order)
ALL_DATASETS=(
  "KG20C"
  "codex-m"
  "WN18RR"
  "FB15k-237"
  "codex-l"
  "YAGO3-10"
  "hetionet"
)

# 12 variants requested
VARIANTS=(
  "baseline::--synergy --redundancy"
  "no-synergy::--redundancy"
  "no-redundancy::--synergy"
  "sign-constraint::--synergy --redundancy --sign_constraint_dependency"
  "lift-init::--synergy --redundancy --init_dep_with_lift"
  "auto-sqrt::--synergy --redundancy --pos auto_sqrt"
  "R2D3::--synergy --redundancy --type_grouping r2d3"
  "R3D6::--synergy --redundancy --type_grouping r3d6"
  "no-dep-scale::--synergy --redundancy --dependency_scale_mode none"
  "dep_l1_1e-4::--synergy --redundancy --dep_l1_lambda 1e-4"
  "dep_l2_1e-5::--synergy --redundancy --dep_l1_lambda 0 --dep_l2_lambda 1e-5"
  "dep_l2_1e-4::--synergy --redundancy --dep_l1_lambda 0 --dep_l2_lambda 1e-4"
)

# Default baseline args for stage2
BASE_ARGS=(
  --train_rule_in_dependency_stage
  --type_grouping rd
  --dependency_scale_mode sqrt_active
  --rule_init_mode surprisal
  --pos auto_ratio
  --dep_l1_lambda 1e-5
)

FAIL_COUNT=0

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"
}

# Run command and continue even if it fails
run_and_record() {
  local label="$1"
  shift
  local log_file="$1"
  shift

  log "START $label"
  (
    "$@"
  ) >"$log_file" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    log "DONE  $label"
  else
    log "FAIL  $label (rc=$rc, log=$log_file)"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
  return 0
}

run_stage4_stage1_for_dataset() {
  local ds="$1"
  local ds_log_dir="$LOG_ROOT/prep_$ds"
  mkdir -p "$ds_log_dir"

  local step4_log="$ds_log_dir/${ds}_step4.log"
  local step1_log="$ds_log_dir/${ds}_step5_stage1.log"

  # Step4
  run_and_record "prep step4 ds=$ds" "$step4_log" bash "$ROOT_DIR/step4_dependency.sh" "$ds"

  # Step5 Stage1 only
  local exp_dir="$ROOT_DIR/data/$ds/aggregation/stage1"
  local stage1_state_dir="$ROOT_DIR/data/$ds/rules/stage1_state"
  mkdir -p "$exp_dir" "$stage1_state_dir"

  local n_proc_per_gpu=2
  if [ "$ds" = "hetionet" ]; then
    n_proc_per_gpu=1
  fi

  run_and_record "prep step5-stage1 ds=$ds" "$step1_log" \
    bash -lc "cd '$ROOT_DIR' && export EXPERIMENT_DIR='$exp_dir' && export PYTHONUNBUFFERED=1 && python -u '$ROOT_DIR/aggregation.py' \
      -d '$ds' \
      --rule_file 'data/$ds/rules/rule.txt' \
      --relation -1 \
      --gpus 4 \
      --n_proc '$n_proc_per_gpu' \
      --train_rule_in_dependency_stage \
      --synergy \
      --redundancy \
      --type_grouping rd \
      --dependency_scale_mode sqrt_active \
      --rule_init_mode surprisal \
      --pos auto_ratio \
      --dep_l1_lambda 1e-5 \
      --stage1_only \
      --save_stage1_state_dir '$stage1_state_dir'"
}

run_one_stage2_variant() {
  local ds="$1"
  local gpu="$2"
  local variant_name="$3"
  shift 3
  local extra_args=("$@")

  local stage1_dir="$ROOT_DIR/data/$ds/rules/stage1_state"
  local exp_dir="$ROOT_DIR/data/$ds/aggregation/$variant_name"
  local log_file="$LOG_ROOT/${ds}_${variant_name}.log"

  mkdir -p "$exp_dir"

  local n_proc_per_gpu=2
  if [ "$ds" = "hetionet" ]; then
    n_proc_per_gpu=1
  fi

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
      --n_proc "$n_proc_per_gpu" \
      --load_stage1_state_dir "$stage1_dir" \
      "${BASE_ARGS[@]}" \
      "${extra_args[@]}"
  ) >"$log_file" 2>&1
}

run_stage2_batch_of_four() {
  local ds="$1"
  shift
  local batch_entries=("$@")
  local gpus=(0 1 2 3)
  local pids=()
  local names=()

  local idx=0
  for entry in "${batch_entries[@]}"; do
    local name="${entry%%::*}"
    local arg_string="${entry#*::}"
    local args=()
    if [ -n "$arg_string" ]; then
      read -r -a args <<< "$arg_string"
    fi

    log "START stage2 ds=$ds variant=$name gpu=${gpus[$idx]}"
    run_one_stage2_variant "$ds" "${gpus[$idx]}" "$name" "${args[@]}" &
    pids+=("$!")
    names+=("$name")
    idx=$((idx + 1))
  done

  idx=0
  for pid in "${pids[@]}"; do
    if wait "$pid"; then
      log "DONE  stage2 ds=$ds variant=${names[$idx]}"
    else
      log "FAIL  stage2 ds=$ds variant=${names[$idx]} (log=$LOG_ROOT/${ds}_${names[$idx]}.log)"
      FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    idx=$((idx + 1))
  done
  return 0
}

run_stage2_for_dataset() {
  local ds="$1"
  local stage1_dir="$ROOT_DIR/data/$ds/rules/stage1_state"

  if [ ! -d "$stage1_dir" ]; then
    log "WARN  ds=$ds stage1_state missing at $stage1_dir, continue to next dataset"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    return 0
  fi

  log "===== STAGE2 DATASET START: $ds ====="

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

    log "BATCH $batch_id START ds=$ds variants=${#current[@]}"
    run_stage2_batch_of_four "$ds" "${current[@]}"
    log "BATCH $batch_id END   ds=$ds"

    start=$end
    batch_id=$((batch_id + 1))
  done

  log "===== STAGE2 DATASET END: $ds ====="
}

main() {
  log "PIPELINE START run_tag=$RUN_TAG"

  # Phase A: prepare YAGO3-10 + hetionet (step4 + stage1)
  for ds in "${PREP_DATASETS[@]}"; do
    log "===== PREP START: $ds ====="
    run_stage4_stage1_for_dataset "$ds"
    log "===== PREP END: $ds ====="
  done

  # Phase B: stage2 variants on all datasets (continue-on-failure)
  for ds in "${ALL_DATASETS[@]}"; do
    run_stage2_for_dataset "$ds"
  done

  if [ "$FAIL_COUNT" -eq 0 ]; then
    log "PIPELINE DONE: all steps succeeded"
  else
    log "PIPELINE DONE WITH FAILURES: fail_count=$FAIL_COUNT"
  fi

  # Always exit 0 so orchestration does not stop prematurely if called by wrapper.
  exit 0
}

main "$@"
