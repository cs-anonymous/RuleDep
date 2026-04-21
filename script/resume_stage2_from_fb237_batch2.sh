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

RUN_TAG="${RUN_TAG:-stage2_pipeline_skip_step4_0416}"
LOG_ROOT="$ROOT_DIR/logs/stage2_pipeline/$RUN_TAG"
mkdir -p "$LOG_ROOT"
MASTER_LOG="$LOG_ROOT/master_resume_from_fb237_batch2.log"

DATASETS=(
  "FB15k-237"
  "codex-l"
  "YAGO3-10"
  "hetionet"
)

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

BASE_ARGS=(
  --train_rule_in_dependency_stage
  --type_grouping rd
  --dependency_scale_mode sqrt_active
  --rule_init_mode surprisal
  --pos auto_ratio
  --dep_l1_lambda 1e-5
  --resume_relation_sweep
)

FAIL_COUNT=0

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"
}

run_one_variant() {
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

run_batch() {
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
    run_one_variant "$ds" "${gpus[$idx]}" "$name" "${args[@]}" &
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
}

pending_variants_for_dataset() {
  local ds="$1"
  local pending=()

  for entry in "${VARIANTS[@]}"; do
    local name="${entry%%::*}"
    local final_json="$ROOT_DIR/data/$ds/aggregation/$name/metrics-final.json"

    if [ ! -f "$final_json" ]; then
      # Resume from interruption point: for FB15k-237 only, start at batch 2 (index >= 4)
      if [ "$ds" = "FB15k-237" ]; then
        case "$name" in
          baseline|no-synergy|no-redundancy|sign-constraint)
            continue
            ;;
        esac
      fi
      pending+=("$entry")
    fi
  done

  printf '%s\n' "${pending[@]}"
}

main() {
  log "RESUME START from FB15k-237 batch 2"

  for ds in "${DATASETS[@]}"; do
    local stage1_dir="$ROOT_DIR/data/$ds/rules/stage1_state"
    if [ ! -d "$stage1_dir" ]; then
      log "WARN ds=$ds missing stage1 state: $stage1_dir"
      FAIL_COUNT=$((FAIL_COUNT + 1))
      continue
    fi

    mapfile -t pending < <(pending_variants_for_dataset "$ds")
    if [ "${#pending[@]}" -eq 0 ]; then
      log "SKIP ds=$ds all variants already done"
      continue
    fi

    log "===== RESUME DATASET START: $ds pending=${#pending[@]} ====="

    local start=0
    local batch_id=1
    local total="${#pending[@]}"
    while [ "$start" -lt "$total" ]; do
      local end=$((start + 4))
      if [ "$end" -gt "$total" ]; then
        end=$total
      fi

      local cur=()
      local i
      for ((i=start; i<end; i++)); do
        cur+=("${pending[$i]}")
      done

      log "BATCH $batch_id START ds=$ds variants=${#cur[@]}"
      run_batch "$ds" "${cur[@]}"
      log "BATCH $batch_id END   ds=$ds"

      start=$end
      batch_id=$((batch_id + 1))
    done

    log "===== RESUME DATASET END: $ds ====="
  done

  if [ "$FAIL_COUNT" -eq 0 ]; then
    log "RESUME DONE: all pending runs succeeded"
  else
    log "RESUME DONE WITH FAILURES: fail_count=$FAIL_COUNT"
  fi
}

main "$@"
