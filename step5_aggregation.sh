#!/usr/bin/env bash
set -euo pipefail

[ "$#" -ge 1 ] || { echo "Usage: $0 <dataset> [multiprocess]" >&2; exit 1; }

dataset="$1"
multiprocess="${2:-2}"
max_parallel_configs="${MAX_PARALLEL_CONFIGS:-4}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs/aggregation_structural/${dataset}"
EXP_ROOT="${ROOT_DIR}/data/${dataset}/aggregation"
MASTER_LOG="${LOG_DIR}/master.log"

mkdir -p "${LOG_DIR}" "${EXP_ROOT}"

if [ -f /home/sy/anaconda3/etc/profile.d/conda.sh ]; then
    source /home/sy/anaconda3/etc/profile.d/conda.sh
    conda activate ruledep
else
    export PATH="/home/sy/anaconda3/envs/ruledep/bin:${PATH}"
fi

echo "======================================"
echo "Step 5: Aggregation for ${dataset}"
echo "======================================"

run_config() {
    local gpu="$1"
    local name="$2"
    shift 2
    local log_path="${LOG_DIR}/${dataset}_${name}.log"
    local exp_dir="${EXP_ROOT}/${name}"

    mkdir -p "${exp_dir}"
    echo "[$(date '+%F %T')] START ${name} gpu=${gpu} args=$*" | tee -a "${MASTER_LOG}"
    (
        cd "${ROOT_DIR}"
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export EXPERIMENT_DIR="${exp_dir}"
        export PYTHONUNBUFFERED=1
        python -u aggregation.py \
            -d "${dataset}" \
            --rule_file "data/${dataset}/rules/rule.txt" \
            --relation -1 \
            --multiprocess "${multiprocess}" \
            --model LinearAggregator \
            --train_rule_in_dependency_stage \
            "$@"
    ) 2>&1 | tee "${log_path}"
    local status=${PIPESTATUS[0]}
    echo "[$(date '+%F %T')] END ${name} status=${status} log=${log_path}" | tee -a "${MASTER_LOG}"
    return "${status}"
}

run_batch() {
    local -a configs=("$@")
    local -a gpus=(0 1 2 3)
    local -a pids=()
    local status=0

    for idx in "${!configs[@]}"; do
        local entry="${configs[$idx]}"
        local name="${entry%%::*}"
        local arg_string="${entry#*::}"
        local -a args=()
        if [ -n "${arg_string}" ]; then
            read -r -a args <<< "${arg_string}"
        fi
        run_config "${gpus[$idx]}" "${name}" "${args[@]}" &
        pids[$idx]=$!
    done

    for pid in "${pids[@]}"; do
        wait "${pid}" || status=1
    done

    return "${status}"
}

run_batched_configs() {
    local batch_size="$1"
    shift
    local -a configs=("$@")
    local total=${#configs[@]}
    local start=0
    local status=0

    while [ "${start}" -lt "${total}" ]; do
        local -a current_batch=()
        local end=$((start + batch_size))
        if [ "${end}" -gt "${total}" ]; then
            end="${total}"
        fi

        for ((idx = start; idx < end; idx++)); do
            current_batch+=("${configs[$idx]}")
        done

        run_batch "${current_batch[@]}" || status=1
        [ "${status}" -eq 0 ] || return "${status}"
        start="${end}"
    done

    return 0
}

batch1=(
    "structural_rd::--synergy --redundancy --rule_grouping none --dependency_grouping none"
    "structural_r2d3::--synergy --redundancy --rule_grouping r2 --dependency_grouping d3"
    "structural_r3d6::--synergy --redundancy --rule_grouping r3 --dependency_grouping d6"
    "synergy::--synergy"
)

batch2=(
    "redundancy::--redundancy"
    "sign_constraint_dependency::--synergy --redundancy --sign_constraint_dependency"
    "init_dep_with_lift::--synergy --redundancy --init_dep_with_lift"
    "pos_auto_ratio::--synergy --redundancy --pos auto_ratio"
)

run_batched_configs "${max_parallel_configs}" "${batch1[@]}"
run_batched_configs "${max_parallel_configs}" "${batch2[@]}"

echo "Step 5 finished for ${dataset}"
