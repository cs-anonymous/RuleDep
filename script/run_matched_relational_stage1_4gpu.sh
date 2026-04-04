#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/sy/anaconda3/envs/ruledep/bin/python}"
LOG_DIR="${ROOT_DIR}/logs/matched_relational_stage1"
mkdir -p "${LOG_DIR}"

DATASETS=(
    "KG20C"
    "WN18RR"
    "codex-m"
    "FB15k-237"
)

GPUS=(0 1 2 3)

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "${LOG_DIR}/master.log"
}

run_one() {
    local dataset="$1"
    local gpu="$2"
    local exp_dir="${ROOT_DIR}/data/${dataset}/aggregation/matched_stage1_oldlike"
    local log_path="${LOG_DIR}/${dataset}.log"

    rm -rf "${exp_dir}"
    mkdir -p "${exp_dir}"

    log "START dataset=${dataset} gpu=${gpu} exp_dir=${exp_dir}"
    (
        cd "${ROOT_DIR}"
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export EXPERIMENT_DIR="${exp_dir}"
        export PYTHONUNBUFFERED=1
        "${PYTHON_BIN}" -u "${ROOT_DIR}/aggregation.py" \
            -d "${dataset}" \
            --data_root data \
            --rule_file "data/${dataset}/rules/rule.txt" \
            --relation -1 \
            --device cuda \
            --batch_size 4096 \
            --max_worker_dataloader 0 \
            --lr 0.005 \
            --max_epoch 40 \
            --evaluate_every 1 \
            --early_stopping -1 \
            --pos 5 \
            --multiprocess 2
    ) 2>&1 | tee "${log_path}"
    local status=${PIPESTATUS[0]}
    log "END dataset=${dataset} gpu=${gpu} status=${status} log=${log_path}"
    return "${status}"
}

main() {
    : > "${LOG_DIR}/master.log"
    declare -a pids=()
    local idx
    for idx in "${!DATASETS[@]}"; do
        run_one "${DATASETS[$idx]}" "${GPUS[$idx]}" &
        pids+=("$!")
        sleep 2
    done

    local status=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || status=1
    done
    return "${status}"
}

main "$@"
