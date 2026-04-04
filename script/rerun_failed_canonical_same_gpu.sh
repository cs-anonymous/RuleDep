#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/aggregation_old_canonical_rerun"
PYTHON_BIN_DEFAULT="/home/sy/anaconda3/envs/ruledep/bin/python"

mkdir -p "${LOG_DIR}"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "${LOG_DIR}/master.log"
}

wait_for_dataset() {
    local dataset="$1"
    while pgrep -af "aggregation_old.py -d ${dataset}|script/run_old.sh ${dataset}" >/dev/null 2>&1; do
        sleep 30
    done
}

run_rerun() {
    local wait_dataset="$1"
    local rerun_dataset="$2"
    local gpu="$3"
    local exp_dir="${ROOT_DIR}/data/${rerun_dataset}/aggregation/canonical"
    local log_path="${LOG_DIR}/${rerun_dataset}.log"
    local status_path="${LOG_DIR}/${rerun_dataset}.status"
    local start_ts end_ts elapsed status

    log "WAIT dataset=${wait_dataset} before rerun dataset=${rerun_dataset} gpu=${gpu}"
    wait_for_dataset "${wait_dataset}"

    start_ts=$(date +%s)
    log "START rerun dataset=${rerun_dataset} gpu=${gpu} exp_dir=${exp_dir}"
    rm -rf "${exp_dir}"
    mkdir -p "${exp_dir}"
    set +e
    (
        cd "${ROOT_DIR}"
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN_DEFAULT}}"
        export EXPERIMENT_DIR="${exp_dir}"
        export PYTHONUNBUFFERED=1
        export MAX_EPOCH_HPO="${MAX_EPOCH_HPO:-40}"
        export MAX_WORKER_DATALOADER="${MAX_WORKER_DATALOADER:-0}"
        bash "${ROOT_DIR}/script/run_old.sh" "${rerun_dataset}"
    ) 2>&1 | tee "${log_path}"
    status=${PIPESTATUS[0]}
    set -e
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))
    cat > "${status_path}" <<EOF
dataset=${rerun_dataset}
wait_dataset=${wait_dataset}
gpu=${gpu}
status=${status}
start_ts=${start_ts}
end_ts=${end_ts}
elapsed_seconds=${elapsed}
log_path=${log_path}
experiment_dir=${exp_dir}
EOF
    log "END rerun dataset=${rerun_dataset} gpu=${gpu} status=${status} elapsed_seconds=${elapsed} log=${log_path}"
}

: > "${LOG_DIR}/master.log"
log "Scheduling reruns: codex-m on GPU0 after KG20C, FB15k-237 on GPU2 after WN18RR"

run_rerun "KG20C" "codex-m" "0" &
pid_codex=$!
run_rerun "WN18RR" "FB15k-237" "2" &
pid_fb=$!

wait "${pid_codex}"
wait "${pid_fb}"

log "All targeted canonical reruns finished"
