#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/aggregation_old_canonical"
PYTHON_BIN_DEFAULT="/home/sy/anaconda3/envs/ruledep/bin/python"
SUMMARY_JSON="${LOG_DIR}/summary.json"
SUMMARY_TXT="${LOG_DIR}/summary.txt"
DATASETS=(
    "KG20C"
    "codex-m"
    "WN18RR"
    "FB15k-237"
    "codex-l"
    "YAGO3-10"
)
GPUS=(0 1 2 3)

mkdir -p "${LOG_DIR}"

is_yago_running() {
    pgrep -f "bash ./step5_aggregation.sh YAGO3-10 2" >/dev/null 2>&1 && return 0
    pgrep -f "python -u aggregation.py -d YAGO3-10" >/dev/null 2>&1 && return 0
    return 1
}

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "${LOG_DIR}/master.log"
}

write_summary() {
    local total_seconds="$1"
    local dataset_rows_json="$2"
    cat > "${SUMMARY_JSON}" <<EOF
{
  "total_seconds": ${total_seconds},
  "total_hours": $(awk "BEGIN { printf \"%.6f\", ${total_seconds}/3600 }"),
  "datasets": ${dataset_rows_json}
}
EOF
}

run_one_dataset() {
    local gpu="$1"
    local dataset="$2"
    local exp_dir="${ROOT_DIR}/data/${dataset}/aggregation/canonical"
    local log_path="${LOG_DIR}/${dataset}.log"
    local status_path="${LOG_DIR}/${dataset}.status"
    local start_ts
    local end_ts
    local elapsed
    start_ts=$(date +%s)
    log "START dataset=${dataset} gpu=${gpu} exp_dir=${exp_dir}"
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
        bash "${ROOT_DIR}/script/run_old.sh" "${dataset}"
    ) 2>&1 | tee "${log_path}"
    local status=${PIPESTATUS[0]}
    set -e
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))
    cat > "${status_path}" <<EOF
dataset=${dataset}
gpu=${gpu}
status=${status}
start_ts=${start_ts}
end_ts=${end_ts}
elapsed_seconds=${elapsed}
log_path=${log_path}
experiment_dir=${exp_dir}
EOF
    log "END dataset=${dataset} gpu=${gpu} status=${status} elapsed_seconds=${elapsed} log=${log_path}"
    return "${status}"
}

schedule_queue() {
    local next_idx=0
    local total="${#DATASETS[@]}"
    declare -A pid_to_gpu=()
    declare -A pid_to_dataset=()
    declare -A gpu_busy=()
    local queue_start_ts
    local queue_end_ts
    local total_seconds
    local dataset_rows_json
    queue_start_ts=$(date +%s)

    for gpu in "${GPUS[@]}"; do
        gpu_busy["${gpu}"]=0
    done

    while [ "${next_idx}" -lt "${total}" ] || [ "${#pid_to_gpu[@]}" -gt 0 ]; do
        while [ "${next_idx}" -lt "${total}" ]; do
            local gpu=""
            for candidate in "${GPUS[@]}"; do
                if [ "${gpu_busy[${candidate}]}" -eq 0 ]; then
                    gpu="${candidate}"
                    break
                fi
            done
            if [ -z "${gpu}" ]; then
                break
            fi
            local dataset="${DATASETS[${next_idx}]}"
            run_one_dataset "${gpu}" "${dataset}" &
            local pid=$!
            pid_to_gpu["${pid}"]="${gpu}"
            pid_to_dataset["${pid}"]="${dataset}"
            gpu_busy["${gpu}"]=1
            next_idx=$((next_idx + 1))
        done

        local finished_pid=""
        while [ -z "${finished_pid}" ]; do
            for pid in "${!pid_to_gpu[@]}"; do
                if ! kill -0 "${pid}" 2>/dev/null; then
                    wait "${pid}" || true
                    finished_pid="${pid}"
                    break
                fi
            done
            if [ -z "${finished_pid}" ]; then
                sleep 10
            fi
        done

        gpu_busy["${pid_to_gpu[${finished_pid}]}"]=0
        unset "pid_to_gpu[${finished_pid}]"
        unset "pid_to_dataset[${finished_pid}]"
    done

    queue_end_ts=$(date +%s)
    total_seconds=$((queue_end_ts - queue_start_ts))
    dataset_rows_json=$(
        /home/sy/anaconda3/envs/ruledep/bin/python - <<'PY'
import json
from pathlib import Path

log_dir = Path("/home/sy/RuleDep/logs/aggregation_old_canonical")
rows = []
for path in sorted(log_dir.glob("*.status")):
    row = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k in {"gpu", "status", "start_ts", "end_ts", "elapsed_seconds"}:
            try:
                row[k] = int(v)
            except Exception:
                row[k] = v
        else:
            row[k] = v
    rows.append(row)
print(json.dumps(rows, ensure_ascii=True))
PY
    )
    write_summary "${total_seconds}" "${dataset_rows_json}"
    {
        echo "total_seconds=${total_seconds}"
        awk "BEGIN { printf \"total_hours=%.6f\n\", ${total_seconds}/3600 }"
        echo "datasets:"
        for dataset in "${DATASETS[@]}"; do
            if [ -f "${LOG_DIR}/${dataset}.status" ]; then
                awk -F= '
                    BEGIN { dataset=""; status=""; elapsed=""; gpu="" }
                    $1=="dataset" { dataset=$2 }
                    $1=="gpu" { gpu=$2 }
                    $1=="status" { status=$2 }
                    $1=="elapsed_seconds" { elapsed=$2 }
                    END { printf "%s gpu=%s status=%s elapsed_seconds=%s\n", dataset, gpu, status, elapsed }
                ' "${LOG_DIR}/${dataset}.status"
            fi
        done
    } | tee "${SUMMARY_TXT}"
}

: > "${LOG_DIR}/master.log"
rm -f "${LOG_DIR}"/*.status "${SUMMARY_JSON}" "${SUMMARY_TXT}"
if [ "${SKIP_WAIT:-0}" != "1" ]; then
    log "Waiting for YAGO3-10 step5 aggregation jobs to finish before starting canonical old aggregation queue"
    while is_yago_running; do
        sleep 60
    done
    log "Detected YAGO3-10 step5 aggregation finished; starting canonical old aggregation queue"
else
    log "SKIP_WAIT=1, starting canonical old aggregation queue immediately"
fi
schedule_queue
log "All canonical old aggregation runs finished; summary_json=${SUMMARY_JSON} summary_txt=${SUMMARY_TXT}"
