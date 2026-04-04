#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN_DEFAULT="/home/sy/anaconda3/envs/ruledep/bin/python"
LOG_DIR="${ROOT_DIR}/logs/aggregation_rd_ablation_queue"
SUMMARY_TXT="${LOG_DIR}/summary.txt"
SUMMARY_JSON="${LOG_DIR}/summary.json"

DATASETS=(
    "FB15k-237"
    "KG20C"
    "WN18RR"
    "YAGO3-10"
    "codex-l"
    "codex-m"
    "hetionet"
)

EXPERIMENTS=(
    "structural_rd_surprisal_filtered"
    "structural_rd_dep_scale_filtered"
    "structural_rd_global_ratio_filtered"
    "structural_rd_rule_mask_filtered"
)

GPUS=(0 1 2 3)

mkdir -p "${LOG_DIR}"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "${LOG_DIR}/master.log"
}

is_old_running() {
    pgrep -f "aggregation_old.py" >/dev/null 2>&1 && return 0
    pgrep -f "script/run_old.sh" >/dev/null 2>&1 && return 0
    pgrep -f "script/wait_and_run_old.sh" >/dev/null 2>&1 && return 0
    pgrep -f "run_old_canonical_after_yago.sh" >/dev/null 2>&1 && return 0
    return 1
}

variant_extra_args() {
    local experiment="$1"
    case "${experiment}" in
        structural_rd_surprisal_filtered)
            echo "--rule_init_mode surprisal"
            ;;
        structural_rd_dep_scale_filtered)
            echo "--dependency_scale_mode sqrt_active"
            ;;
        structural_rd_global_ratio_filtered)
            echo "--type_grouping rd"
            ;;
        structural_rd_rule_mask_filtered)
            echo "--dependency_mask_low_rule_weight"
            ;;
        *)
            return 1
            ;;
    esac
}

run_one_experiment() {
    local dataset="$1"
    local experiment="$2"
    local gpu="$3"
    local exp_dir="${ROOT_DIR}/data/${dataset}/aggregation/${experiment}"
    local log_path="${LOG_DIR}/${dataset}_${experiment}.log"
    local status_path="${LOG_DIR}/${dataset}_${experiment}.status"
    local python_bin
    local start_ts
    local end_ts
    local elapsed

    start_ts=$(date +%s)
    log "START dataset=${dataset} experiment=${experiment} gpu=${gpu} exp_dir=${exp_dir}"

    rm -rf "${exp_dir}"
    mkdir -p "${exp_dir}"

    local extra_args
    extra_args="$(variant_extra_args "${experiment}")"
    python_bin="${PYTHON_BIN:-${PYTHON_BIN_DEFAULT}}"

    set +e
    (
        cd "${ROOT_DIR}"
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export EXPERIMENT_DIR="${exp_dir}"
        export PYTHONUNBUFFERED=1
        # shellcheck disable=SC2086
        "${python_bin}" "${ROOT_DIR}/aggregation.py" \
            -d "${dataset}" \
            --data_root data \
            --rule_file "data/${dataset}/rules/rule.txt" \
            --relation -1 \
            --device cuda \
            --batch_size 4096 \
            --max_worker_dataloader 0 \
            --lr 0.01,0.005,0.001 \
            --max_epoch 60 \
            --evaluate_every 4,2,1 \
            --pos auto_sqrt \
            --synergy \
            --redundancy \
            --train_rule_in_dependency_stage \
            --type_grouping none \
            --multiprocess 2 \
            ${extra_args}
    ) 2>&1 | tee "${log_path}"
    local status=${PIPESTATUS[0]}
    set -e

    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))
    cat > "${status_path}" <<EOF
dataset=${dataset}
experiment=${experiment}
gpu=${gpu}
status=${status}
start_ts=${start_ts}
end_ts=${end_ts}
elapsed_seconds=${elapsed}
log_path=${log_path}
experiment_dir=${exp_dir}
EOF
    log "END dataset=${dataset} experiment=${experiment} gpu=${gpu} status=${status} elapsed_seconds=${elapsed}"
    return "${status}"
}

write_summary() {
    local total_seconds="$1"
    local rows_json="$2"
    cat > "${SUMMARY_JSON}" <<EOF
{
  "total_seconds": ${total_seconds},
  "total_hours": $(awk "BEGIN { printf \"%.6f\", ${total_seconds}/3600 }"),
  "rows": ${rows_json}
}
EOF
}

main() {
    : > "${LOG_DIR}/master.log"
    rm -f "${LOG_DIR}"/*.status "${SUMMARY_TXT}" "${SUMMARY_JSON}"

    log "Waiting for old aggregation jobs to finish before starting RD ablation queue"
    while is_old_running; do
        sleep 60
    done
    log "Detected old aggregation finished; starting RD ablation queue"

    local queue_start_ts
    queue_start_ts=$(date +%s)

    for dataset in "${DATASETS[@]}"; do
        log "DATASET_BEGIN dataset=${dataset}"
        declare -a pids=()
        for idx in "${!EXPERIMENTS[@]}"; do
            run_one_experiment "${dataset}" "${EXPERIMENTS[${idx}]}" "${GPUS[${idx}]}" &
            pids+=("$!")
            sleep 3
        done
        for pid in "${pids[@]}"; do
            wait "${pid}" || true
        done
        log "DATASET_END dataset=${dataset}"
    done

    local queue_end_ts
    local total_seconds
    local rows_json
    queue_end_ts=$(date +%s)
    total_seconds=$((queue_end_ts - queue_start_ts))
    rows_json=$(
        /home/sy/anaconda3/envs/ruledep/bin/python - <<'PY'
import json
from pathlib import Path

log_dir = Path("/home/sy/RuleDep/logs/aggregation_rd_ablation_queue")
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
    write_summary "${total_seconds}" "${rows_json}"
    {
        echo "total_seconds=${total_seconds}"
        awk "BEGIN { printf \"total_hours=%.6f\n\", ${total_seconds}/3600 }"
        echo "runs:"
        for dataset in "${DATASETS[@]}"; do
            for experiment in "${EXPERIMENTS[@]}"; do
                local status_path="${LOG_DIR}/${dataset}_${experiment}.status"
                if [ -f "${status_path}" ]; then
                    awk -F= '
                        BEGIN { dataset=""; experiment=""; gpu=""; status=""; elapsed="" }
                        $1=="dataset" { dataset=$2 }
                        $1=="experiment" { experiment=$2 }
                        $1=="gpu" { gpu=$2 }
                        $1=="status" { status=$2 }
                        $1=="elapsed_seconds" { elapsed=$2 }
                        END { printf "%s %s gpu=%s status=%s elapsed_seconds=%s\n", dataset, experiment, gpu, status, elapsed }
                    ' "${status_path}"
                fi
            done
        done
    } | tee "${SUMMARY_TXT}"
    log "All RD ablation runs finished; summary_json=${SUMMARY_JSON} summary_txt=${SUMMARY_TXT}"
}

main "$@"
