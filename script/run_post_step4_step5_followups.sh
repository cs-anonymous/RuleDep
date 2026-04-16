#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/post_step4_step5_followups"
PYTHON_BIN="${PYTHON_BIN:-/home/sy/anaconda3/envs/ruledep/bin/python}"
STEP4_THREADS="${STEP4_THREADS:-$(nproc)}"
MULTIPROCESS="${MULTIPROCESS:-2}"
WAIT_INTERVAL="${WAIT_INTERVAL:-10}"
DATASETS_OVERRIDE="${DATASETS_OVERRIDE:-}"
DATASETS=("FB15k-237" "KG20C" "WN18RR" "YAGO3-10" "codex-l" "codex-m" "hetionet")
GPUS=(0 1 2 3)

mkdir -p "${LOG_DIR}"

if [ -f /home/sy/anaconda3/etc/profile.d/conda.sh ]; then
    source /home/sy/anaconda3/etc/profile.d/conda.sh
    conda activate ruledep
else
    export PATH="/home/sy/anaconda3/envs/ruledep/bin:${PATH}"
fi

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "${LOG_DIR}/master.log"
}

if [ -n "${DATASETS_OVERRIDE}" ]; then
    IFS=',' read -r -a DATASETS <<< "${DATASETS_OVERRIDE}"
fi

current_sweep_running() {
    if tmux has-session -t ruledep_step4_sweep 2>/dev/null; then
        return 0
    fi
    pgrep -f "run_step4_filter_sweep_then_step5_none.sh" >/dev/null 2>&1 && return 0
    return 1
}

step4_threads_for_dataset() {
    case "$1" in
        hetionet) echo "8" ;;
        *) echo "${STEP4_THREADS}" ;;
    esac
}

run_step4_variant() {
    local dataset="$1"
    local variant="$2"
    local step4_threads
    local dep_dir="${ROOT_DIR}/data/${dataset}/dependency_variants/${variant}"
    local log_path="${LOG_DIR}/${dataset}_${variant}_step4.log"
    mkdir -p "${dep_dir}"
    step4_threads="$(step4_threads_for_dataset "${dataset}")"

    log "START step4 dataset=${dataset} variant=${variant}"
    (
        cd "${ROOT_DIR}"
        case "${variant}" in
            top500_rule_combo)
                DEPENDENCY_DIR="data/${dataset}/dependency_variants/${variant}" \
                RUN_DEPLEARN_LOG="data/${dataset}/dependency_variants/${variant}/run_deplearn.log" \
                    TOP_K=500 \
                ./step4_dependency.sh "${dataset}" "${step4_threads}"
                ;;
            unified_formula)
                DEPENDENCY_DIR="data/${dataset}/dependency_variants/${variant}" \
                RUN_DEPLEARN_LOG="data/${dataset}/dependency_variants/${variant}/run_deplearn.log" \
                DEPENDENCY_FORMULA_MODE=unified \
                ./step4_dependency.sh "${dataset}" "${step4_threads}"
                ;;
            *)
                echo "Unknown step4 variant: ${variant}" >&2
                exit 1
                ;;
        esac
    ) 2>&1 | tee "${log_path}"
    local status=${PIPESTATUS[0]}
    log "END step4 dataset=${dataset} variant=${variant} status=${status} log=${log_path}"
    return "${status}"
}

run_step5_config() {
    local gpu="$1"
    local dataset="$2"
    local exp_name="$3"
    shift 3
    local log_path="${LOG_DIR}/${dataset}_${exp_name}.log"
    local exp_dir="${ROOT_DIR}/data/${dataset}/aggregation/${exp_name}"

    log "START step5 dataset=${dataset} exp=${exp_name} gpu=${gpu}"
    (
        cd "${ROOT_DIR}"
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export EXPERIMENT_DIR="${exp_dir}"
        export PYTHONUNBUFFERED=1
        "${PYTHON_BIN}" -u aggregation.py \
            -d "${dataset}" \
            --rule_file "data/${dataset}/rules/rule.txt" \
            --relation -1 \
            --multiprocess "${MULTIPROCESS}" \
            --train_rule_in_dependency_stage \
            "$@"
    ) 2>&1 | tee "${log_path}"
    local status=${PIPESTATUS[0]}
    log "END step5 dataset=${dataset} exp=${exp_name} status=${status} log=${log_path}"
    return "${status}"
}

resolve_best_combination_json() {
    "${PYTHON_BIN}" "${ROOT_DIR}/script/resolve_best_combination.py" "$1"
}

best_combination_args() {
    local dataset="$1"
    local json_payload
    json_payload="$(resolve_best_combination_json "${dataset}")"
    DATASET_JSON="${json_payload}" "${PYTHON_BIN}" - <<'PY'
import json, os, shlex
payload = json.loads(os.environ["DATASET_JSON"])
args = ["--synergy", "--redundancy", "--type_grouping", payload["type_grouping"]]
if payload.get("use_surprisal_init"):
    args.extend(["--rule_init_mode", "surprisal"])
if payload.get("use_pos_auto_ratio"):
    args.extend(["--pos", "auto_ratio"])
if payload.get("synergy_file"):
    args.extend(["--synergy_file", payload["synergy_file"]])
if payload.get("redundancy_file"):
    args.extend(["--redundancy_file", payload["redundancy_file"]])
print(" ".join(shlex.quote(a) for a in args))
PY
}

run_step5_followups_for_dataset() {
    local dataset="$1"
    local top500_synergy="data/${dataset}/dependency_variants/top500_rule_combo/synergy_filtered.txt"
    local top500_redundancy="data/${dataset}/dependency_variants/top500_rule_combo/redundancy_filtered.txt"
    local unified_synergy="data/${dataset}/dependency_variants/unified_formula/synergy_filtered.txt"
    local unified_redundancy="data/${dataset}/dependency_variants/unified_formula/redundancy_filtered.txt"
    local best_args
    best_args="$(best_combination_args "${dataset}")"

    local -a configs=(
        "top500_rule_combo::--synergy --redundancy --type_grouping none --synergy_file ${top500_synergy} --redundancy_file ${top500_redundancy}"
        "unified_formula::--synergy --redundancy --type_grouping none --synergy_file ${unified_synergy} --redundancy_file ${unified_redundancy}"
        "dep_scale_surprisal_init::--synergy --redundancy --type_grouping none --dependency_scale_mode sqrt_active --rule_init_mode surprisal"
        "best_combination::${best_args}"
    )

    local -a pids=()
    local idx=0
    for entry in "${configs[@]}"; do
        local name="${entry%%::*}"
        local arg_string="${entry#*::}"
        local -a args=()
        read -r -a args <<< "${arg_string}"
        run_step5_config "${GPUS[$idx]}" "${dataset}" "${name}" "${args[@]}" &
        pids+=($!)
        idx=$((idx + 1))
    done

    local status=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || status=1
    done
    return "${status}"
}

: > "${LOG_DIR}/master.log"
log "Queue started; waiting for current filter sweep to finish"
while current_sweep_running; do
    sleep "${WAIT_INTERVAL}"
done
log "Current filter sweep finished; starting follow-up runs"

for dataset in "${DATASETS[@]}"; do
    log "DATASET START ${dataset}"
    run_step4_variant "${dataset}" "top500_rule_combo"
    run_step4_variant "${dataset}" "unified_formula"
    run_step5_followups_for_dataset "${dataset}"
    log "DATASET END ${dataset}"
done

log "All follow-up runs finished"
