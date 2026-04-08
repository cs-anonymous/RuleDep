#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/targeted_followups_and_hetionet"
PYTHON_BIN="${PYTHON_BIN:-/home/sy/anaconda3/envs/ruledep/bin/python}"
MULTIPROCESS="${MULTIPROCESS:-2}"
HETIONET_FILTER_JOBS="${HETIONET_FILTER_JOBS:-4}"
GPUS=(2 3)
NON_HETIONET_DATASETS=("FB15k-237" "KG20C" "WN18RR" "YAGO3-10" "codex-l" "codex-m")
HETIONET_VARIANTS=(
    "lift 1"
    "lift 4"
    "ratio 1"
    "ratio 2"
    "ratio 4"
    "mix 1"
    "mix 2"
    "mix 4"
)

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

run_aggregation() {
    local gpu="$1"
    local dataset="$2"
    local exp_name="$3"
    shift 3
    local log_path="${LOG_DIR}/${dataset}_${exp_name}.log"
    local exp_dir="${ROOT_DIR}/data/${dataset}/aggregation/${exp_name}"
    local -a extra_args=()

    if [ -f "${exp_dir}/metrics-final.json" ]; then
        log "SKIP step5 dataset=${dataset} exp=${exp_name} reason=metrics-final-exists"
        return 0
    fi
    if compgen -G "${exp_dir}/metric-*.json" >/dev/null 2>&1; then
        extra_args+=(--resume_relation_sweep)
    fi

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
            "${extra_args[@]}" \
            "$@"
    ) 2>&1 | tee "${log_path}"
    local status=${PIPESTATUS[0]}
    log "END step5 dataset=${dataset} exp=${exp_name} status=${status} log=${log_path}"
    return "${status}"
}

best_combination_args() {
    local dataset="$1"
    local json_payload
    json_payload="$("${PYTHON_BIN}" "${ROOT_DIR}/script/resolve_best_combination.py" "${dataset}")"
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

run_non_hetionet_dataset() {
    local dataset="$1"
    local best_args
    local -a args_best=()
    best_args="$(best_combination_args "${dataset}")"
    read -r -a args_best <<< "${best_args}"

    log "DATASET START ${dataset}"
    run_aggregation "${GPUS[0]}" "${dataset}" "dep_scale_surprisal_init" \
        --synergy --redundancy --type_grouping none \
        --dependency_scale_mode sqrt_active --rule_init_mode surprisal &
    local pid1=$!

    run_aggregation "${GPUS[1]}" "${dataset}" "best_combination" "${args_best[@]}" &
    local pid2=$!

    local status=0
    wait "${pid1}" || status=1
    wait "${pid2}" || status=1
    log "DATASET END ${dataset} status=${status}"
    return "${status}"
}

run_hetionet_filter() {
    local log_path="${LOG_DIR}/hetionet_filter_sweep.log"
    local rules_dir="${ROOT_DIR}/data/hetionet/rules"
    local -a expected_outputs=(
        "${rules_dir}/synergy_filtered_lift_k1.txt"
        "${rules_dir}/synergy_filtered_lift_k2.txt"
        "${rules_dir}/synergy_filtered_lift_k4.txt"
        "${rules_dir}/synergy_filtered_ratio_k1.txt"
        "${rules_dir}/synergy_filtered_ratio_k2.txt"
        "${rules_dir}/synergy_filtered_ratio_k4.txt"
        "${rules_dir}/synergy_filtered_mix_k1.txt"
        "${rules_dir}/synergy_filtered_mix_k2.txt"
        "${rules_dir}/synergy_filtered_mix_k4.txt"
        "${rules_dir}/redundancy_filtered_lift_k1.txt"
        "${rules_dir}/redundancy_filtered_lift_k2.txt"
        "${rules_dir}/redundancy_filtered_lift_k4.txt"
        "${rules_dir}/redundancy_filtered_ratio_k1.txt"
        "${rules_dir}/redundancy_filtered_ratio_k2.txt"
        "${rules_dir}/redundancy_filtered_ratio_k4.txt"
        "${rules_dir}/redundancy_filtered_mix_k1.txt"
        "${rules_dir}/redundancy_filtered_mix_k2.txt"
        "${rules_dir}/redundancy_filtered_mix_k4.txt"
    )
    local missing=0
    for path in "${expected_outputs[@]}"; do
        if [ ! -f "${path}" ]; then
            missing=1
            break
        fi
    done
    if [ "${missing}" -eq 0 ]; then
        log "SKIP hetionet filter reason=all_expected_outputs_exist"
        return 0
    fi

    log "START hetionet filter jobs=${HETIONET_FILTER_JOBS} target_split=valid variants=default_9"
    (
        cd "${ROOT_DIR}"
        "${PYTHON_BIN}" -u filter_dependency.py \
            -d hetionet \
            --target_split valid \
            --min_supp 5 \
            --variant_sweep default_9 \
            --jobs "${HETIONET_FILTER_JOBS}"
    ) 2>&1 | tee "${log_path}"
    local status=${PIPESTATUS[0]}
    log "END hetionet filter status=${status} log=${log_path}"
    return "${status}"
}

run_hetionet_variants() {
    local batch_start=0
    local -a pids=()
    while [ "${batch_start}" -lt "${#HETIONET_VARIANTS[@]}" ]; do
        pids=()
        for idx in "${!GPUS[@]}"; do
            local variant_idx=$((batch_start + idx))
            [ "${variant_idx}" -lt "${#HETIONET_VARIANTS[@]}" ] || break
            local ranking_mode multiplier suffix exp_name
            read -r ranking_mode multiplier <<< "${HETIONET_VARIANTS[${variant_idx}]}"
            suffix="${ranking_mode}_k${multiplier}"
            exp_name="structural_none_${suffix}"
            run_aggregation "${GPUS[${idx}]}" "hetionet" "${exp_name}" \
                --synergy --redundancy --type_grouping none \
                --synergy_file "data/hetionet/rules/synergy_filtered_${suffix}.txt" \
                --redundancy_file "data/hetionet/rules/redundancy_filtered_${suffix}.txt" &
            pids+=($!)
        done
        local status=0
        for pid in "${pids[@]}"; do
            wait "${pid}" || status=1
        done
        [ "${status}" -eq 0 ] || return "${status}"
        batch_start=$((batch_start + ${#GPUS[@]}))
    done
}

: > "${LOG_DIR}/master.log"
log "Queue started"

overall_status=0
for dataset in "${NON_HETIONET_DATASETS[@]}"; do
    run_non_hetionet_dataset "${dataset}" || overall_status=1
done

log "HETIONET START"
run_hetionet_filter || overall_status=1
run_hetionet_variants || overall_status=1
log "HETIONET END status=${overall_status}"

log "All targeted runs finished status=${overall_status}"
exit "${overall_status}"
