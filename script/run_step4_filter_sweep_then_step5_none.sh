#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/step4_filter_sweep_then_step5_none"
PYTHON_BIN="${PYTHON_BIN:-/home/sy/anaconda3/envs/ruledep/bin/python}"
FILTER_JOBS="${FILTER_JOBS:-24}"
HETIONET_FILTER_JOBS="${HETIONET_FILTER_JOBS:-4}"
START_DATASET="${START_DATASET:-}"
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

canonical_running() {
    pgrep -f "aggregation_old.py .*aggregation/canonical" >/dev/null 2>&1 && return 0
    pgrep -f "bash /home/sy/RuleDep/script/run_old.sh .*" >/dev/null 2>&1 && return 0
    return 1
}

filter_target_split() {
    case "$1" in
        hetionet) echo "valid" ;;
        *) echo "train" ;;
    esac
}

filter_min_supp() {
    case "$1" in
        KG20C|WN18RR) echo "2" ;;
        *) echo "5" ;;
    esac
}

filter_jobs_for_dataset() {
    case "$1" in
        hetionet) echo "${HETIONET_FILTER_JOBS}" ;;
        *) echo "${FILTER_JOBS}" ;;
    esac
}

run_filter_sweep() {
    local dataset="$1"
    local target_split
    local min_supp
    local filter_jobs
    target_split="$(filter_target_split "${dataset}")"
    min_supp="$(filter_min_supp "${dataset}")"
    filter_jobs="$(filter_jobs_for_dataset "${dataset}")"
    local log_path="${LOG_DIR}/${dataset}_filter_sweep.log"
    local rules_dir="${ROOT_DIR}/data/${dataset}/rules"
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
        log "SKIP filter dataset=${dataset} reason=all_expected_outputs_exist"
        return 0
    fi

    log "START filter dataset=${dataset} target_split=${target_split} jobs=${filter_jobs} variants=default_9"
    (
        cd "${ROOT_DIR}"
        "${PYTHON_BIN}" -u filter_dependency.py \
            -d "${dataset}" \
            --target_split "${target_split}" \
            --min_supp "${min_supp}" \
            --variant_sweep default_9 \
            --jobs "${filter_jobs}"
    ) 2>&1 | tee "${log_path}"
    local status=${PIPESTATUS[0]}
    log "END filter dataset=${dataset} status=${status} log=${log_path}"
    return "${status}"
}

run_step5_variant() {
    local dataset="$1"
    local gpu="$2"
    local ranking_mode="$3"
    local multiplier="$4"
    local suffix="${ranking_mode}_k${multiplier}"
    local name="structural_none_${suffix}"
    local log_path="${LOG_DIR}/${dataset}_${name}.log"
    local exp_dir="${ROOT_DIR}/data/${dataset}/aggregation/${name}"
    local synergy_file="data/${dataset}/rules/synergy_filtered_${suffix}.txt"
    local redundancy_file="data/${dataset}/rules/redundancy_filtered_${suffix}.txt"
    local -a extra_args=()

    if [ -f "${exp_dir}/metrics-final.json" ]; then
        log "SKIP step5 dataset=${dataset} exp=${name} reason=metrics-final-exists"
        return 0
    fi
    if compgen -G "${exp_dir}/metric-*.json" >/dev/null 2>&1; then
        extra_args+=(--resume_relation_sweep)
    fi

    log "START step5 dataset=${dataset} exp=${name} gpu=${gpu}"
    (
        cd "${ROOT_DIR}"
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export EXPERIMENT_DIR="${exp_dir}"
        export PYTHONUNBUFFERED=1
        "${PYTHON_BIN}" -u aggregation.py \
            -d "${dataset}" \
            --rule_file "data/${dataset}/rules/rule.txt" \
            --relation -1 \
            --multiprocess 2 \
            --train_rule_in_dependency_stage \
            "${extra_args[@]}" \
            --synergy \
            --redundancy \
            --type_grouping none \
            --synergy_file "${synergy_file}" \
            --redundancy_file "${redundancy_file}"
    ) 2>&1 | tee "${log_path}"
    local status=${PIPESTATUS[0]}
    log "END step5 dataset=${dataset} exp=${name} status=${status} log=${log_path}"
    return "${status}"
}

run_step5_batches() {
    local dataset="$1"
    local -a variants=(
        "lift 1"
        "lift 4"
        "ratio 1"
        "ratio 2"
        "ratio 4"
        "mix 1"
        "mix 2"
        "mix 4"
    )
    local -a pids=()
    local batch_start=0

    while [ "${batch_start}" -lt "${#variants[@]}" ]; do
        pids=()
        for idx in "${!GPUS[@]}"; do
            local variant_idx=$((batch_start + idx))
            [ "${variant_idx}" -lt "${#variants[@]}" ] || break
            read -r ranking_mode multiplier <<< "${variants[${variant_idx}]}"
            run_step5_variant "${dataset}" "${GPUS[${idx}]}" "${ranking_mode}" "${multiplier}" &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do
            wait "${pid}"
        done
        batch_start=$((batch_start + ${#GPUS[@]}))
    done
}

: > "${LOG_DIR}/master.log"
log "Queue started; waiting for canonical runs to finish"
while canonical_running; do
    sleep 60
done
log "Canonical runs finished; starting filter+step5 sweep"

started=0
for dataset in "${DATASETS[@]}"; do
    if [ -n "${START_DATASET}" ] && [ "${started}" -eq 0 ]; then
        if [ "${dataset}" != "${START_DATASET}" ]; then
            continue
        fi
        started=1
    fi
    log "DATASET START ${dataset}"
    run_filter_sweep "${dataset}"
    run_step5_batches "${dataset}"
    log "DATASET END ${dataset}"
done

log "All filter+step5 sweep runs finished"
