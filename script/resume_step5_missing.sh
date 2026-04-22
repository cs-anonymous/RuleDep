#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <dataset> [dataset ...] [multiprocess]" >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
max_parallel_configs="${MAX_PARALLEL_CONFIGS:-2}"

last_arg="${!#}"
if [[ "${last_arg}" =~ ^[0-9]+$ ]]; then
    multiprocess="${last_arg}"
    dataset_count=$(($# - 1))
else
    multiprocess="${MULTIPROCESS:-2}"
    dataset_count="$#"
fi

datasets=()
for ((i = 1; i <= dataset_count; i++)); do
    datasets+=("${!i}")
done

if [ -f /home/sy/anaconda3/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /home/sy/anaconda3/etc/profile.d/conda.sh
    conda activate ruledep
else
    export PATH="/home/sy/anaconda3/envs/ruledep/bin:${PATH}"
fi

build_48_configs() {
    local -a type_groupings=("rd" "r2d3" "r3d6")
    local -a pos_modes=("auto_ratio" "auto_sqrt")
    local -a rule_inits=("conf" "surprisal")
    local -a static_norms=("none" "per_rule_degree")
    local -a l1_values=("0" "1e-5")

    local tg pos ri dn l1
    for tg in "${type_groupings[@]}"; do
        for pos in "${pos_modes[@]}"; do
            for ri in "${rule_inits[@]}"; do
                for dn in "${static_norms[@]}"; do
                    for l1 in "${l1_values[@]}"; do
                        printf 'tg_%s__pos_%s__ri_%s__dn_%s__dl1_%s::--synergy --redundancy --type_grouping %s --pos %s --rule_init_mode %s --dependency_static_norm %s --dep_l1_lambda %s\n' \
                            "${tg}" "${pos}" "${ri}" "${dn}" "${l1}" \
                            "${tg}" "${pos}" "${ri}" "${dn}" "${l1}"
                    done
                done
            done
        done
    done
}

run_config() {
    local dataset="$1"
    local log_dir="$2"
    local exp_root="$3"
    local master_log="$4"
    local gpu="$5"
    local name="$6"
    shift 6

    local exp_dir="${exp_root}/${name}"
    local log_path="${log_dir}/${dataset}_${name}_resume.log"

    if [ -f "${exp_dir}/metrics-final.json" ]; then
        echo "[$(date '+%F %T')] SKIP ds=${dataset} ${name} metrics-final.json exists" | tee -a "${master_log}"
        return 0
    fi

    mkdir -p "${exp_dir}"
    echo "[$(date '+%F %T')] START ds=${dataset} ${name} gpu=${gpu} args=$*" | tee -a "${master_log}"
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
            --train_rule_in_dependency_stage \
            "$@"
    ) 2>&1 | tee "${log_path}"
    local status=${PIPESTATUS[0]}
    echo "[$(date '+%F %T')] END ds=${dataset} ${name} status=${status} log=${log_path}" | tee -a "${master_log}"
    return "${status}"
}

run_batch() {
    local dataset="$1"
    local log_dir="$2"
    local exp_root="$3"
    local master_log="$4"
    local batch_id="$5"
    shift 5

    local -a configs=("$@")
    local -a gpus=(0 1 2 3)
    local -a pids=()
    local status=0

    echo "[$(date '+%F %T')] BATCH START ds=${dataset} batch=${batch_id} size=${#configs[@]}" | tee -a "${master_log}"
    for idx in "${!configs[@]}"; do
        local entry="${configs[$idx]}"
        local name="${entry%%::*}"
        local arg_string="${entry#*::}"
        local -a args=()
        read -r -a args <<< "${arg_string}"
        run_config "${dataset}" "${log_dir}" "${exp_root}" "${master_log}" "${gpus[$idx]}" "${name}" "${args[@]}" &
        pids[$idx]=$!
    done

    for pid in "${pids[@]}"; do
        wait "${pid}" || status=1
    done
    echo "[$(date '+%F %T')] BATCH END   ds=${dataset} batch=${batch_id} status=${status}" | tee -a "${master_log}"
    return "${status}"
}

run_for_dataset() {
    local dataset="$1"
    local log_dir="${ROOT_DIR}/logs/aggregation_structural/${dataset}"
    local exp_root="${ROOT_DIR}/data/${dataset}/aggregation"
    local master_log="${log_dir}/master_resume.log"
    local -a missing=()

    mkdir -p "${log_dir}" "${exp_root}"

    local entry name
    while IFS= read -r entry; do
        name="${entry%%::*}"
        if [ ! -f "${exp_root}/${name}/metrics-final.json" ]; then
            missing+=("${entry}")
        fi
    done < <(build_48_configs)

    echo "======================================" | tee -a "${master_log}"
    echo "Step 5 resume: ${dataset}" | tee -a "${master_log}"
    echo "Missing configs: ${#missing[@]}/48, batch size=${max_parallel_configs}, multiprocess=${multiprocess}" | tee -a "${master_log}"
    echo "======================================" | tee -a "${master_log}"

    if [ "${#missing[@]}" -eq 0 ]; then
        echo "[$(date '+%F %T')] Nothing to do for ${dataset}" | tee -a "${master_log}"
        return 0
    fi

    local start=0
    local batch_id=1
    while [ "${start}" -lt "${#missing[@]}" ]; do
        local -a current_batch=()
        local end=$((start + max_parallel_configs))
        if [ "${end}" -gt "${#missing[@]}" ]; then
            end="${#missing[@]}"
        fi

        for ((idx = start; idx < end; idx++)); do
            current_batch+=("${missing[$idx]}")
        done

        run_batch "${dataset}" "${log_dir}" "${exp_root}" "${master_log}" "${batch_id}" "${current_batch[@]}"
        start="${end}"
        batch_id=$((batch_id + 1))
    done

    echo "[$(date '+%F %T')] Step 5 resume finished for ${dataset}" | tee -a "${master_log}"
}

for dataset in "${datasets[@]}"; do
    run_for_dataset "${dataset}"
done
