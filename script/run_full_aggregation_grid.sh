#!/usr/bin/env bash
set -euo pipefail

[ "$#" -ge 1 ] || { echo "Usage: $0 <dataset|all> [multiprocess]" >&2; exit 1; }

dataset_arg="$1"
multiprocess="${2:-2}"
max_parallel_configs="${MAX_PARALLEL_CONFIGS:-4}"
run_tag="${RUN_TAG:-}"
gpu_ids_raw="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a AVAILABLE_GPUS <<< "${gpu_ids_raw}"

[ "${#AVAILABLE_GPUS[@]}" -gt 0 ] || { echo "GPU_IDS must contain at least one GPU index" >&2; exit 1; }
for gpu in "${AVAILABLE_GPUS[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "Invalid GPU index in GPU_IDS: ${gpu}" >&2; exit 1; }
done
[[ "${max_parallel_configs}" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_PARALLEL_CONFIGS must be a positive integer" >&2; exit 1; }
if [ "${max_parallel_configs}" -gt "${#AVAILABLE_GPUS[@]}" ]; then
    echo "Capping MAX_PARALLEL_CONFIGS=${max_parallel_configs} to ${#AVAILABLE_GPUS[@]} available GPU(s)" >&2
    max_parallel_configs="${#AVAILABLE_GPUS[@]}"
fi

ALL_DATASETS=(
    "KG20C"
    "codex-m"
    "WN18RR"
    "FB15k-237"
    "codex-l"
    "YAGO3-10"
)

if [ -n "${run_tag}" ] && [ "${run_tag#_}" = "${run_tag}" ]; then
    run_tag="_${run_tag}"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f /home/sy/anaconda3/etc/profile.d/conda.sh ]; then
    source /home/sy/anaconda3/etc/profile.d/conda.sh
    conda activate ruledep
else
    export PATH="/home/sy/anaconda3/envs/ruledep/bin:${PATH}"
fi

run_config() {
    local dataset="$1"
    local log_dir="$2"
    local exp_root="$3"
    local master_log="$4"
    local gpu="$5"
    local name="$6"
    shift 6
    local log_path="${log_dir}/${dataset}_${name}.log"
    local exp_dir="${exp_root}/${name}${run_tag}"

    if [ -n "${run_tag}" ]; then
        log_path="${log_dir}/${dataset}_${name}${run_tag}.log"
    fi

    mkdir -p "${exp_dir}"
    echo "[$(date '+%F %T')] START ds=${dataset} ${name} gpu=${gpu} args=$*" | tee -a "${master_log}"
    (
        cd "${ROOT_DIR}"
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export EXPERIMENT_DIR="${exp_dir}"
        export PYTHONUNBUFFERED=1
        python -u src/ruledep/aggregation.py \
            -d "${dataset}" \
            --rule_file "data/${dataset}/rules/rule.txt" \
            --relation -1 \
            --multiprocess "${multiprocess}" \
            --train_rule_in_dependency_stage \
            --resume_relation_sweep \
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
    local -a pids=()
    local status=0

    echo "[$(date '+%F %T')] BATCH START ds=${dataset} batch=${batch_id} size=${#configs[@]}" | tee -a "${master_log}"

    for idx in "${!configs[@]}"; do
        local entry="${configs[$idx]}"
        local name="${entry%%::*}"
        local arg_string="${entry#*::}"
        local -a args=()
        if [ -n "${arg_string}" ]; then
            read -r -a args <<< "${arg_string}"
        fi
        run_config "${dataset}" "${log_dir}" "${exp_root}" "${master_log}" "${AVAILABLE_GPUS[$idx]}" "${name}" "${args[@]}" &
        pids[$idx]=$!
    done

    for pid in "${pids[@]}"; do
        wait "${pid}" || status=1
    done

    echo "[$(date '+%F %T')] BATCH END   ds=${dataset} batch=${batch_id} status=${status}" | tee -a "${master_log}"

    return "${status}"
}

run_batched_configs() {
    local dataset="$1"
    local log_dir="$2"
    local exp_root="$3"
    local master_log="$4"
    local batch_size="$5"
    shift 5
    local -a configs=("$@")
    local total=${#configs[@]}
    local start=0
    local status=0
    local batch_id=1

    while [ "${start}" -lt "${total}" ]; do
        local -a current_batch=()
        local end=$((start + batch_size))
        if [ "${end}" -gt "${total}" ]; then
            end="${total}"
        fi

        for ((idx = start; idx < end; idx++)); do
            current_batch+=("${configs[$idx]}")
        done

        run_batch "${dataset}" "${log_dir}" "${exp_root}" "${master_log}" "${batch_id}" "${current_batch[@]}" || status=1
        [ "${status}" -eq 0 ] || return "${status}"
        start="${end}"
        batch_id=$((batch_id + 1))
    done

    return 0
}

build_48_configs() {
    local -a type_groupings=("rd" "r2d3" "r3d6")
    local -a pos_modes=("auto_ratio" "auto_sqrt")
    local -a rule_inits=("conf" "surprisal")
    local -a static_norms=("none" "per_rule_degree")
    local -a l1_values=("0" "1e-5")

    local -a out=()
    local tg pos ri dn l1
    for tg in "${type_groupings[@]}"; do
        for pos in "${pos_modes[@]}"; do
            for ri in "${rule_inits[@]}"; do
                for dn in "${static_norms[@]}"; do
                    for l1 in "${l1_values[@]}"; do
                        local name="tg_${tg}__pos_${pos}__ri_${ri}__dn_${dn}__dl1_${l1}"
                        local arg_string="--synergy --redundancy --type_grouping ${tg} --pos ${pos} --rule_init_mode ${ri} --dependency_static_norm ${dn} --dep_l1_lambda ${l1}"
                        out+=("${name}::${arg_string}")
                    done
                done
            done
        done
    done

    printf '%s\n' "${out[@]}"
}

run_for_dataset() {
    local dataset="$1"
    local log_dir="${ROOT_DIR}/logs/aggregation_structural/${dataset}"
    local exp_root="${ROOT_DIR}/data/${dataset}/aggregation"
    local master_log="${log_dir}/master${run_tag}.log"

    mkdir -p "${log_dir}" "${exp_root}"

    echo "======================================" | tee -a "${master_log}"
    echo "Step 5: Aggregation for ${dataset}" | tee -a "${master_log}"
    echo "Config space: 48 (=3*2*2*2*2), batch size=${max_parallel_configs}, multiprocess=${multiprocess}, GPUs=${gpu_ids_raw}" | tee -a "${master_log}"
    echo "======================================" | tee -a "${master_log}"

    mapfile -t configs < <(build_48_configs)
    run_batched_configs "${dataset}" "${log_dir}" "${exp_root}" "${master_log}" "${max_parallel_configs}" "${configs[@]}"
    echo "Step 5 finished for ${dataset}" | tee -a "${master_log}"
}

if [ "${dataset_arg}" = "all" ]; then
    for ds in "${ALL_DATASETS[@]}"; do
        run_for_dataset "${ds}"
    done
else
    run_for_dataset "${dataset_arg}"
fi
