#!/usr/bin/env bash
set -euo pipefail

[ "$#" -ge 1 ] || { echo "Usage: $0 <dataset|all> [multiprocess]" >&2; exit 1; }

dataset_arg="$1"
multiprocess="${2:-2}"
gpu_id="${GPU_ID:-0}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS=(KG20C WN18RR codex-m FB15k-237 codex-l YAGO3-10)

if [ -f /home/sy/anaconda3/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /home/sy/anaconda3/etc/profile.d/conda.sh
    conda activate ruledep
else
    export PATH="/home/sy/anaconda3/envs/ruledep/bin:${PATH}"
fi

is_supported_dataset() {
    local requested="$1"
    local dataset
    for dataset in "${DATASETS[@]}"; do
        [ "${requested}" = "${dataset}" ] && return 0
    done
    return 1
}

require_inputs() {
    local dataset="$1"
    local required=(
        "data/${dataset}/train.txt"
        "data/${dataset}/valid.txt"
        "data/${dataset}/test.txt"
        "data/${dataset}/rules/rule.txt"
        "data/${dataset}/rules/synergy_filtered.txt"
        "data/${dataset}/rules/redundancy_filtered.txt"
        "data/${dataset}/application/processed_sp_train.pkl"
        "data/${dataset}/application/processed_sp_valid.pkl"
        "data/${dataset}/application/processed_sp_test.pkl"
        "data/${dataset}/application/processed_po_train.pkl"
        "data/${dataset}/application/processed_po_valid.pkl"
        "data/${dataset}/application/processed_po_test.pkl"
        "data/${dataset}/datasets"
    )
    local path
    for path in "${required[@]}"; do
        if [ ! -e "${ROOT_DIR}/${path}" ]; then
            echo "Missing Step 1-4 artifact: ${path}" >&2
            echo "Download the complete RuleDepData release before running Step 5." >&2
            return 1
        fi
    done
}

config_for_dataset() {
    case "$1" in
        KG20C)
            CONFIG_NAME="tg_r2d3__pos_auto_ratio__ri_conf__dn_per_rule_degree__dl1_1e-5"
            CONFIG_ARGS=(--synergy --redundancy --type_grouping r2d3 --pos auto_ratio --rule_init_mode conf --dependency_static_norm per_rule_degree --dep_l1_lambda 1e-5)
            ;;
        WN18RR)
            CONFIG_NAME="structural_rd"
            CONFIG_ARGS=(--synergy --redundancy --type_grouping rd --pos auto_sqrt --rule_init_mode conf --dependency_static_norm none --dep_l1_lambda 0)
            ;;
        codex-m)
            CONFIG_NAME="tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5"
            CONFIG_ARGS=(--synergy --redundancy --type_grouping rd --pos auto_sqrt --rule_init_mode conf --dependency_static_norm per_rule_degree --dep_l1_lambda 1e-5)
            ;;
        FB15k-237)
            CONFIG_NAME="tg_r2d3__pos_auto_ratio__ri_conf__dn_none__dl1_1e-5"
            CONFIG_ARGS=(--synergy --redundancy --type_grouping r2d3 --pos auto_ratio --rule_init_mode conf --dependency_static_norm none --dep_l1_lambda 1e-5)
            ;;
        codex-l)
            CONFIG_NAME="tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5"
            CONFIG_ARGS=(--synergy --redundancy --type_grouping rd --pos auto_sqrt --rule_init_mode conf --dependency_static_norm per_rule_degree --dep_l1_lambda 1e-5)
            ;;
        YAGO3-10)
            CONFIG_NAME="tg_r3d6__pos_auto_sqrt__ri_surprisal__dn_none__dl1_1e-5"
            CONFIG_ARGS=(--synergy --redundancy --type_grouping r3d6 --pos auto_sqrt --rule_init_mode surprisal --dependency_static_norm none --dep_l1_lambda 1e-5)
            ;;
    esac
}

run_dataset() {
    local dataset="$1"
    local output_dir="${ROOT_DIR}/data/${dataset}/aggregation/reproduction"
    local log_dir="${ROOT_DIR}/logs/aggregation_reproduction/${dataset}"
    local log_path="${log_dir}/step5.log"

    require_inputs "${dataset}"
    config_for_dataset "${dataset}"
    mkdir -p "${output_dir}" "${log_dir}"

    echo "[$(date '+%F %T')] START dataset=${dataset} config=${CONFIG_NAME} gpu=${gpu_id} output=${output_dir}" | tee -a "${log_path}"
    (
        cd "${ROOT_DIR}"
        export CUDA_VISIBLE_DEVICES="${gpu_id}"
        export EXPERIMENT_DIR="${output_dir}"
        export PYTHONUNBUFFERED=1
        python -u src/ruledep/aggregation.py \
            -d "${dataset}" \
            --rule_file "data/${dataset}/rules/rule.txt" \
            --relation -1 \
            --multiprocess "${multiprocess}" \
            --resume_relation_sweep \
            --train_rule_in_dependency_stage \
            "${CONFIG_ARGS[@]}"
    ) 2>&1 | tee -a "${log_path}"
    local status=${PIPESTATUS[0]}
    echo "[$(date '+%F %T')] END dataset=${dataset} status=${status}" | tee -a "${log_path}"
    return "${status}"
}

if [ "${dataset_arg}" = "all" ]; then
    for dataset in "${DATASETS[@]}"; do
        run_dataset "${dataset}"
    done
elif is_supported_dataset "${dataset_arg}"; then
    run_dataset "${dataset_arg}"
else
    echo "Unsupported dataset: ${dataset_arg}" >&2
    echo "Supported datasets: ${DATASETS[*]}" >&2
    exit 1
fi
