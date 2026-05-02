#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPORT_DIR="${EXPORT_DIR:-${ROOT_DIR}/reports/official_query_subset/true_official_per_query_rr}"
RUN_SUFFIX="${RUN_SUFFIX:-true_rr_rerun_20260501}"
GPU="${GPU:-0}"
MULTIPROCESS="${MULTIPROCESS:-2}"
STAGE1_ONLY="${STAGE1_ONLY:-1}"

if [ -f /home/sy/anaconda3/etc/profile.d/conda.sh ]; then
    source /home/sy/anaconda3/etc/profile.d/conda.sh
    conda activate ruledep
else
    export PATH="/home/sy/anaconda3/envs/ruledep/bin:${PATH}"
fi

run_one() {
    local dataset="$1"
    local experiment="$2"
    shift 2
    local exp_dir="${ROOT_DIR}/data/${dataset}/aggregation/${experiment}_${RUN_SUFFIX}"
    local log_dir="${ROOT_DIR}/logs/true_official_per_query_rr"
    local stage_args=()
    mkdir -p "${exp_dir}" "${log_dir}" "${EXPORT_DIR}"
    if [ "${STAGE1_ONLY}" = "1" ]; then
        stage_args+=(--stage1_only)
    else
        stage_args+=(--train_rule_in_dependency_stage)
    fi

    echo "[$(date '+%F %T')] START dataset=${dataset} experiment=${experiment} stage1_only=${STAGE1_ONLY}" | tee -a "${log_dir}/master.log"
    (
        cd "${ROOT_DIR}"
        export CUDA_VISIBLE_DEVICES="${GPU}"
        export EXPERIMENT_DIR="${exp_dir}"
        export PYTHONUNBUFFERED=1
        python -u aggregation.py \
            -d "${dataset}" \
            --rule_file "data/${dataset}/rules/rule.txt" \
            --relation -1 \
            --multiprocess "${MULTIPROCESS}" \
            --resume_relation_sweep \
            "${stage_args[@]}" \
            --export_per_query_rr_dir "${EXPORT_DIR}" \
            --export_experiment_name "${experiment}" \
            "$@"
    ) 2>&1 | tee "${log_dir}/${dataset}_${experiment}.log"
    local status=${PIPESTATUS[0]}
    echo "[$(date '+%F %T')] END dataset=${dataset} experiment=${experiment} status=${status}" | tee -a "${log_dir}/master.log"
    return "${status}"
}

run_one "KG20C" "tg_r2d3__pos_auto_ratio__ri_conf__dn_per_rule_degree__dl1_1e-5" \
    --synergy --redundancy --type_grouping r2d3 --pos auto_ratio --rule_init_mode conf --dependency_static_norm per_rule_degree --dep_l1_lambda 1e-5

run_one "WN18RR" "best_combination_dep_l1_regularization_dep_fix_topk8_0412" \
    --synergy --redundancy --type_grouping rd --pos auto_sqrt --rule_init_mode conf --dependency_static_norm none --dep_l1_lambda 1e-5 \
    --synergy_file data/WN18RR/rules/synergy_filtered_ratio_k1.txt \
    --redundancy_file data/WN18RR/rules/redundancy_filtered_ratio_k1.txt

run_one "YAGO3-10" "tg_r3d6__pos_auto_sqrt__ri_surprisal__dn_none__dl1_1e-5" \
    --synergy --redundancy --type_grouping r3d6 --pos auto_sqrt --rule_init_mode surprisal --dependency_static_norm none --dep_l1_lambda 1e-5

run_one "codex-l" "tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5" \
    --synergy --redundancy --type_grouping rd --pos auto_sqrt --rule_init_mode conf --dependency_static_norm per_rule_degree --dep_l1_lambda 1e-5

run_one "codex-m" "tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5" \
    --synergy --redundancy --type_grouping rd --pos auto_sqrt --rule_init_mode conf --dependency_static_norm per_rule_degree --dep_l1_lambda 1e-5

run_one "FB15k-237" "tg_r2d3__pos_auto_ratio__ri_conf__dn_none__dl1_1e-5" \
    --synergy --redundancy --type_grouping r2d3 --pos auto_ratio --rule_init_mode conf --dependency_static_norm none --dep_l1_lambda 1e-5

run_one "hetionet" "tg_rd__pos_auto_ratio__ri_conf__dn_per_rule_degree__dl1_1e-5" \
    --synergy --redundancy --type_grouping rd --pos auto_ratio --rule_init_mode conf --dependency_static_norm per_rule_degree --dep_l1_lambda 1e-5
