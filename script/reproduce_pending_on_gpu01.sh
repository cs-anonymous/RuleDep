#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SUFFIX="${RUN_SUFFIX:-main_table_per_query_rr_20260809}"
MULTIPROCESS="${MULTIPROCESS:-2}"
EXPORT_DIR="${EXPORT_DIR:-${ROOT_DIR}/reports/official_query_subset/true_official_per_query_rr/${RUN_SUFFIX}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/main_table_per_query_rr_${RUN_SUFFIX}}"
STATUS_LOG="${LOG_DIR}/gpu01_reassignment.log"

if [ -f /home/sy/anaconda3/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /home/sy/anaconda3/etc/profile.d/conda.sh
    conda activate ruledep
else
    export PATH="/home/sy/anaconda3/envs/ruledep/bin:${PATH}"
fi

mkdir -p "${EXPORT_DIR}" "${LOG_DIR}"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "${STATUS_LOG}"
}

run_aggregation() {
    local dataset="$1"
    local gpu="$2"
    local experiment="$3"
    local log_name="$4"
    shift 4

    local experiment_dir="${ROOT_DIR}/data/${dataset}/aggregation/${experiment}_${RUN_SUFFIX}"
    mkdir -p "${experiment_dir}"
    log "START dataset=${dataset} gpu=${gpu} output=${experiment_dir}"
    (
        cd "${ROOT_DIR}"
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export EXPERIMENT_DIR="${experiment_dir}"
        export PYTHONUNBUFFERED=1
        python -u src/ruledep/aggregation.py \
            -d "${dataset}" \
            --rule_file "data/${dataset}/rules/rule.txt" \
            --relation -1 \
            --multiprocess "${MULTIPROCESS}" \
            --resume_relation_sweep \
            --train_rule_in_dependency_stage \
            --export_per_query_rr_dir "${EXPORT_DIR}" \
            --export_experiment_name "${experiment}" \
            "$@"
    ) 2>&1 | tee "${LOG_DIR}/${log_name}.log"
    local status=${PIPESTATUS[0]}
    log "END dataset=${dataset} gpu=${gpu} status=${status}"
    return "${status}"
}

run_gpu0_queue() {
    run_aggregation WN18RR 0 structural_rd WN18RR_structural_rd_gpu0_early \
        --synergy --redundancy --type_grouping rd

    run_aggregation KG20C 0 \
        tg_r2d3__pos_auto_ratio__ri_conf__dn_per_rule_degree__dl1_1e-5 \
        KG20C_tg_r2d3_gpu0_early \
        --synergy --redundancy --type_grouping r2d3 \
        --pos auto_ratio --rule_init_mode conf \
        --dependency_static_norm per_rule_degree --dep_l1_lambda 1e-5
}

run_gpu1_queue() {
    run_aggregation codex-m 1 \
        tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5 \
        codex-m_tg_rd_gpu1_early \
        --synergy --redundancy --type_grouping rd \
        --pos auto_sqrt --rule_init_mode conf \
        --dependency_static_norm per_rule_degree --dep_l1_lambda 1e-5
}

log "Starting reassigned queues: GPU0=WN18RR->KG20C GPU1=codex-m"
run_gpu0_queue &
gpu0_pid=$!
run_gpu1_queue &
gpu1_pid=$!

status=0
wait "${gpu0_pid}" || status=1
wait "${gpu1_pid}" || status=1
log "Reassigned queues complete status=${status}"
exit "${status}"
