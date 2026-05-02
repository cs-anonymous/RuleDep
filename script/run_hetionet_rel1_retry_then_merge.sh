#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="hetionet"
EXPERIMENT="tg_rd__pos_auto_ratio__ri_conf__dn_per_rule_degree__dl1_1e-5"
RUN_SUFFIX="true_rr_full_20260501"
GPU="${GPU:-1}"

EXP_DIR="${ROOT_DIR}/data/${DATASET}/aggregation/${EXPERIMENT}_${RUN_SUFFIX}"
EXPORT_DIR="${ROOT_DIR}/reports/official_query_subset/true_official_per_query_rr"
LOG_DIR="${ROOT_DIR}/logs/true_official_per_query_rr_full"
RETRY_LOG="${LOG_DIR}/${DATASET}_${EXPERIMENT}_relation1_retry.log"
MERGE_LOG="${LOG_DIR}/merge_after_relation1_retry.log"

mkdir -p "${EXP_DIR}" "${EXPORT_DIR}" "${LOG_DIR}"

if [ -f /home/sy/anaconda3/etc/profile.d/conda.sh ]; then
    source /home/sy/anaconda3/etc/profile.d/conda.sh
    conda activate ruledep
else
    export PATH="/home/sy/anaconda3/envs/ruledep/bin:${PATH}"
fi

echo "[$(date '+%F %T')] START hetionet relation=1 retry on GPU=${GPU}" | tee -a "${RETRY_LOG}"
(
    cd "${ROOT_DIR}"
    export CUDA_VISIBLE_DEVICES="${GPU}"
    export EXPERIMENT_DIR="${EXP_DIR}"
    export PYTHONUNBUFFERED=1
    python -u aggregation.py \
        -d "${DATASET}" \
        --rule_file "data/${DATASET}/rules/rule.txt" \
        --relation 1 \
        --synergy --redundancy \
        --type_grouping rd \
        --pos auto_ratio \
        --rule_init_mode conf \
        --dependency_static_norm per_rule_degree \
        --dep_l1_lambda 1e-5 \
        --train_rule_in_dependency_stage \
        --export_per_query_rr_dir "${EXPORT_DIR}" \
        --export_experiment_name "${EXPERIMENT}"
) 2>&1 | tee -a "${RETRY_LOG}"
RETRY_STATUS=${PIPESTATUS[0]}
echo "[$(date '+%F %T')] END retry status=${RETRY_STATUS}" | tee -a "${RETRY_LOG}"

STAGE2_CSV="${EXPORT_DIR}/${DATASET}/${EXPERIMENT}/relation-1-stage2.csv"
if [ "${RETRY_STATUS}" != "0" ] || [ ! -s "${STAGE2_CSV}" ]; then
    echo "[$(date '+%F %T')] retry failed (status=${RETRY_STATUS}, stage2_csv_exists=$( [ -s "${STAGE2_CSV}" ] && echo yes || echo no )); skipping merge" | tee -a "${RETRY_LOG}"
    exit 1
fi

echo "[$(date '+%F %T')] START merge with run_suffix=${RUN_SUFFIX}" | tee -a "${MERGE_LOG}"
python -u "${ROOT_DIR}/script/merge_true_official_per_query_rr.py" \
    --run_suffix "${RUN_SUFFIX}" 2>&1 | tee -a "${MERGE_LOG}"
MERGE_STATUS=${PIPESTATUS[0]}
echo "[$(date '+%F %T')] END merge status=${MERGE_STATUS}" | tee -a "${MERGE_LOG}"
exit "${MERGE_STATUS}"
