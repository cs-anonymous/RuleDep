#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${ROOT_DIR}/logs/rerun_step2_4"
MASTER_LOG="${LOG_ROOT}/hetionet_step4_to_step5_master.log"
mkdir -p "${LOG_ROOT}"

if [ -f /home/sy/anaconda3/etc/profile.d/conda.sh ]; then
    source /home/sy/anaconda3/etc/profile.d/conda.sh
    conda activate ruledep
else
    export PATH="/home/sy/anaconda3/envs/ruledep/bin:${PATH}"
fi

run_stage() {
    local stage="$1"
    shift
    local log_path="${LOG_ROOT}/hetionet_${stage}_restart.log"
    echo "[$(date '+%F %T')] START hetionet ${stage}" | tee -a "${MASTER_LOG}"
    "$@" 2>&1 | tee "${log_path}"
    local status=${PIPESTATUS[0]}
    echo "[$(date '+%F %T')] END hetionet ${stage} status=${status} log=${log_path}" | tee -a "${MASTER_LOG}"
    return "${status}"
}

echo "[$(date '+%F %T')] Queue launcher started for hetionet step4-5." | tee -a "${MASTER_LOG}"
cd "${ROOT_DIR}"
rm -f data/hetionet/rules/synergy_filtered.txt data/hetionet/rules/redundancy_filtered.txt
run_stage step4 ./step4_dependency.sh hetionet
run_stage step5 ./step5_aggregation.sh hetionet 2

echo "[$(date '+%F %T')] Queue finished with status=0" | tee -a "${MASTER_LOG}"
