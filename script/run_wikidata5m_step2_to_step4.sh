#!/usr/bin/env bash
set -euo pipefail

export PATH="/home/sy/anaconda3/envs/ruledep/bin:${PATH}"

log_root="logs/rerun_step2_4"
master_log="${log_root}/wikidata5m_step2_to_step4_master.log"
mkdir -p "${log_root}"

run_stage() {
  local dataset="$1"
  local stage="$2"
  shift 2
  local log_path="${log_root}/${dataset}_${stage}.log"

  echo "[$(date '+%F %T')] START ${dataset} ${stage}" | tee -a "${master_log}"
  "$@" 2>&1 | tee "${log_path}"
  local cmd_status=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END ${dataset} ${stage} status=${cmd_status} log=${log_path}" | tee -a "${master_log}"
  return "${cmd_status}"
}

echo "[$(date '+%F %T')] Queue launcher started for wikidata5m step2-4." | tee -a "${master_log}"
run_stage "wikidata5m" "step2" ./step2_application.sh "wikidata5m"
run_stage "wikidata5m" "step3" ./step3_dataset.sh "wikidata5m"
run_stage "wikidata5m" "step4" ./step4_dependency.sh "wikidata5m"
echo "[$(date '+%F %T')] Queue finished with status=0" | tee -a "${master_log}"
