#!/usr/bin/env bash
set -euo pipefail

log_root="logs/rerun_step2_4"
mkdir -p "${log_root}"

datasets=(
  KG20C
  WN18RR
  YAGO3-10
  codex-l
  codex-m
  hetionet
  wikidata5m
)

run_stage() {
  local dataset="$1"
  local stage="$2"
  shift 2
  local log_path="${log_root}/${dataset}_${stage}.log"

  echo "[$(date '+%F %T')] START ${dataset} ${stage}" | tee -a "${log_root}/master.log"
  "$@" 2>&1 | tee "${log_path}"
  local cmd_status=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] END ${dataset} ${stage} status=${cmd_status} log=${log_path}" | tee -a "${log_root}/master.log"
  return "${cmd_status}"
}

for dataset in "${datasets[@]}"; do
  echo "======================================"
  echo "[$(date '+%F %T')] Re-running step2-step4 for ${dataset}"
  echo "======================================"
  echo "[$(date '+%F %T')] DATASET ${dataset} begin" | tee -a "${log_root}/master.log"
  run_stage "${dataset}" "step2" ./step2_application.sh "${dataset}"
  run_stage "${dataset}" "step3" ./step3_dataset.sh "${dataset}"
  run_stage "${dataset}" "step4" ./step4_dependency.sh "${dataset}"
  echo "[$(date '+%F %T')] DATASET ${dataset} finished" | tee -a "${log_root}/master.log"
done
