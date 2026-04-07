#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/nonhetionet_then_hetionet"

mkdir -p "${LOG_DIR}"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "${LOG_DIR}/master.log"
}

: > "${LOG_DIR}/master.log"

cd "${ROOT_DIR}"

log "START non-hetionet follow-up queue"
DATASETS_OVERRIDE="FB15k-237,KG20C,WN18RR,YAGO3-10,codex-l,codex-m" \
    bash "${ROOT_DIR}/script/run_post_step4_step5_followups.sh"
log "END non-hetionet follow-up queue"

log "START hetionet main sweep"
DATASETS_OVERRIDE="hetionet" START_DATASET="hetionet" \
    bash "${ROOT_DIR}/script/run_step4_filter_sweep_then_step5_none.sh"
log "END hetionet main sweep"

log "START hetionet follow-up queue"
DATASETS_OVERRIDE="hetionet" \
    bash "${ROOT_DIR}/script/run_post_step4_step5_followups.sh"
log "END hetionet follow-up queue"

log "All scheduled queues finished"
