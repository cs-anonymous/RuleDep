#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
wait_dataset="$1"
rerun_dataset="$2"
gpu="$3"
log_dir="${ROOT_DIR}/logs/aggregation_old_manual"
mkdir -p "$log_dir"
log_file="${log_dir}/${rerun_dataset}.log"
while pgrep -af "aggregation_old.py -d ${wait_dataset}|script/run_old.sh ${wait_dataset}" >/dev/null 2>&1; do
  sleep 30
done
cd "$ROOT_DIR"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHON_BIN="/home/sy/anaconda3/envs/ruledep/bin/python"
export EXPERIMENT_DIR="${ROOT_DIR}/data/${rerun_dataset}/aggregation/canonical"
export PYTHONUNBUFFERED=1
export MAX_EPOCH_HPO=40
export MAX_WORKER_DATALOADER=0
rm -rf "$EXPERIMENT_DIR"
mkdir -p "$EXPERIMENT_DIR"
bash "${ROOT_DIR}/script/run_old.sh" "$rerun_dataset" 2>&1 | tee "$log_file"
