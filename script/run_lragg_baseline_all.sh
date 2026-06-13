#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

datasets=("$@")
if [ "${#datasets[@]}" -eq 0 ]; then
    datasets=(
        "KG20C"
        "codex-m"
        "WN18RR"
        "FB15k-237"
        "codex-l"
        "YAGO3-10"
    )
fi

for dataset in "${datasets[@]}"; do
    echo "=================================================="
    echo "Running LR-Agg baseline for ${dataset}"
    echo "=================================================="
    PYTHON_BIN="${PYTHON_BIN:-/home/sy/anaconda3/bin/python}" "${ROOT_DIR}/script/run_lragg_baseline.sh" "${dataset}"
done
