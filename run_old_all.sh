#!/usr/bin/env bash
set -euo pipefail

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
    echo "Running old aggregation for ${dataset}"
    echo "=================================================="
    PYTHON_BIN="${PYTHON_BIN:-/home/sy/anaconda3/bin/python}" ./run_old.sh "${dataset}"
done
