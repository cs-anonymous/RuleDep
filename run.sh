#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
    datasets=("$@")
else
    datasets=(
        "KG20C"
        "codex-m"
        "codex-l"
        "WN18RR"
        "YAGO3-10"
    )
fi

for dataset in "${datasets[@]}"; do
    echo "=================================================="
    echo "Running steps 0-5 for ${dataset}"
    echo "=================================================="

    python "${PREPROCESS_SCRIPT:-preprocess.py}" "data/${dataset}"
    ./step1_learning.sh "${dataset}" "${SUPPORT_THRESHOLD:-5}" "${SNAPSHOTS:-10,100,400,1000}" "${LEARNING_WORKER_THREADS:-20}"
    ./step2_application.sh "${dataset}" "${RULESET:-rules-1000-5}" "${TOPK:-100}" "${APPLICATION_WORKER_THREADS:-20}"
    ./step3_dataset.sh "${dataset}" "${RULESET:-rules-1000-5}"
    ./step4_dependency.sh "${dataset}" "${RULESET:-rules-1000-5}"
    ./step5_aggregation.sh "${dataset}" "${RULESET:-rules-1000-5}" "${AGGREGATION_MULTIPROCESS:-2}" "${AGGREGATION_POS_VALUES:-5,10,15}"
done
