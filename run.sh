#!/usr/bin/env bash
set -euo pipefail

default_support_threshold_for_dataset() {
    case "$1" in
        KG20C|WN18RR) echo 3 ;;
        *) echo 5 ;;
    esac
}

if [ "$#" -gt 0 ]; then
    datasets=("$@")
else
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
    support_threshold="${SUPPORT_THRESHOLD:-$(default_support_threshold_for_dataset "${dataset}")}"

    echo "=================================================="
    echo "Running steps 0-5 for ${dataset}"
    echo "=================================================="

    python "${PREPROCESS_SCRIPT:-preprocess.py}" "data/${dataset}"
    # ./step1_learning.sh "${dataset}" "${support_threshold}" "${SNAPSHOTS:-10,100,400,1000}" "${LEARNING_WORKER_THREADS:-20}"
    # ./step2_application.sh "${dataset}" "${TOPK:-100}" "${APPLICATION_WORKER_THREADS:-20}"
    # ./step3_dataset.sh "${dataset}"
    ./step4_dependency.sh "${dataset}"
    # ./step5_aggregation.sh "${dataset}" "${AGGREGATION_MULTIPROCESS:-2}"
done
