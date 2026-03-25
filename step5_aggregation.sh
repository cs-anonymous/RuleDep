#!/usr/bin/env bash
set -euo pipefail

[ "$#" -ge 1 ] || { echo "Usage: $0 <dataset> [multiprocess]" >&2; exit 1; }

dataset="$1"
multiprocess="${2:-2}"

echo "======================================"
echo "Step 5: Aggregation for ${dataset}"
echo "======================================"

mkdir -p "data/${dataset}/aggregation"

run_aggregation() {
    local model="$1"
    shift

    python aggregation.py -d "${dataset}" --rule_file "data/${dataset}/rules/rule.txt" \
        --relation -1 --multiprocess "${multiprocess}" \
        --model "${model}" \
        --train_rule_in_dependency_stage \
        "$@"
}

run_aggregation_with_control() {
    local model="$1"
    shift

    run_aggregation "${model}" "$@"
    run_aggregation "${model}" "$@" --pos auto_ratio
}

for model in LinearAggregator SurprisalAggregator; do
    run_aggregation_with_control "${model}" --synergy --redundancy
    run_aggregation_with_control "${model}" --synergy
    run_aggregation_with_control "${model}" --redundancy
    run_aggregation_with_control "${model}" --synergy --redundancy --sign_constraint_dependency
    run_aggregation_with_control "${model}" --synergy --redundancy --init_dep_with_lift
done

echo "Step 5 finished for ${dataset}"
