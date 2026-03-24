#!/usr/bin/env bash
set -euo pipefail

[ "$#" -ge 1 ] || { echo "Usage: $0 <dataset> [ruleset] [multiprocess]" >&2; exit 1; }

dataset="$1"
ruleset="${2:-rule.txt}"
multiprocess="${3:-2}"

echo "======================================"
echo "Step 5: Aggregation for ${dataset}"
echo "======================================"

mkdir -p "data/${dataset}/aggregation"

run_aggregation() {
    local model="$1"
    shift

    python aggregation.py -d "${dataset}" --rule_file "data/${dataset}/rules/${ruleset}" \
        --relation -1 --multiprocess "${multiprocess}" \
        --model "${model}" \
        --train_rule_in_dependency_stage \
        "$@"
}

for model in LinearAggregator SurprisalAggregator; do
    run_aggregation "${model}" --synergy --redundancy
    run_aggregation "${model}" --synergy
    run_aggregation "${model}" --redundancy
    run_aggregation "${model}" --synergy --redundancy --sign_constraint_dependency
    run_aggregation "${model}" --synergy --redundancy --init_dep_with_lift
done

echo "Step 5 finished for ${dataset}"
