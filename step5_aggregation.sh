#!/usr/bin/env bash
set -euo pipefail

[ "$#" -ge 1 ] || { echo "Usage: $0 <dataset> [ruleset] [multiprocess] [pos_values]" >&2; exit 1; }

dataset="$1"
ruleset="${2:-rules-1000-5}"
multiprocess="${3:-2}"
pos_values="${4:-5,10,15}"

echo "======================================"
echo "Step 5: Aggregation for ${dataset}"
echo "======================================"

mkdir -p "data/${dataset}/aggregation"

OLD_IFS="$IFS"
IFS=',' read -r -a pos_list <<< "${pos_values}"
IFS="$OLD_IFS"

for pos in "${pos_list[@]}"; do
    pos_trimmed="${pos// /}"
    [ -n "${pos_trimmed}" ] || continue
    echo "Running aggregation with pos=${pos_trimmed}..."
    python aggregation.py -d "${dataset}" --rule_file "data/${dataset}/rules/${ruleset}" \
        --relation -1 --multiprocess "${multiprocess}" \
        --synergy --redundancy --no_sign_constraint_dependency \
        --train_rule_in_dependency_stage --pos "${pos_trimmed}"
done

echo "Step 5 finished for ${dataset}"
