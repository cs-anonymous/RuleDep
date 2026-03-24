#!/usr/bin/env bash
set -euo pipefail

[ "$#" -ge 1 ] || { echo "Usage: $0 <dataset> [ruleset]" >&2; exit 1; }

dataset="$1"
ruleset="${2:-rule.txt}"

echo "======================================"
echo "Step 3: Dataset generation for ${dataset}"
echo "======================================"

mkdir -p "data/${dataset}/datasets"

for split in train valid test; do
    python process_rules.py \
        --data_dir "data/${dataset}" \
        --split "${split}" \
        --target_file "data/${dataset}/${split}.txt" \
        --applied_rules_file "data/${dataset}/application/applied_rules_${split}.json" \
        --save_dir "data/${dataset}/application"
done

python create_datasets.py -d "${dataset}" \
    --applied_rules "data/${dataset}/application/applied_rules_train.json" \
    --rule_file "data/${dataset}/rules/${ruleset}" \
    --output "data/${dataset}/datasets"

echo "Step 3 finished for ${dataset}"
