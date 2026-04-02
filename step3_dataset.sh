#!/usr/bin/env bash
set -euo pipefail

[ "$#" -ge 1 ] || { echo "Usage: $0 <dataset>" >&2; exit 1; }

dataset="$1"
worker_threads="${2:-$(nproc)}"

echo "======================================"
echo "Step 3: Dataset generation for ${dataset}"
echo "======================================"

mkdir -p "data/${dataset}/datasets"

for split in train valid test; do
    python process_rules.py \
        --data_dir "data/${dataset}" \
        --split "${split}" \
        --target_file "data/${dataset}/${split}.txt" \
        --applied_rules_dir "data/${dataset}/applied_rules" \
        --save_dir "data/${dataset}/application" \
        --num_workers "${worker_threads}"
done

echo "Generating relation-local processed explanation files..."
python script/split_processed_by_relation.py \
    -d "${dataset}" \
    --application_dir "data/${dataset}/application"

python create_datasets.py -d "${dataset}" \
    --applied_rules_dir "data/${dataset}/applied_rules" \
    --rule_file "data/${dataset}/rules/rule.txt" \
    --output "data/${dataset}/datasets" \
    --num_workers "${worker_threads}"

echo "Step 3 finished for ${dataset}"
