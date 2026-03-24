#!/usr/bin/env bash
set -euo pipefail

[ "$#" -ge 1 ] || { echo "Usage: $0 <dataset> [ruleset] [topk] [worker_threads]" >&2; exit 1; }

dataset="$1"
ruleset="${2:-rules-1000-5}"
topk="${3:-100}"
worker_threads="${4:-20}"

mkdir -p "data/${dataset}/application"
: > "data/${dataset}/application/empty.txt"

echo "======================================"
echo "Step 2: Application for ${dataset}"
echo "======================================"

python script/eval.py --dataset "${dataset}" --rules "data/${dataset}/rules/${ruleset}" \
    --aggregation_function noisyor > "data/${dataset}/application/eval-noisyor.log"
python script/eval.py --dataset "${dataset}" --rules "data/${dataset}/rules/${ruleset}" \
    --aggregation_function maxplus > "data/${dataset}/application/eval-maxplus.log"

python script/apply_pyclause.py \
    --filter-w-data 0 \
    --train "data/${dataset}/train.txt" \
    --valid "data/${dataset}/application/empty.txt" \
    --target "data/${dataset}/train.txt" \
    --rules "data/${dataset}/rules/${ruleset}" \
    --output "data/${dataset}/application/applied_rules_train.json" \
    --topk "${topk}" \
    --worker-threads "${worker_threads}" \
    --aggregation maxplus \
    --min-correct-predictions 5 \
    --read-cyclic-rules 1 \
    --read-acyclic1-rules 1 \
    --read-acyclic2-rules 0 \
    --read-zero-rules 0 \
    --read-uxxc-rules 0 \
    --read-uxxd-rules 0

python script/apply_pyclause.py \
    --filter-w-data 1 \
    --train "data/${dataset}/train.txt" \
    --valid "data/${dataset}/application/empty.txt" \
    --target "data/${dataset}/valid.txt" \
    --rules "data/${dataset}/rules/${ruleset}" \
    --output "data/${dataset}/application/applied_rules_valid.json" \
    --topk "${topk}" \
    --worker-threads "${worker_threads}" \
    --aggregation maxplus \
    --min-correct-predictions 5 \
    --read-cyclic-rules 1 \
    --read-acyclic1-rules 1 \
    --read-acyclic2-rules 0 \
    --read-zero-rules 0 \
    --read-uxxc-rules 0 \
    --read-uxxd-rules 0

python script/apply_pyclause.py \
    --filter-w-data 1 \
    --train "data/${dataset}/train.txt" \
    --valid "data/${dataset}/valid.txt" \
    --target "data/${dataset}/test.txt" \
    --rules "data/${dataset}/rules/${ruleset}" \
    --output "data/${dataset}/application/applied_rules_test.json" \
    --topk "${topk}" \
    --worker-threads "${worker_threads}" \
    --aggregation maxplus \
    --min-correct-predictions 5 \
    --read-cyclic-rules 1 \
    --read-acyclic1-rules 1 \
    --read-acyclic2-rules 0 \
    --read-zero-rules 0 \
    --read-uxxc-rules 0 \
    --read-uxxd-rules 0

echo "Step 2 finished for ${dataset}"
