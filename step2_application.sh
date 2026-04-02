#!/usr/bin/env bash
set -euo pipefail

[ "$#" -ge 1 ] || { echo "Usage: $0 <dataset> [topk] [worker_threads] [num_top_rules] [query_batch_size]" >&2; exit 1; }

default_min_correct_predictions() {
    case "$1" in
        KG20C|WN18RR) echo 2 ;;
        *) echo 5 ;;
    esac
}

dataset="$1"
topk="${2:-100}"
default_worker_threads="$(nproc)"
worker_threads="${3:-${default_worker_threads}}"
num_top_rules="${4:-200}"
min_correct_predictions="${MIN_CORRECT_PREDICTIONS:-$(default_min_correct_predictions "${dataset}")}"
b_max_length="${B_MAX_LENGTH:-}"

mkdir -p "data/${dataset}/application"
: > "data/${dataset}/application/empty.txt"

echo "======================================"
echo "Step 2: Application for ${dataset}"
echo "======================================"

python script/eval.py --dataset "${dataset}" --rules "data/${dataset}/rules/rule.txt" \
    --aggregation_function noisyor > "data/${dataset}/application/eval-noisyor.log"
python script/eval.py --dataset "${dataset}" --rules "data/${dataset}/rules/rule.txt" \
    --aggregation_function maxplus > "data/${dataset}/application/eval-maxplus.log"

for split in train valid test; do
    case "${split}" in
        train)
            filter_w_data=0
            valid_file="data/${dataset}/application/empty.txt"
            target_file="data/${dataset}/train.txt"
            ;;
        valid)
            filter_w_data=1
            valid_file="data/${dataset}/application/empty.txt"
            target_file="data/${dataset}/valid.txt"
            ;;
        test)
            filter_w_data=1
            valid_file="data/${dataset}/valid.txt"
            target_file="data/${dataset}/test.txt"
            ;;
    esac

    python script/apply_pyclause.py \
        --filter-w-data "${filter_w_data}" \
        --train "data/${dataset}/train.txt" \
        --valid "${valid_file}" \
        --target "${target_file}" \
        --rules "data/${dataset}/rules/rule.txt" \
        --output "data/${dataset}/application/applied_rules_${split}.json" \
        --topk "${topk}" \
        --worker-threads "${worker_threads}" \
        --num_top_rules "${num_top_rules}" \
        --aggregation maxplus \
        --read-cyclic-rules 1 \
        --read-acyclic1-rules 1 \
        --read-acyclic2-rules 1 \
        --read-zero-rules 0 \
        --read-uxxc-rules 0 \
        --read-uxxd-rules 0 \
        ${b_max_length:+--b-max-length "${b_max_length}"} \
        --min-correct-predictions "${min_correct_predictions}"
        
        # --min-correct-predictions 5 \
done

python script/eval_base_ranker.py \
    --dataset "${dataset}" \
    --rules "data/${dataset}/rules/rule.txt" \
    --applied_rules "data/${dataset}/application/applied_rules_test.json" \
    --aggregation noisyor > "data/${dataset}/application/eval_base_ranker_noisyor.log"

python script/eval_base_ranker.py \
    --dataset "${dataset}" \
    --rules "data/${dataset}/rules/rule.txt" \
    --applied_rules "data/${dataset}/application/applied_rules_test.json" \
    --aggregation maxplus > "data/${dataset}/application/eval_base_ranker_maxplus.log"

echo "Step 2 finished for ${dataset}"
