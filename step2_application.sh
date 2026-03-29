#!/usr/bin/env bash
set -euo pipefail

[ "$#" -ge 1 ] || { echo "Usage: $0 <dataset> [topk] [worker_threads] [num_top_rules] [query_batch_size]" >&2; exit 1; }

dataset="$1"
topk="${2:-100}"
worker_threads="${3:-20}"
num_top_rules="${4:-200}"

split_value() {
    local split="$1"
    local default_value="$2"
    local key="${split^^}_$3"
    local value="${!key:-}"
    if [ -n "${value}" ]; then
        echo "${value}"
    else
        echo "${default_value}"
    fi
}

mkdir -p "data/${dataset}/application"
: > "data/${dataset}/application/empty.txt"

echo "======================================"
echo "Step 2: Application for ${dataset}"
echo "======================================"

# python script/eval.py --dataset "${dataset}" --rules "data/${dataset}/rules/rule.txt" \
#     --aggregation_function noisyor > "data/${dataset}/application/eval-noisyor.log"
# python script/eval.py --dataset "${dataset}" --rules "data/${dataset}/rules/rule.txt" \
#     --aggregation_function maxplus > "data/${dataset}/application/eval-maxplus.log"

for split in train valid test; do
    split_topk="$(split_value "${split}" "${topk}" "TOPK")"
    split_worker_threads="$(split_value "${split}" "${worker_threads}" "APPLICATION_WORKER_THREADS")"
    split_num_top_rules="$(split_value "${split}" "${num_top_rules}" "NUM_TOP_RULES")"
    split_hard_stop_at="$(split_value "${split}" "" "HARD_STOP_AT")"
    split_adapt_topk="$(split_value "${split}" "" "ADAPT_TOPK")"
    split_b_max_branching_factor="$(split_value "${split}" "${B_MAX_BRANCHING_FACTOR:-}" "B_MAX_BRANCHING_FACTOR")"

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

    pyclause_args=(
        --filter-w-data "${filter_w_data}"
        --train "data/${dataset}/train.txt"
        --valid "${valid_file}"
        --target "${target_file}"
        --rules "data/${dataset}/rules/rule.txt"
        --output "data/${dataset}/application/applied_rules_${split}.json"
        --topk "${split_topk}"
        --worker-threads "${split_worker_threads}"
        --num_top_rules "${split_num_top_rules}"
        --aggregation maxplus
        --b-max-branching-factor "${split_b_max_branching_factor:--1}"
        --read-cyclic-rules 1
        --read-acyclic1-rules 1
        --read-acyclic2-rules 0
        --read-zero-rules 0
        --read-uxxc-rules 0
        --read-uxxd-rules 0
    )
    if [ -n "${split_hard_stop_at}" ]; then
        pyclause_args+=(--hard-stop-at "${split_hard_stop_at}")
    fi
    if [ -n "${split_adapt_topk}" ]; then
        pyclause_args+=(--adapt-topk "${split_adapt_topk}")
    fi

    python script/apply_pyclause.py "${pyclause_args[@]}"
        
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
