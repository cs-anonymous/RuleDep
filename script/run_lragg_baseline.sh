#!/usr/bin/env bash
set -euo pipefail

[ "$#" -ge 1 ] || {
    echo "Usage: $0 <dataset> [relation]" >&2
    echo "  relation: relation id, default -1 (train all relations)" >&2
    exit 1
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    echo "Usage: $0 <dataset> [relation]"
    echo "  relation: relation id, default -1 (train all relations)"
    echo
    echo "Optional env vars:"
    echo "  PYTHON_BIN, DEVICE, MODEL, BATCH_SIZE, MAX_WORKER_DATALOADER"
    echo "  LR_HPO, MAX_EPOCH_HPO, POS_HPO, NUM_UNSEEN, SIGN_CONSTRAINT, EXPERIMENT_DIR"
    exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

dataset="$1"
relation="${2:--1}"
if [ "$#" -ge 2 ]; then
    shift 2
else
    shift 1
fi

device="${DEVICE:-cuda}"
model="${MODEL:-LinearAggregator}"
batch_size="${BATCH_SIZE:-4096}"
max_workers="${MAX_WORKER_DATALOADER:-8}"
python_bin="${PYTHON_BIN:-python3}"

lr_hpo="${LR_HPO:-0.005}"
max_epoch_hpo="${MAX_EPOCH_HPO:-40}"
pos_hpo="${POS_HPO:-5}"
num_unseen="${NUM_UNSEEN:-0}"
sign_constraint="${SIGN_CONSTRAINT:-1}"
cmd=(
    "${python_bin}" "${ROOT_DIR}/src/baselines/aggregation_old.py"
    -d "${dataset}"
    --data_root "${ROOT_DIR}/data"
    --rule_file "${ROOT_DIR}/data/${dataset}/rules/rule.txt"
    --directory_explanations "${ROOT_DIR}/data/${dataset}/application/"
    --directory_preprocessed_datasets "${ROOT_DIR}/data/${dataset}/datasets/"
    --relation "${relation}"
    --device "${device}"
    --model "${model}"
    --batch_size "${batch_size}"
    --max_worker_dataloader "${max_workers}"
    --lr_hpo ${lr_hpo}
    --max_epoch_hpo ${max_epoch_hpo}
    --pos_hpo ${pos_hpo}
    --num_unseen ${num_unseen}
)

if [ "${sign_constraint}" != "0" ]; then
    cmd+=(--sign_constraint)
fi

if [ -n "${EXPERIMENT_DIR:-}" ]; then
    mkdir -p "$(dirname "${EXPERIMENT_DIR}")"
    cmd+=(-e "${EXPERIMENT_DIR}")
fi

cmd+=("$@")
"${cmd[@]}"
