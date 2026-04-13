#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [ -f /home/sy/anaconda3/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /home/sy/anaconda3/etc/profile.d/conda.sh
    conda activate ruledep
else
    export PATH="/home/sy/anaconda3/envs/ruledep/bin:${PATH}"
fi

multiprocess="${AGGREGATION_MULTIPROCESS:-2}"
run_tag="${RUN_TAG:-dep_fix_$(date +%m%d_%H%M%S)}"
datasets=("$@")
if [ "${#datasets[@]}" -eq 0 ]; then
    datasets=(
        "KG20C"
        "codex-m"
        "WN18RR"
        "FB15k-237"
        "codex-l"
        "YAGO3-10"
        "hetionet"
    )
fi

variant_args() {
    case "$1" in
        static_per_rule_dep_norm)
            printf '%s\n' "--dependency_static_norm per_rule_degree"
            ;;
        dep_l1_regularization)
            printf '%s\n' "--dep_l1_lambda 1e-5"
            ;;
        dep_score_clip)
            printf '%s\n' "--dep_score_clip_gamma 1.0"
            ;;
        global_preselected_dep_topk_per_kind)
            printf '%s\n' "--dependency_topk_per_kind 8 --dependency_topk_score abs_lift"
            ;;
        *)
            echo "Unknown variant: $1" >&2
            return 1
            ;;
    esac
}

base_args_for_dataset() {
    local dataset="$1"
    python - "$dataset" <<'PY'
import json
import os
import shlex
import sys
from pathlib import Path

dataset = sys.argv[1]
path = Path("data") / dataset / "aggregation" / "best_combination" / "config.json"
cfg = json.load(open(path))
args = [
    "-d", dataset,
    "--rule_file", cfg.get("rule_file") or f"data/{dataset}/rules/rule.txt",
    "--relation", "-1",
    "--multiprocess", os.environ.get("MULTIPROCESS", "2"),
    "--batch_size", str(cfg.get("batch_size", 4096)),
    "--lr", str(cfg.get("lr", "0.01,0.005,0.001")),
    "--max_epoch", str(cfg.get("max_epoch", 60)),
    "--evaluate_every", str(cfg.get("evaluate_every", "4,2,1")),
    "--early_stopping", str(cfg.get("early_stopping", 3)),
    "--pos", str(cfg.get("pos", "auto_sqrt")),
    "--rule_init_mode", str(cfg.get("rule_init_mode", "conf")),
    "--dependency_scale_mode", str(cfg.get("dependency_scale_mode", "none")),
    "--type_grouping", str(cfg.get("type_grouping", "none")),
    "--eval_key_batch_size", str(cfg.get("eval_key_batch_size", 64)),
    "--dependency_chunk_size", str(cfg.get("dependency_chunk_size", 4096)),
]
if cfg.get("train_rule_in_dependency_stage"):
    args.append("--train_rule_in_dependency_stage")
if cfg.get("synergy"):
    args.append("--synergy")
if cfg.get("redundancy"):
    args.append("--redundancy")
if cfg.get("sign_constraint") is False:
    args.append("--no_sign_constraint")
if cfg.get("sign_constraint_dependency"):
    args.append("--sign_constraint_dependency")
else:
    args.append("--no_sign_constraint_dependency")
if cfg.get("init_dep_with_lift"):
    args.append("--init_dep_with_lift")
if cfg.get("dependency_mask_low_rule_weight"):
    args.append("--dependency_mask_low_rule_weight")
if cfg.get("synergy_file"):
    args.extend(["--synergy_file", str(cfg["synergy_file"])])
if cfg.get("redundancy_file"):
    args.extend(["--redundancy_file", str(cfg["redundancy_file"])])
print(" ".join(shlex.quote(x) for x in args))
PY
}

export MULTIPROCESS="${multiprocess}"
variants=(
    "static_per_rule_dep_norm"
    "dep_l1_regularization"
    "dep_score_clip"
    "global_preselected_dep_topk_per_kind"
)
gpus=(0 1 2 3)

for dataset in "${datasets[@]}"; do
    config_path="data/${dataset}/aggregation/best_combination/config.json"
    if [ ! -f "${config_path}" ]; then
        echo "[$(date '+%F %T')] SKIP ${dataset}: missing ${config_path}" >&2
        continue
    fi

    log_dir="logs/aggregation_dep_fix/${run_tag}/${dataset}"
    mkdir -p "${log_dir}" "data/${dataset}/aggregation"
    base_args="$(base_args_for_dataset "${dataset}")"
    echo "[$(date '+%F %T')] DATASET ${dataset} base_args=${base_args}"

    pids=()
    status=0
    for idx in "${!variants[@]}"; do
        variant="${variants[$idx]}"
        gpu="${gpus[$idx]}"
        exp_dir="${ROOT_DIR}/data/${dataset}/aggregation/best_combination_${variant}_${run_tag}"
        log_path="${ROOT_DIR}/${log_dir}/${variant}.log"
        args="${base_args} $(variant_args "${variant}")"
        echo "[$(date '+%F %T')] START ${dataset}/${variant} gpu=${gpu} exp=${exp_dir}"
        (
            export CUDA_VISIBLE_DEVICES="${gpu}"
            export EXPERIMENT_DIR="${exp_dir}"
            export PYTHONUNBUFFERED=1
            mkdir -p "${exp_dir}"
            # shellcheck disable=SC2086
            python -u aggregation.py ${args}
        ) 2>&1 | tee "${log_path}" &
        pids[$idx]=$!
    done

    for pid in "${pids[@]}"; do
        wait "${pid}" || status=1
    done
    if [ "${status}" -ne 0 ]; then
        echo "[$(date '+%F %T')] FAIL ${dataset}" >&2
        exit "${status}"
    fi
    echo "[$(date '+%F %T')] DONE ${dataset}"
done

echo "[$(date '+%F %T')] ALL DONE run_tag=${run_tag}"
