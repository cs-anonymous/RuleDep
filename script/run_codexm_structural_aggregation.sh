#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/aggregation_structural"
EXP_ROOT="${ROOT_DIR}/data/codex-m/aggregation"
SUMMARY_PATH="${LOG_DIR}/codex-m_structural_summary.json"
MULTIPROCESS="${MULTIPROCESS:-2}"

mkdir -p "${LOG_DIR}" "${EXP_ROOT}"

source /home/sy/anaconda3/etc/profile.d/conda.sh
conda activate ruledep

run_variant() {
    local gpu="$1"
    local name="$2"
    local rule_grouping="$3"
    local dependency_grouping="$4"
    local log_path="${LOG_DIR}/codex-m_${name}.log"
    local exp_dir="${EXP_ROOT}/structural_${name}"

    mkdir -p "${exp_dir}"
    echo "[`date '+%F %T'`] START ${name} gpu=${gpu} rule_grouping=${rule_grouping} dependency_grouping=${dependency_grouping}" | tee -a "${LOG_DIR}/master.log"
    (
        cd "${ROOT_DIR}"
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export EXPERIMENT_DIR="${exp_dir}"
        export PYTHONUNBUFFERED=1
        python -u aggregation.py \
            -d codex-m \
            --rule_file "data/codex-m/rules/rule.txt" \
            --relation -1 \
            --multiprocess "${MULTIPROCESS}" \
            --model LinearAggregator \
            --train_rule_in_dependency_stage \
            --synergy \
            --redundancy \
            --rule_grouping "${rule_grouping}" \
            --dependency_grouping "${dependency_grouping}"
    ) 2>&1 | tee "${log_path}"
    local status=${PIPESTATUS[0]}
    echo "[`date '+%F %T'`] END ${name} status=${status} log=${log_path}" | tee -a "${LOG_DIR}/master.log"
    return "${status}"
}

run_variant 0 rd none none &
pid_rd=$!
run_variant 1 r2d3 r2 d3 &
pid_r2d3=$!
run_variant 2 r3d3 r3 d3 &
pid_r3d3=$!
run_variant 3 r3d6 r3 d6 &
pid_r3d6=$!

status=0
wait "${pid_rd}" || status=1
wait "${pid_r2d3}" || status=1
wait "${pid_r3d3}" || status=1
wait "${pid_r3d6}" || status=1

(
    cd "${ROOT_DIR}"
    python script/summarize_structural_suite.py --root "${EXP_ROOT}" --out "${SUMMARY_PATH}"
) 2>&1 | tee "${LOG_DIR}/summary.log"

exit "${status}"
