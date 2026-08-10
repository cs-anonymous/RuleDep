#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_SUFFIX="${RUN_SUFFIX:-main_table_per_query_rr_20260809}"
MULTIPROCESS="${MULTIPROCESS:-2}"
EXPORT_DIR="${EXPORT_DIR:-${ROOT_DIR}/reports/official_query_subset/true_official_per_query_rr/${RUN_SUFFIX}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/main_table_per_query_rr_${RUN_SUFFIX}}"

if [ -f /home/sy/anaconda3/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /home/sy/anaconda3/etc/profile.d/conda.sh
    conda activate ruledep
else
    export PATH="/home/sy/anaconda3/envs/ruledep/bin:${PATH}"
fi

mkdir -p "${EXPORT_DIR}" "${LOG_DIR}"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "${LOG_DIR}/master.log"
}

require_dataset_inputs() {
    local dataset="$1"
    local required=(
        "${ROOT_DIR}/data/${dataset}/rules/rule.txt"
        "${ROOT_DIR}/data/${dataset}/rules/synergy_filtered.txt"
        "${ROOT_DIR}/data/${dataset}/rules/redundancy_filtered.txt"
        "${ROOT_DIR}/data/${dataset}/application/processed_sp_test.pkl"
        "${ROOT_DIR}/data/${dataset}/application/processed_po_test.pkl"
        "${ROOT_DIR}/data/${dataset}/datasets"
    )
    local path
    for path in "${required[@]}"; do
        if [ ! -e "${path}" ]; then
            echo "Missing required input: ${path}" >&2
            return 1
        fi
    done
}

for dataset in KG20C WN18RR codex-m FB15k-237 codex-l YAGO3-10; do
    require_dataset_inputs "${dataset}"
done

run_aggregation() {
    local dataset="$1"
    local gpu="$2"
    local experiment="$3"
    local experiment_dir="$4"
    local log_name="$5"
    shift 5

    mkdir -p "${experiment_dir}"
    log "START dataset=${dataset} experiment=${experiment} gpu=${gpu} output=${experiment_dir}"
    (
        cd "${ROOT_DIR}"
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export EXPERIMENT_DIR="${experiment_dir}"
        export PYTHONUNBUFFERED=1
        python -u src/ruledep/aggregation.py \
            -d "${dataset}" \
            --rule_file "data/${dataset}/rules/rule.txt" \
            --relation -1 \
            --multiprocess "${MULTIPROCESS}" \
            --resume_relation_sweep \
            --train_rule_in_dependency_stage \
            --export_per_query_rr_dir "${EXPORT_DIR}" \
            --export_experiment_name "${experiment}" \
            "$@"
    ) 2>&1 | tee "${LOG_DIR}/${log_name}.log"
    local status=${PIPESTATUS[0]}
    log "END dataset=${dataset} experiment=${experiment} gpu=${gpu} status=${status}"
    return "${status}"
}

run_main_table_config() {
    local dataset="$1"
    local gpu="$2"
    local experiment="$3"
    shift 3
    local experiment_dir="${ROOT_DIR}/data/${dataset}/aggregation/reproduction"
    run_aggregation "${dataset}" "${gpu}" "${experiment}" "${experiment_dir}" "${dataset}_${experiment}" "$@"
}

build_codex_l_shards() {
    python - "${ROOT_DIR}/data/codex-l/datasets" <<'PY'
from pathlib import Path
import sys

dataset_dir = Path(sys.argv[1])
relations = []
for path in dataset_dir.glob("dataset_*.p"):
    relation = int(path.stem.rsplit("_", 1)[1])
    relations.append((relation, path.stat().st_size))

if not relations:
    raise SystemExit(f"No relation datasets found under {dataset_dir}")

loads = [0, 0]
shards = [[], []]
for relation, size in sorted(relations, key=lambda item: item[1], reverse=True):
    shard = 0 if loads[0] <= loads[1] else 1
    shards[shard].append(relation)
    loads[shard] += size

for shard in shards:
    print(",".join(str(relation) for relation in sorted(shard)))
PY
}

run_codex_l_sharded() {
    local dataset="codex-l"
    local experiment="tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5"
    local canonical_dir="${ROOT_DIR}/data/${dataset}/aggregation/reproduction"
    local shard0_dir="${canonical_dir}_shard0"
    local shard1_dir="${canonical_dir}_shard1"
    local config_args=(
        --synergy --redundancy --type_grouping rd
        --pos auto_sqrt --rule_init_mode conf
        --dependency_static_norm per_rule_degree --dep_l1_lambda 1e-5
    )
    local shards=()
    mapfile -t shards < <(build_codex_l_shards)
    if [ "${#shards[@]}" -ne 2 ]; then
        echo "Expected two Codex-L relation shards, found ${#shards[@]}" >&2
        return 1
    fi

    log "Codex-L shard0 relations=${shards[0]}"
    log "Codex-L shard1 relations=${shards[1]}"

    run_aggregation "${dataset}" 0 "${experiment}" "${shard0_dir}" \
        "${dataset}_${experiment}_shard0" --relation_ids "${shards[0]}" "${config_args[@]}" &
    local shard0_pid=$!
    run_aggregation "${dataset}" 1 "${experiment}" "${shard1_dir}" \
        "${dataset}_${experiment}_shard1" --relation_ids "${shards[1]}" "${config_args[@]}" &
    local shard1_pid=$!

    local status=0
    wait "${shard0_pid}" || status=1
    wait "${shard1_pid}" || status=1
    if [ "${status}" -ne 0 ]; then
        log "Codex-L shard run failed; preserving shard outputs for resume"
        return "${status}"
    fi

    mkdir -p "${canonical_dir}"
    cp -a "${shard0_dir}/." "${canonical_dir}/"
    cp -a "${shard1_dir}/." "${canonical_dir}/"

    # Re-enter the complete sweep in resume mode. Existing relation metrics are
    # reused; this writes one complete metrics-final.json and fills any relation
    # that did not produce a valid metric in either shard.
    run_aggregation "${dataset}" 0 "${experiment}" "${canonical_dir}" \
        "${dataset}_${experiment}_finalize" "${config_args[@]}"

    rm -r -- "${shard0_dir}" "${shard1_dir}"
}

run_gpu2_queue() {
    run_main_table_config FB15k-237 2 \
        tg_r2d3__pos_auto_ratio__ri_conf__dn_none__dl1_1e-5 \
        --synergy --redundancy --type_grouping r2d3 \
        --pos auto_ratio --rule_init_mode conf \
        --dependency_static_norm none --dep_l1_lambda 1e-5

    run_main_table_config WN18RR 2 structural_rd \
        --synergy --redundancy --type_grouping rd

    run_main_table_config KG20C 2 \
        tg_r2d3__pos_auto_ratio__ri_conf__dn_per_rule_degree__dl1_1e-5 \
        --synergy --redundancy --type_grouping r2d3 \
        --pos auto_ratio --rule_init_mode conf \
        --dependency_static_norm per_rule_degree --dep_l1_lambda 1e-5
}

run_gpu3_queue() {
    run_main_table_config YAGO3-10 3 \
        tg_r3d6__pos_auto_sqrt__ri_surprisal__dn_none__dl1_1e-5 \
        --synergy --redundancy --type_grouping r3d6 \
        --pos auto_sqrt --rule_init_mode surprisal \
        --dependency_static_norm none --dep_l1_lambda 1e-5

    run_main_table_config codex-m 3 \
        tg_rd__pos_auto_sqrt__ri_conf__dn_per_rule_degree__dl1_1e-5 \
        --synergy --redundancy --type_grouping rd \
        --pos auto_sqrt --rule_init_mode conf \
        --dependency_static_norm per_rule_degree --dep_l1_lambda 1e-5
}

log "Starting six-dataset main-table reproduction; RUN_SUFFIX=${RUN_SUFFIX}"
run_codex_l_sharded &
codex_l_pid=$!
run_gpu2_queue &
gpu2_pid=$!
run_gpu3_queue &
gpu3_pid=$!

status=0
wait "${codex_l_pid}" || status=1
wait "${gpu2_pid}" || status=1
wait "${gpu3_pid}" || status=1
if [ "${status}" -ne 0 ]; then
    log "At least one GPU queue failed. Re-run the same command to resume completed relations."
    exit "${status}"
fi

log "Exporting saved direction-specific final ranks"
(
    GPU=0 RUN_SUFFIX="${RUN_SUFFIX}" EXPORT_DIR="${EXPORT_DIR}" \
        bash "${ROOT_DIR}/script/export_saved_final_official_per_query_rr.sh" FB15k-237
) &
export_fb_pid=$!
(
    GPU=1 RUN_SUFFIX="${RUN_SUFFIX}" EXPORT_DIR="${EXPORT_DIR}" \
        bash "${ROOT_DIR}/script/export_saved_final_official_per_query_rr.sh" WN18RR
    GPU=1 RUN_SUFFIX="${RUN_SUFFIX}" EXPORT_DIR="${EXPORT_DIR}" \
        bash "${ROOT_DIR}/script/export_saved_final_official_per_query_rr.sh" KG20C
) &
export_small_pid=$!
(
    GPU=2 RUN_SUFFIX="${RUN_SUFFIX}" EXPORT_DIR="${EXPORT_DIR}" \
        bash "${ROOT_DIR}/script/export_saved_final_official_per_query_rr.sh" YAGO3-10
    GPU=2 RUN_SUFFIX="${RUN_SUFFIX}" EXPORT_DIR="${EXPORT_DIR}" \
        bash "${ROOT_DIR}/script/export_saved_final_official_per_query_rr.sh" codex-m
) &
export_medium_pid=$!
(
    GPU=3 RUN_SUFFIX="${RUN_SUFFIX}" EXPORT_DIR="${EXPORT_DIR}" \
        bash "${ROOT_DIR}/script/export_saved_final_official_per_query_rr.sh" codex-l
) &
export_large_pid=$!

status=0
wait "${export_fb_pid}" || status=1
wait "${export_small_pid}" || status=1
wait "${export_medium_pid}" || status=1
wait "${export_large_pid}" || status=1
if [ "${status}" -ne 0 ]; then
    log "At least one saved-rank export failed"
    exit "${status}"
fi

log "Merging official per-query rank/RR exports"
(
    cd "${ROOT_DIR}"
    python src/query_analysis/merge_true_official_per_query_rr.py \
        --export_dir "${EXPORT_DIR}" \
        --run_suffix "${RUN_SUFFIX}" \
        --aggregation_dir_name reproduction \
        --stages stage1,final \
        --out_rows "${EXPORT_DIR}/true_official_per_query_rr_long.csv" \
        --out_wide "${EXPORT_DIR}/true_official_per_query_rr_wide.csv" \
        --out_checks "${EXPORT_DIR}/true_official_per_query_rr_checks.csv"
)

python - "${EXPORT_DIR}/true_official_per_query_rr_checks.csv" <<'PY'
from pathlib import Path
import sys

import pandas as pd

path = Path(sys.argv[1])
checks = pd.read_csv(path)
missing_metrics = int(checks["metric_mrr"].isna().sum())
row_mismatches = int((checks["exported_rows"] != checks["expected_rows"]).sum())
max_abs_diff = float(checks["abs_diff"].dropna().max()) if checks["abs_diff"].notna().any() else float("nan")
print(
    f"validation rows={len(checks)} missing_metrics={missing_metrics} "
    f"row_mismatches={row_mismatches} max_abs_diff={max_abs_diff}"
)
if missing_metrics or row_mismatches or max_abs_diff > 1e-3:
    raise SystemExit("Per-query export validation failed")
PY

log "Reproduction complete: ${EXPORT_DIR}/true_official_per_query_rr_wide.csv"
