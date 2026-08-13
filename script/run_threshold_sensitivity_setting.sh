#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <dataset> <e_min> <g_min> <gpu> [multiprocess] [min_support]" >&2
    exit 1
fi

dataset="$1"
e_min="$2"
g_min="$3"
gpu="$4"
multiprocess="${5:-2}"

case "${dataset}" in
    KG20C)
        default_min_supp=2
        config_args=(
            --synergy --redundancy --type_grouping r2d3
            --pos auto_ratio --rule_init_mode conf
            --dependency_static_norm per_rule_degree --dep_l1_lambda 1e-5
        )
        ;;
    codex-m)
        default_min_supp=5
        config_args=(
            --synergy --redundancy --type_grouping rd
            --pos auto_sqrt --rule_init_mode conf
            --dependency_static_norm per_rule_degree --dep_l1_lambda 1e-5
        )
        ;;
    YAGO3-10)
        default_min_supp=5
        config_args=(
            --synergy --redundancy --type_grouping r3d6
            --pos auto_sqrt --rule_init_mode surprisal
            --dependency_static_norm none --dep_l1_lambda 1e-5
        )
        ;;
    *)
        echo "Unsupported threshold-sensitivity dataset: ${dataset}" >&2
        exit 1
        ;;
esac

min_supp="${6:-${default_min_supp}}"
if ! [[ "${min_supp}" =~ ^[1-9][0-9]*$ ]]; then
    echo "min_support must be a positive integer: ${min_supp}" >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
format_value() {
    printf '%s' "$1" | tr '.' 'p'
}
tag="emin_$(format_value "${e_min}")__gmin_$(format_value "${g_min}")"
if [ "${min_supp}" != "${default_min_supp}" ]; then
    tag="${tag}__nmin_${min_supp}"
fi
dependency_dir="${ROOT_DIR}/dependency_runs/threshold_sensitivity/${dataset}/${tag}"
support_index_dir="${ROOT_DIR}/dependency_runs/threshold_sensitivity/support_index/${dataset}/nmin_${min_supp}"
experiment_dir="${ROOT_DIR}/data/${dataset}/aggregation/threshold_sensitivity/${tag}"
log_dir="${ROOT_DIR}/logs/threshold_sensitivity/${dataset}/${tag}"
mining_log="${log_dir}/dependency_mining.log"
training_log="${log_dir}/dependency_training.log"

mkdir -p "${dependency_dir}" "${experiment_dir}" "${log_dir}"

if [ -f /home/sy/anaconda3/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /home/sy/anaconda3/etc/profile.d/conda.sh
    conda activate ruledep
else
    export PATH="/home/sy/anaconda3/envs/ruledep/bin:${PATH}"
fi

log() {
    echo "[$(date '+%F %T')] dataset=${dataset} e_min=${e_min} g_min=${g_min} n_min=${min_supp} gpu=${gpu} $*"
}

if [ ! -f "${dependency_dir}/raw_mining.done" ]; then
    mining_start="$(date +%s)"
    log "START dependency mining" | tee "${mining_log}"
    (
        cd "${ROOT_DIR}"
        export DATASET="${dataset}"
        export MIN_SUPP="${min_supp}"
        export WORKER_THREADS="${WORKER_THREADS:-8}"
        export TOP_K="${TOP_K:-500}"
        export MIN_SURPRISAL_LIFT="${e_min}"
        export MIN_ABS_LIFT_DEPENDENCY="${g_min}"
        export WRITE_DEPENDENCY_JSON="false"
        export DEPENDENCY_FORMULA_MODE="unified"
        export PATH_TRAINING="data/${dataset}/train.txt"
        export PATH_VALID="data/${dataset}/valid.txt"
        export PATH_TEST="data/${dataset}/test.txt"
        export PATH_RULES="data/${dataset}/rules/rule.txt"
        export PATH_DEPENDENCY="${dependency_dir}/dependency.txt"
        export MAVEN_OPTS="${MAVEN_OPTS:--Xms8g -Xmx56g -XX:MaxMetaspaceSize=2g}"
        mvn -DskipTests exec:java
    ) 2>&1 | tee -a "${mining_log}"
    mining_status=${PIPESTATUS[0]}
    if [ "${mining_status}" -ne 0 ]; then
        log "FAIL dependency mining status=${mining_status}" | tee -a "${mining_log}"
        exit "${mining_status}"
    fi
    touch "${dependency_dir}/raw_mining.done"
else
    log "SKIP raw dependency mining reason=raw_mining.done"
    mining_start="$(date +%s)"
fi

if [ ! -f "${dependency_dir}/mining.done" ]; then
    if [ ! -f "${support_index_dir}/complete" ]; then
        echo "Missing support index: ${support_index_dir}" >&2
        exit 1
    fi
    python "${ROOT_DIR}/src/ruledep/filter_dependency_fast.py" \
        -d "${dataset}" \
        --support_index "${support_index_dir}" \
        --synergy_file "${dependency_dir}/synergy.txt" \
        --redundancy_file "${dependency_dir}/redundancy.txt" \
        2>&1 | tee -a "${mining_log}"
    filter_status=${PIPESTATUS[0]}
    if [ "${filter_status}" -ne 0 ]; then
        log "FAIL dependency filtering status=${filter_status}" | tee -a "${mining_log}"
        exit "${filter_status}"
    fi
    mining_end="$(date +%s)"
    {
        printf 'dataset\te_min\tg_min\tn_min\tmining_seconds\traw_synergy\traw_redundancy\tfiltered_synergy\tfiltered_redundancy\n'
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${dataset}" "${e_min}" "${g_min}" "${min_supp}" "$((mining_end - mining_start))" \
            "$(wc -l < "${dependency_dir}/synergy.txt")" \
            "$(wc -l < "${dependency_dir}/redundancy.txt")" \
            "$(wc -l < "${dependency_dir}/synergy_filtered.txt")" \
            "$(wc -l < "${dependency_dir}/redundancy_filtered.txt")"
    } > "${dependency_dir}/mining_summary.tsv"
    touch "${dependency_dir}/mining.done"
    log "END dependency mining seconds=$((mining_end - mining_start))" | tee -a "${mining_log}"
else
    log "SKIP dependency filtering reason=mining.done"
fi

# Serialize attempts for the same setting. This lets queued duplicate attempts
# safely skip after the first process writes its final metrics.
exec 9>"${experiment_dir}/.run.lock"
if ! flock -n 9; then
    log "SKIP dependency training reason=setting-already-running"
    exit 0
fi

if [ -f "${experiment_dir}/metrics-final.json" ]; then
    log "SKIP dependency training reason=metrics-final-exists"
    exit 0
fi

log "START dependency training output=${experiment_dir}" | tee "${training_log}"
(
    cd "${ROOT_DIR}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export EXPERIMENT_DIR="${experiment_dir}"
    export PYTHONUNBUFFERED=1
    python -u src/ruledep/aggregation.py \
        -d "${dataset}" \
        --rule_file "data/${dataset}/rules/rule.txt" \
        --relation -1 \
        --multiprocess "${multiprocess}" \
        --resume_relation_sweep \
        --train_rule_in_dependency_stage \
        --synergy_file "${dependency_dir}/synergy_filtered.txt" \
        --redundancy_file "${dependency_dir}/redundancy_filtered.txt" \
        "${config_args[@]}"
) 2>&1 | tee -a "${training_log}"
training_status=${PIPESTATUS[0]}
log "END dependency training status=${training_status}" | tee -a "${training_log}"
exit "${training_status}"
