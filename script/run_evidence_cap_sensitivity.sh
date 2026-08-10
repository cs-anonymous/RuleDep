#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${ROOT_DIR}/reports/evidence_cap_sensitivity"
WORK_ROOT="${OUTPUT_ROOT}/dependency_runs"
LOG_ROOT="${OUTPUT_ROOT}/logs"
WORKER_THREADS="${WORKER_THREADS:-24}"
CPU_AFFINITY="${CPU_AFFINITY:-12-35}"
MAVEN_OPTS="${MAVEN_OPTS:--Xms240g -Xmx240g -XX:MaxMetaspaceSize=2g}"

DATASETS=(KG20C WN18RR codex-m FB15k-237 codex-l YAGO3-10)
CAPS=(5 6 7 8 9 no-cap)

default_min_supp() {
    case "$1" in
        KG20C|WN18RR) echo 2 ;;
        *) echo 5 ;;
    esac
}

run_one() {
    local dataset="$1"
    local cap_label="$2"
    local cap_value="$3"
    local work_dir="${WORK_ROOT}/${dataset}/cap_${cap_label}"
    local log_path="${LOG_ROOT}/${dataset}/cap_${cap_label}.log"
    mkdir -p "${work_dir}" "${LOG_ROOT}/${dataset}"

    if [ -s "${work_dir}/synergy.txt" ] && [ -s "${work_dir}/redundancy.txt" ]; then
        echo "[$(date '+%F %T')] SKIP dataset=${dataset} cap=${cap_label} outputs exist"
        return 0
    fi

    echo "[$(date '+%F %T')] START dataset=${dataset} cap=${cap_label}"
    (
        cd "${ROOT_DIR}"
        export DATASET="${dataset}"
        export MIN_SUPP="$(default_min_supp "${dataset}")"
        export WORKER_THREADS
        export TOP_K=500
        export PATH_TRAINING="data/${dataset}/train.txt"
        export PATH_VALID="data/${dataset}/valid.txt"
        export PATH_TEST="data/${dataset}/test.txt"
        export PATH_RULES="data/${dataset}/rules/rule.txt"
        export PATH_DEPENDENCY="${work_dir}/dependency.txt"
        export DEPENDENCY_FORMULA_MODE=unified
        export MAX_SURPRISAL="${cap_value}"
        export MAVEN_OPTS
        taskset -c "${CPU_AFFINITY}" mvn -q exec:java
    ) >"${log_path}" 2>&1
    echo "[$(date '+%F %T')] END dataset=${dataset} cap=${cap_label}"
}

mkdir -p "${WORK_ROOT}" "${LOG_ROOT}"
cd "${ROOT_DIR}"
mvn -q -DskipTests compile

for dataset in "${DATASETS[@]}"; do
    for cap in "${CAPS[@]}"; do
        if [ "${cap}" = "no-cap" ]; then
            value=0
        else
            value="${cap}"
        fi
        run_one "${dataset}" "${cap}" "${value}"
    done
done

python src/reporting/analyze_evidence_cap_sensitivity.py \
    --work-root "${WORK_ROOT}" \
    --rule-root "${ROOT_DIR}/data" \
    --output-root "${OUTPUT_ROOT}" \
    --parallel-datasets 2
