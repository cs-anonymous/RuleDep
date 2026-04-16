#!/usr/bin/env bash
set -euo pipefail

[ "$#" -ge 1 ] || { echo "Usage: $0 <dataset>" >&2; exit 1; }

default_dep_min_supp() {
    case "$1" in
        KG20C|WN18RR) echo 2 ;;
        *) echo 5 ;;
    esac
}

default_filter_target_split() {
    case "$1" in
        hetionet|wikidata5m) echo valid ;;
        *) echo train ;;
    esac
}

dataset="$1"
worker_threads="${20:-$(nproc)}"
filter_target_split="${TARGET_SPLIT:-$(default_filter_target_split "${dataset}")}"
filter_min_supp="${FILTER_MIN_SUPP:-$(default_dep_min_supp "${dataset}")}"
dependency_dir="${DEPENDENCY_DIR:-data/${dataset}/rules}"
path_dependency_default="${dependency_dir}/dependency.txt"
synergy_file_default="${dependency_dir}/synergy.txt"
redundancy_file_default="${dependency_dir}/redundancy.txt"
run_deplearn_log="${RUN_DEPLEARN_LOG:-${dependency_dir}/run_deplearn.log}"

export DATASET="${dataset}"
export MIN_SUPP="${MIN_SUPP:-$(default_dep_min_supp "${dataset}")}"
export WORKER_THREADS="${worker_threads}"
export TOP_K="${TOP_K:-500}"
export PATH_TRAINING="data/${dataset}/train.txt"
export PATH_VALID="data/${dataset}/valid.txt"
export PATH_TEST="data/${dataset}/test.txt"
export PATH_RULES="data/${dataset}/rules/rule.txt"
export PATH_DEPENDENCY="${PATH_DEPENDENCY:-${path_dependency_default}}"
export MAVEN_OPTS="${MAVEN_OPTS:--Xms240g -Xmx240g -XX:MaxMetaspaceSize=2g}"

echo "======================================"
echo "Step 4: Dependency learning for ${dataset}"
echo "======================================"
echo "TOP_K=${TOP_K}"

mkdir -p "${dependency_dir}"

mvn -DskipTests compile exec:java > "${run_deplearn_log}" 2>&1 || {
    tail -n 120 "${run_deplearn_log}" >&2
    exit 1
}
python filter_dependency.py \
    -d "${dataset}" \
    --jobs "${worker_threads}" \
    --target_split "${filter_target_split}" \
    --min_supp "${filter_min_supp}" \
    --synergy_file "${FILTER_SYNERGY_FILE:-${synergy_file_default}}" \
    --redundancy_file "${FILTER_REDUNDANCY_FILE:-${redundancy_file_default}}"

echo "Step 4 finished for ${dataset}"
