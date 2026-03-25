#!/usr/bin/env bash
set -euo pipefail

[ "$#" -ge 1 ] || { echo "Usage: $0 <dataset>" >&2; exit 1; }

default_dep_min_supp() {
    case "$1" in
        KG20C|WN18RR) echo 3 ;;
        *) echo 5 ;;
    esac
}

dataset="$1"

export DATASET="${dataset}"
export MIN_SUPP="${MIN_SUPP:-$(default_dep_min_supp "${dataset}")}"
export PATH_TRAINING="data/${dataset}/train.txt"
export PATH_VALID="data/${dataset}/valid.txt"
export PATH_TEST="data/${dataset}/test.txt"
export PATH_RULES="data/${dataset}/rules/rule.txt"
export PATH_DEPENDENCY="data/${dataset}/rules/dependency.txt"
export MAVEN_OPTS="${MAVEN_OPTS:--Xms240g -Xmx240g -XX:MaxMetaspaceSize=2g}"

echo "======================================"
echo "Step 4: Dependency learning for ${dataset}"
echo "======================================"

mkdir -p "data/${dataset}/rules"

mvn -DskipTests compile exec:java > "data/${dataset}/rules/run_deplearn.log" 2>&1 || {
    tail -n 120 "data/${dataset}/rules/run_deplearn.log" >&2
    exit 1
}
python filter_dependency.py -d "${dataset}"

echo "Step 4 finished for ${dataset}"
