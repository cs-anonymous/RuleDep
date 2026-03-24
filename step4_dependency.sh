#!/usr/bin/env bash
set -euo pipefail

[ "$#" -ge 1 ] || { echo "Usage: $0 <dataset> [ruleset]" >&2; exit 1; }

dataset="$1"
ruleset="${2:-rules-1000-5}"

export DATASET="${dataset}"
export PATH_TRAINING="data/${dataset}/train.txt"
export PATH_VALID="data/${dataset}/valid.txt"
export PATH_TEST="data/${dataset}/test.txt"
export PATH_RULES="data/${dataset}/rules/${ruleset}"
export PATH_DEPENDENCY="data/${dataset}/rules/dependency.txt"
export PATH_H2B2metric="data/${dataset}/rules/H2B2metric.json"
export PATH_RULES_TXT="data/${dataset}/rules/rule.txt"
export PATH_H2F2metric="data/${dataset}/rules/H2F2metric.json"
export PATH_OUTPUT="data/${dataset}/application/predictions"
export MAVEN_OPTS="${MAVEN_OPTS:--Xms240g -Xmx240g -XX:MaxMetaspaceSize=2g}"

echo "======================================"
echo "Step 4: Dependency learning for ${dataset}"
echo "======================================"

mkdir -p "data/${dataset}/rules" "data/${dataset}/application"

mvn -q -DskipTests compile exec:java > "data/${dataset}/rules/run_depgraph.log" 2>&1
python filter_dependency.py -d "${dataset}" --rule_file "data/${dataset}/rules/${ruleset}"

echo "Step 4 finished for ${dataset}"
