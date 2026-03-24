#!/usr/bin/env bash
set -euo pipefail

[ "$#" -ge 1 ] || { echo "Usage: $0 <dataset> [support_threshold] [snapshots] [worker_threads]" >&2; exit 1; }

dataset="$1"
support_threshold="${2:-5}"

echo "======================================"
echo "Step 1: Rule learning for ${dataset}"
echo "======================================"

mkdir -p "data/${dataset}/rules"

java -Xmx240G -cp script/AnyBURL-23-1x.jar de.unima.ki.anyburl.Learn <(
cat <<EOF
PATH_TRAINING = data/${dataset}/train.txt

PATH_OUTPUT   = data/${dataset}/rules/rules

SNAPSHOTS_AT = ${3:-10,100,400,1000}

WORKER_THREADS = ${4:-20}
EOF
)

(
    cd "data/${dataset}/rules"
    rm -f "rules-*-${support_threshold}"
    shopt -s nullglob
    for input_file in rules-*; do
        case "${input_file}" in
            *-"${support_threshold}") continue ;;
        esac
        awk -v threshold="${support_threshold}" '$2 >= threshold' "${input_file}" > "${input_file}-${support_threshold}"
    done
)

echo "Step 1 finished for ${dataset}"
