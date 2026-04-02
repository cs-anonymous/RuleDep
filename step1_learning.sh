#!/usr/bin/env bash
set -euo pipefail

[ "$#" -ge 1 ] || { echo "Usage: $0 <dataset> [support_threshold] [snapshots] [worker_threads]" >&2; exit 1; }

default_support_threshold() {
    case "$1" in
        KG20C|WN18RR) echo 2 ;;
        *) echo 5 ;;
    esac
}

dataset="$1"
support_threshold="${2:-$(default_support_threshold "${dataset}")}"
snapshots="${3:-10,100,400,1000}"
default_worker_threads="$(nproc)"
worker_threads="${4:-${default_worker_threads}}"
final_snapshot="${snapshots##*,}"
max_length_cyclic="${MAX_LENGTH_CYCLIC:-}"

echo "======================================"
echo "Step 1: Rule learning for ${dataset}"
echo "======================================"

mkdir -p "data/${dataset}/rules"

java_log="$(mktemp)"
set +e
java -Xmx240G -cp script/AnyBURL-23-1x.jar de.unima.ki.anyburl.Learn <(
cat <<EOF
PATH_TRAINING = data/${dataset}/train.txt

PATH_OUTPUT   = data/${dataset}/rules/rules

SNAPSHOTS_AT = ${snapshots}

WORKER_THREADS = ${worker_threads}
${max_length_cyclic:+MAX_LENGTH_CYCLIC = ${max_length_cyclic}}
EOF
) 2>&1 | tee "${java_log}"
java_status="${PIPESTATUS[0]}"
set -e

if [ "${java_status}" -ne 0 ]; then
    if [ "${java_status}" -eq 1 ] \
        && grep -Fq '>>> Bye, bye.' "${java_log}" \
        && [ -f "data/${dataset}/rules/rules-${final_snapshot}" ]; then
        echo "Learner exited with code 1 after final snapshot; continuing."
    else
        rm -f "${java_log}"
        exit "${java_status}"
    fi
fi
rm -f "${java_log}"

(
    cd "data/${dataset}/rules"
    rm -f "rules-10-${support_threshold}" "rules-100-${support_threshold}" "rules-400-${support_threshold}" "rules-1000-${support_threshold}" "rule.txt"
    for input_file in rules-10 rules-100 rules-400 rules-1000; do
        [ -f "${input_file}" ] || continue
        awk -v threshold="${support_threshold}" '$2 >= threshold' "${input_file}" > "${input_file}-${support_threshold}"
        echo "Created ${input_file}-${support_threshold}"
    done

    mv "rules-${final_snapshot}-${support_threshold}" "rule.txt"
    echo "Created rule.txt from rules-${final_snapshot}-${support_threshold}"
)

echo "Step 1 finished for ${dataset}"
