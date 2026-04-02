#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

run_tag="filtered"
multiprocess="${AGGREGATION_MULTIPROCESS:-2}"
max_parallel_configs="${MAX_PARALLEL_CONFIGS:-4}"
hetionet_dataset="hetionet"
other_datasets=(
  "KG20C"
  "codex-m"
  "WN18RR"
  "FB15k-237"
  "codex-l"
  "YAGO3-10"
)

ensure_no_filtered_outputs() {
  local dataset="$1"
  local root="data/${dataset}/aggregation"
  if [ ! -d "${root}" ]; then
    return 0
  fi
  mapfile -t existing < <(find "${root}" -maxdepth 1 -mindepth 1 -type d -name '*_filtered' | sort)
  if [ "${#existing[@]}" -gt 0 ]; then
    echo "Refusing to run ${dataset}: existing _filtered outputs found:" >&2
    printf '  %s\n' "${existing[@]}" >&2
    return 1
  fi
}

run_dataset() {
  local dataset="$1"
  echo "[$(date '+%F %T')] START step5 ${dataset} (_filtered)"
  RUN_TAG="${run_tag}" MAX_PARALLEL_CONFIGS="${max_parallel_configs}" ./step5_aggregation.sh "${dataset}" "${multiprocess}"
  echo "[$(date '+%F %T')] END step5 ${dataset} (_filtered)"
}

echo "[$(date '+%F %T')] Preflight check for _filtered outputs"
ensure_no_filtered_outputs "${hetionet_dataset}"
for dataset in "${other_datasets[@]}"; do
  ensure_no_filtered_outputs "${dataset}"
done

echo "[$(date '+%F %T')] Queue start: hetionet first, then remaining datasets"
run_dataset "${hetionet_dataset}"
for dataset in "${other_datasets[@]}"; do
  run_dataset "${dataset}"
done

echo "[$(date '+%F %T')] All filtered step5 reruns completed"
