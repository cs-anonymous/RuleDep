#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TAG="${RUN_TAG:-ens_full_20260809}"
MULTIPROCESS="${MULTIPROCESS:-2}"

SESSION_FB237="ruledep-ens-fb237"
SESSION_CODEXL="ruledep-ens-codexl"
SESSION_YAGO="ruledep-ens-yago"
SESSION_SMALL="ruledep-ens-small"
SESSIONS=("${SESSION_FB237}" "${SESSION_CODEXL}" "${SESSION_YAGO}" "${SESSION_SMALL}")

command -v tmux >/dev/null 2>&1 || { echo "tmux is not installed" >&2; exit 1; }

for session in "${SESSIONS[@]}"; do
    if tmux has-session -t "${session}" 2>/dev/null; then
        echo "Refusing to start: tmux session already exists: ${session}" >&2
        exit 1
    fi
done

launch() {
    local session="$1"
    local command="$2"
    tmux new-session -d -s "${session}" \
        "cd '${ROOT_DIR}' && export RUN_TAG='${RUN_TAG}' MAX_PARALLEL_CONFIGS=1; ${command}"
}

launch "${SESSION_FB237}" \
    "GPU_IDS=0 ./script/run_full_aggregation_grid.sh FB15k-237 '${MULTIPROCESS}'"
launch "${SESSION_CODEXL}" \
    "GPU_IDS=1 ./script/run_full_aggregation_grid.sh codex-l '${MULTIPROCESS}'"
launch "${SESSION_YAGO}" \
    "GPU_IDS=2 ./script/run_full_aggregation_grid.sh YAGO3-10 '${MULTIPROCESS}'"
launch "${SESSION_SMALL}" \
    "export GPU_IDS=3; ./script/run_full_aggregation_grid.sh KG20C '${MULTIPROCESS}' && ./script/run_full_aggregation_grid.sh WN18RR '${MULTIPROCESS}' && ./script/run_full_aggregation_grid.sh codex-m '${MULTIPROCESS}'"

echo "Started full ensemble sweep (RUN_TAG=${RUN_TAG}, MULTIPROCESS=${MULTIPROCESS}):"
tmux list-sessions -F '#{session_name}' | grep '^ruledep-ens-' | sort
