#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${SESSION:-threshold_sensitivity}"
MULTIPROCESS="${MULTIPROCESS:-2}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    exit 1
fi

cd "${ROOT_DIR}"
mvn -q -DskipTests compile

runner="${ROOT_DIR}/script/run_threshold_sensitivity_setting.sh"

tmux new-session -d -s "${SESSION}" -n kg20c \
    "cd '${ROOT_DIR}' && '${runner}' KG20C 0.01 0.01 0 '${MULTIPROCESS}' && '${runner}' KG20C 0.03 0.01 0 '${MULTIPROCESS}' && '${runner}' KG20C 0.05 0.01 0 '${MULTIPROCESS}' && '${runner}' KG20C 0.1 0.01 0 '${MULTIPROCESS}' && '${runner}' KG20C 0.05 0.005 0 '${MULTIPROCESS}' && '${runner}' KG20C 0.05 0.03 0 '${MULTIPROCESS}' && '${runner}' KG20C 0.05 0.05 0 '${MULTIPROCESS}'; exec bash"

tmux new-window -t "${SESSION}" -n codex-m-1 \
    "cd '${ROOT_DIR}' && '${runner}' codex-m 0.01 0.01 1 '${MULTIPROCESS}' && '${runner}' codex-m 0.1 0.01 1 '${MULTIPROCESS}'; exec bash"

tmux new-window -t "${SESSION}" -n codex-m-2 \
    "cd '${ROOT_DIR}' && '${runner}' codex-m 0.03 0.01 2 '${MULTIPROCESS}' && '${runner}' codex-m 0.05 0.005 2 '${MULTIPROCESS}'; exec bash"

tmux new-window -t "${SESSION}" -n codex-m-3 \
    "cd '${ROOT_DIR}' && '${runner}' codex-m 0.05 0.01 3 '${MULTIPROCESS}' && '${runner}' codex-m 0.05 0.03 3 '${MULTIPROCESS}' && '${runner}' codex-m 0.05 0.05 3 '${MULTIPROCESS}'; exec bash"

echo "Started tmux session: ${SESSION}"
echo "Attach with: tmux attach -t ${SESSION}"
