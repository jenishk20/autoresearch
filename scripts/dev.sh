#!/usr/bin/env bash
# Local dev launcher: node agent (local backend, NOT sandboxed) + control plane UI.
# Usage: scripts/dev.sh            then open http://127.0.0.1:8000
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -x "$PY" ] || { echo "Create the venv first: see README (uv venv --python 3.11 .venv && uv pip install ...)"; exit 1; }
mkdir -p runs
SLOTS="${NODE_SLOTS:-3}"
echo "starting node agent on :8100 (local backend, $SLOTS slots)"
NODE_BACKEND=local NODE_SLOTS="$SLOTS" $PY -m uvicorn node_agent.app:app --host 127.0.0.1 --port 8100 --log-level warning > runs/node_agent.log 2>&1 &
NODE_PID=$!
trap 'kill $NODE_PID 2>/dev/null || true' EXIT INT TERM
sleep 1.5
PORT="${PORT:-8000}"
echo "starting control plane on http://127.0.0.1:$PORT"
$PY -m uvicorn control.app:app --host 127.0.0.1 --port "$PORT" --log-level warning
