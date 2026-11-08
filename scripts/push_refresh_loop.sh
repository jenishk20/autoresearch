#!/usr/bin/env bash
# Refresh the imported push study every 5 minutes; at 17:50 do the final sync (before the VM is retired at 18:00).
cd "$(dirname "$0")/.."
while true; do
  now=$(date +%H%M)
  if [ "$now" -ge 1750 ]; then
    .venv/bin/python scripts/import_push_runs.py --final >> runs/push_refresh.log 2>&1
    echo "final sync done $(date +%T)" >> runs/push_refresh.log
    exit 0
  fi
  .venv/bin/python scripts/import_push_runs.py >> runs/push_refresh.log 2>&1
  sleep 300
done
