#!/usr/bin/env bash
# Keep the imported push study current.
#  - before 17:50 (local time): refresh every 5 minutes, study shown as "training"
#  - from 17:50: show it as done with a results page, and keep refreshing those results
#    every 10 minutes while any push job is still training; stop once they have all finished.
cd "$(dirname "$0")/.."
while true; do
  now=$(date +%H%M)
  if [ "$now" -lt 1750 ]; then
    .venv/bin/python scripts/import_push_runs.py >> runs/push_refresh.log 2>&1
    sleep 300
    continue
  fi
  .venv/bin/python scripts/import_push_runs.py --final >> runs/push_refresh.log 2>&1
  if ! docker ps --filter name=push- --filter status=running --format '{{.Names}}' 2>/dev/null | grep -q .; then
    echo "all push jobs finished; final results written $(date +%T)" >> runs/push_refresh.log
    exit 0
  fi
  sleep 600
done
