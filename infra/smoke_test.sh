#!/usr/bin/env bash
# Runs a 4k-step PPO job + a verify job under gVisor with the exact production flags, plus an egress probe.
set -euo pipefail
R=/srv/runs/smoke && rm -rf $R && mkdir -p $R/job $R/out && chmod 777 $R/out
cat > $R/job/config.json <<'J'
{"run_id":"smoke","task":"rl","dataset":"Hopper-v4","model":"ppo","params":{},"metric":"return","seed":1,
 "budget":{"unit":"env_steps","total":4096,"anneal_total":4096},"ckpt_every":4096}
J
FLAGS="--rm --runtime=runsc --network none --cpus 2 --memory 3g --memory-swap 3g --pids-limit 256 --read-only
 --tmpfs /tmp:size=256m --cap-drop ALL --security-opt no-new-privileges --user 1000:1000
 -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e HOME=/tmp -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa"
t=$(date +%s)
docker run $FLAGS -v $R/job:/job:ro -v $R/out:/out:rw autoresearch-worker:latest python -m worker.run --config /job/config.json
echo "train exit=$? in $(( $(date +%s) - t ))s"; tail -2 $R/out/metrics.jsonl
docker run $FLAGS autoresearch-worker:latest python -c "
import socket
try:
    socket.create_connection(('1.1.1.1', 80), 2); print('EGRESS OPEN (bad)')
except OSError as e:
    print('egress blocked:', e)
try:
    open('/etc/x', 'w')
except OSError as e:
    print('rootfs write blocked:', e)
"
