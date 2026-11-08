#!/usr/bin/env bash
# 3 parallel 30k-step PPO jobs under gVisor with production flags; prints per-run sps and container start-up time.
set -u
FLAGS="--runtime=runsc --network none --cpus 2 --memory 3g --memory-swap 3g --pids-limit 256 --read-only --tmpfs /tmp:size=256m --cap-drop ALL --security-opt no-new-privileges --user 1000:1000 -e OMP_NUM_THREADS=2 -e HOME=/tmp -e MUJOCO_GL=osmesa"
B=/srv/bench/$(date +%s)
for i in 1 2 3; do
  mkdir -p $B/$i/job $B/$i/out && chmod 777 $B/$i/out
  printf '{"run_id":"b%s","task":"rl","dataset":"Hopper-v4","model":"ppo","params":{},"metric":"return","seed":%s,"budget":{"unit":"env_steps","total":30720,"anneal_total":30720},"ckpt_every":30720}' $i $i > $B/$i/job/config.json
  docker run -d --name bench$i-$(basename $B) $FLAGS -v $B/$i/job:/job:ro -v $B/$i/out:/out:rw autoresearch-worker:latest python -m worker.run --config /job/config.json >/dev/null
done
t=$(date +%s)
for i in 1 2 3; do docker wait bench$i-$(basename $B) >/dev/null; done
echo "3 parallel x 30720 steps: $(( $(date +%s) - t ))s"
for i in 1 2 3; do grep '"update"' $B/$i/out/metrics.jsonl | tail -1 | python3 -c "import json,sys; print('sps', json.load(sys.stdin)['extra']['sps'])"; done
t=$(date +%s%N); docker run --rm $FLAGS autoresearch-worker:latest python -c "import torch, gymnasium, mujoco"
echo "container start + imports: $(( ($(date +%s%N) - t) / 1000000 )) ms"
docker container prune -f >/dev/null
