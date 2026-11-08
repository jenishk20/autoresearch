#!/usr/bin/env bash
# Launch the Panda push comparison on the worker VM: SAC+HER (sparse), SAC (dense), PPO (dense, 1M steps).
# Each job is its own gVisor sandbox with the production flags. Outputs under /srv/push/<name>/out.
set -eu
B=/srv/push
mkdir -p $B/build
cat > $B/build/Dockerfile <<'DF'
FROM autoresearch-worker:latest
USER root
RUN pip install --no-cache-dir "stable-baselines3==2.3.2" 2>&1 | tail -1
USER 1000
DF
docker build -q -t autoresearch-sb3 $B/build >/dev/null
FLAGS="--runtime=runsc --network none --cpus 2 --memory 4g --memory-swap 4g --pids-limit 256 --read-only --tmpfs /tmp:size=256m --cap-drop ALL --security-opt no-new-privileges --user 1000:1000 -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e HOME=/tmp -e PYTHONUNBUFFERED=1"
launch() {  # name, image, cmd...
  local name=$1 img=$2; shift 2
  mkdir -p $B/$name/out $B/$name/job && chmod 777 $B/$name/out
  docker rm -f push-$name >/dev/null 2>&1 || true
  docker run -d --name push-$name $FLAGS -v $B/$name/out:/out:rw -v $B/$name/job:/job:ro "$img" "$@" >/dev/null
  echo "launched $name"
}
launch sac_her autoresearch-sb3 python -m worker.sac_push --env PandaPush-v3 --her --steps 600000 --every 25000
launch sac_dense autoresearch-sb3 python -m worker.sac_push --env PandaPushDense-v3 --steps 600000 --every 25000
printf '{"run_id":"ppo_push_1m","task":"rl","dataset":"PandaPushDense-v3","model":"ppo","params":{},"metric":"return","seed":1,"budget":{"unit":"env_steps","total":1000000,"anneal_total":1000000},"ckpt_every":100000,"video_every":100000}' > $B/ppo_dense/job/config.json 2>/dev/null || { mkdir -p $B/ppo_dense/job; printf '{"run_id":"ppo_push_1m","task":"rl","dataset":"PandaPushDense-v3","model":"ppo","params":{},"metric":"return","seed":1,"budget":{"unit":"env_steps","total":1000000,"anneal_total":1000000},"ckpt_every":100000,"video_every":100000}' > $B/ppo_dense/job/config.json; }
launch ppo_dense autoresearch-worker:latest python -m worker.run --config /job/config.json
docker ps --filter name=push- --format '{{.Names}} {{.Status}}'
