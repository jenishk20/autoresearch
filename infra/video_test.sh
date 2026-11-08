#!/usr/bin/env bash
# Train 20k steps, then run the verify job and check that a rollout video is rendered under gVisor (osmesa).
set -u
FLAGS="--runtime=runsc --network none --cpus 2 --memory 3g --memory-swap 3g --pids-limit 256 --read-only --tmpfs /tmp:size=256m --cap-drop ALL --security-opt no-new-privileges --user 1000:1000 -e OMP_NUM_THREADS=2 -e HOME=/tmp -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa"
B=/srv/videotest/$(date +%s); mkdir -p $B/t/job $B/t/out $B/v/job $B/v/out && chmod 777 $B/t/out $B/v/out
printf '{"run_id":"vt","task":"rl","dataset":"Hopper-v4","model":"ppo","params":{},"metric":"return","seed":1,"budget":{"unit":"env_steps","total":20480,"anneal_total":20480},"ckpt_every":20480}' > $B/t/job/config.json
docker run --rm $FLAGS -v $B/t/job:/job:ro -v $B/t/out:/out autoresearch-worker:latest python -m worker.run --config /job/config.json
CK=$(ls $B/t/out/ckpt | tail -1)
printf '{"run_id":"vv","task":"rl","dataset":"Hopper-v4","model":"ppo","params":{},"metric":"return","seed":1,"budget":{"unit":"env_steps","total":1},"resume_from":"%s"}' $CK > $B/v/job/config.json
t=$(date +%s)
docker run --rm $FLAGS -v $B/v/job:/job:ro -v $B/v/out:/out -v $B/t/out/ckpt:/parent:ro autoresearch-worker:latest python -m worker.run --config /job/config.json --evaluate
echo "verify took $(( $(date +%s) - t ))s"; ls -la $B/v/out; python3 -c "import json; d=json.load(open('$B/v/out/eval.json')); print('mean', d['mean_return'], 'video', d['video'])"
