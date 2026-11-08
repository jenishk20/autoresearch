#!/usr/bin/env bash
# Probe: can panda-gym (Franka Panda, PyBullet) run + render inside gVisor? Builds a throwaway image layer.
set -eu
D=/srv/probe/panda-$(date +%s); mkdir -p $D
cat > $D/Dockerfile <<'DF'
FROM autoresearch-worker:latest
USER root
RUN pip install --no-cache-dir "panda-gym==3.0.7" 2>&1 | tail -2
USER 1000
DF
docker build -q -t autoresearch-panda-probe $D
cat > $D/probe.py <<'PY'
import time, gymnasium as gym, panda_gym, numpy as np
for env_id in ["PandaReachDense-v3", "PandaPushDense-v3", "PandaPickAndPlaceDense-v3"]:
    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    print(env_id, "obs keys", {k: v.shape for k, v in obs.items()}, "act", env.action_space.shape, "max_steps", env.spec.max_episode_steps)
    t = time.time(); n = 0
    while time.time() - t < 3:
        obs, r, te, tr, info = env.step(env.action_space.sample()); n += 1
        if te or tr: env.reset()
    t2 = time.time(); f = env.render(); rt = time.time() - t2
    print(f"  {n/3:.0f} steps/s | frame {f.shape} in {rt*1000:.0f} ms")
PY
docker run --rm --runtime=runsc --network none --cpus 2 --memory 3g --read-only --tmpfs /tmp:size=256m --cap-drop ALL --user 1000:1000 -e HOME=/tmp -v $D/probe.py:/probe.py:ro autoresearch-panda-probe python /probe.py
