"""SAC (optionally with Hindsight Experience Replay) on a Panda goal task, run INSIDE the gVisor sandbox.

  python -m worker.sac_push --env PandaPush-v3 --her --steps 600000 --every 25000

Writes /out/metrics.jsonl ({"type":"eval"} every --every steps: success rate over 20 unseen episodes, mean return),
a labelled rollout video per eval under /out/videos/, and the model under /out/ckpt/.
Hyper-parameters follow the RL Zoo settings for panda-gym (SAC/TQC + HER): gamma 0.95, tau 0.05, lr 1e-3,
batch 256 (512 with HER), 3x256 networks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401
import torch
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback

OUT = Path(os.environ.get("OUT_DIR", "/out"))
CAMERA = dict(render_width=480, render_height=360, render_target_position=[0.0, 0.0, 0.05],
              render_distance=1.1, render_yaw=60, render_pitch=-35)


def log(rec):
    with open(OUT / "metrics.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def caption(frame, text, ok):
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(frame); d = ImageDraw.Draw(im)
        d.rectangle([8, 8, 16 + 7 * len(text), 30], fill=(20, 20, 19) if ok is None else ((60, 120, 60) if ok else (150, 60, 50)))
        d.text((14, 12), text, fill=(255, 255, 255))
        return np.asarray(im)
    except Exception:
        return frame


class EvalAndFilm(BaseCallback):
    def __init__(self, env_id, every, tag):
        super().__init__()
        self.env_id, self.every, self.tag = env_id, every, tag
        self.eval_env = gym.make(env_id)
        self.film_env = gym.make(env_id, render_mode="rgb_array", **CAMERA)
        self.next = every
        self.t0 = time.time()

    def _evaluate(self, n=20):
        succ, rets = [], []
        for i in range(n):
            o, _ = self.eval_env.reset(seed=10_000 + i)   # held-out seeds
            done, ret, info = False, 0.0, {}
            while not done:
                a, _ = self.model.predict(o, deterministic=True)
                o, r, te, tr, info = self.eval_env.step(a)
                ret += float(r); done = te or tr
            succ.append(float(info.get("is_success", 0.0))); rets.append(ret)
        return float(np.mean(succ)), float(np.mean(rets))

    def _film(self, step):
        frames = []
        for ep in range(2):
            o, _ = self.film_env.reset(seed=20_000 + ep)
            ep_frames, done, info = [self.film_env.render()], False, {}
            while not done:
                a, _ = self.model.predict(o, deterministic=True)
                o, r, te, tr, info = self.film_env.step(a)
                done = te or tr
                ep_frames.append(self.film_env.render())
            ok = bool(info.get("is_success", 0.0))
            ep_frames = [caption(f, f"{self.tag} · {step // 1000}k · attempt {ep + 1}", None) for f in ep_frames]
            ep_frames += [caption(ep_frames[-1], f"attempt {ep + 1}: {'success' if ok else 'missed'}", ok)] * 14
            frames += ep_frames
        import imageio
        rel = f"videos/step_{step:09d}.mp4"
        (OUT / "videos").mkdir(parents=True, exist_ok=True)
        imageio.mimsave(OUT / rel, frames, fps=12, macro_block_size=8)
        return rel

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next:
            self.next += self.every
            sr, ret = self._evaluate()
            rel = self._film(self.num_timesteps)
            (OUT / "ckpt").mkdir(parents=True, exist_ok=True)
            self.model.save(OUT / "ckpt" / f"step_{self.num_timesteps:09d}")
            sps = self.num_timesteps / max(1e-6, time.time() - self.t0)
            log({"type": "eval", "step": self.num_timesteps, "success": round(sr, 3), "ret": round(ret, 2),
                 "video": rel, "sps": round(sps, 1), "t": round(time.time() - self.t0, 1)})
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="PandaPush-v3")
    ap.add_argument("--her", action="store_true")
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--every", type=int, default=25_000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    OUT.mkdir(parents=True, exist_ok=True)
    env = gym.make(a.env)
    kw = {}
    if a.her:
        kw = dict(replay_buffer_class=HerReplayBuffer,
                  replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy="future"))
    model = SAC("MultiInputPolicy", env, buffer_size=1_000_000, batch_size=512 if a.her else 256, learning_rate=1e-3,
                gamma=0.95, tau=0.05, learning_starts=1000, policy_kwargs=dict(net_arch=[256, 256, 256]),
                seed=a.seed, verbose=0, **kw)
    tag = "SAC+HER" if a.her else "SAC"
    log({"type": "start", "algo": tag, "env": a.env, "steps": a.steps})
    model.learn(total_timesteps=a.steps, callback=EvalAndFilm(a.env, a.every, tag))
    log({"type": "done", "step": a.steps})


if __name__ == "__main__":
    sys.exit(main())
