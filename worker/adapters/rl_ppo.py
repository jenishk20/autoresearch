"""PPO for continuous control. Structure follows CleanRL's ppo_continuous_action.py
(https://docs.cleanrl.dev/rl-algorithms/ppo/); the changes are the ones listed in
docs/rl-autoresearch-plan.md §4.1:

  1 pinned stack (gymnasium 0.29.1 / mujoco 2.3.7 / numpy<2)   6 wrapper order with TrueReturn
  2 config from JSON, clamped by the worker                      7 metrics.jsonl, flushed per line
  3 torch threads = OMP_NUM_THREADS                              8 checkpoints + sha256
  4 num_envs=4 x num_steps=512 (batch 2048)                      9 STOP flag -> graceful exit
  5 obs normalisation inside the agent (RunningNorm)            10 resume / fork with lineage LR schedule
"""
from __future__ import annotations

import math
import os
import random
import sys
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from worker.common.ckpt import ckpt_name, sha256_file
from worker.common.jsonl import MetricsWriter
from worker.common.norm import RunningNorm
from worker.common.paths import OUT, PARENT, stop_requested


# ------------------------------------------------------------------ wrappers
class TrueReturn(gym.Wrapper):
    """Sits directly on the base env, so the raw env return is always logged even when a shaped
    reward or reward normalisation is applied further out."""

    def __init__(self, env):
        super().__init__(env)
        self._ret, self._len = 0.0, 0

    def reset(self, **kw):
        self._ret, self._len = 0.0, 0
        return self.env.reset(**kw)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self._ret += float(r)
        self._len += 1
        if term or trunc:
            info["true_return"], info["true_len"] = self._ret, self._len
            if "is_success" in info:   # goal-reaching tasks (Panda): did the robot finish the task?
                info["true_success"] = float(info["is_success"])
        return obs, r, term, trunc, info


class ShapedReward(gym.Wrapper):
    def __init__(self, env, fn):
        super().__init__(env)
        self.fn, self._obs = fn, None

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self._obs = obs
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        shaped = float(self.fn(self._obs, action, obs, float(r), info))
        self._obs = obs
        return obs, shaped, term, trunc, info


PANDA_CAMERA = dict(render_width=480, render_height=360, render_target_position=[0.0, 0.0, 0.05],
                    render_distance=1.1, render_yaw=60, render_pitch=-35)   # table, cube and target all in frame


def base_env(env_id, render=False):
    """Plain env. Panda tasks (panda-gym, PyBullet) have dict observations -> flattened to one vector."""
    if env_id.startswith("Panda"):
        import panda_gym  # noqa: F401  (registers the Panda envs)
        kw = dict(render_mode="rgb_array", **PANDA_CAMERA) if render else {}
        return gym.wrappers.FlattenObservation(gym.make(env_id, **kw))
    return gym.make(env_id, render_mode="rgb_array") if render else gym.make(env_id)


def can_render(env_id) -> bool:
    if env_id.startswith("Panda"):
        return True   # PyBullet's CPU renderer needs no OpenGL
    return os.environ.get("MUJOCO_GL", "") not in ("", "disable", "disabled", "off", "0")


_RENDER_ENVS = {}


def _caption(frame, text, ok):
    """Burn a small status label into the frame (top-left)."""
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(frame); d = ImageDraw.Draw(im)
        w = 8 + 7 * len(text)
        d.rectangle([8, 8, 8 + w, 30], fill=(20, 20, 19) if ok is None else ((60, 120, 60) if ok else (150, 60, 50)))
        d.text((14, 12), text, fill=(255, 255, 255))
        return np.asarray(im)
    except Exception:
        return frame


def record_rollout(agent, env_id, seed, path, episodes=None) -> tuple[float, float | None]:
    """Deterministic rollouts with the current policy, saved as one mp4 (Panda: 2 attempts, slowed down,
    each labelled with its outcome). Returns (mean true return, success rate or None)."""
    import imageio
    env = _RENDER_ENVS.get(env_id) or _RENDER_ENVS.setdefault(env_id, base_env(env_id, render=True))
    panda = env_id.startswith("Panda")
    episodes = episodes or (2 if panda else 1)
    fps = env.metadata.get("render_fps") or 25
    k = 1 if panda else max(1, round(fps / 30))
    out_fps = 12 if panda else min(30, fps / k)        # Panda episodes are 2 s at real speed; slow them down
    lo, hi = env.action_space.low, env.action_space.high
    frames, rets, succ = [], [], []
    for ep in range(episodes):
        o, _ = env.reset(seed=seed + ep)
        ep_frames, ret, done, n, info = [env.render()], 0.0, False, 0, {}
        while not done:
            a = agent.act_deterministic(torch.tensor(o, dtype=torch.float32).unsqueeze(0))[0].numpy()
            o, r, term, trunc, info = env.step(np.clip(a, lo, hi))
            ret += float(r); n += 1; done = term or trunc
            if n % k == 0:
                ep_frames.append(env.render())
        ok = float(info["is_success"]) if "is_success" in info else None
        rets.append(ret)
        if ok is not None:
            succ.append(ok)
        if panda:
            label = f"attempt {ep + 1}"
            ep_frames = [_caption(f, label, None) for f in ep_frames]
            end = _caption(ep_frames[-1], f"attempt {ep + 1}: {'success' if ok else 'missed'}", bool(ok))
            ep_frames += [end] * int(out_fps * 1.2)      # hold the outcome for ~1 s
        frames += ep_frames
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=out_fps, macro_block_size=8)
    return float(np.mean(rets)), (float(np.mean(succ)) if succ else None)


def make_env(env_id, gamma, reward_fn=None):
    def thunk():
        env = base_env(env_id)
        env = TrueReturn(env)
        if reward_fn is not None:
            env = ShapedReward(env, reward_fn)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk


# ------------------------------------------------------------------ agent
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def mlp(sizes, out, out_std):
    layers, d = [], sizes[0]
    for h in sizes[1:]:
        layers += [layer_init(nn.Linear(d, h)), nn.Tanh()]
        d = h
    layers.append(layer_init(nn.Linear(d, out), std=out_std))
    return nn.Sequential(*layers)


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(64, 64), norm_obs=True):
        super().__init__()
        self.norm = RunningNorm(obs_dim, enabled=norm_obs)
        self.critic = mlp([obs_dim, *hidden], 1, 1.0)
        self.actor_mean = mlp([obs_dim, *hidden], act_dim, 0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    # x is already normalised (normalisation happens once, at rollout time)
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    @torch.no_grad()
    def act_deterministic(self, raw_obs):
        return self.actor_mean(self.norm(raw_obs))


# ------------------------------------------------------------------ checkpoint
def save_ckpt(writer, agent, optimizer, global_step, params, env_id):
    d = OUT / "ckpt"
    d.mkdir(parents=True, exist_ok=True)
    path = d / ckpt_name(global_step, "pt")
    torch.save({
        "agent": agent.state_dict(), "optimizer": optimizer.state_dict(), "global_step": global_step,
        "params": params, "env_id": env_id,
        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()},
    }, path)
    writer.write("checkpoint", step=global_step, path=f"ckpt/{path.name}", sha256=sha256_file(path))


# ------------------------------------------------------------------ train
def train(cfg) -> int:
    from control.models import worker_clamp

    p = worker_clamp("ppo", cfg.params)          # change 2: never trust input
    threads = int(os.environ.get("OMP_NUM_THREADS", "2"))
    torch.set_num_threads(threads)               # change 3

    env_id, seed = cfg.dataset, cfg.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cpu")

    reward_fn = None
    if cfg.feature_fn:
        from worker.common.harness import load_function
        reward_fn = load_function(cfg.feature_fn, "shaped_reward")

    envs = gym.vector.SyncVectorEnv([make_env(env_id, p["gamma"], reward_fn) for _ in range(p["num_envs"])])
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    agent = Agent(obs_dim, act_dim, tuple(p["hidden"]), p["norm_obs"]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=p["learning_rate"], eps=1e-5)

    writer = MetricsWriter()
    global_step = cfg.budget.start_at
    if cfg.resume_from:                          # change 10: resume / fork
        ck = torch.load(PARENT / os.path.basename(cfg.resume_from), map_location=device, weights_only=False)
        agent.load_state_dict(ck["agent"])
        optimizer.load_state_dict(ck["optimizer"])
        global_step = int(ck["global_step"]) if not cfg.budget.start_at else cfg.budget.start_at
        # overrides (lr / ent / clip) come from the new config's params, already in p
    total = cfg.budget.total
    anneal_total = cfg.budget.anneal_total or total
    lr0 = p["learning_rate"]

    num_envs, num_steps = p["num_envs"], p["num_steps"]
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // p["num_minibatches"]
    num_iterations = max(1, math.ceil((total - global_step) / batch_size))
    ckpt_every = max(batch_size, int(cfg.ckpt_every))
    next_ckpt = (global_step // ckpt_every + 1) * ckpt_every

    obs = torch.zeros((num_steps, num_envs, obs_dim))
    actions = torch.zeros((num_steps, num_envs, act_dim))
    logprobs = torch.zeros((num_steps, num_envs))
    rewards = torch.zeros((num_steps, num_envs))
    dones = torch.zeros((num_steps, num_envs))
    values = torch.zeros((num_steps, num_envs))

    next_obs_raw, _ = envs.reset(seed=seed + global_step)
    next_obs_raw = torch.tensor(next_obs_raw, dtype=torch.float32)
    next_done = torch.zeros(num_envs)
    recent_true, recent_len, recent_succ = [], [], []
    video_every = int(getattr(cfg, "video_every", 0) or 0) if can_render(env_id) else 0
    next_video = (global_step // video_every + 1) * video_every if video_every else None

    def checkpoint_video(step):
        try:
            rel = f"videos/step_{step:09d}.mp4"
            ret, succ = record_rollout(agent, env_id, 10_000, OUT / rel)
            writer.write("video", step=step, path=rel, ret_true=round(ret, 2), success=succ)
        except Exception as e:  # videos are optional; never kill training over them
            print(f"video skipped: {e}", file=sys.stderr)
    t_start, steps_start = time.time(), global_step

    for iteration in range(1, num_iterations + 1):
        if stop_requested():                     # change 9
            save_ckpt(writer, agent, optimizer, global_step, p, env_id)
            writer.write("done", step=global_step, reason="stop_flag")
            return 0

        lr_now = lr0 * max(0.0, 1.0 - global_step / anneal_total)
        for g in optimizer.param_groups:
            g["lr"] = max(lr_now, lr0 * 0.01)

        for step in range(num_steps):
            global_step += num_envs
            agent.norm.update(next_obs_raw)
            next_obs = agent.norm(next_obs_raw)
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step], logprobs[step] = action, logprob

            o, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward, dtype=torch.float32).view(-1)
            next_obs_raw = torch.tensor(o, dtype=torch.float32)
            next_done = torch.tensor(next_done_np, dtype=torch.float32)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        rt = float(info.get("true_return", info["episode"]["r"]))
                        ln = int(info.get("true_len", info["episode"]["l"]))
                        recent_true = (recent_true + [rt])[-10:]
                        recent_len = (recent_len + [ln])[-10:]
                        succ = info.get("true_success")
                        if succ is not None:
                            recent_succ = (recent_succ + [succ])[-20:]
                        writer.write("episode", step=global_step, ret_true=round(rt, 3),
                                     ret_shaped=round(float(info["episode"]["r"]), 3), len=ln, success=succ)

        # bootstrap value (GAE)
        with torch.no_grad():
            next_value = agent.get_value(agent.norm(next_obs_raw)).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal, nextvalues = 1.0 - next_done, next_value
                else:
                    nextnonterminal, nextvalues = 1.0 - dones[t + 1], values[t + 1]
                delta = rewards[t] + p["gamma"] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + p["gamma"] * p["gae_lambda"] * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, act_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(p["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start:start + minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > p["clip_coef"]).float().mean().item()]
                mb_adv = b_advantages[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - p["clip_coef"], 1 + p["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -p["clip_coef"], p["clip_coef"])
                v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - p["ent_coef"] * entropy_loss + v_loss * 0.5
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), p["max_grad_norm"])
                optimizer.step()

        if not torch.isfinite(loss).item():
            writer.write("done", step=global_step, reason="nan")
            return 3

        y_pred, y_true = b_values.numpy(), b_returns.numpy()
        var_y = np.var(y_true)
        ev = float("nan") if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)
        sps = (global_step - steps_start) / max(1e-6, time.time() - t_start)
        writer.write("update", step=global_step, unit="env_steps", train_loss=round(loss.item(), 5),
                     val_metric=round(float(np.mean(recent_true)), 3) if recent_true else None,
                     extra={"sps": round(sps, 1), "approx_kl": round(approx_kl.item(), 5),
                            "clipfrac": round(float(np.mean(clipfracs)), 4), "ev": round(ev, 4) if math.isfinite(ev) else 0.0,
                            "lr": optimizer.param_groups[0]["lr"], "entropy": round(entropy_loss.item(), 4),
                            "pg_loss": round(pg_loss.item(), 5), "v_loss": round(v_loss.item(), 5),
                            "ep_len": round(float(np.mean(recent_len)), 1) if recent_len else 0.0,
                            **({"success": round(float(np.mean(recent_succ)), 3)} if recent_succ else {})})

        if global_step >= next_ckpt and global_step < total:   # change 8 (final one saved below)
            save_ckpt(writer, agent, optimizer, global_step, p, env_id)
            next_ckpt += ckpt_every
        if next_video is not None and global_step >= next_video and global_step < total:
            checkpoint_video(global_step)
            next_video += video_every

    save_ckpt(writer, agent, optimizer, global_step, p, env_id)
    if video_every:
        checkpoint_video(global_step)
    writer.write("done", step=global_step, reason="budget")
    envs.close()
    return 0


# ------------------------------------------------------------------ evaluate (verifier job)
def evaluate(cfg, episodes=10, seed0=10_000) -> dict:
    """Held-out seeds (10_000+), deterministic actions, raw env reward. Optional median-seed video."""
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    ck = torch.load(PARENT / os.path.basename(cfg.resume_from), map_location="cpu", weights_only=False)
    p = ck["params"]
    env = base_env(cfg.dataset)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    agent = Agent(obs_dim, act_dim, tuple(p["hidden"]), p["norm_obs"])
    agent.load_state_dict(ck["agent"])
    agent.eval()
    lo, hi = env.action_space.low, env.action_space.high

    def run_episode(e, s):
        o, _ = e.reset(seed=s)
        ret, n, done, info = 0.0, 0, False, {}
        while not done:
            a = agent.act_deterministic(torch.tensor(o, dtype=torch.float32).unsqueeze(0))[0].numpy()
            o, r, term, trunc, info = e.step(np.clip(a, lo, hi))
            ret += float(r); n += 1; done = term or trunc
        return ret, n, (float(info["is_success"]) if "is_success" in info else None)

    rets, lens, succ = [], [], []
    for i in range(episodes):
        r, n, sc = run_episode(env, seed0 + i)
        rets.append(round(r, 3)); lens.append(n)
        if sc is not None:
            succ.append(sc)
    out = {"env_id": cfg.dataset, "seeds": [seed0 + i for i in range(episodes)], "returns": rets,
           "lengths": lens, "mean_return": float(np.mean(rets)), "video": None,
           "success_rate": float(np.mean(succ)) if succ else None}
    if can_render(cfg.dataset):
        try:
            med = int(np.argsort(rets)[len(rets) // 2])   # median seed, not the best one
            record_rollout(agent, cfg.dataset, seed0 + med, OUT / "video.mp4")
            out["video"] = {"path": "video.mp4", "seed": seed0 + med, "note": "median seed"}
        except Exception as e:  # rendering is optional
            print(f"video skipped: {e}", file=sys.stderr)
    return out
