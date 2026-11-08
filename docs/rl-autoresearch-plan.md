# RL auto-research agent — build plan

> **Status: this is now the RL adapter spec (Demo 2).** The main plan is `docs/autoresearch-plan.md`, and it wins wherever the two differ.
> Still authoritative here: §4.1 (PPO worker changes 1–10), §4.2 (sandbox flags), §4.3 (node agent API, plus the additions in main plan §10.5), §6.1–6.3 (envs, search space, halving maths) and the RL parts of §4.7–4.8.
> Superseded by the main plan: §0 pitch, §1 checklist, §3 repo layout, §5 schemas (now generic, main plan §10), §6.4–6.5 presets, §7 budget, §8 timeline, §11 video script, §12 open items.

**Event:** Vultr Agent Rush (lablab.ai), Challenge 1 "Blast Radius Zero" · **Due:** today, 8 Nov 2026 (check the exact cutoff time on the lablab page and write it here: `DEADLINE = __:__`)
**Team:** 2 people · **Stack:** CleanRL-style PyTorch workers, Docker + gVisor on Vultr CPU VMs, Vultr Serverless Inference for every LLM call · **No GPUs**

---

## 0. One-paragraph pitch

A user types a task ("find the best PPO setup for locomotion on HalfCheetah and Walker2d, check it transfers to Hopper, 30-minute budget"). The agent writes a research plan with **pre-registered, machine-checkable predictions**. It dispatches each training run into its own **gVisor sandbox with no network**, watches the live learning curves and **kills, continues or forks runs** with logged reasons. It **re-verifies the winners in fresh sandboxes on held-out seeds with the true environment reward**, then publishes a report: winning config per environment, IQM ± bootstrap CI, which predictions held, rollout videos, checkpoint hashes, and every containment event. The LLM never executes code on the control plane. It only emits JSON, which is validated and bounded by code.

What makes it more than a sweep:
1. The agent intervenes mid-run (successive halving plus LLM-proposed forks).
2. Its predictions get scored against outcomes (a calibration score).
3. Every number in the report is computed by code and traceable to a run ID and a checkpoint SHA-256.

---

## 1. Constraints and submission checklist

| Requirement | How we meet it |
|---|---|
| Agent LLM calls use Vultr Serverless Inference (required for Challenge 1) | `control/llm.py` uses the OpenAI SDK with `base_url=https://api.vultrinference.com/v1` |
| Isolated execution (Docker recommended; gVisor listed) | Every run is `docker run --runtime=runsc --network none …` |
| Vultr is the central control layer, dispatching work and returning verifiable output | Control VM orchestrates worker VMs over a Vultr VPC; report carries hashes + re-verification |
| Public demo URL | Control VM behind Caddy, `https://<ip>.sslip.io` |
| Recorded demo video, with ≥1 containment moment for this track | Section 11 script; red-team presets (network egress + memory bomb) |
| Clear architecture and use-case explanation | README + slide PDF (lablab judges presentation on PDF + video) |
| No GPUs | CPU-only MuJoCo (state observations); no pixel/CNN policies |
| $200 credits per person | Budget math in §7; destroy worker VMs when done |

LLM note: per Vultr docs, tool calling is only supported on `kimi-k2-instruct`. **We don't use tool calling.** Every agent step is "prompt → JSON → pydantic validation → bounded action". So any chat model works. At kickoff, run `GET https://api.vultrinference.com/v1/chat/models`, pick the strongest instruct model, and default to `kimi-k2-instruct`.

---

## 2. Architecture

```
                         ┌──────────────────────── Control VM (2–4 vCPU) ────────────────────────┐
 User / judge ──HTTPS──▶ │ FastAPI + htmx UI ── SSE ──▶ browser (live curves, decisions, report)  │
                         │   │                                                                     │
                         │   ├── Planner ───┐                                                      │
                         │   ├── Monitor ───┼──▶ llm.py ──HTTPS──▶ Vultr Serverless Inference     │
                         │   ├── Reporter ──┘                                                      │
                         │   ├── Scheduler (queue, slots, waves)    SQLite (WAL): sessions, runs,  │
                         │   ├── Verifier / Stats                   metrics, decisions, events     │
                         │   └── Safety (AST check on generated reward code)                       │
                         └───────────────┬─────────────────────────────────────────────────────────┘
                                         │ Vultr VPC (private), bearer token
                     ┌───────────────────┴───────────────────┐
             ┌───────▼─────── Worker VM 1 (32 vCPU) ─┐   ┌────▼──── Worker VM 2 (optional) ──┐
             │ node_agent (trusted, outside sandbox) │   │ same image, cloned from snapshot  │
             │   docker run --runtime=runsc          │   └───────────────────────────────────┘
             │   ┌─────────┐┌─────────┐┌─────────┐   │
             │   │ run 001 ││ run 002 ││ run 003 │…  │  each: 2 vCPU, 3 GB, no network,
             │   └─────────┘└─────────┘└─────────┘   │  read-only rootfs, non-root, caps dropped
             │   /srv/runs/<id>/{job:ro, out:rw}     │  metrics leave via the /out bind mount
             └───────────────────────────────────────┘
```

Trust boundaries:
- **Untrusted:** everything inside a container, including LLM-written reward code and anything the policy does.
- **Trusted:** `node_agent` (tails files, starts and kills containers) and the control plane.
- **Secrets:** the inference API key lives only on the control VM. Worker VMs hold only the node-agent token. Containers hold nothing.

MVP topology: 1 control VM + 1 worker VM. Set up worker 1, take a Vultr snapshot, and clone worker 2 from it only if time and quota allow. The node-agent API is the same either way, so adding a worker is one line in `control/config.yaml`.

---

## 3. Repo layout

```
rl-autoresearch/
  worker/                    # runs INSIDE the sandbox
    Dockerfile
    requirements.txt
    train_ppo.py             # CleanRL ppo_continuous_action.py + the changes in §4.1
    train_sac.py             # stretch goal (CleanRL sac_continuous_action.py)
    evaluate.py              # verifier job: held-out-seed eval + mp4
    common/
      norm.py                # in-agent RunningNorm (saved in state_dict)
      jsonl.py               # metrics writer (flush every line)
      ckpt.py                # save/load + sha256
      reward.py              # loads optional /job/reward.py, TrueReturn wrapper
  node_agent/
    app.py                   # FastAPI: start / stop / kill / status / metrics / files
  control/
    app.py                   # FastAPI routes + SSE
    config.yaml              # nodes, slots, model name, presets
    db.py  models.py         # SQLite + pydantic schemas (§5)
    llm.py                   # Vultr client, JSON extraction, retry, logging
    planner.py  monitor.py  features.py  scheduler.py
    verifier.py  stats.py  predictions.py  reporter.py
    safety.py                # AST allowlist for reward code
    replay.py                # replays a stored session at N× speed
    templates/  static/      # htmx + uPlot (or Chart.js)
  infra/
    setup_worker.sh  setup_control.sh  Caddyfile
  scripts/
    bench.py                 # steps/sec benchmark, runc vs runsc
    redteam/                 # malicious reward files for the containment demo
  README.md
```

---

## 4. Components

### 4.1 Worker: `train_ppo.py` (Person B)

Start from CleanRL `ppo_continuous_action.py` and keep its structure. Changes:

1. **Pin the stack CleanRL was written against** to avoid Gymnasium 1.x vector-env API changes (`final_info` vs the `_episode` mask) and NumPy 2 breakage:
   `gymnasium==0.29.1`, `mujoco==2.3.7`, `numpy<2`, `torch` (CPU wheel), `imageio[ffmpeg]`. Use `HalfCheetah-v4`, `Walker2d-v4`, `Hopper-v4`, and `InvertedDoublePendulum-v4` as the smoke test.
2. **Config from JSON**: `python train_ppo.py --config /job/config.json` (schema in §5.1). Code clamps every hyperparameter to the bounds in §6.2. The worker never trusts its input.
3. **Threads**: put `torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))` at the top. The container also sets `OMP_NUM_THREADS=MKL_NUM_THREADS=2`. Without this, torch spawns a thread per *host* core inside a 2-CPU cgroup and throughput collapses.
4. **Vectorised envs**: `num_envs=4`, `num_steps=512`, so batch = 4 × 512 = 2048, the same as CleanRL's default `num_envs=1 × num_steps=2048`. Keep `num_minibatches=32` (minibatch 64), `update_epochs=10`.
5. **Obs normalisation inside the agent**, not as a per-env Gymnasium wrapper. CleanRL wraps each env separately, so there are 4 different normalisers and none of them are saved. That's the classic "eval is garbage after loading the checkpoint" bug. Use a module whose buffers are saved in `state_dict` automatically:

   ```python
   class RunningNorm(nn.Module):
       def __init__(self, dim, eps=1e-8, clip=10.0):
           super().__init__()
           self.register_buffer("mean", torch.zeros(dim))
           self.register_buffer("var", torch.ones(dim))
           self.register_buffer("count", torch.tensor(1e-4))
           self.eps, self.clip = eps, clip
       @torch.no_grad()
       def update(self, x):                       # x: [num_envs, dim], call once per env step
           bm, bv, bn = x.mean(0), x.var(0, unbiased=False), x.shape[0]
           d, tot = bm - self.mean, self.count + bn
           m2 = self.var * self.count + bv * bn + d.pow(2) * self.count * bn / tot
           self.mean.copy_(self.mean + d * bn / tot); self.var.copy_(m2 / tot); self.count.copy_(tot)
       def forward(self, x):
           return ((x - self.mean) / torch.sqrt(self.var + self.eps)).clamp(-self.clip, self.clip)
   ```
   During training, call `update` before each action. In `evaluate.py`, never call it (frozen). Keep `NormalizeReward` as a per-env wrapper; it only affects training. On resume it re-warms within ~10k steps, which is acceptable and documented.
6. **Wrapper order**, so ground-truth return is always logged:
   `base env → TrueReturn (logs raw env return into info) → [optional ShapedReward from /job/reward.py] → RecordEpisodeStatistics → ClipAction → NormalizeReward → TransformReward(clip)`.
7. **Metrics**: append JSON lines to `/out/metrics.jsonl` and flush every line (schema in §5.2). One `update` line per PPO iteration (every 2048 steps) and one `episode` line per finished episode.
8. **Checkpoints**: every 100k steps and on stop, save `{agent, optimizer, global_step, rng states, params}` to `/out/ckpt/step_XXXXXXXXX.pt`, then log the sha256.
9. **Stop flag**: at the top of each iteration, `if os.path.exists("/job/STOP")`, save a checkpoint, log `done(reason="stop_flag")` and exit 0. The host writes `STOP` into the read-only-for-the-container job directory. That's a graceful kill. `docker kill` is the fallback after 15 s.
10. **Resume / fork**: if `resume_from` is set, load the agent, optimizer and normaliser, set `global_step = start_step`, and apply overrides (lr, ent_coef, clip_coef). LR annealing continues on the lineage schedule: `lr_t = lr_0 · (1 − step / anneal_total_steps)`, with `anneal_total_steps` passed in config.

`evaluate.py` (verifier job): load a checkpoint and run 10 episodes on **eval seeds 10_000+** (training seeds are 1–5) with deterministic actions (the Gaussian mean) and the **raw env reward**. Write `eval.json` with per-episode returns and lengths. Optionally render one episode to mp4 with `MUJOCO_GL=osmesa` (CPU software rendering; add `libosmesa6` to the image).

Video timing:
- Use `env.metadata["render_fps"]`: HalfCheetah is 20 fps (dt = 0.01 × 5). Hopper and Walker2d are 125 fps (dt = 0.002 × 4).
- Render every k-th frame, with k = round(fps / 30): k = 1 for HalfCheetah, k = 4 for Hopper/Walker. That cuts render time 4× and keeps playback real-time.
- Render the **median seed**, not the best seed. That's honest, and the report says so.

### 4.2 Sandbox runner: Docker + gVisor (Person A)

Worker VM setup (Ubuntu 24.04):

```bash
# docker
curl -fsSL https://get.docker.com | sh
# gVisor
curl -fsSL https://gvisor.dev/archive.key | sudo gpg --dearmor -o /usr/share/keyrings/gvisor-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" \
  | sudo tee /etc/apt/sources.list.d/gvisor.list
sudo apt-get update && sudo apt-get install -y runsc
sudo runsc install && sudo systemctl restart docker
docker run --rm --runtime=runsc hello-world
# image (build once, then snapshot the VM)
docker build -t rlagent-worker:latest worker/
```

Per-run launch (built by `node_agent`):

```bash
docker run -d --name run-$ID --label rlagent.run=$ID \
  --runtime=runsc --network none \
  --cpus 2 --memory 3g --memory-swap 3g --pids-limit 256 \
  --read-only --tmpfs /tmp:size=256m \
  --cap-drop ALL --security-opt no-new-privileges --user 1000:1000 \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
  -v /srv/runs/$ID/job:/job:ro \
  -v /srv/runs/$ID/out:/out:rw \
  -v /srv/runs/$PARENT/out/ckpt:/parent:ro \          # only for resume/fork
  rlagent-worker:latest python train_ppo.py --config /job/config.json
```

- **Wall-clock cap**: the node agent runs a watchdog and kills the container at `max_seconds = 1.5 × expected` (`expected = steps / measured_sps`).
- **Exit classification** from `docker inspect`: `State.ExitCode` and `State.OOMKilled`. Under runsc the OOM flag may not be set, so also treat exit 137 with memory near the cap as an OOM. Verify this in the M1 smoke test.
- **Memory**: a PPO run uses about 0.6–1 GB, so the 3 GB cap gives headroom. A 32 vCPU / 64 GB worker with 2 vCPUs reserved for the host gives (32 − 2) / 2 = **15 slots**, using 15 × 3 = 45 GB.
- **Fallback** if runsc breaks MuJoCo or osmesa (test at M1): use the default runtime with every other flag kept. Say so honestly in the submission. That's still no network, no caps and resource-capped, just a weaker kernel boundary.

### 4.3 Node agent API (Person A)

Bind to the VPC IP only. Every request carries `Authorization: Bearer $NODE_TOKEN`.

| Method & path | Body / response |
|---|---|
| `POST /runs` | `{run_id, config, reward_py?, parent_run_id?, cpus, mem_gb, max_seconds}` → writes `/srv/runs/<id>/job/`, starts the container → `{ok}` |
| `POST /runs/{id}/stop` | creates `job/STOP`; after 15 s, `docker kill` |
| `POST /runs/{id}/kill` | immediate `docker kill` |
| `GET /runs/{id}/status` | `{state: running/exited, exit_code, oom_killed, started_at, finished_at}` |
| `GET /runs/{id}/metrics?offset=N` | `{lines: [...], offset: M}` (byte-offset tail of `metrics.jsonl`) |
| `GET /runs/{id}/files/{path}` | streams `eval.json`, `video.mp4`, checkpoints |
| `GET /runs/{id}/logs?tail=200` | container stderr, used to show containment errors |
| `GET /health` | `{free_slots, running}` |

### 4.4 Scheduler (Person A)

- **Run state machine**: `PENDING → STARTING → RUNNING → {DONE, STOPPED, FAILED, CONTAINED}`. `CONTAINED` means it was killed by a sandbox limit or exited on a blocked syscall.
- **Main loop** (async, every 2 s): assign `PENDING` runs to nodes with free slots, FIFO within the current wave.
- **Metrics poller** (every 5 s per running run): fetch new lines, insert them into SQLite, push them to the session's SSE channel.
- **Waves are sequential.** Screening "segments" are implemented as resumes (see §6.3), so there are no barriers inside a run. A run either finishes its segment or gets stopped.
- **Every state change** is written to `events` with a timestamp. That table is what replay mode uses.

### 4.5 LLM layer: `llm.py` (Person B)

```python
client = OpenAI(base_url="https://api.vultrinference.com/v1", api_key=os.environ["VULTR_INFERENCE_API_KEY"])

def ask_json(system, user, schema: type[BaseModel], temperature=0.2, retries=1):
    msgs = [{"role": "system", "content": system + "\nReturn ONLY one JSON object matching the schema."},
            {"role": "user", "content": user}]
    for attempt in range(retries + 1):
        r = client.chat.completions.create(model=MODEL, messages=msgs, temperature=temperature, timeout=60)
        txt = r.choices[0].message.content
        try:
            return schema.model_validate_json(extract_first_json_object(txt))
        except (ValidationError, ValueError) as e:
            msgs += [{"role": "assistant", "content": txt},
                     {"role": "user", "content": f"Invalid: {e}. Fix and return only JSON."}]
    raise LLMFailed
```

- Log every call to `events` (`kind="llm_call"`, latency, tokens, prompt hash). The UI shows a small "Serverless Inference: N calls, X s" badge, which proves the required tech is used.
- **The agent never blocks on the LLM.** If the planner fails, use the default plan in `config.yaml`. If the monitor fails, apply the rules only. If the reporter fails, use the templated report.

### 4.6 Planner (Person B)

Input:
- the user prompt
- the env registry (obs/act dims, termination behaviour, horizon 1000, the priors in §6.1)
- the bounded search space (§6.2)
- the compute budget: slots and minutes
- the **measured** steps/sec

Output: a `Plan` (§5.3).

Code then:
- clamps every value to its bounds
- computes `estimate_wall_minutes(plan, sps, slots)` (formula in §7)
- trims configs or steps if over budget, logging "planner asked for X, trimmed to Y"

Planner system prompt (sketch):

> You are an RL research lead planning experiments on Gymnasium MuJoCo v4 tasks under a fixed CPU budget. Choose ≤8 PPO configs from the allowed space, chosen to test specific hypotheses, not to cover the grid. Write 3–5 falsifiable predictions in the given schema with a probability you would bet at. Facts: Hopper and Walker2d terminate when the robot falls, so short episodes mean falling. HalfCheetah never terminates. Results at 150k steps are noisy with one seed. Observation normalisation usually matters a lot for PPO on these tasks. Do not invent hyperparameters outside the allowed values.

### 4.7 Monitor (Person B)

Every 30 s, `features.py` computes per run (in code; the LLM never computes numbers):

| Feature | Definition |
|---|---|
| `frac` | step / segment_total |
| `ret_recent` | mean true return over the last 10 episodes |
| `ret_best` | best 10-episode mean so far |
| `slope` | OLS slope of episode true return vs step over the last 30% of steps, per 100k steps |
| `ep_len_recent` | mean length of the last 10 episodes (Hopper/Walker: < 1000 means falling) |
| `kl`, `clipfrac`, `ev` | means over the last 5 updates (approx_kl, clipfrac, explained variance) |
| `rank` | rank of `ret_recent` within its (wave, env) cohort |
| `flags` | `nan`, `kl_high` (kl > 0.05 for 3 updates), `collapse` (ret_recent < 0.5·ret_best and ret_best > cohort median), `stalled` (frac > 0.5, slope ≤ 0, rank in bottom half), `slow` (sps < 0.5 × benchmark) |

Rules (code, always on):
- `nan` → stop
- `FAILED`/`CONTAINED` → record and surface it, never retry the same reward code
- halving rungs (§6.3) are decided by code

LLM calls happen only at rungs, when a flag is raised, and every 3 min during wave 2. The LLM sees a compact table of at most 45 rows plus the remaining fork budget, and returns a `MonitorDecision` (§5.4). Allowed actions:
- `stop(run, reason)`
- `fork(parent_run, overrides, extra_steps, reason)`
- `rescue(run, reason)`: keep one config per env that the rung rule would have cut, for example the one with the steepest slope
- `note(text)`

Guardrails (code):
- ≤ 2 forks per tick
- total fork steps ≤ 15% of wave-2 steps
- override values within the fork bounds in §6.2
- can't stop the last running seed of a wave-2 config (that would make the stats meaningless)
- every decision is stored with its evidence, i.e. the feature row the LLM saw

### 4.8 Verifier and stats (Person B logic, Person A wiring)

For each env, verify the winning wave-2 config. Also verify the runner-up when the CIs overlap. Launch `evaluate.py` jobs in **fresh sandboxes**, mounting only the final checkpoint.

`stats.py`:

```python
from scipy.stats import trim_mean
def iqm(x): return trim_mean(x, 0.25)          # n=5 → drops 1 from each end, mean of the middle 3
def boot_ci(x, f=iqm, B=10_000, a=0.05, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    s = np.array([f(x[rng.integers(0, len(x), len(x))]) for _ in range(B)])
    return np.quantile(s, [a/2, 1-a/2])
def diff_ci(xa, xb, f=iqm, B=10_000, a=0.05, seed=0):
    rng = np.random.default_rng(seed); xa, xb = map(np.asarray, (xa, xb))
    d = [f(xa[rng.integers(0,len(xa),len(xa))]) - f(xb[rng.integers(0,len(xb),len(xb))]) for _ in range(B)]
    return np.quantile(d, [a/2, 1-a/2])
```

The unit of analysis is the **seed**: each seed's score is the mean of its 10 eval episodes. With n = 5 the CIs are coarse, and the report says so.

**Gaming check**: if the verifier's true return is less than 0.8 × the training-logged true return, or if the shaped return keeps rising while the true return doesn't, flag `eval_mismatch / reward_gaming`.

**Overall winner across envs**: average rank of each wave-2 config across envs (Borda count). Break ties by the mean of `(R − R_min) / (R_max − R_min)` per env.

### 4.9 Predictions scorer and reporter (Person B)

Scoring a prediction of kind `a_beats_b` with ratio r:
- **confirmed** if IQM_a / IQM_b ≥ r and `diff_ci(a, b)` lies entirely above 0
- **refuted** if the CI lies entirely below 0, or if the ratio fails while the CI excludes 0 in a's favour, i.e. a is better but by less than claimed
- **inconclusive** otherwise, or whenever n < 3 seeds

**Calibration**: Brier score over decided predictions, `B = mean((p_i − o_i)²)` with o = 1 for confirmed and 0 for refuted. Show B next to the constant-0.5 baseline of 0.25.

**Report**: code assembles `report.json` (every number). The LLM writes narrative sections from that JSON under the instruction "use only numbers that appear in the JSON". Code then extracts every number from the narrative with a regex and checks that it matches a JSON value within rounding. If any number fails, regenerate once, then fall back to the template. Put this check in the slides; it's a strong verification story.

Report contents:
- task, plan, predictions
- per-env table: config, IQM, 95% CI, seeds, steps
- the winner and why
- predictions with outcomes and the Brier score
- decision timeline: kills, forks, rescues, with reasons
- compute: env steps, vCPU-hours, steps saved by halving vs the naive sweep
- containment events
- artifacts: checkpoint SHA-256s, download links, median-seed videos for each env

### 4.10 Web UI (Person A)

FastAPI + Jinja + htmx + uPlot, with SSE at `/sessions/{id}/stream`.

Pages:
- `/`: new session form (prompt, envs, preset) plus a pinned "hero session" link
- `/sessions/{id}`:
  - live curve grid: one panel per env, one line per run, dashed lines for forks
  - run table with status pills
  - decision feed
  - containment panel
  - predictions panel
- `/sessions/{id}/report`
- `/sessions/{id}/replay?speed=10`: re-streams stored events with time scaled down. Use this for judges and for recording the video, since a full session takes 20–30 min.

Deployment:
- Caddy on the control VM for `https://<ip>.sslip.io`, giving automatic TLS without buying a domain
- the full presets sit behind an access code
- the public "quick" preset runs at most one at a time and is rate-limited, so judges can't burn the credits

### 4.11 Containment demo path (Person A + B)

- **Reward-shaping proposals**: the planner may propose at most one extra reward term per env as `def shaped_reward(obs, action, next_obs, env_reward, info) -> float`.
- **Layer 1, `safety.py`**: an AST allowlist. Only `math` and `numpy as np` may be imported. No `open`, `exec`, `eval`, `compile`, `__import__`, `getattr`, `globals`, dunder attributes, `while True`, or classes. Max 2 KB.
- **Layer 2, the sandbox**: no network, no capabilities, read-only rootfs, pids/memory/time caps.
- **Layer 3, the verifier**: scores only on the true env reward.

Red-team presets in `scripts/redteam/`, triggered from a "Red-team" button that sends them with layer 1 disabled to show that layer 2 holds by itself:

| File | What it does | Expected outcome, shown in the containment panel |
|---|---|---|
| `egress.py` | `socket.create_connection(("1.1.1.1", 80), 2)` | `OSError: Network is unreachable` → run `CONTAINED`, stderr line shown |
| `membomb.py` | allocates 8 GB | killed at the 3 GB cap (exit 137 / OOMKilled) → `CONTAINED` |
| `forkbomb.py` | `os.fork()` loop | stops at the pids limit (256) → `CONTAINED` |
| `escape_write.py` | writes `/etc/x`, `/job/x` | `EROFS` / permission denied → `CONTAINED` |
| `gaming.py` | rewards vertical speed | trains "well" on the shaped return, verifier flags `reward_gaming` |

The video shows `egress.py` and `membomb.py`, and `gaming.py` if there's time.

---

## 5. Schemas (agree on these in the first 30 min so you can split up)

### 5.1 `config.json` (control → worker)
```json
{"run_id": "s12-w1-hc-c3-s1-seg1", "algo": "ppo", "env_id": "HalfCheetah-v4", "seed": 1,
 "total_steps": 150000, "start_step": 0, "anneal_total_steps": 300000,
 "params": {"hidden": [64, 64], "learning_rate": 3e-4, "ent_coef": 0.0, "norm_obs": true,
            "clip_coef": 0.2, "num_envs": 4, "num_steps": 512, "update_epochs": 10,
            "num_minibatches": 32, "gamma": 0.99, "gae_lambda": 0.95, "max_grad_norm": 0.5},
 "resume_from": null, "reward_fn": null, "ckpt_every": 100000}
```

### 5.2 `metrics.jsonl` (worker → node agent → control)
```json
{"type":"update","step":204800,"t":131.2,"sps":1561,"lr":2.1e-4,"pg_loss":-0.01,"v_loss":0.42,"entropy":4.1,"approx_kl":0.012,"clipfrac":0.11,"ev":0.83}
{"type":"episode","step":205312,"t":131.6,"ret_true":1834.2,"ret_shaped":1834.2,"len":1000}
{"type":"checkpoint","step":300000,"path":"ckpt/step_000300000.pt","sha256":"9f2c…"}
{"type":"done","step":300000,"reason":"budget"}
```

### 5.3 `Plan` (planner → control)
```python
class ConfigSpec(BaseModel):
    name: str; hidden: Literal["64x64","256x256"]; learning_rate: Literal[1e-4, 3e-4, 1e-3]
    ent_coef: Literal[0.0, 0.01]; norm_obs: bool; rationale: str
class Prediction(BaseModel):
    id: str; statement: str; kind: Literal["a_beats_b", "reaches_threshold"]
    env: Literal["HalfCheetah-v4","Walker2d-v4","Hopper-v4","all"]
    a: str; b: str | None = None; min_ratio: float | None = None; threshold: float | None = None
    at_step: int; p: confloat(ge=0.05, le=0.95)
class Plan(BaseModel):
    task_summary: str; envs: list[str]; configs: conlist(ConfigSpec, min_length=2, max_length=8)
    hypothesis_config: str          # carried to wave 2 to test one key prediction
    predictions: conlist(Prediction, min_length=3, max_length=5)
    reward_proposal: dict[str, str] | None = None   # env -> python source, optional
```

### 5.4 `MonitorDecision`
```python
class Action(BaseModel):
    kind: Literal["stop","fork","rescue","note"]; run_id: str | None = None; reason: str
    overrides: dict[str, float] | None = None; extra_steps: int | None = None
class MonitorDecision(BaseModel):
    actions: conlist(Action, max_length=6); summary: str
```

### 5.5 SQLite tables
`sessions(id, prompt, preset, status, plan_json, report_json, created_at)` ·
`runs(id, session_id, wave, segment, env_id, config_name, seed, params_json, parent_run_id, start_step, total_steps, node, status, exit_code, oom_killed, started_at, ended_at)` ·
`metrics(run_id, type, step, t, json)` with an index on `(run_id, step)` ·
`decisions(id, session_id, ts, actor, run_id, action, reason, evidence_json)` ·
`predictions(id, session_id, json, outcome, evidence_json)` ·
`events(id, session_id, ts, kind, payload_json)` ·
`artifacts(run_id, kind, path, sha256)`

---

## 6. Experiment design

### 6.1 Environments

| Env | obs / act dims | Termination | Role | Rough PPO prior @1M steps* |
|---|---|---|---|---|
| HalfCheetah-v4 | 17 / 6 | none (always 1000 steps) | primary | ~1.5k–3k, high seed variance (some seeds learn to run flipped on their back) |
| Walker2d-v4 | 17 / 6 | on fall | primary | ~2.5k–4k |
| Hopper-v4 | 11 / 3 | on fall | transfer check, quick preset | ~2k–3k |
| InvertedDoublePendulum-v4 | 11 / 1 | on fall | 1-min smoke test | ~9k max |

\*Priors only, from typical CleanRL-style PPO curves. Write your own predictions down before the M1 reference runs, then replace these with what you measure. Expect lower numbers at 500k.

You were right that HalfCheetah and Walker2d are cheap enough on CPU. The MuJoCo step cost is in the same range as Hopper, so per-run time is set by PPO overhead, not physics.

### 6.2 Search space (enforced in code)

| Param | Planner may choose | Forks may set |
|---|---|---|
| hidden | 64x64, 256x256 | fixed (inherited) |
| learning_rate | 1e-4, 3e-4, 1e-3 | × 0.5 or × 2, clamped to [3e-5, 3e-3] |
| ent_coef | 0.0, 0.01 | [0, 0.02] |
| norm_obs | true, false | fixed |
| clip_coef | 0.2 (fixed) | [0.1, 0.3] |
| everything else | CleanRL defaults (§5.1) | fixed |

That's 2 × 3 × 2 × 2 = 24 combinations, and the planner picks ≤ 8, each with a rationale. `norm_obs=false` makes a good "known-bad control": if the agent's prediction about it fails, something is broken.

### 6.3 Waves for the "full" hero session (3 envs, PPO only)

Notation: s = measured env steps/sec per run (assume **s = 1,500** until the M1 benchmark), S = concurrent slots.

**Wave 1, screening with successive halving** (8 configs × 3 envs × 1 seed = 24 lineages):
- Segment 1: 24 runs × 150k steps → each takes 150k / 1.5k = **100 s**
- Rung (code): per env, keep the top 4 of 8 by `ret_recent`. The monitor LLM may `rescue` one.
- Segment 2: 12 survivors resume from their 150k checkpoint for +150k (another 100 s each)
- Steps: 24 × 150k + 12 × 150k = **5.4M**, vs 24 × 300k = 7.2M for the naive sweep, so halving saves **25%** of screening compute. The report states this.

**Wave 2, confirmation** (3 configs × 3 envs × 5 seeds = 45 runs × 500k steps):
- The 3 configs: the top 2 by Borda rank from wave 1, plus the planner's `hypothesis_config`, so at least one prediction is tested with 5 seeds.
- Each run: 500k / 1.5k = **333 s ≈ 5.6 min**. Steps: 45 × 500k = **22.5M**.
- Fork reserve: ≤ 15% of 22.5M ≈ **3.4M** steps.

**Wave 3, verification**: 3 envs × 5 seeds × 10 episodes × ≤1000 steps = 150k inference-only steps (well under a minute), plus 3 videos (~250 rendered frames each).

Why 5 seeds in wave 2: the minimum detectable gap for a two-sided 95% t-test is t₀.₉₇₅,₂ₙ₋₂ · σ·√(2/n):
- n = 3: 2.776 × 0.816σ ≈ **2.27σ**
- n = 5: 2.306 × 0.632σ ≈ **1.46σ**
- n = 10: 2.101 × 0.447σ ≈ **0.94σ**

Wave 1 (n = 1) is only used to discard clear losers, never to declare winners.

### 6.4 Presets

| Preset | Envs | Configs | Wave 1 | Wave 2 | Wall time at s = 1.5k, S = 15 |
|---|---|---|---|---|---|
| quick (public) | Hopper | 4 | 4 × 150k, no rung | none, verify all 4 | ~3–4 min |
| demo | Walker2d + HalfCheetah | 6 | 12 × (150k + 150k), keep 3/6 | 2 configs × 2 envs × 3 seeds × 300k | ~15 min |
| full (hero) | all three | 8 | as in §6.3 | as in §6.3 | ~27 min (§7) |

### 6.5 Example pre-registered predictions for the hero run

(These are the agent's predictions, but write yours down too and compare.)
- **P0 (infra)**: gVisor costs < 10% of steps/sec vs runc for this compute-bound workload (p = 0.7). Measured in `bench.py`.
- **P1**: `norm_obs=true` beats `norm_obs=false` by ≥ 1.3× IQM on Walker2d at 500k (p = 0.85).
- **P2**: 256x256 beats 64x64 on HalfCheetah by ≥ 1.1× (p = 0.55).
- **P3**: lr = 1e-3 raises `kl_high` flags in ≥ 2 of 3 envs during wave 1 (p = 0.7). Scored from the decision log.
- **P4**: the wave-2 winner on Walker2d is also top-2 on Hopper (p = 0.6).

---

## 7. Compute and cost math

Formulas (in `planner.estimate_wall_minutes`):
- run time `T = steps / s`
- wall time per wave ≈ `ceil(N_runs / S) · T_run`
- slots `S = floor((vCPU_total − 2·n_nodes) / cpus_per_run)`

Hero session at s = 1,500:

| Item | Steps | Run-seconds (= steps / 1500) |
|---|---|---|
| Wave 1 | 5.4M | 3,600 |
| Wave 2 | 22.5M | 15,000 |
| Fork reserve (max) | 3.4M | 2,250 |
| **Total** | **31.3M** | **20,850 ≈ 5.8 run-hours ≈ 11.6 vCPU-hours** (2 vCPU/run) |

Wall clock with 1 worker (S = 15):
- wave 1: ceil(24/15) × 100 s + ceil(12/15) × 100 s = 200 + 100 = 300 s
- wave 2: ceil(45/15) × 333 s = 1,000 s
- forks: ~150 s
- verify: ~120 s
- total ≈ **26 min**

With 2 workers (S = 30): 200 + 666 + 150 + 120 s ≈ **19 min**.

If the benchmark gives a different s, scale every time by 1,500 / s. If s < 800, switch the hero run to the "demo" preset.

Cost: (hourly price of the worker VM) × (hours it exists), plus the small control VM. Check prices in the console. Create workers only when needed, snapshot after setup, and **destroy workers right after submission**. Also check the account's instance-size quota at kickoff; new accounts are sometimes capped. If you can't get a 32 vCPU instance, use 2 × 16 vCPU.

SAC (stretch only): one gradient step per env step, so expect roughly 10× lower throughput on CPU (~150–300 steps/sec with 256×256 nets). 300k steps would take 17–33 min. Only add a single SAC arm on HalfCheetah, and only if `bench.py` shows ≥ 250 steps/sec. The interesting finding would be return per vCPU-minute vs per env step.

---

## 8. Today's timeline (2 people)

**Person A** = infra + web. **Person B** = RL + agent logic (probably you). Times are relative to start. Shift the whole schedule so "submit" lands ≥ 30 min before the deadline.

| Time | Person A | Person B | Milestone |
|---|---|---|---|
| 0:00–0:30 | Claim credits; create the control VM (2–4 vCPU) and worker VM (32 vCPU) in one region with a VPC; firewall: 80/443 public on control only | Get the inference key; `GET /v1/chat/models`, choose the model; write the §5 schemas into `control/models.py` | Interfaces frozen |
| 0:30–2:00 | `setup_worker.sh` (Docker, gVisor), Dockerfile, build image, node agent, test with a dummy container plus `egress.py` | `train_ppo.py` changes 1–10, test 20k steps on 3 envs locally (Mac is fine), `evaluate.py` + video | |
| 2:00 | Run `bench.py` on the worker: s per env, runc vs runsc → plug s into the planner. Start one 1M reference run per env in the background (real curves for priors and UI testing) | | **M1**: real PPO run in gVisor writing metrics |
| 2:00–4:00 | `control/app.py`, DB, scheduler, metrics poller, SSE, live curve page, run table | `llm.py`, planner (+ budget trim), `features.py`, monitor rules + rung + LLM decisions + guardrails, `stats.py` | |
| 4:00 | | | **M2**: quick preset end-to-end. **Immediately start a full hero session** to shake out bugs |
| 4:00–5:30 | Verifier job wiring, report page, videos, replay mode, Caddy + sslip.io, access code | Predictions scorer, reporter + number check, `safety.py`, red-team presets, reward-proposal path | |
| 5:30 | Start the **final hero session**; screen-record it, this is your video footage | | **M3**: report renders with real numbers |
| 5:30–6:30 | Bug fixes, README, architecture diagram | Check the report numbers by hand against the DB for one env | |
| 6:30–7:30 | Edit the video, make the slide PDF | Write the submission text; test the public URL from a phone | **Submit** (≥ 30 min before the deadline) |

Cut lines (decide at the clock times, don't drift):
- **Behind at 3:30** → drop the LLM monitor (rules + rung only; the LLM does the planner and report), drop forks and drop reward proposals. Keep red-team presets with a fixed reward file.
- **Behind at 5:00** → hero run uses the "demo" preset; videos only for the overall winner; skip replay mode and record a live run instead.
- **Never cut**: the gVisor sandbox, held-out-seed verification, one containment moment, and the public URL. They're the scoring core of this track.

---

## 9. Testing

- `stats.py` unit tests: IQM of `[1,2,3,4,100]` = 3.0; the bootstrap CI contains the point estimate; `diff_ci` of identical arrays straddles 0.
- `safety.py`: every file in `scripts/redteam/` is rejected; three legitimate rewards (torso-upright term, action-magnitude penalty, forward-velocity bonus) are accepted.
- Schemas: feed the planner and monitor deliberately malformed JSON; check one retry, then the fallback.
- Resume correctness: run 150k, resume to 300k, compare with a straight 300k run on the same seed. Curves should be similar, not identical (the reward normaliser re-warms).
- Chaos: kill the node agent mid-run → the control plane marks runs `FAILED` and the UI stays up. Kill a container → status updates within 10 s.
- **Eval sanity** (catches the normaliser bug): evaluate the M1 reference checkpoint. The verifier return should be within ~20% of the last training returns.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| s much lower than 1,500 | Measured at M1; budgets scale by formula; fall back to the demo preset |
| MuJoCo/osmesa misbehaves under runsc | Test at M1; fallback is the default runtime with all other limits, stated honestly |
| Gymnasium/CleanRL API mismatch | Pin gymnasium 0.29.1 / mujoco 2.3.7 / numpy<2 and use -v4 envs |
| Torch thread oversubscription | `OMP_NUM_THREADS` + `torch.set_num_threads` + `--cpus` |
| LLM returns bad JSON or is slow | Pydantic + 1 retry + rules fallback; async calls, runs never wait |
| A "winner" that's just seed noise | 5 seeds, IQM + bootstrap CI, ties declared as ties |
| Reward gaming by LLM-written reward | Verifier uses true reward only; mismatch flag |
| Judges burn credits | Access code on full presets; quick preset rate-limited; replay mode for viewing |
| Instance quota / capacity | Check at 0:00; 2 × 16 vCPU fallback |
| Running out of time | Hero session started by 4:00 and again at 5:30; cut lines above |

---

## 11. Demo video script (~3 min; check the lablab length limit)

1. **0:00–0:20 Problem.** RL research is babysitting sweeps; agents that run code need containment.
2. **0:20–0:50 Prompt → plan.** Show the configs, rationales and the pre-registered predictions with probabilities.
3. **0:50–1:50 Live runs** (replay at 10×). Curves per env. The rung cuts half; the monitor rescues one ("steepest slope"); a fork with half the lr. Reasons appear in the feed.
4. **1:50–2:20 Containment moment.** Red-team button → `egress.py` blocked (network unreachable) and `membomb.py` killed at 3 GB. The containment panel and the stderr line show it; the session keeps going.
5. **2:20–3:00 Report.** Winner per env with IQM ± CI, predictions confirmed/refuted/inconclusive, Brier score, compute saved by halving, median-seed videos side by side, checkpoint hashes.
6. **3:00–3:20 Architecture.** Vultr control VM, VPC, gVisor workers, Serverless Inference call count.

---

## 12. Open items to settle at kickoff

- Exact deadline time (and whether the lablab submission can be edited after the first submit; if yes, submit a draft at 6:30).
- Model choice from `/v1/chat/models` (default `kimi-k2-instruct`).
- Instance sizes your accounts are allowed to create.
- Video length limit and slide format from the lablab rules.

## Sources
- Vultr Agent Rush event page: https://lablab.ai/ai-hackathons/vultr-hackathon
- Vultr Serverless Inference chat endpoint (tool-calling note): https://docs.vultr.com/products/compute/serverless-inference/management/usage/chat
- Vultr Serverless Inference in Python (OpenAI SDK, model list URL): https://docs.vultr.com/how-to-use-vultr-serverless-inference-in-python
- gVisor install docs: https://gvisor.dev/docs/user_guide/install/
- CleanRL PPO continuous: https://docs.cleanrl.dev/rl-algorithms/ppo/
- rliable (IQM, interval estimates): https://github.com/google-research/rliable
