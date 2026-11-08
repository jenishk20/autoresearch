# Autoresearch

A query-adaptive ML research agent for the Vultr Agent Rush hackathon (Challenge 1, "Blast Radius Zero").
You ask a research question in plain English. The agent writes a plan with pre-registered predictions, trains every
candidate in its own sandbox, intervenes while runs are live (cut / rescue / fork / stop), re-scores the winners on
held-out seeds, and writes a report where every number is computed by code and traceable to a run ID and a
checkpoint SHA-256.

Plan: [`docs/autoresearch-plan.md`](docs/autoresearch-plan.md) · PPO worker spec: [`docs/rl-autoresearch-plan.md`](docs/rl-autoresearch-plan.md)

## Run it locally

Requirements: Python 3.11 and [uv](https://docs.astral.sh/uv/). No Docker needed for local dev.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -r worker/requirements.txt -r control/requirements.txt pytest
cp .env.example .env    # then put your Vultr Serverless Inference key in it
scripts/dev.sh          # node agent on :8100 + UI on http://127.0.0.1:8000
```

Open http://127.0.0.1:8000, keep the default query (*"Find a good PPO setup for Hopper and check whether it transfers
to Walker2d. 15-minute budget."*) and press **Start research**. The session page shows the brief, plan and predictions,
live learning curves per wave and env, the run table, the agent's decisions and flags. When it finishes, **Open report**.

- **Local dev runs are NOT sandboxed.** `scripts/dev.sh` starts the node agent with `NODE_BACKEND=local` (plain
  subprocesses). The header says so, and the red-team buttons are disabled. The sandbox is the Docker + gVisor backend on a Linux worker.
- **No LLM? It still runs.** If the Vultr key is missing or rejected, the header shows "LLM: off" and every agent step
  uses its rule-based fallback (brief, plan, rescue, monitor, report narrative). Restart after fixing `.env`.
- `.env` keys: `VULTR_INFERENCE_API_KEY` (Serverless Inference key, not the account API key), `LLM_MODEL` (optional;
  ignored if Vultr doesn't serve it, then `control/config.yaml` picks), `NODE_TOKEN` (shared secret for node agents).
- Tests: `.venv/bin/python -m pytest -q tests`

## Run it on Vultr (sandboxed)

Worker VM (Ubuntu 24.04, e.g. 32 vCPU):

```bash
curl -fsSL https://get.docker.com | sh
curl -fsSL https://gvisor.dev/archive.key | sudo gpg --dearmor -o /usr/share/keyrings/gvisor-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" | sudo tee /etc/apt/sources.list.d/gvisor.list
sudo apt-get update && sudo apt-get install -y runsc && sudo runsc install && sudo systemctl restart docker
docker build -f worker/Dockerfile -t autoresearch-worker:latest .
NODE_BACKEND=docker NODE_SLOTS=15 NODE_TOKEN=... uvicorn node_agent.app:app --host <vpc-ip> --port 8100
```

Control VM: set `NODE_URL=http://<worker-vpc-ip>:8100` and `NODE_TOKEN` in `.env`, then
`uvicorn control.app:app --host 127.0.0.1 --port 8000` behind Caddy (`https://<ip>.sslip.io`).

Each job runs as `docker run --runtime=runsc --network none --read-only --cap-drop ALL --pids-limit 256
--memory 3g --cpus 2 --user 1000:1000` with only `/job` (read-only), `/out` and, for resumes and verification, the parent
checkpoint (read-only) mounted.

## Layout

```
worker/      runs INSIDE the sandbox: run.py (entry), adapters/rl_ppo.py (CleanRL-style PPO + evaluate), common/
node_agent/  trusted per-VM agent: start/stop/kill containers, tail metrics, serve files (docker or local backend)
control/     FastAPI app + agent: understand, planner, session (waves, rung, monitor), engine (scheduler), verify,
             stats, predictions, reporter (number-checked narrative), safety (AST allowlist), llm (Vultr client)
scripts/     dev.sh, redteam/ (malicious reward functions for the containment demo)
tests/       contract tests
```

## Status

| Piece | State |
|---|---|
| PPO worker (train / resume / fork / STOP / checkpoints / held-out eval) | working, tested locally |
| Node agent, local backend | working |
| Node agent, Docker + gVisor backend | written, **not yet run on a Linux worker** |
| Agent loop for RL: brief → plan → wave 1 halving + rescue → wave 2 + monitor (stop/fork, guardrails) → verify → report | working end to end |
| LLM steps on Vultr Serverless Inference | wired with fallbacks; needs a valid Serverless Inference key |
| Web UI: live SSE curves, runs, decisions, flags, containment panel, report | working |
| Tabular adapter + churn demo (ingest, profile, sealed test set) | not built yet |
| Worker Dockerfile, Caddy, replay mode | not built yet |
