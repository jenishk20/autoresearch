# Auto-research agent: build plan

**Event:** Vultr Agent Rush (lablab.ai), Challenge 1 "Blast Radius Zero" · **Due:** today, 8 Nov 2026 (`DEADLINE = __:__`, fill in from the lablab page)
**Team:** 2 people · **Stack:** Python workers (scikit-learn, PyTorch, CleanRL-style PPO), Docker + gVisor on Vultr CPU VMs, Vultr Serverless Inference for every LLM call, FastAPI + htmx, SQLite · **No GPUs**

This is the main plan. The PPO worker details live in `docs/rl-autoresearch-plan.md` (the **RL adapter spec**); this file overrides it wherever they differ.

---

## 0. Pitch

You ask a business or research question in plain English. The agent turns it into a real ML study: it works out what kind of problem it is, profiles the data, writes a plan with **pre-registered, machine-checkable predictions**, trains every candidate in its own **gVisor sandbox with no network**, stops, forks or rescues runs as the curves come in, **re-scores the winners in fresh sandboxes on a sealed test set that training never had access to**, and writes a report where every number is computed by code and traceable to a run ID, a checkpoint SHA-256 and a dataset SHA-256.

The LLM never executes anything on the control plane. It emits JSON (validated and bounded by code), and, where it writes code, that code only ever runs inside a sandbox.

**Two demos, one engine:**

| | Demo 1: Customer churn with dollar costs (hero) | Demo 2: PPO on MuJoCo (capability) |
|---|---|---|
| Query | *"Here's our customer CSV. A retention offer costs $50, a lost customer costs $600, and the offer keeps about 30% of customers who would otherwise leave. Who should we target, how much will it save, and can we trust the model?"* | *"Find a good PPO setup for Hopper and check whether it transfers to Walker2d. 20-minute budget."* |
| Problem type | Tabular binary classification, cost-sensitive | Continuous-control RL |
| Data | Uploaded CSV (IBM Telco Customer Churn, ~7k rows) | Simulated envs (no dataset) |
| Main metric | Expected savings per 1,000 customers on the sealed test set | Mean true env return on held-out seeds |
| What it shows | Business-metric framing, leakage detection, calibration and trust, **containment of agent-written feature code touching customer data** | **Mid-run kill/fork/rescue**, successive halving, transfer check |

Demo 1 is the hero (it carries the enterprise story and the containment moment). Demo 2 shows the same engine adapting to a completely different kind of research.

---

## 1. Business value

**The problem in an enterprise.** Questions like "who should get a retention offer" turn into weeks of analyst time, and the results are often hard to trust:
- Models are tuned for AUC or accuracy, while the business cares about dollars.
- Data leakage (a column that's only filled in after the outcome) makes offline results look great and production results collapse.
- Many configurations get tried and the best one is reported, with no pre-registration and no honest error bars.
- Security and data-governance teams won't let an AI agent run code on customer data, because nothing stops that code from reading what it shouldn't or sending data out.

**What this agent gives you.**
1. **Answers in business terms.** The agent builds the cost function from the query and reports "targeting the top X% saves $Y per 1,000 customers, 95% CI [a, b]", compared against the obvious baselines (target nobody, target everyone).
2. **Results you can audit.** Every number in the report is computed by code and linked to a run, a checkpoint hash and a dataset hash. The LLM-written narrative is checked number by number against the computed results.
3. **Honest science.** Predictions are written down before training and scored afterwards (Brier score). Winners are decided on a sealed test set with confidence intervals; ties are reported as ties.
4. **Leakage caught automatically.** The data profiler flags columns that look like they encode the outcome; the verifier flags results that drop sharply on the sealed test set.
5. **Containment by design ("Blast Radius Zero").** Agent-written code runs only in gVisor sandboxes with no network, a read-only filesystem, no capabilities, and memory/process/time caps. Customer data can't leave the sandbox, and training code can't see the sealed test labels.
6. **One engine, many problems.** The same loop handles tabular business problems and control/optimisation problems (Demo 2), because problem-specific logic lives in adapters.

**Why Vultr.** Cheap CPU VMs are enough for this class of work (no GPUs), the VPC keeps workers private, and Serverless Inference provides the LLM without the API key ever reaching a worker. The report shows the compute cost of each session (vCPU-hours × price).

---

## 2. Constraints and submission checklist

| Requirement | How we meet it |
|---|---|
| Agent LLM calls use Vultr Serverless Inference | `control/llm.py`, OpenAI SDK with `base_url=https://api.vultrinference.com/v1` |
| Isolated execution | Every job is `docker run --runtime=runsc --network none …` |
| Vultr is the central control layer, returning verifiable output | Control VM dispatches to worker VMs over a Vultr VPC; report carries hashes + sealed-test re-verification |
| Public demo URL | Control VM behind Caddy, `https://<ip>.sslip.io` |
| Recorded video with ≥1 containment moment | §15 script; red-team presets in §9 |
| Architecture and use-case explanation | README + slide PDF (§1 is the use-case story) |
| No GPUs | sklearn / small PyTorch models; CPU MuJoCo with state observations |
| $200 credits per person | §12 budget; destroy worker VMs after submission |

**LLM:** no tool calling (on Vultr only `kimi-k2-instruct` supports it). Every agent step is prompt → JSON → pydantic validation → bounded action, with a rule-based fallback. At kickoff, `GET /v1/chat/models` and pick the strongest instruct model; default `kimi-k2-instruct`.

---

## 3. Architecture

```
                     ┌──────────────────────────── Control VM (2–4 vCPU) ─────────────────────────────┐
 User / judge ─HTTPS▶│ FastAPI + htmx UI ── SSE ──▶ browser (brief, profile, live curves, decisions,   │
   (query, CSV)      │   │                                     containment panel, report)             │
                     │   ├── Ingest (trusted): upload / catalog / OpenML → parquet, sha256, sealed split│
                     │   ├── Agent stages: Understand · Plan · Monitor · Reflect · Report ─▶ llm.py ──▶ Vultr Serverless Inference
                     │   ├── Scheduler (queue, slots, waves, rounds)   SQLite (WAL)                     │
                     │   ├── Verifier / Stats / Predictions                                             │
                     │   └── Safety (AST check on agent-written code)                                   │
                     └───────────────┬───────────────────────────────────────────────────────────────────┘
                                     │ Vultr VPC (private), bearer token
                    ┌────────────────▼──────────── Worker VM (32 vCPU) ─────────────────────────┐
                    │ node_agent (trusted): datasets store, start/stop/kill containers, tail metrics │
                    │   /srv/datasets/<sha>/train.parquet      ← mounted into training + profile jobs │
                    │   /srv/sealed/<sha>/test.parquet         ← mounted ONLY into verify jobs        │
                    │   docker run --runtime=runsc --network none  (2 vCPU, 3 GB, ro rootfs, non-root)│
                    │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐                           │
                    │   │ profile │ │ train … │ │ train … │ │ verify … │                           │
                    │   └─────────┘ └─────────┘ └─────────┘ └──────────┘                           │
                    └────────────────────────────────────────────────────────────────────────────────┘
```

**Trust boundaries**
- **Untrusted:** everything inside a container: agent-written feature code, model code, anything a policy does, and uploaded data.
- **Trusted:** ingest, node_agent, and the control plane. They never execute agent-written code.
- **Secrets:** the inference API key lives only on the control VM. Workers hold the node token. Containers hold nothing.
- **Sealed test sets** live only on the worker host under `/srv/sealed/`, are never baked into the image, and are mounted read-only only into verify containers.

MVP topology: 1 control VM + 1 worker VM. Snapshot the worker; clone a second only if needed.

---

## 4. The research loop

```
query → 1 Understand → 2 Ingest → 3 Profile → 4 Plan → 5 Run + Monitor → 6 Verify → 7 Reflect ─┐
          (LLM)        (trusted)   (sandbox)   (LLM)    (sandbox + LLM)    (sandbox)   (LLM)     │
                                                ▲                                              │
                                                └──── follow-up round (≤ 1 extra, budget-capped)┘ → Report
```

1. **Understand → `ResearchBrief`.** The LLM reads the query and returns problem type, data source, target, metric (from a fixed list), business costs if given, constraints, and 2–4 research questions phrased in the user's terms. **Code**, not the LLM, sets the metric direction. If the request doesn't fit (needs GPUs, needs live data), the brief says so and proposes the closest feasible version.
2. **Ingest (trusted code).** Gets the data from one of three sources:
   - an uploaded CSV
   - the built-in catalog (each entry has a description the LLM can match)
   - OpenML: trusted code searches OpenML by keyword, the LLM picks an ID **from the real results**, trusted code downloads it.

   Then: normalise to parquet with a typed schema, compute the SHA-256, and cut the sealed test split (stratified 20%, fixed seed, or the last 20% by time for time-ordered data). Size cap: 200 MB / 2M rows.
3. **Profile (sandbox job).** Rows, column types, missing values, cardinality, class balance, target stats, and **leakage suspects**: columns whose single-feature AUC on the train pool is > 0.95, ID-like columns (unique per row), and columns that are missing almost exactly when the target is negative. The output is a `DataProfile`.
4. **Plan → `Plan`.** The LLM picks configs from the **model registry for that problem type** (§5), states which research question each one answers, chooses which suspect columns to drop and why, and writes 3–5 predictions. Code clamps every value, checks the budget, and trims if needed (logging "asked for X, trimmed to Y").
5. **Run + Monitor.** Waves with successive halving (§7.1), live curves, rung decisions, and LLM stop/fork/rescue with guardrails.
6. **Verify.** Fresh sandboxes, final checkpoint only, sealed test set (tabular) or held-out seeds with the true reward (RL).
7. **Reflect → `Reflection`.** The LLM reads the code-computed results and either proposes **one follow-up round** (for example "boosting overfits; test stronger regularisation and a calibration step") or stops. Code caps it at 1 extra round and the remaining budget.

Then the **Report** answers the user's original questions directly (§8.4).

**Two levels of flexibility (hybrid)**

| | Level 1: structured (default) | Level 2: agent-written code |
|---|---|---|
| LLM emits | JSON: registry model, preprocessing, hyperparameters | Python `build_features(df) -> df` (Demo 1) or `shaped_reward(...)` (Demo 2) |
| Runs | Trusted adapter code inside the sandbox | Inside the sandbox through a fixed harness |
| Checks | Pydantic + bounds | `safety.py` AST allowlist + sandbox + sealed-test verifier |
| If it fails | Can't (validated before launch) | One repair attempt from the stack trace, then the run is dropped and logged |

Level 1 makes the demo reliable. Level 2 lets the agent adapt beyond the registry, and it's what the sandbox is there to contain.

---

## 5. Adapters

The worker image has one entry point, `python run.py --config /job/config.json`, which dispatches on `task`. Each adapter defines its search space, metrics, budget unit, checkpoint/resume, and an `evaluate` function.

### 5.1 Tabular adapter (Demo 1)

**Preprocessing** (fixed pipeline, options chosen by the plan): drop listed columns → impute (median / most-frequent) → one-hot for low-cardinality categoricals (≤ 20 levels), ordinal codes otherwise → standard scaling for linear and MLP models. Fitted on the train split only and saved in the checkpoint.

**Model registry (binary classification)**

| Model | Planner may choose | Budget unit (halving / forks act on this) | Forks may set |
|---|---|---|---|
| `logreg` | C ∈ {0.01, 0.1, 1, 10}; class_weight ∈ {none, balanced} | none (fits in ~1 s, used as the baseline) | n/a |
| `random_forest` | max_depth ∈ {None, 8, 16}; min_samples_leaf ∈ {1, 5, 20} | trees, +50 per chunk (`warm_start`) | n/a |
| `hgb` (HistGradientBoosting) | learning_rate ∈ {0.03, 0.1, 0.3}; max_leaf_nodes ∈ {15, 31, 63}; l2 ∈ {0, 1} | boosting rounds, +25 per chunk (`warm_start`) | l2 ∈ [0, 10]; learning_rate × 0.5 |
| `mlp` (PyTorch) | hidden ∈ {64x64, 256x256}; lr ∈ {1e-4, 1e-3, 1e-2}; weight_decay ∈ {0, 1e-4}; dropout ∈ {0, 0.2} | epochs | lr × 0.5 or × 2 within [1e-5, 3e-2]; weight_decay ∈ [0, 1e-3]; dropout ∈ [0, 0.5] |
| any | calibration ∈ {none, isotonic} (fitted on a slice of the train split, not on val) | n/a | n/a |

Regression (RMSE/MAE) and multiclass use the same registry with the matching sklearn estimators; they're supported by the schemas but not demoed.

**Metrics logged** (one `update` line per chunk or epoch): `train_loss` (log loss), `val_loss`, `val_auc`, `val_savings` (the business metric, when costs are given), `val_metric` (= the brief's main metric).

**Business metric (cost-sensitive targeting).** With offer cost C, loss from a churned customer L, and save rate s:
- Targeting customer *i* is worth `y_i · s · L − C` (y = 1 if they churn).
- `savings(T) = Σ_{i∈T} (y_i · s · L − C)`, reported per 1,000 customers.
- **The decision threshold is chosen on the validation split only** and stored in the checkpoint; the verifier applies it unchanged to the sealed test set.
- For a calibrated model the theoretical threshold is `C / (s · L)` (= 50 / 180 ≈ 0.28 in Demo 1); the report shows both.
- Baselines in the report: target nobody ($0), target everyone, and the oracle (target exactly the churners).

**Checkpoint:** `joblib` of `{preprocessor, model, threshold, params, feature_list}` plus its SHA-256. It's only ever loaded inside a sandbox (pickles are code).

**Seeds:** seed *k* changes the train/val split (stratified 80/20 of the train pool) and model initialisation.

### 5.2 RL adapter (Demo 2)

The PPO worker is exactly as specified in `docs/rl-autoresearch-plan.md` §4.1 (CleanRL `ppo_continuous_action.py` with changes 1–10, pins `gymnasium==0.29.1`, `mujoco==2.3.7`, `numpy<2`, in-agent `RunningNorm`, `TrueReturn` wrapper, STOP flag, resume/fork), with its search space from §6.2 there. The only changes are that it runs as `run.py` → `adapters/rl_ppo.py` and reads the generic `config.json` (§10.1). `val_metric` = mean true return of the last 10 episodes. Verification = 10 episodes on eval seeds 10_000+ with the raw env reward, plus a median-seed video.

Local dev on macOS: set `MUJOCO_GL=disable` (mujoco 2.3.7 can't load OpenGL there). Video rendering only works on the Linux worker with osmesa.

---

## 6. Data

- **Demo 1 dataset:** IBM Telco Customer Churn (public sample, ~7,043 rows, ~26% churn). Download the CSV once on a dev machine and keep it in `data/demo/` (it's uploaded through the UI during the demo, not baked into the image).
- **Injected leakage column (staged, said so in the video):** `contract_end_reason`, filled only for churned customers (values like "competitor offer", "moved"), empty otherwise. Real CRM exports often have columns like this. The profiler should flag it and the plan should drop it with a stated reason. Keep the original CSV next to the modified one.
- **Catalog:** a small `control/catalog.yaml` (name, description, source, target, problem type). For today it holds the Telco dataset plus 2–3 OpenML entries for testing the OpenML path.
- **Sealed split:** done once at ingest by trusted code. The report shows both the train-pool and sealed-test SHA-256s.

---

## 7. Experiment design

### 7.1 Demo 1: churn

| Wave | What | Runs | Approx. cost |
|---|---|---|---|
| 0. Profile | 1 profile job | 1 | ~10 s |
| 1. Screening | ≤ 8 configs × 1 seed, halving on rounds/epochs/trees: chunk 1, keep the top half by `val_savings`, chunk 2 for survivors | ~12 run-segments | ~2 min wall |
| 2. Confirmation | top 2 + the plan's `hypothesis_config` × 5 seeds | 15 | ~2 min wall |
| 3. Reflection round (optional) | ≤ 3 new configs × 5 seeds | ≤ 15 | ~2 min wall |
| 4. Verify | winner (+ runner-up if the CIs overlap) per round, fresh sandboxes, sealed test | ≤ 10 | ~1 min |

Each run takes seconds to about a minute on 2 vCPUs (7k rows), so container start-up (~2–4 s under gVisor with torch imported) is a noticeable share. That's fine. **Expected session time: ~6–10 min including LLM calls**, short enough to run live for judges. Because runs are short, mid-run decisions in Demo 1 happen at the rungs between chunks (implemented as resume-from-checkpoint, as in the RL plan). Live kills mid-curve are shown in Demo 2.

**Example pre-registered predictions** (the agent writes its own; these are for comparison):
- **P1** (`a_beats_b`): `hgb` beats `logreg` on sealed-test savings by ≥ 10% (p = 0.45). Logistic regression is famously competitive on Telco, so this is a genuinely uncertain bet.
- **P2** (`reaches_threshold`): the winning model saves ≥ $10,000 per 1,000 customers on the sealed test set (p = 0.6).
- **P3** (`top_feature`): `tenure` or `Contract` is among the top 2 features by permutation importance (p = 0.75).
- **P4** (`a_beats_b`): isotonic calibration improves `hgb` savings by ≥ 3% (p = 0.5).
- **P5** (`flag_count`): `mlp` with lr = 1e-2 raises the `diverging` flag in ≥ 2 of 5 seeds (p = 0.5).

### 7.2 Demo 2: PPO

This is the "demo" preset from the RL spec, scaled to the query:
- **Wave 1:** 6 configs × Hopper × 1 seed, 150k + 150k steps with halving (keep 3 of 6). The monitor LLM may rescue one.
- **Wave 2:** top 2 (+ hypothesis config) × {Hopper, Walker2d} × 3 seeds × 300k steps. Mid-run kills (`nan`, `collapse`, `stalled`) and ≤ 2 forks per tick happen here.
- **Wave 3:** verification on held-out seeds, plus median-seed videos.
- **Wall time:** ~15 min at s = 1,500 steps/sec and S = 15 slots (scale by 1,500 / measured s). **For the video, use replay mode at 10×.**
- **Example predictions:** norm_obs=true beats false by ≥ 1.3× on Hopper (p = 0.85); the Hopper winner is top-2 on Walker2d (`rank_transfer`, p = 0.6); lr = 1e-3 raises `kl_high` in ≥ 2 runs (`flag_count`, p = 0.7).

---

## 8. Monitor, verifier, stats, predictions, report

### 8.1 Monitor
Every 15 s (tabular) / 30 s (RL), `features.py` computes per run, in code:
- `frac`, `val_recent`, `val_best`, `slope` (OLS over the last 30% of the budget), `rank` within the (wave, dataset/env) cohort
- tabular: `gap` = train − val loss; RL: `ep_len_recent`, `kl`, `clipfrac`, `ev`

**Flags:** `nan`; `diverging` (val loss up 3 updates in a row while train loss falls, or kl > 0.05 for 3 updates in RL); `overfitting` (gap > 2× the cohort median); `collapse` (RL: val_recent < 0.5 × val_best); `stalled` (frac > 0.5, slope ≤ 0, bottom half); `slow`.

**Rules (code, always on):** `nan` → stop. `FAILED` / `CONTAINED` → record and surface, never retry the same agent code. Rungs are decided by code.

**LLM calls** happen at rungs, when a flag fires, and every 3 min in wave 2. The input is a compact table (≤ 45 rows) plus the remaining fork budget. The output is a `MonitorDecision`: `stop`, `fork`, `rescue` or `note`.

**Guardrails:** ≤ 2 forks per tick; fork budget ≤ 15% of wave-2 budget; overrides within the adapter's fork bounds; the last running seed of a wave-2 config can't be stopped; every decision is stored with the feature row the LLM saw.

### 8.2 Verifier
- **Tabular:** fresh sandbox with `/ckpt:ro` and `/sealed:ro` mounted (no train pool). It loads the checkpoint, scores the sealed test set with the **stored** threshold, and writes `eval.json` (per-row predictions, AUC, savings, calibration bins).
- **RL:** as in the RL spec (10 episodes, eval seeds 10_000+, raw reward, median-seed video).
- **Flags:** `eval_mismatch` if the test metric is < 0.8 × the validation metric (savings) or the AUC drops by > 0.05; `suspected_leakage` if val AUC > 0.98; `reward_gaming` (RL) if the shaped return rises while the true return doesn't.

### 8.3 Stats (`stats.py`)
- **Across seeds:** IQM + bootstrap CI (as in the RL spec, `trim_mean(x, 0.25)`, B = 10,000). With n = 5 the CIs are coarse, and the report says so.
- **Winner vs runner-up (tabular):** **paired bootstrap over sealed-test rows** on the savings difference (resample rows, compute both models' savings on the same resample). This gives much tighter intervals than seeds alone.
- **Metric direction** is looked up in code (`savings`, `auc`, `accuracy`, `return` higher; `logloss`, `rmse`, `mae` lower). "a beats b by r" means `m_a / m_b ≥ r` for higher-is-better and `m_b / m_a ≥ r` for lower-is-better.
- **Savings can be negative**, so ratio predictions on savings are computed against the "target nobody" baseline of $0 and are only decided when both values are positive; otherwise the prediction is `inconclusive` and the report says why.

### 8.4 Predictions and report
**Scoring:**
- `a_beats_b`: confirmed if the ratio holds **and** the difference CI is entirely in a's favour; refuted if the CI is entirely in b's favour, or a is better but by less than claimed (with the CI excluding 0); otherwise inconclusive (always inconclusive with n < 3).
- `reaches_threshold`: confirmed if the lower CI bound passes the threshold; refuted if the upper bound doesn't; otherwise inconclusive.
- `top_feature`: from permutation importance on the sealed test set, averaged over seeds.
- `flag_count`: counted from the decisions/events log.
- `rank_transfer`: rank of the named config in the second dataset/env.

**Calibration of the agent:** Brier score over decided predictions, shown next to the 0.25 baseline.

**Report:** code builds `report.json` with every number. The LLM writes the narrative using only numbers from the JSON. Code then pulls every number out of the narrative with a regex and checks each against the JSON within rounding; on failure it regenerates once, then falls back to a template.

**Demo 1 report sections:**
1. **The answer:** who to target (threshold and share of customers), savings per 1,000 customers with 95% CI on the sealed test set, versus target-nobody / target-everyone / oracle.
2. **Can we trust it?** Calibration plot; sealed-test vs validation gap; leakage findings (what was dropped and why); stability across seeds; top drivers.
3. **What we tried:** plan, predictions with outcomes, Brier score, the decision timeline, the reflection round and why.
4. **Containment events.**
5. **Provenance:** dataset hashes, checkpoint hashes, run IDs, compute used (vCPU-hours and $).

**Demo 2 report:** as in the RL spec: per-env table with IQM ± CI, transfer result, predictions, decision timeline (kills/forks/rescues), compute saved by halving, videos, hashes.

---

## 9. Containment

**Layers**
1. `safety.py`: AST allowlist for Level-2 code. Allowed imports: `math`, `numpy as np`, `pandas as pd`. Forbidden: `open`, `exec`, `eval`, `compile`, `__import__`, `getattr`, `globals`, dunder attributes, `while True`, classes, file/OS/network modules. Max 4 KB.
2. **Sandbox:** gVisor, `--network none`, read-only rootfs, `--cap-drop ALL`, no-new-privileges, uid 1000, `--pids-limit 256`, 3 GB memory, wall-clock watchdog. The sealed test set isn't mounted into training jobs at all.
3. **Verifier:** scores on the sealed test set / true env reward only.

**Red-team presets** (`scripts/redteam/`, sent from a "Red-team" button with layer 1 turned off, to show the sandbox holds by itself)

| File | Demo | Attempt | Expected outcome in the containment panel |
|---|---|---|---|
| `exfil_customers.py` | churn | `build_features` posts customer rows to an external IP | `OSError: Network is unreachable` → `CONTAINED` |
| `peek_sealed.py` | churn | searches `/srv`, `/data`, `/` for test files and labels | not mounted → nothing found / `FileNotFoundError` → `CONTAINED` |
| `membomb.py` | churn | builds an 8 GB "interaction features" array | killed at 3 GB (exit 137 / OOM) → `CONTAINED` |
| `escape_write.py` | churn | writes to `/etc`, `/job`, `/data` | `EROFS` / permission denied → `CONTAINED` |
| `egress.py` | PPO | reward function opens a socket | network unreachable → `CONTAINED` |
| `gaming.py` | PPO | rewards vertical speed | trains "well" on the shaped return; verifier flags `reward_gaming` |

The leakage column in the Telco CSV is the non-adversarial version of this story: the profiler catches it, and if it were kept, the verifier would flag `suspected_leakage`.

**Video:** `exfil_customers.py` and `peek_sealed.py` (Demo 1). `egress.py` if there's time.

---

## 10. Schemas (the contract between Person A and Person B)

Implemented in `control/models.py`; this section is the summary.

### 10.1 `config.json` (control → worker)
```json
{"run_id": "s3-r1-w1-hgb2-s1-seg1", "task": "tabular", "dataset": "telco-3f9c…", "model": "hgb",
 "params": {"learning_rate": 0.1, "max_leaf_nodes": 31, "l2": 0.0, "calibration": "none"},
 "preprocessing": {"drop": ["customerID", "contract_end_reason"]},
 "target": "Churn", "metric": "savings", "costs": {"offer_cost": 50, "churn_loss": 600, "save_rate": 0.3},
 "seed": 1, "budget": {"unit": "rounds", "total": 100, "start_at": 0, "anneal_total": 200, "chunk": 25},
 "resume_from": null, "feature_fn": null, "ckpt_every": 1}
```
For `task: "rl"`: `dataset` = env id, `params` = the PPO params from the RL spec, `budget.unit = "env_steps"`, `feature_fn` = reward file.

### 10.2 `metrics.jsonl` (worker → node agent → control)
```json
{"type":"update","step":50,"unit":"rounds","t":3.1,"train_loss":0.41,"val_loss":0.43,"val_metric":11840.0,"extra":{"val_auc":0.846}}
{"type":"update","step":204800,"unit":"env_steps","t":131.2,"train_loss":0.42,"val_metric":1834.2,"extra":{"sps":1561,"approx_kl":0.012,"clipfrac":0.11,"ev":0.83,"lr":2.1e-4}}
{"type":"episode","step":205312,"t":131.6,"ret_true":1834.2,"ret_shaped":1834.2,"len":1000}
{"type":"checkpoint","step":100,"path":"ckpt/step_000000100.joblib","sha256":"9f2c…"}
{"type":"done","step":100,"reason":"budget"}
```

### 10.3 Agent outputs
- `ResearchBrief {problem_type, data_source{kind, ref}, target, metric, costs?, constraints[], questions[2–4], feasible, scope_note?}`
- `DataProfile {n_rows, n_cols, columns[{name, dtype, missing_frac, cardinality, single_feature_auc?}], target_stats{}, leakage_suspects[{column, reason}], id_like[], time_column?}`
- `Plan {round, task_summary, drop_columns[{column, reason}], configs[2–8 ConfigSpec{name, model, params{}, preprocessing{}, answers_question, rationale}], hypothesis_config, predictions[3–5], feature_code?}`
- `Prediction {id, statement, kind ∈ {a_beats_b, reaches_threshold, top_feature, flag_count, rank_transfer}, dataset, a, b?, min_ratio?, threshold?, feature?, flag?, min_count?, at_budget, p ∈ [0.05, 0.95]}`
- `MonitorDecision {actions[≤6 Action{kind, run_id?, reason, overrides?, extra_budget?}], summary}`
- `Reflection {findings[], next_round: Plan | null, stop_reason}`

### 10.4 SQLite tables
`sessions(id, prompt, status, brief_json, profile_json, report_json, created_at)` ·
`datasets(sha256, name, source, n_rows, train_path, sealed_path, created_at)` ·
`plans(session_id, round, plan_json)` ·
`runs(id, session_id, round, wave, segment, task, dataset, model, config_name, seed, params_json, parent_run_id, start_at, budget_total, node, status, exit_code, oom_killed, started_at, ended_at)` ·
`metrics(run_id, type, step, t, json)` (index on `(run_id, step)`) ·
`decisions(id, session_id, ts, actor, run_id, action, reason, evidence_json)` ·
`predictions(id, session_id, json, outcome, evidence_json)` ·
`events(id, session_id, ts, kind, payload_json)` ·
`artifacts(run_id, kind, path, sha256)`

### 10.5 Node agent API (additions to the RL spec §4.3)
- `PUT /datasets/{sha}` with `{train: bytes, sealed: bytes}` → stores under `/srv/datasets/<sha>/` and `/srv/sealed/<sha>/`.
- `POST /runs` gains `kind ∈ {profile, train, verify}`. Only `verify` mounts `/sealed`. `train` and `profile` mount `/data` (train pool) read-only.

**Run states:** `PENDING → STARTING → RUNNING → {DONE, STOPPED, FAILED, CONTAINED}`.

---

## 11. Repo layout

```
autoresearch/
  worker/                   # runs INSIDE the sandbox
    Dockerfile  requirements.txt
    run.py                  # entry point: dispatch on task
    profile.py              # DataProfile job
    evaluate.py             # verify job (tabular + rl)
    adapters/  tabular.py  rl_ppo.py
    common/    jsonl.py  ckpt.py  norm.py  harness.py (loads /job/features.py or /job/reward.py)
  node_agent/  app.py
  control/
    app.py  config.yaml  catalog.yaml
    models.py  db.py  llm.py
    ingest.py               # trusted: upload / catalog / OpenML → parquet, sha, sealed split
    understand.py  planner.py  monitor.py  features.py  reflect.py  reporter.py
    scheduler.py  verifier.py  stats.py  predictions.py  safety.py  replay.py
    templates/  static/
  infra/       setup_worker.sh  setup_control.sh  Caddyfile
  scripts/     bench.py  redteam/
  data/demo/   telco_churn.csv  telco_churn_with_leak.csv
  docs/        autoresearch-plan.md (this file)  rl-autoresearch-plan.md (RL adapter spec)
  tests/
```

---

## 12. Compute and cost

- **Demo 1:** about 50 short runs; roughly 0.5–1 vCPU-hour per session. It costs little, so judges can be allowed one live run at a time (rate-limited).
- **Demo 2:** wave 1 = 6 × 300k (with halving ≈ 1.35M steps); wave 2 = 3 configs × 2 envs × 3 seeds × 300k = 5.4M; forks ≤ 0.8M. Total ≈ 7.6M steps ≈ 1.4 run-hours ≈ 2.8 vCPU-hours at s = 1,500. Full RL sessions sit behind an access code; the public page shows the replay.
- **Formulas:** run time `T = budget / s`; wave wall time ≈ `ceil(N / S) · T`; slots `S = floor((vCPU_total − 2·n_nodes) / cpus_per_run)` (= 15 on a 32 vCPU worker).
- **Cost:** worker hourly price × hours it exists, plus the control VM. Check the console; create workers only when needed; destroy them right after submission. Check the instance-size quota at kickoff (fallback 2 × 16 vCPU).

---

## 13. Timeline (relative to now; shift so "submit" is ≥ 30 min before the deadline)

**Person A** = infra + web. **Person B** = agent + ML (Arjun).

| Time | Person A | Person B | Milestone |
|---|---|---|---|
| 0:00–0:30 | Control + worker VMs in one VPC; firewall (80/443 on control only); inference key; `/v1/chat/models` | `control/models.py` (done), get the Telco CSV and add the leak column, `llm.py` | Contract frozen |
| 0:30–2:00 | `setup_worker.sh` (Docker + gVisor), Dockerfile, node agent incl. datasets + sealed mount, test with a dummy container and `exfil_customers.py` | Tabular adapter (`run.py`, `adapters/tabular.py`, `profile.py`, `evaluate.py`), tested locally on Telco | |
| 2:00 | `bench.py` on the worker (PPO steps/sec, container start-up time, runc vs runsc) | | **M1:** tabular train + verify job runs in gVisor |
| 2:00–3:30 | `control/app.py`, DB, `ingest.py` + upload, scheduler, metrics poller, SSE, live curves, run table | `understand.py`, `planner.py` (+ budget trim), `features.py`, monitor rules + rungs + LLM + guardrails, `stats.py` (incl. paired bootstrap) | |
| 3:30 | | | **M2:** churn query end to end (brief → profile → plan → waves → verify). Start PPO worker work |
| 3:30–5:00 | Brief/profile panels, containment panel, report page, Caddy + sslip.io, access code, replay mode | `predictions.py`, `reflect.py`, `reporter.py` + number check, `safety.py`, Level-2 harness, red-team files; `adapters/rl_ppo.py` (RL spec §4.1), start the PPO demo session in the background | **M3:** churn report with real numbers + containment |
| 5:00–6:00 | Bug fixes, README, architecture diagram | PPO verify + videos; check the churn report numbers by hand against the DB | **M4:** PPO session done (replayable) |
| 6:00–7:00 | Record the video (live churn run + PPO replay), slide PDF | Submission text; test the public URL from a phone | **Submit** |

**Cut lines (decide at the clock times):**
- **Behind at 3:30:** drop the reflection round, OpenML (keep upload + catalog), and LLM-written feature code (keep the fixed red-team files for containment).
- **Behind at 5:00:** the PPO demo becomes Hopper-only with rule-based kills and no forks, shown as a replay; videos only for the winner.
- **Never cut:** gVisor sandbox, sealed-test verification, the dollar answer with CI, one containment moment, the public URL.

---

## 14. Testing

- **models:** schema round-trips, clamping of hostile params, fork bounds, metric directions (`tests/test_models.py`).
- **stats:** IQM of `[1, 2, 3, 4, 100]` = 3.0; bootstrap CI contains the point estimate; `diff_ci` of identical arrays straddles 0; paired bootstrap of a model against itself straddles 0.
- **savings:** a hand-computed 10-row example; threshold chosen on val is never refit on test; the target-everyone baseline matches the formula.
- **leakage:** the profiler flags `contract_end_reason` and `customerID` in the Telco file with the injected column; it flags nothing else.
- **safety:** every red-team file is rejected; three legitimate feature functions (tenure buckets, charges per month of tenure, service count) are accepted.
- **schemas under failure:** malformed planner/monitor/reflect JSON → one retry, then the fallback.
- **resume:** tabular chunked training (25 + 25 rounds) matches a straight 50-round run exactly (HGB `warm_start` is deterministic for a fixed seed); RL as in the RL spec.
- **chaos:** kill the node agent mid-run → runs marked `FAILED`, UI stays up; kill a container → status updates within 10 s.
- **eval sanity:** the verifier's sealed-test AUC is within ~0.03 of validation AUC for the M1 run (catches preprocessing not being saved in the checkpoint).

---

## 15. Demo video script (~3 min; check the lablab length limit)

1. **0:00–0:20 Problem.** Enterprise ML questions take weeks, results optimise the wrong metric, and nobody lets an agent run code on customer data.
2. **0:20–0:45 Query → brief.** The churn question in plain English; the agent turns it into a brief with the cost model and research questions.
3. **0:45–1:05 Profile → plan.** The profiler flags `contract_end_reason` as leakage; the plan drops it with a reason; predictions with probabilities appear.
4. **1:05–1:35 Runs.** Live curves, the rung cuts half, the reflection round adds calibration.
5. **1:35–2:00 Containment.** Red-team button: `exfil_customers.py` → network unreachable; `peek_sealed.py` → the sealed test set isn't there. The session continues.
6. **2:00–2:25 The answer.** "Target the top X%: saves $Y per 1,000 customers [CI]" versus target everyone (loses money) and the oracle; calibration plot; predictions scored; hashes.
7. **2:25–2:50 Same engine, different research.** PPO query in replay at 10×: a mid-run kill, a fork with half the learning rate, the transfer check to Walker2d, a median-seed video.
8. **2:50–3:10 Architecture.** Vultr control VM, VPC, gVisor workers, Serverless Inference call count.

---

## 16. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Container start-up dominates short tabular runs | Measure at M1; use larger chunks; reuse one container per run lineage if needed |
| Telco results too close to call | That's a legitimate finding; the report says "tie" with the CI. The dollar answer still stands |
| LLM picks a bad metric or target | Code validates the brief against the profile (target must exist and be binary); fallback to the catalog's defaults |
| Agent-written feature code breaks often | One repair attempt, then drop; Level 1 always works |
| Leakage detector false positives | Report suspects with reasons; the plan must justify every drop |
| PPO steps/sec lower than planned | Scale budgets by 1,500 / s; Hopper-only fallback |
| MuJoCo/osmesa issues under runsc | Test at M1; fall back to the default runtime with all other limits, stated honestly |
| LLM returns bad JSON or is slow | Pydantic + 1 retry + rule-based fallbacks; runs never wait on the LLM |
| Judges burn credits | Churn: one live session at a time; PPO behind an access code; replay mode |
| Running out of time | Cut lines in §13 |

---

## 17. Open items

- Exact deadline time; whether the submission can be edited after the first submit.
- Model choice from `/v1/chat/models` (default `kimi-k2-instruct`).
- Instance sizes the accounts can create.
- Video length limit and slide format.
- Confirm the Telco CSV source and licence before committing it to the repo.

## Sources
- Vultr Agent Rush event page: https://lablab.ai/ai-hackathons/vultr-hackathon
- Vultr Serverless Inference (chat, tool-calling note): https://docs.vultr.com/products/compute/serverless-inference/management/usage/chat
- Vultr Serverless Inference in Python: https://docs.vultr.com/how-to-use-vultr-serverless-inference-in-python
- gVisor install: https://gvisor.dev/docs/user_guide/install/
- CleanRL PPO continuous: https://docs.cleanrl.dev/rl-algorithms/ppo/
- rliable (IQM, interval estimates): https://github.com/google-research/rliable
- OpenML API: https://www.openml.org/apis
