# Autoresearch: pitch

**One line:** an AI research engineer that tunes robot control policies for you, proves which setup really wins, and
does all of it inside sandboxes it cannot escape.

## Use case: autonomous policy tuning for robotics and control teams

Who: robotics and automation teams that train control policies in simulation before deploying them, for example
legged and warehouse robots, drones, industrial control, or game AI. Our demo robots (Hopper, Walker2d, HalfCheetah) are the standard
simulated stand-ins for this work.

### The problem
1. **Tuning is slow, manual babysitting.** Reinforcement learning is very sensitive to settings (learning rate,
   network size, exploration). Engineers launch sweeps, watch curves for days, and kill runs by hand.
2. **"Wins" are often luck.** One lucky random seed can make a bad setup look best. Teams ship policies that fail
   when conditions change.
3. **Compute is wasted.** Runs that are clearly failing keep burning CPU/GPU until someone notices.
4. **Letting an AI agent run code is a security risk.** An agent that writes reward functions or training code can
   leak data, reach the internet, or damage the machine it runs on. Most teams won't allow it.
5. **No audit trail.** Hard to say later which exact model produced which number.

### How Autoresearch solves it
| Problem | What the agent does |
|---|---|
| Babysitting | Plans the study from one plain-English question, launches every run, and watches the live curves: it cuts weak runs, rescues promising ones, and branches leaders with gentler settings, all with logged reasons |
| Luck | Confirms finalists with several seeds, then **re-tests the saved models on unseen conditions** (held-out seeds, true reward). Reports ranges, and says "too close to call" when it is |
| Wasted compute | Successive halving cuts the weaker half early (~25% less screening compute) |
| Unsafe agents | Every experiment runs in its own **gVisor sandbox on Vultr**: no internet, read-only system, CPU/memory/process caps. Live demo: we ask it to reach the internet, write outside, fork-bomb, grab 8 GB of memory, and it's blocked every time |
| No audit trail | Every number in the report is computed by code and linked to a run and a SHA-256 of the saved model. Predictions are written **before** training and scored after (Brier score), so the AI can't move the goalposts |
| Transfer risk | Checks whether the winner on one robot still works on another body: a cheap proxy for robustness before sim-to-real |

### Business value
- **Engineer time:** a tuning study goes from days of babysitting to one question and a ~30-minute wait.
- **Compute cost:** runs on cheap Vultr CPU VMs (no GPUs). A full study uses ~1–2 vCPU-hours (~$0.10–0.30). Early cutting
  saves ~25% of screening compute.
- **Fewer bad deploys:** multi-seed confirmation and unseen-condition re-tests catch lucky winners before they ship.
- **Safe to adopt:** "blast radius zero": the agent's code can't touch the network, the host, or other experiments,
  which is what security teams need before letting an AI run experiments.
- **Audit-ready:** pre-registered bets, a full decision log and model fingerprints, which matter for safety cases in robotics.

### Same engine, more use cases (next adapters)
- **Business ML studies:** e.g. "which customers should get a retention offer, and how much will it save?" (tabular
  models, answers in dollars, data leakage detection). Planned in `docs/autoresearch-plan.md`.
- Any "which setup is best, and can we trust it?" question: ranking/recommendation models, forecasting, ablations.

## Demo (3 minutes)
1. **0:00 Problem:** tuning robots is babysitting, wins are luck, AI agents running code are risky.
2. **0:20 Ask:** "Train the Hopper robot to hop as far as possible… check it also works on Walker2d. 30-minute budget."
3. **0:40 Plan:** the agent picks setups and writes its bets with confidence levels.
4. **1:00 Live:** curves climb; the agent cuts the weak half, rescues one, branches a leader. Plain-English activity feed.
5. **1:40 Containment:** click "Reach the internet" and "Write outside the sandbox": both blocked, the study keeps running.
6. **2:10 Results:** videos of the trained robots hopping and walking, the winning setup with its likely range, which
   bets were right, compute saved, model fingerprints.
7. **2:45 Architecture:** Vultr Serverless Inference (agent brain) → control plane → Vultr VM running gVisor sandboxes.
