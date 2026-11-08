"""Stage 4: brief -> Plan + budget (main plan §4.4, §7.2). The LLM chooses configs and predictions;
code clamps, sizes the waves to the budget, and trims (logging what it trimmed)."""
from __future__ import annotations

import json
import math

from control.llm import LLMFailed
from control.models import PLANNER_CHOICES, Plan, ResearchBrief

ENV_FACTS = {
    "Hopper-v4": "obs 11 / act 3, terminates on fall (short episodes = falling), PPO ~2k-3k at 1M steps",
    "Walker2d-v4": "obs 17 / act 6, terminates on fall, PPO ~2.5k-4k at 1M steps",
    "HalfCheetah-v4": "obs 17 / act 6, never terminates (always 1000 steps), PPO ~1.5k-3k at 1M, high seed variance",
    "InvertedDoublePendulum-v4": "obs 11 / act 1, terminates on fall, max ~9.3k, smoke test",
    "PandaReachDense-v3": "Franka Panda arm must move its gripper to a random target point; 50-step episodes; dense reward = "
                          "-distance (0 is perfect); success = within 5 cm. PPO usually solves it in ~50k-100k steps",
    "PandaPushDense-v3": "Franka Panda arm must push a cube across the table to a target; 50-step episodes; dense reward = "
                         "-cube-to-target distance; success = cube within 5 cm. Hard for PPO in small budgets",
}

SYSTEM = """You are an RL research lead planning PPO experiments on simulated robot tasks under a fixed CPU budget.
Choose 2-{max_configs} PPO configs (at most {max_configs}) from the allowed space to test specific hypotheses (not to cover the grid). Each config
gives only the params it changes from the default (hidden 64x64, learning_rate 3e-4, ent_coef 0.0, norm_obs true).
Write 3-5 falsifiable predictions with a probability you would bet at (0.05-0.95).

Prediction kinds (dataset = an env id):
- a_beats_b: config a's mean held-out return beats b's by >= min_ratio (> 1.0) on dataset.
- reaches_threshold: config a's held-out return reaches threshold on dataset.
- flag_count: runs of config a raise `flag` (one of kl_high, collapse, stalled, nan) at least min_count times.
- rank_transfer: config a (or "winner" = the winner on dataset) is in the top_k on env transfer_to.

Facts: results after a few hundred thousand steps with few seeds are noisy. Observation normalisation usually matters a lot
for PPO here. lr 1e-3 often makes approx_kl spike. Do not invent hyperparameters or values outside the allowed space.
hypothesis_config is the config whose key prediction you most want tested with several seeds; it is carried to wave 2.

Schema:
{"round": 1, "task_summary": str, "configs": [{"name": str (letters, digits, _ or -), "model": "ppo",
  "params": {...}, "answers_question": int|null, "rationale": str}], "hypothesis_config": str,
 "predictions": [{"id": "P1", "statement": str, "kind": str, "dataset": str, "a": str, "b": str|null,
  "min_ratio": float|null, "threshold": float|null, "flag": str|null, "min_count": int|null,
  "transfer_to": str|null, "top_k": int|null, "p": float}]}"""


def default_plan(brief: ResearchBrief, max_configs: int = 6) -> Plan:
    env = brief.data_source.ref
    preds = [
        {"id": "P1", "statement": f"Observation normalisation helps: base beats no_norm by >=1.3x on {env}.",
         "kind": "a_beats_b", "dataset": env, "a": "base", "b": "no_norm", "min_ratio": 1.3, "p": 0.8},
        {"id": "P2", "statement": "lr=1e-3 raises the kl_high flag in at least one run.", "kind": "flag_count",
         "dataset": env, "a": "lr_high", "flag": "kl_high", "min_count": 1, "p": 0.6},
        {"id": "P3", "statement": f"base reaches a held-out reward of {-3 if env.startswith('Panda') else 500} on {env}.",
         "kind": "reaches_threshold", "dataset": env, "a": "base", "threshold": -3.0 if env.startswith("Panda") else 500.0, "p": 0.5},
    ]
    if brief.also_evaluate_on:
        preds.append({"id": "P4", "statement": f"The {env} winner is top-2 on {brief.also_evaluate_on[0]}.",
                      "kind": "rank_transfer", "dataset": env, "a": "winner", "transfer_to": brief.also_evaluate_on[0],
                      "top_k": 2, "p": 0.6})
    cfgs = [
        ("base", {}, "CleanRL defaults; the reference point"),
        ("no_norm", {"norm_obs": False}, "known-bad control: tests whether obs normalisation matters"),
        ("lr_high", {"learning_rate": 1e-3}, "faster learning, risk of KL spikes"),
        ("lr_low", {"learning_rate": 1e-4}, "stabler but slower"),
        ("wide", {"hidden": "256x256"}, "more capacity"),
        ("explore", {"ent_coef": 0.01}, "entropy bonus for exploration"),
    ][:max_configs]
    names = {c[0] for c in cfgs}
    preds = [p for p in preds if p["a"] in names | {"winner"} and p.get("b") in names | {None}]
    thr = -3.0 if env.startswith("Panda") else 500.0
    for c in [c[0] for c in cfgs]:   # keep >= 3 bets when the team is small
        if len(preds) >= 3:
            break
        if not any(p["kind"] == "reaches_threshold" and p["a"] == c for p in preds):
            preds.append({"id": f"P{len(preds) + 1}", "statement": f"{c} reaches a held-out reward of {thr:g} on {env}.",
                          "kind": "reaches_threshold", "dataset": env, "a": c, "threshold": thr, "p": 0.5})
    for i, p in enumerate(preds, 1):
        p["id"] = f"P{i}"
    return Plan.model_validate({
        "task_summary": f"Screen {len(cfgs)} PPO variants on {env}, confirm the best with several seeds"
                        + (f", check transfer to {', '.join(brief.also_evaluate_on)}" if brief.also_evaluate_on else ""),
        "configs": [{"name": n, "model": "ppo", "params": p, "rationale": r} for n, p, r in cfgs],
        "hypothesis_config": "no_norm", "predictions": preds})


def wave2_configs(plan: Plan, ranked: list[str], cap: int = 4) -> list[str]:
    """Top 2 from wave 1 + the hypothesis config + the configs its a_beats_b prediction needs."""
    out = ranked[:2]
    extra = [plan.hypothesis_config]
    for p in plan.predictions:
        if p.kind == "a_beats_b" and plan.hypothesis_config in (p.a, p.b):
            extra += [p.a, p.b]
    for c in extra:
        if c and c not in out and len(out) < cap:
            out.append(c)
    for c in ranked:  # pad to 3 if fewer
        if len(out) >= 3:
            break
        if c not in out:
            out.append(c)
    return out


def size_budget(n_cfg: int, n_envs: int, minutes: float, sps: float, slots: int, cfg: dict) -> dict:
    """Pick the largest (wave-1 segment, wave-2) step ladder rung that fits (formulas from plan §12)."""
    seeds, verify_s = cfg["seeds"], cfg["verify_seconds"]
    avail = minutes * 60 - 45   # LLM + startup overhead
    notes = []

    def est(b1, b2, s_):
        n2 = min(n_cfg, 3) * n_envs * s_
        w1 = math.ceil(n_cfg / slots) * b1 / sps + math.ceil(math.ceil(n_cfg / 2) / slots) * b1 / sps
        w2 = math.ceil(n2 / slots) * b2 / sps
        ver = math.ceil(n2 / slots) * verify_s
        return w1 + w2 + ver + 0.1 * w2  # + fork reserve

    for s_ in (seeds, 2):
        for b1, b2 in cfg["ladder"]:
            t = est(b1, b2, s_)
            if t <= avail:
                if s_ != seeds:
                    notes.append(f"seeds trimmed {seeds} -> {s_} to fit {minutes:.0f} min")
                return {"b1": b1, "b2": b2, "seeds": s_, "est_minutes": round(t / 60, 1), "sps": round(sps),
                        "slots": slots, "minutes": minutes, "notes": notes}
    b1, b2 = cfg["ladder"][-1]
    notes.append("budget too small even for the smallest preset; running the smallest anyway")
    return {"b1": b1, "b2": b2, "seeds": 2, "est_minutes": round(est(b1, b2, 2) / 60, 1), "sps": round(sps),
            "slots": slots, "minutes": minutes, "notes": notes}


def trim_plan(plan: Plan, max_configs: int) -> tuple[Plan, str | None]:
    """Keep the hypothesis config and configs that predictions reference first; drop predictions that lose a config."""
    if len(plan.configs) <= max_configs:
        return plan, None
    order = [plan.hypothesis_config] + [r for p in plan.predictions for r in (p.a, p.b) if r] + [c.name for c in plan.configs]
    keep = []
    for n in order:
        if n not in keep and any(c.name == n for c in plan.configs):
            keep.append(n)
    keep = keep[:max_configs]
    preds = [p for p in plan.predictions if (p.a in keep or p.a == "winner") and (p.b is None or p.b in keep)]
    new = plan.model_copy(update={"configs": [c for c in plan.configs if c.name in keep], "predictions": preds})
    return new, f"planner asked for {len(plan.configs)} configs, trimmed to {len(keep)}: {keep}"


def plan_rl(llm, session_id: str, brief: ResearchBrief, budget_hint: dict, max_configs: int = 3) -> tuple[Plan, str]:
    envs = [brief.data_source.ref, *brief.also_evaluate_on]
    user = json.dumps({
        "brief": brief.model_dump(), "env_facts": {e: ENV_FACTS[e] for e in envs},
        "allowed_space": {k: list(v) for k, v in PLANNER_CHOICES["ppo"].items()},
        "compute": budget_hint,
        "design": "wave 1 screens every config on the first env with 1 seed and successive halving; wave 2 runs "
                  "the top 2 + hypothesis_config on all envs with several seeds; winners are re-scored on held-out seeds.",
    }, indent=1)
    try:
        plan = llm.ask_json("plan", session_id, SYSTEM.replace("{max_configs}", str(max_configs)), user, Plan)
        if any(c.model != "ppo" for c in plan.configs):
            raise LLMFailed("non-ppo model in RL plan")
        return plan, "llm"
    except LLMFailed:
        return default_plan(brief, max_configs), "rules"
