"""Monitor (main plan §8.1): rules always on, LLM at rungs / on new flags / periodically, guardrails in code."""
from __future__ import annotations

import json

from pydantic import BaseModel

from control.llm import LLMFailed
from control.models import MonitorDecision, bound_fork_overrides

SYSTEM = """You are the monitor of a running PPO study. You see a table of live runs (features computed by code).
Decide whether to intervene. Allowed actions:
- stop(run_id, reason): stop a run that is clearly broken or hopeless (diverging, collapsed).
- fork(run_id, overrides, extra_budget, reason): branch a promising run from its latest checkpoint with changed
  hyperparameters. overrides may only set learning_rate (x0.5 or x2 of the parent), ent_coef (0-0.02), clip_coef (0.1-0.3).
- note(reason): record an observation without acting.
Prefer few, well-justified actions. You cannot stop the last running seed of a config. Code enforces all limits.
Schema: {"actions": [{"kind": "stop"|"fork"|"note", "run_id": str|null, "reason": str,
  "overrides": {str: float}|null, "extra_budget": int|null}], "summary": str}"""

RESCUE_SYSTEM = """You are the monitor of a PPO screening wave. Code has ranked the configs at a successive-halving rung
and will cut the bottom half. You may rescue AT MOST ONE cut config if its curve suggests it will overtake
(for example the steepest recent slope). Return {"rescue": str|null, "reason": str}."""


class Rescue(BaseModel):
    rescue: str | None = None
    reason: str


def table(feats: dict) -> list[dict]:
    keep = ("run_id", "config", "env", "seed", "frac", "ret_recent", "ret_best", "slope", "ep_len_recent", "kl",
            "clipfrac", "ev", "rank", "flags")
    return [{k: f.get(k) for k in keep} for f in feats.values()][:45]


def ask_monitor(llm, session_id, feats, fork_budget_left) -> MonitorDecision | None:
    user = json.dumps({"runs": table(feats), "fork_budget_left_steps": fork_budget_left})
    try:
        return llm.ask_json("monitor", session_id, SYSTEM, user, MonitorDecision)
    except LLMFailed:
        return None


def ask_rescue(llm, session_id, rows, cut) -> Rescue | None:
    try:
        r = llm.ask_json("rescue", session_id, RESCUE_SYSTEM, json.dumps({"ranked": rows, "cut": cut}), Rescue)
        return r if (r.rescue is None or r.rescue in cut) else None
    except LLMFailed:
        return None


def rule_decision(feats: dict, runs: dict, forked_envs: set, fork_budget_left: int) -> MonitorDecision:
    """Fallback when the LLM is unavailable: stop clear divergence; fork each env's leader once at ~half way."""
    actions = []
    for f in feats.values():
        if {"collapse", "kl_high"} <= set(f["flags"]):
            actions.append({"kind": "stop", "run_id": f["run_id"],
                            "reason": "rule: diverging (KL high for 3 updates and return collapsed below half its best)"})
    leaders = {}
    for f in feats.values():
        rs = runs[f["run_id"]]
        if rs.is_fork or f["ret_recent"] is None:
            continue
        if f["env"] not in leaders or f["ret_recent"] > leaders[f["env"]]["ret_recent"]:
            leaders[f["env"]] = f
    for env, f in leaders.items():
        if env in forked_envs or not (0.3 <= f["frac"] <= 0.8) or fork_budget_left <= 0:
            continue
        actions.append({"kind": "fork", "run_id": f["run_id"], "overrides": {"learning_rate": 0.0},
                        "extra_budget": 1, "reason": f"rule: exploit the {env} leader - branch it with half the "
                                                     "learning rate for the remaining steps"})
    return MonitorDecision.model_validate({"actions": actions[:6], "summary": "rule-based monitor (LLM unavailable)"})


def fork_overrides(parent_params: dict, requested: dict) -> dict:
    parent = {"learning_rate": 3e-4, "ent_coef": 0.0, "clip_coef": 0.2, **parent_params}
    req = dict(requested)
    if req.get("learning_rate", None) == 0.0:   # rule fallback asks for "half"
        req["learning_rate"] = parent["learning_rate"] * 0.5
    return bound_fork_overrides("ppo", parent, req)
