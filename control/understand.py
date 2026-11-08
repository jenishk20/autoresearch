"""Stage 1: query -> ResearchBrief (main plan §4.1). LLM first, rule-based fallback."""
from __future__ import annotations

import re

from control.llm import LLMFailed
from control.models import RL_ENVS, ResearchBrief

SYSTEM = f"""You are the intake step of an ML research agent that runs on CPU-only sandboxes.
Turn the user's request into a research brief.

problem_type: one of binary, multiclass, regression, rl.
metric (code decides the direction): binary -> savings (only if the user gives costs), auc, logloss, accuracy, f1;
  multiclass -> accuracy, logloss, f1; regression -> rmse, mae; rl -> return.
data_source.kind: upload (user CSV), catalog, openml, or env (RL). For rl, data_source.ref is ONE env id from
  {list(RL_ENVS)} and also_evaluate_on lists other env ids to check transfer on (may be empty).
  Map names: "hopper" -> Hopper-v4, "walker"/"walker2d" -> Walker2d-v4, "cheetah"/"halfcheetah" -> HalfCheetah-v4.
costs (binary churn-style problems): offer_cost, churn_loss, save_rate (0-1).
budget_minutes: the time budget the user asked for, or null.
questions: 1-4 research questions phrased in the user's terms.
feasible: false if it needs GPUs, internet access from the sandbox, or anything outside these problem types;
  then explain in scope_note and propose the closest feasible version.

Schema:
{{"problem_type": str, "data_source": {{"kind": str, "ref": str}}, "target": str|null, "metric": str,
  "costs": {{"offer_cost": float, "churn_loss": float, "save_rate": float}}|null, "also_evaluate_on": [str],
  "budget_minutes": int|null, "constraints": [str], "questions": [str], "feasible": bool, "scope_note": str|null}}"""

_ENV_WORDS = [("halfcheetah", "HalfCheetah-v4"), ("cheetah", "HalfCheetah-v4"), ("walker", "Walker2d-v4"),
              ("hopper", "Hopper-v4"), ("pendulum", "InvertedDoublePendulum-v4")]


def _envs_in(text: str) -> list[str]:
    t, found = text.lower(), []
    hits = sorted((t.find(w), env) for w, env in _ENV_WORDS if w in t)
    for _, env in hits:
        if env not in found:
            found.append(env)
    return found


def _minutes(text: str) -> int | None:
    m = re.search(r"(\d{1,3})\s*-?\s*(?:min|minute)", text.lower())
    return max(2, min(240, int(m.group(1)))) if m else None


def fallback_brief(query: str) -> ResearchBrief:
    envs = _envs_in(query)
    rl_words = any(w in query.lower() for w in ("ppo", "reinforcement", " rl ", "policy", "mujoco", "locomotion"))
    if envs or rl_words:
        envs = envs or ["Hopper-v4"]
        q = [f"Which PPO configuration learns best on {envs[0]}?"]
        if len(envs) > 1:
            q.append(f"Does the {envs[0]} winner transfer to {', '.join(envs[1:])}?")
        return ResearchBrief(problem_type="rl", data_source={"kind": "env", "ref": envs[0]}, metric="return",
                             also_evaluate_on=envs[1:3], budget_minutes=_minutes(query), questions=q,
                             constraints=["rule-based intake (LLM unavailable)"])
    return ResearchBrief(problem_type="binary", data_source={"kind": "upload", "ref": "none"}, target="unknown",
                         metric="auc", questions=[query[:200] or "?"], feasible=False,
                         scope_note="Tabular research needs an uploaded dataset; the tabular adapter is not wired "
                                    "into this build yet. Try an RL query (e.g. PPO on Hopper).")


def understand(llm, session_id: str, query: str) -> tuple[ResearchBrief, str]:
    """Returns (brief, actor) where actor is 'llm' or 'rules'."""
    try:
        brief = llm.ask_json("understand", session_id, SYSTEM, f"User request:\n{query}", ResearchBrief)
        if brief.problem_type != "rl" and brief.feasible:
            brief.feasible = False
            brief.scope_note = ("Tabular research (upload/catalog/OpenML) is planned but not wired into this build "
                                "yet; RL queries run end to end.")
        return brief, "llm"
    except LLMFailed:
        return fallback_brief(query), "rules"
