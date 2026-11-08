"""Verifier + report assembly (main plan §8.2-8.4). Every number in report.json is computed here."""
from __future__ import annotations

import asyncio
import time

import numpy as np

from control.predictions import brier, score_predictions
from control.reporter import narrate
from control.stats import boot_ci, diff_ci, iqm


async def verify_runs(sess):
    targets = [r for r in sess.my_runs(2) if r.last_ckpt() and r.status in ("DONE", "STOPPED")]
    jobs = []
    for r in targets:
        v = sess._mk(3, 1, r.env, r.config, r.params, r.seed, 0, 1, 1, kind="verify", parent=r.id,
                     resume_from=r.last_ckpt()[1], is_fork=r.is_fork)
        v.id = r.id.replace("-w2-", "-v3-")
        jobs.append((r, v))
        sess.e.submit(v)
    await sess.e.wait([v.id for _, v in jobs])
    for r, v in jobs:
        r.eval = v.eval
        if v.eval:
            sess.e._publish(sess.id, {"type": "run", "run": sess.e.run_view(r)})


def _r(x, d=1):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


async def build_report(sess) -> dict:
    e, plan, brief = sess.e, sess.plan, sess.brief
    envs = sess.budget["envs"]
    w2 = [r for r in sess.my_runs(2)]
    base = [r for r in w2 if not r.is_fork]

    # ---- per-env tables (unit = seed; score = mean of 10 held-out episodes)
    scores: dict = {}
    per_env = {}
    flags_eval = []
    for env in envs:
        scores[env] = {}
        for r in base:
            if r.env == env and r.eval:
                scores[env].setdefault(r.config, []).append(r.eval["mean_return"])
        rows = []
        for cfg, xs in scores[env].items():
            lo, hi = boot_ci(xs) if len(xs) > 1 else (xs[0], xs[0])
            train = [r.ret_recent() for r in base if r.env == env and r.config == cfg and r.ret_recent() is not None]
            rows.append({"config": cfg, "n": len(xs), "iqm": _r(iqm(xs)), "ci": [_r(lo), _r(hi)],
                         "seed_scores": [_r(x) for x in xs], "train_return": _r(np.mean(train)) if train else None,
                         "steps": sess.budget["b2"]})
        rows.sort(key=lambda r: -(r["iqm"] or -1e9))
        winner, tie, gap_ci = (rows[0]["config"] if rows else None), False, None
        if len(rows) > 1 and rows[0]["n"] > 1 and rows[1]["n"] > 1:
            gap_ci = [_r(v) for v in diff_ci(scores[env][rows[0]["config"]], scores[env][rows[1]["config"]])]
            tie = gap_ci[0] <= 0
        per_env[env] = {"table": rows, "winner": winner, "runner_up": rows[1]["config"] if len(rows) > 1 else None,
                        "tie": tie, "winner_minus_runner_up_ci": gap_ci}
        for r in base:
            if r.env == env and r.eval and r.ret_recent():
                if r.ret_recent() > 0 and r.eval["mean_return"] < 0.8 * r.ret_recent():
                    flags_eval.append({"run_id": r.id, "flag": "eval_mismatch", "train": _r(r.ret_recent()),
                                       "verify": _r(r.eval["mean_return"])})

    # ---- overall winner across envs (Borda) + transfer
    cfgs = sorted({c for env in envs for c in scores[env]})
    borda = {c: [] for c in cfgs}
    for env in envs:
        order = [row["config"] for row in per_env[env]["table"]]
        for c in cfgs:
            borda[c].append(order.index(c) + 1 if c in order else len(order) + 1)
    overall = sorted(cfgs, key=lambda c: np.mean(borda[c])) if cfgs else []

    # ---- forks
    forks = []
    for r in w2:
        if r.is_fork:
            parent = e.runs.get(r.parent)
            forks.append({"run_id": r.id, "parent": r.parent, "env": r.env, "config": r.config,
                          "from_step": r.start_at, "params": {k: r.params.get(k) for k in ("learning_rate", "ent_coef", "clip_coef") if k in r.params},
                          "verify": _r(r.eval["mean_return"]) if r.eval else None,
                          "parent_verify": _r(parent.eval["mean_return"]) if parent and parent.eval else None})

    # ---- flags per config (for flag_count predictions)
    flag_runs: dict = {}
    for r in sess.my_runs():
        d = flag_runs.setdefault(r.config.split("~")[0], {})
        for fl in r.flags_ever:
            d[fl] = d.get(fl, 0) + 1
    winners = {env: per_env[env]["winner"] for env in envs}
    preds = score_predictions(plan, scores, flag_runs, winners)
    with e.db_lock:
        for p in preds:
            e.db.execute("UPDATE predictions SET outcome=?, evidence_json=? WHERE session_id=? AND id=?",
                         (p["outcome"], str(p["evidence"]), sess.id, p["id"]))
        e.db.commit()

    # ---- compute
    train_runs = sess.my_runs()
    steps = sum(max(0, r.last_step - r.start_at) for r in train_runs)
    run_s = sum((r.ended_at or time.time()) - (r.started_at or time.time()) for r in e.runs.values() if r.session_id == sess.id)
    n1 = len(plan.configs)
    seg2 = len(sess.my_runs(1)) - n1
    halving = {"steps_used": (n1 + seg2) * sess.budget["b1"], "naive_steps": n1 * 2 * sess.budget["b1"]}
    halving["saved_pct"] = _r(100 * (1 - halving["steps_used"] / halving["naive_steps"]))

    with e.db_lock:
        decisions = [dict(r) for r in e.db.execute(
            "SELECT ts, actor, run_id, action, reason FROM decisions WHERE session_id=? ORDER BY ts", (sess.id,))]
        contain = [dict(r) for r in e.db.execute(
            "SELECT ts, payload_json FROM events WHERE session_id=? AND kind='containment' ORDER BY ts", (sess.id,))]
        llm_calls = e.db.execute("SELECT COUNT(*) FROM events WHERE session_id=? AND kind='llm_call'", (sess.id,)).fetchone()[0]

    artifacts = [{"run_id": r.id, "env": r.env, "config": r.config, "seed": r.seed,
                  "checkpoint": r.last_ckpt()[1], "sha256": r.last_ckpt()[2],
                  "verify_mean": _r(r.eval["mean_return"]) if r.eval else None}
                 for r in w2 if r.last_ckpt()]

    report = {
        "session_id": sess.id, "query": sess.query, "task_summary": plan.task_summary,
        "brief": brief.model_dump(), "envs": envs, "budget": sess.budget,
        "per_env": per_env, "overall_ranking": overall, "borda": {c: _r(np.mean(v), 2) for c, v in borda.items()},
        "transfer": {"from": envs[0], "to": envs[1:], "winner": per_env[envs[0]]["winner"],
                     "rank_on": {env: ([row["config"] for row in per_env[env]["table"]].index(per_env[envs[0]]["winner"]) + 1
                                       if per_env[envs[0]]["winner"] in [row["config"] for row in per_env[env]["table"]] else None)
                                 for env in envs[1:]}},
        "forks": forks, "eval_flags": flags_eval,
        "predictions": preds, "calibration": brier(preds),
        "halving": halving,
        "compute": {"env_steps": steps, "run_seconds": _r(run_s), "vcpu_hours": _r(run_s * e.cfg["run"]["cpus"] / 3600, 2),
                    "sandboxed": e.sandboxed},
        "decisions": decisions, "containment": contain, "artifacts": artifacts,
        "llm": {**e.llm.status, "session_calls": llm_calls},
        "verification": "final checkpoints re-scored in fresh jobs on eval seeds 10000-10009, deterministic actions, raw env reward",
    }
    narrative, actor, _ = await asyncio.to_thread(narrate, e.llm, sess.id, report)
    report["narrative"] = narrative.model_dump()
    report["narrative_source"] = actor
    return report
