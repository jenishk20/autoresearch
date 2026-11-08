"""Score pre-registered predictions against code-computed results (main plan §8.4)."""
from __future__ import annotations

from control.models import Plan, improvement_ratio
from control.stats import boot_ci, diff_ci, iqm


def score_predictions(plan: Plan, scores: dict, flag_runs: dict, winners: dict, metric: str = "return") -> list[dict]:
    """
    scores:    {env: {config: [per-seed score, ...]}}   (wave-2, non-fork runs)
    flag_runs: {config: {flag: n_runs_that_raised_it}}
    winners:   {env: config_or_None}
    """
    out = []
    for p in plan.predictions:
        res = {"id": p.id, "statement": p.statement, "kind": p.kind, "p": p.p, "outcome": "inconclusive",
               "evidence": {}}
        env_scores = scores.get(p.dataset, {})
        if p.kind == "a_beats_b":
            xa, xb = env_scores.get(p.a, []), env_scores.get(p.b, [])
            if len(xa) < 3 or len(xb) < 3:
                res["evidence"] = {"why": f"needs >=3 seeds of both configs on {p.dataset}",
                                   "n_a": len(xa), "n_b": len(xb)}
            else:
                ia, ib = iqm(xa), iqm(xb)
                lo, hi = diff_ci(xa, xb)
                ratio = improvement_ratio(metric, ia, ib)
                res["evidence"] = {"iqm_a": round(ia, 1), "iqm_b": round(ib, 1), "diff_ci": [round(lo, 1), round(hi, 1)],
                                   "ratio": None if ratio is None else round(ratio, 3), "min_ratio": p.min_ratio}
                if ratio is not None and ratio >= p.min_ratio and lo > 0:
                    res["outcome"] = "confirmed"
                elif hi < 0 or (ratio is not None and ratio < p.min_ratio and lo > 0):
                    res["outcome"] = "refuted"
        elif p.kind == "reaches_threshold":
            xa = env_scores.get(p.a, [])
            if len(xa) < 3:
                res["evidence"] = {"why": f"needs >=3 seeds of {p.a} on {p.dataset}", "n": len(xa)}
            else:
                lo, hi = boot_ci(xa)
                res["evidence"] = {"iqm": round(iqm(xa), 1), "ci": [round(lo, 1), round(hi, 1)], "threshold": p.threshold}
                if lo >= p.threshold:
                    res["outcome"] = "confirmed"
                elif hi < p.threshold:
                    res["outcome"] = "refuted"
        elif p.kind == "flag_count":
            n = flag_runs.get(p.a, {}).get(p.flag, 0)
            ran = p.a in flag_runs
            res["evidence"] = {"runs_with_flag": n, "min_count": p.min_count, "config_ran": ran}
            if ran:
                res["outcome"] = "confirmed" if n >= p.min_count else "refuted"
        elif p.kind == "rank_transfer":
            cfg = winners.get(p.dataset) if p.a == "winner" else p.a
            tgt = scores.get(p.transfer_to, {})
            if not cfg or cfg not in tgt or len(tgt[cfg]) < 2:
                res["evidence"] = {"why": "config not run on the transfer env", "config": cfg}
            else:
                ranking = sorted(tgt, key=lambda c: -iqm(tgt[c]))
                rank = ranking.index(cfg) + 1
                res["evidence"] = {"config": cfg, "rank": rank, "of": len(ranking), "top_k": p.top_k}
                res["outcome"] = "confirmed" if rank <= p.top_k else "refuted"
        else:
            res["evidence"] = {"why": f"{p.kind} does not apply to this task"}
        out.append(res)
    return out


def brier(results: list[dict]) -> dict:
    decided = [r for r in results if r["outcome"] in ("confirmed", "refuted")]
    if not decided:
        return {"brier": None, "n_decided": 0, "baseline": 0.25}
    b = sum((r["p"] - (1.0 if r["outcome"] == "confirmed" else 0.0)) ** 2 for r in decided) / len(decided)
    return {"brier": round(b, 4), "n_decided": len(decided), "baseline": 0.25}
