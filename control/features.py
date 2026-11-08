"""Per-run live state and monitor features (main plan §8.1). All numbers here are computed in code;
the LLM only ever sees the resulting table."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RunState:
    id: str
    session_id: str
    round: int
    wave: int
    segment: int
    env: str
    config: str
    seed: int
    params: dict
    start_at: int
    total: int
    anneal_total: int
    kind: str = "train"            # train | verify | redteam
    parent: str | None = None
    resume_from: str | None = None
    code_files: dict = field(default_factory=dict)
    status: str = "PENDING"
    node: str | None = None
    offset: int = 0
    last_step: int = 0
    updates: list = field(default_factory=list)    # dicts
    episodes: list = field(default_factory=list)   # (step, ret_true, len)
    ckpts: list = field(default_factory=list)      # (step, path, sha)
    done_reason: str | None = None
    exit_code: int | None = None
    best10: float = float("-inf")
    flags: set = field(default_factory=set)        # current flags
    flags_ever: set = field(default_factory=set)
    note: str | None = None                        # containment / failure detail
    is_fork: bool = False
    eval: dict | None = None
    started_at: float | None = None
    ended_at: float | None = None

    @property
    def lineage_key(self) -> tuple:
        return (self.round, self.wave, self.env, self.config, self.seed)

    def ret_recent(self) -> float | None:
        if not self.episodes:
            return None
        return float(np.mean([e[1] for e in self.episodes[-10:]]))

    def add_episode(self, step, ret, ln):
        self.episodes.append((step, ret, ln))
        if len(self.episodes) >= 3:
            self.best10 = max(self.best10, float(np.mean([e[1] for e in self.episodes[-10:]])))

    def last_ckpt(self):
        return self.ckpts[-1] if self.ckpts else None


def slope_per_100k(rs: RunState) -> float:
    if len(rs.episodes) < 6:
        return 0.0
    steps = np.array([e[0] for e in rs.episodes], dtype=float)
    lo = steps.min() + 0.7 * (steps.max() - steps.min())
    m = steps >= lo
    if m.sum() < 4 or np.ptp(steps[m]) == 0:
        return 0.0
    x, y = steps[m], np.array([e[1] for e in rs.episodes])[m]
    return float(np.polyfit(x, y, 1)[0] * 100_000)


def compute_features(runs: list[RunState], sps_benchmark: float) -> dict[str, dict]:
    """Features for every run in `runs` (a cohort = same wave & env is formed inside)."""
    feats: dict[str, dict] = {}
    for rs in runs:
        ups = rs.updates[-5:]
        span = max(1, rs.total - rs.start_at)
        feats[rs.id] = {
            "run_id": rs.id, "config": rs.config, "env": rs.env, "seed": rs.seed, "wave": rs.wave,
            "status": rs.status, "step": rs.last_step,
            "frac": round(min(1.0, (rs.last_step - rs.start_at) / span), 3),
            "ret_recent": None if rs.ret_recent() is None else round(rs.ret_recent(), 1),
            "ret_best": None if rs.best10 == float("-inf") else round(rs.best10, 1),
            "slope": round(slope_per_100k(rs), 1),
            "ep_len_recent": round(float(np.mean([e[2] for e in rs.episodes[-10:]])), 1) if rs.episodes else None,
            "kl": round(float(np.mean([u["extra"].get("approx_kl", 0) for u in ups])), 4) if ups else None,
            "clipfrac": round(float(np.mean([u["extra"].get("clipfrac", 0) for u in ups])), 3) if ups else None,
            "ev": round(float(np.mean([u["extra"].get("ev", 0) for u in ups])), 3) if ups else None,
            "sps": ups[-1]["extra"].get("sps") if ups else None,
        }
    # cohort ranks + flags
    cohorts: dict[tuple, list[str]] = {}
    for rs in runs:
        cohorts.setdefault((rs.round, rs.wave, rs.env), []).append(rs.id)
    by_id = {r.id: r for r in runs}
    for ids in cohorts.values():
        scored = sorted(ids, key=lambda i: -(feats[i]["ret_recent"] if feats[i]["ret_recent"] is not None else -1e9))
        bests = [feats[i]["ret_best"] for i in ids if feats[i]["ret_best"] is not None]
        med_best = float(np.median(bests)) if bests else 0.0
        for rank, i in enumerate(scored, 1):
            f, rs = feats[i], by_id[i]
            f["rank"] = rank
            f["cohort_size"] = len(ids)
            flags = set()
            if rs.done_reason == "nan":
                flags.add("nan")
            if len(rs.updates) >= 3 and all(u["extra"].get("approx_kl", 0) > 0.05 for u in rs.updates[-3:]):
                flags.add("kl_high")
            if (f["ret_recent"] is not None and f["ret_best"] is not None and f["ret_best"] > med_best
                    and f["ret_best"] > 0 and f["ret_recent"] < 0.5 * f["ret_best"]):
                flags.add("collapse")
            if f["frac"] > 0.5 and f["slope"] <= 0 and rank > len(ids) / 2 and len(ids) > 1:
                flags.add("stalled")
            if f["sps"] and sps_benchmark and f["sps"] < 0.5 * sps_benchmark:
                flags.add("slow")
            f["flags"] = sorted(flags)
            rs.flags = flags
            rs.flags_ever |= flags
    return feats
