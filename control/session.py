"""One research session: Understand -> Plan -> wave 1 (halving) -> wave 2 (+monitor) -> verify -> report."""
from __future__ import annotations

import asyncio
import json
import math
import time
import traceback
import uuid

from control.engine import Engine
from control.features import RunState, compute_features
from control.models import Plan, ResearchBrief
from control.monitor import ask_monitor, ask_rescue, fork_overrides, rule_decision
from control.planner import plan_rl, size_budget, wave2_configs
from control.understand import understand

ENV_SHORT = {"HalfCheetah-v4": "hc", "Walker2d-v4": "w2d", "Hopper-v4": "hop", "InvertedDoublePendulum-v4": "idp"}


class SessionRunner:
    def __init__(self, engine: Engine, query: str, session_id: str | None = None):
        self.e = engine
        self.query = query
        self.id = session_id or uuid.uuid4().hex[:8]
        self.brief: ResearchBrief | None = None
        self.plan: Plan | None = None
        self.budget: dict = {}
        self.phase = "created"
        self.status = "running"
        self.fork_steps_used = 0
        self.fork_budget = 0
        self.forked_envs: set = set()
        self.n_forks = 0
        self.report: dict | None = None
        self.task: asyncio.Task | None = None
        with engine.db_lock:
            engine.db.execute("INSERT INTO sessions(id, prompt, status, created_at) VALUES (?,?,?,?)",
                              (self.id, query, "running", time.time()))
            engine.db.commit()

    # ------------------------------------------------------------ helpers
    def _save(self, **cols):
        sets = ", ".join(f"{k}=?" for k in cols)
        with self.e.db_lock:
            self.e.db.execute(f"UPDATE sessions SET {sets} WHERE id=?", (*cols.values(), self.id))
            self.e.db.commit()

    def set_phase(self, phase: str, detail: str = ""):
        self.phase = phase
        self.e.log_event(self.id, "phase", {"phase": phase, "detail": detail})

    def my_runs(self, wave=None, kind="train"):
        return [r for r in self.e.runs.values() if r.session_id == self.id and r.kind == kind
                and (wave is None or r.wave == wave)]

    def _mk(self, wave, segment, env, cfg_name, params, seed, start, total, anneal, **kw) -> RunState:
        rid = f"{self.id}-w{wave}-{ENV_SHORT[env]}-{cfg_name}-s{seed}-seg{segment}".replace("~", "-")
        return RunState(id=rid, session_id=self.id, round=1, wave=wave, segment=segment, env=env, config=cfg_name,
                        seed=seed, params=params, start_at=start, total=total, anneal_total=anneal, **kw)

    def start(self):
        self.task = asyncio.create_task(self.run())
        return self

    # ------------------------------------------------------------ main
    async def run(self):
        prov = self.e.provisioner
        self.vm_task = None
        try:
            if prov.mode == "vultr":   # boot the session's VM while the LLM works on the brief and plan
                self.e.log_event(self.id, "worker", {"msg": "connecting to the gVisor worker VM"})
                self.vm_task = asyncio.create_task(prov.acquire(self.id))
            self.set_phase("understand", "turning the query into a research brief")
            self.brief, actor = await asyncio.to_thread(understand, self.e.llm, self.id, self.query)
            self._save(brief_json=self.brief.model_dump_json())
            self.e.decision(self.id, actor, None, "brief", f"problem={self.brief.problem_type}, "
                            f"env={self.brief.data_source.ref}, transfer={self.brief.also_evaluate_on}",
                            self.brief.model_dump())
            if not self.brief.feasible or self.brief.problem_type != "rl":
                self.status = "unsupported"
                self._save(status="unsupported")
                self.set_phase("unsupported", self.brief.scope_note or "not supported")
                return
            await self.run_rl()
            self.status = "done"
            self._save(status="done")
            self.set_phase("done", "report ready")
        except asyncio.CancelledError:
            self.status = "cancelled"
            self._save(status="interrupted")
            raise
        except Exception as ex:
            self.status = "failed"
            self._save(status="failed")
            self.e.log_event(self.id, "error", {"error": str(ex), "trace": traceback.format_exc()[-2000:]})
            self.set_phase("failed", str(ex))
        finally:
            if self.vm_task is not None:
                if not self.vm_task.done():
                    self.vm_task.cancel()
                await asyncio.shield(prov.release(self.id))

    async def run_rl(self):
        b = self.brief
        envs = [b.data_source.ref, *[x for x in b.also_evaluate_on if x != b.data_source.ref]][:3]
        primary = envs[0]
        minutes = b.budget_minutes or self.e.cfg["rl"]["default_budget_minutes"]
        sps, slots = self.e.sps_estimate(), self.e.total_slots

        self.set_phase("plan", "choosing configs and pre-registering predictions")
        max_cfg = self.e.cfg["rl"].get("max_configs", 3)
        self.plan, actor = await asyncio.to_thread(plan_rl, self.e.llm, self.id, b,
                                                   {"slots": slots, "sps_per_run": round(sps), "minutes": minutes}, max_cfg)
        from control.planner import trim_plan
        self.plan, trim_note = trim_plan(self.plan, max_cfg)
        self.budget = size_budget(len(self.plan.configs), len(envs), minutes, sps, slots, self.e.cfg["rl"])
        self.budget["envs"] = envs
        with self.e.db_lock:
            self.e.db.execute("INSERT OR REPLACE INTO plans(session_id, round, plan_json) VALUES (?,?,?)",
                              (self.id, 1, self.plan.model_dump_json()))
            for p in self.plan.predictions:
                self.e.db.execute("INSERT OR REPLACE INTO predictions(id, session_id, json) VALUES (?,?,?)",
                                  (p.id, self.id, p.model_dump_json()))
            self.e.db.commit()
        self.e.decision(self.id, actor, None, "plan", f"{len(self.plan.configs)} configs, "
                        f"{len(self.plan.predictions)} predictions; budget {self.budget}", self.plan.model_dump())
        if trim_note:
            self.budget["notes"].append(trim_note)
        for note in self.budget["notes"]:
            self.e.decision(self.id, "code", None, "trim", note)
        self.e._publish(self.id, {"type": "plan", "plan": self.plan.model_dump(), "budget": self.budget})

        B1, B2, seeds = self.budget["b1"], self.budget["b2"], self.budget["seeds"]
        cfgs = {c.name: c.resolved_params() for c in self.plan.configs}

        if self.vm_task is not None:
            self.set_phase("provision", "waiting for the gVisor worker VM")
            await self.vm_task

        # ---------------- wave 1, segment 1
        self.set_phase("wave1", f"screening {len(cfgs)} configs on {primary}, {B1:,} steps each")
        seg1 = [self._mk(1, 1, primary, n, p, 1, 0, B1, 2 * B1) for n, p in cfgs.items()]
        await self.run_wave(seg1, wave=1)

        # ---------------- rung (code), rescue (LLM or rule)
        ok = [r for r in seg1 if r.status == "DONE" and r.last_ckpt()]
        ranked = sorted(ok, key=lambda r: -(r.ret_recent() or -1e9))
        keep_n = math.ceil(len(ranked) / 2)
        survivors, cut = ranked[:keep_n], ranked[keep_n:]
        feats = compute_features(seg1, sps)
        rows = [{"config": r.config, "ret_recent": feats[r.id]["ret_recent"], "slope": feats[r.id]["slope"],
                 "kl": feats[r.id]["kl"]} for r in ranked]
        self.e.decision(self.id, "code", None, "rung",
                        f"keep top {keep_n} of {len(ranked)} by mean return of the last 10 episodes: "
                        f"{[r.config for r in survivors]}; cut {[r.config for r in cut]}", rows)
        if cut:
            res = await asyncio.to_thread(ask_rescue, self.e.llm, self.id, rows, [r.config for r in cut])
            actor = "llm"
            if res is None:  # rule: rescue the cut config with the steepest slope if it beats the survivors' median
                actor = "rules"
                surv_slopes = sorted(feats[r.id]["slope"] for r in survivors)
                med = surv_slopes[len(surv_slopes) // 2] if surv_slopes else 0
                best = max(cut, key=lambda r: feats[r.id]["slope"])
                from control.monitor import Rescue
                res = (Rescue(rescue=best.config, reason=f"rule: steepest slope among cut configs "
                              f"({feats[best.id]['slope']} per 100k vs survivors' median {med})")
                       if feats[best.id]["slope"] > med else Rescue(rescue=None, reason="rule: no cut config is climbing faster"))
            if res.rescue:
                r = next(r for r in cut if r.config == res.rescue)
                survivors.append(r)
                cut.remove(r)
                self.e.decision(self.id, actor, r.id, "rescue", res.reason, feats[r.id])
            else:
                self.e.decision(self.id, actor, None, "note", res.reason)
        for r in cut:
            self.e.decision(self.id, "code", r.id, "cut", f"bottom half at the rung (return {feats[r.id]['ret_recent']})")

        # ---------------- wave 1, segment 2 (resume survivors)
        self.set_phase("wave1-seg2", f"{len(survivors)} survivors continue to {2 * B1:,} steps")
        seg2 = [self._mk(1, 2, primary, r.config, r.params, 1, B1, 2 * B1, 2 * B1, parent=r.id,
                         resume_from=r.last_ckpt()[1]) for r in survivors]
        await self.run_wave(seg2, wave=1)
        final = {r.config: r for r in seg2 if r.ret_recent() is not None}
        order = sorted(final, key=lambda c: -final[c].ret_recent()) + [r.config for r in cut]
        w2cfgs = wave2_configs(self.plan, order)
        self.e.decision(self.id, "code", None, "select", f"wave 2 configs: {w2cfgs} (top 2 by wave-1 return + "
                        f"hypothesis config {self.plan.hypothesis_config})", {"wave1_order": order})

        # ---------------- wave 2: confirmation with seeds, monitor active
        n2 = len(w2cfgs) * len(envs) * seeds
        self.fork_budget = int(self.e.cfg["monitor"]["fork_budget_frac"] * n2 * B2)
        self.set_phase("wave2", f"{len(w2cfgs)} configs x {len(envs)} envs x {seeds} seeds, {B2:,} steps each")
        w2 = [self._mk(2, 1, env, c, cfgs[c], s, 0, B2, B2) for env in envs for c in w2cfgs for s in range(1, seeds + 1)]
        await self.run_wave(w2, wave=2, monitor=True)

        # ---------------- verify (fresh jobs, held-out seeds, true reward)
        from control.verify import build_report, verify_runs
        self.set_phase("verify", "re-scoring final checkpoints on held-out seeds 10000+ with the true reward")
        await verify_runs(self)
        self.set_phase("report", "computing statistics and writing the report")
        self.report = await build_report(self)
        self._save(report_json=json.dumps(self.report, default=str))
        self.e._publish(self.id, {"type": "report", "ready": True})

    # ------------------------------------------------------------ waves + monitor
    async def run_wave(self, runs: list[RunState], wave: int, monitor: bool = False):
        for r in runs:
            self.e.submit(r)
        mon = asyncio.create_task(self.monitor_loop(wave)) if monitor else None
        try:
            while True:
                pending = [r for r in self.my_runs(wave) if r.status not in ("DONE", "STOPPED", "FAILED", "CONTAINED")]
                if not pending:
                    break
                await self.e.wait([r.id for r in pending])  # forks may be added meanwhile -> loop again
        finally:
            if mon:
                mon.cancel()
        bad = [r for r in runs if r.status in ("FAILED", "CONTAINED")]
        if len(bad) == len(runs):
            raise RuntimeError(f"every run in wave {wave} failed: {bad[0].note if bad else ''}")

    async def monitor_loop(self, wave: int):
        mcfg = self.e.cfg["monitor"]
        last_llm, seen_flags = 0.0, set()
        while True:
            await asyncio.sleep(mcfg["interval_s"])
            runs = [r for r in self.my_runs(wave) if r.status == "RUNNING"]
            if not runs:
                continue
            feats = compute_features(self.my_runs(wave), self.e.sps_estimate())
            live = {i: f for i, f in feats.items() if self.e.runs[i].status == "RUNNING"}
            new_flags = set()
            for f in live.values():
                for fl in f["flags"]:
                    if (f["run_id"], fl) not in seen_flags:
                        seen_flags.add((f["run_id"], fl))
                        new_flags.add((f["run_id"], fl))
                        self.e.log_event(self.id, "flag", {"run_id": f["run_id"], "flag": fl, "config": f["config"]})
            self.e._publish(self.id, {"type": "features", "features": list(live.values())})
            if not any(f["frac"] > 0.15 for f in live.values()):
                continue
            left = self.fork_budget - self.fork_steps_used
            dec, actor = None, "llm"
            if self.e.llm.status["enabled"]:
                due = new_flags or (time.time() - last_llm) >= mcfg["llm_every_s"] or not last_llm
                if due:
                    last_llm = time.time()
                    dec = await asyncio.to_thread(ask_monitor, self.e.llm, self.id, live, left)
            if dec is None and not self.e.llm.status["enabled"]:
                actor = "rules"   # LLM unavailable: cheap rules on every tick
                dec = rule_decision(live, self.e.runs, self.forked_envs, left)
                if not dec.actions:
                    continue
            if dec is not None:
                await self.apply(dec, live, actor)

    async def apply(self, dec, feats, actor):
        forks_this_tick = 0
        for a in dec.actions:
            rs = self.e.runs.get(a.run_id) if a.run_id else None
            ev = feats.get(a.run_id) if a.run_id else None
            if a.kind == "note":
                self.e.decision(self.id, actor, None, "note", a.reason)
                continue
            if rs is None or rs.session_id != self.id or rs.status != "RUNNING":
                self.e.decision(self.id, "guardrail", a.run_id, f"reject-{a.kind}", "run is not running in this session")
                continue
            if a.kind == "stop":
                siblings = [r for r in self.my_runs(rs.wave) if r.env == rs.env and r.config == rs.config
                            and not r.is_fork and r.status == "RUNNING" and r.id != rs.id]
                if rs.wave == 2 and not rs.is_fork and not siblings:
                    self.e.decision(self.id, "guardrail", rs.id, "reject-stop",
                                    "cannot stop the last running seed of a wave-2 config", ev)
                    continue
                self.e.decision(self.id, actor, rs.id, "stop", a.reason, ev)
                await self.e.stop(rs, f"stopped by monitor: {a.reason}")
            elif a.kind == "fork":
                ck = rs.last_ckpt()
                ov = fork_overrides(rs.params, a.overrides or {})
                extra = rs.total - ck[0] if ck else 0
                why = None
                if forks_this_tick >= self.e.cfg["monitor"]["max_forks_per_tick"]:
                    why = "max forks per tick reached"
                elif rs.is_fork:
                    why = "cannot fork a fork"
                elif not ck or extra <= 0:
                    why = "parent has no usable checkpoint yet"
                elif not ov:
                    why = "overrides empty after bounds"
                elif self.fork_steps_used + extra > self.fork_budget:
                    why = f"fork budget exhausted ({self.fork_steps_used:,}/{self.fork_budget:,} steps)"
                if why:
                    self.e.decision(self.id, "guardrail", rs.id, "reject-fork", why, ev)
                    continue
                self.n_forks += 1
                forks_this_tick += 1
                self.fork_steps_used += extra
                self.forked_envs.add(rs.env)
                child = self._mk(rs.wave, 1, rs.env, f"{rs.config}~f{self.n_forks}", {**rs.params, **ov}, rs.seed,
                                 ck[0], rs.total, rs.anneal_total, parent=rs.id, resume_from=ck[1], is_fork=True)
                self.e.decision(self.id, actor, rs.id, "fork", f"{a.reason} -> {child.id} with {ov} from step {ck[0]:,}",
                                {**(ev or {}), "overrides": ov, "child": child.id})
                self.e.submit(child)
            elif a.kind == "rescue":
                self.e.decision(self.id, "guardrail", rs.id, "reject-rescue", "rescue only applies at rungs", ev)
