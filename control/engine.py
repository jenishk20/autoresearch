"""Run manager (scheduler, RL spec §4.4): queue -> node slots -> metrics poller -> exit classification.
Also the single place that writes events/decisions and publishes them to the browser."""
from __future__ import annotations

import asyncio
import json
import threading
import time

from control import db as dbmod
from control.bus import bus
from control.features import RunState
from control.llm import LLM
from control.models import RunConfig, TERMINAL_STATUSES
from control.nodes import make_nodes

# Log lines that mean "a sandbox limit stopped this" rather than "the code crashed".
CONTAINMENT_SIGNATURES = [
    ("Network is unreachable", "network blocked (--network none)"),
    ("Temporary failure in name resolution", "DNS blocked (--network none)"),
    ("Name or service not known", "DNS blocked (--network none)"),
    ("Read-only file system", "write blocked (read-only filesystem)"),
    ("Permission denied", "write blocked (permission denied)"),
    ("Resource temporarily unavailable", "process limit hit (--pids-limit)"),
    ("BlockingIOError", "process limit hit (--pids-limit)"),
    ("Cannot allocate memory", "memory limit hit"),
    ("MemoryError", "memory limit hit"),
]
TERMINAL = {s.value for s in TERMINAL_STATUSES}


class Engine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.db = dbmod.connect(cfg["db_path"])
        self.db_lock = threading.Lock()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.worker_mode = cfg.get("worker", {}).get("mode", "local")
        self.nodes = make_nodes(cfg) if self.worker_mode == "local" else []
        self.llm = LLM(cfg["llm"], self.log_event)
        self.runs: dict[str, RunState] = {}
        self.queue: list[str] = []
        self.done_events: dict[str, asyncio.Event] = {}
        self.on_node: dict[str, int] = {n.name: 0 for n in self.nodes}
        self.observed_sps: list[float] = []
        self._tasks = []

    # ------------------------------------------------------------ lifecycle
    async def start(self):
        self.loop = asyncio.get_running_loop()
        with self.db_lock:
            self.db.execute("UPDATE sessions SET status='interrupted' WHERE status NOT IN ('done','failed','unsupported','interrupted')")
            self.db.commit()
        for n in self.nodes:
            await n.health()
        await asyncio.to_thread(self.llm.probe)
        from control.provision import Provisioner
        self.provisioner = Provisioner(self)
        await self.provisioner.start()
        self._tasks = [asyncio.create_task(self._dispatch_loop()), asyncio.create_task(self._poll_loop()),
                       asyncio.create_task(self._health_loop())]

    def add_node(self, node):
        self.nodes.append(node)
        self.on_node[node.name] = 0

    def remove_node(self, node):
        self.nodes = [n for n in self.nodes if n is not node]

    @property
    def total_slots(self) -> int:
        if self.worker_mode == "vultr":
            return int(self.cfg["worker"].get("slots", 3))
        return sum(n.slots for n in self.nodes if n.healthy) or 1

    @property
    def sandboxed(self) -> bool:
        if self.worker_mode == "vultr":
            return True   # every job runs on a per-session gVisor VM
        return bool(self.nodes) and all(n.sandboxed for n in self.nodes if n.healthy)

    def sps_estimate(self) -> float:
        if len(self.observed_sps) >= 5:
            xs = sorted(self.observed_sps[-50:])
            return xs[len(xs) // 2]
        if self.worker_mode == "vultr":
            return float(self.cfg["worker"].get("sps_estimate", 950))
        return float(self.cfg["rl"]["sps_estimate"])

    # ------------------------------------------------------------ events / decisions (thread-safe)
    def _publish(self, session_id, ev):
        if self.loop is None or session_id is None:
            return
        if threading.current_thread() is threading.main_thread():
            bus.publish(session_id, ev)
        else:
            self.loop.call_soon_threadsafe(bus.publish, session_id, ev)

    def log_event(self, session_id, kind: str, payload: dict):
        ts = time.time()
        with self.db_lock:
            self.db.execute("INSERT INTO events(session_id, ts, kind, payload_json) VALUES (?,?,?,?)",
                            (session_id, ts, kind, json.dumps(payload, default=str)))
            self.db.commit()
        self._publish(session_id, {"type": "event", "kind": kind, "ts": ts, "payload": payload})

    def decision(self, session_id, actor: str, run_id, action: str, reason: str, evidence=None):
        ts = time.time()
        with self.db_lock:
            self.db.execute("INSERT INTO decisions(session_id, ts, actor, run_id, action, reason, evidence_json) "
                            "VALUES (?,?,?,?,?,?,?)", (session_id, ts, actor, run_id, action, reason,
                                                        json.dumps(evidence, default=str)))
            self.db.commit()
        self._publish(session_id, {"type": "decision", "ts": ts, "actor": actor, "run_id": run_id,
                                   "action": action, "reason": reason, "evidence": evidence})

    def run_view(self, rs: RunState) -> dict:
        return {"id": rs.id, "round": rs.round, "wave": rs.wave, "segment": rs.segment, "env": rs.env,
                "config": rs.config, "seed": rs.seed, "status": rs.status, "step": rs.last_step,
                "start_at": rs.start_at, "total": rs.total, "kind": rs.kind, "parent": rs.parent,
                "is_fork": rs.is_fork, "ret_recent": None if rs.ret_recent() is None else round(rs.ret_recent(), 1),
                "flags": sorted(rs.flags), "note": rs.note, "node": rs.node,
                "eval_mean": None if not rs.eval else round(rs.eval["mean_return"], 1)}

    def _save_run(self, rs: RunState):
        with self.db_lock:
            self.db.execute(
                "INSERT OR REPLACE INTO runs(id, session_id, round, wave, segment, kind, task, dataset, model, "
                "config_name, seed, params_json, parent_run_id, start_at, budget_total, node, status, exit_code, "
                "oom_killed, started_at, ended_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rs.id, rs.session_id, rs.round, rs.wave, rs.segment, rs.kind, "rl", rs.env, "ppo", rs.config,
                 rs.seed, json.dumps(rs.params), rs.parent, rs.start_at, rs.total, rs.node, rs.status,
                 rs.exit_code, 0, rs.started_at, rs.ended_at))
            self.db.commit()
        self._publish(rs.session_id, {"type": "run", "run": self.run_view(rs)})

    def set_status(self, rs: RunState, status: str, note: str | None = None):
        rs.status = status
        if note:
            rs.note = note
        if status in TERMINAL:
            rs.ended_at = time.time()
        self._save_run(rs)

    # ------------------------------------------------------------ submit / wait
    def submit(self, rs: RunState):
        self.runs[rs.id] = rs
        self.done_events[rs.id] = asyncio.Event()
        self.queue.append(rs.id)
        self._save_run(rs)

    async def wait(self, ids):
        await asyncio.gather(*(self.done_events[i].wait() for i in ids))

    def config_for(self, rs: RunState) -> dict:
        cfg = {"run_id": rs.id, "task": "rl", "dataset": rs.env, "model": "ppo", "params": rs.params,
               "metric": "return", "seed": rs.seed,
               "budget": {"unit": "env_steps", "total": rs.total, "start_at": rs.start_at,
                          "anneal_total": rs.anneal_total},
               "resume_from": rs.resume_from, "ckpt_every": max(2048, (rs.total - rs.start_at) // 4)}
        if rs.code_files:
            cfg["feature_fn"] = next(iter(rs.code_files))
        RunConfig.model_validate(cfg)  # the contract, checked before anything leaves the control plane
        return cfg

    # ------------------------------------------------------------ loops
    async def _health_loop(self):
        while True:
            await asyncio.sleep(10)
            for n in self.nodes:
                await n.health()

    async def _dispatch_loop(self):
        while True:
            await asyncio.sleep(0.5)
            for node in self.nodes:
                if not node.healthy:
                    continue
                while self.queue and self.on_node[node.name] < node.slots:
                    rs = self.runs[self.queue.pop(0)]
                    if rs.status != "PENDING":
                        continue
                    rs.node = node.name
                    self.set_status(rs, "STARTING")
                    payload = {"run_id": rs.id, "kind": "verify" if rs.kind == "verify" else "train",
                               "config": self.config_for(rs), "code_files": rs.code_files,
                               "parent_run_id": rs.parent if rs.resume_from else None,
                               "cpus": self.cfg["run"]["cpus"], "mem_gb": self.cfg["run"]["mem_gb"],
                               "max_seconds": self._max_seconds(rs)}
                    try:
                        await node.start(payload)
                        self.on_node[node.name] += 1
                        rs.started_at = time.time()
                        self.set_status(rs, "RUNNING")
                    except Exception as e:
                        self._finish(rs, "FAILED", f"could not start: {e}")

    def _max_seconds(self, rs: RunState) -> int:
        if rs.kind == "verify":
            return 300
        expected = (rs.total - rs.start_at) / max(200.0, self.sps_estimate() / 2)
        return int(max(120, 1.5 * expected + 60))

    def _finish(self, rs: RunState, status: str, note: str | None = None):
        if rs.status in ("STARTING", "RUNNING") and rs.node:
            self.on_node[rs.node] = max(0, self.on_node[rs.node] - 1)
        self.set_status(rs, status, note)
        if status == "CONTAINED":
            self.log_event(rs.session_id, "containment", {"run_id": rs.id, "reason": note, "config": rs.config})
        self.done_events[rs.id].set()

    def node_of(self, rs: RunState):
        return next(n for n in self.nodes if n.name == rs.node)

    def node_or_none(self, rs: RunState):
        return next((n for n in self.nodes if n.name == rs.node), None)

    async def _poll_loop(self):
        while True:
            await asyncio.sleep(2)
            running = [r for r in self.runs.values() if r.status == "RUNNING"]
            for rs in running:
                if self.node_or_none(rs) is None:   # its VM is gone
                    self._finish(rs, "FAILED", "worker VM went away")
                    continue
                try:
                    await self._poll_one(rs)
                except Exception as e:  # a flaky node must not kill the poller
                    rs.note = f"poll error: {e}"

    async def _ingest(self, rs: RunState):
        node = self.node_of(rs)
        m = await node.metrics(rs.id, rs.offset)
        rs.offset = m["offset"]
        rows, points = [], []
        for ln in m["lines"]:
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            t = d.get("type")
            if t == "update":
                rs.updates.append(d)
                rs.last_step = d["step"]
                if d.get("extra", {}).get("sps"):
                    self.observed_sps.append(float(d["extra"]["sps"]))
                if d.get("val_metric") is not None:
                    points.append([d["step"], d["val_metric"]])
            elif t == "episode":
                rs.add_episode(d["step"], d["ret_true"], d["len"])
            elif t == "checkpoint":
                rs.ckpts.append((d["step"], d["path"], d["sha256"]))
                with self.db_lock:
                    self.db.execute("INSERT INTO artifacts(run_id, kind, path, sha256) VALUES (?,?,?,?)",
                                    (rs.id, "checkpoint", d["path"], d["sha256"]))
            elif t == "done":
                rs.done_reason = d.get("reason")
            rows.append((rs.id, t, d.get("step"), d.get("t"), ln))
        if rows:
            with self.db_lock:
                self.db.executemany("INSERT INTO metrics(run_id, type, step, t, json) VALUES (?,?,?,?,?)", rows)
                self.db.commit()
        if points:
            self._publish(rs.session_id, {"type": "points", "run_id": rs.id, "points": points,
                                          "run": self.run_view(rs)})

    async def _poll_one(self, rs: RunState):
        node = self.node_of(rs)
        await self._ingest(rs)
        st = await node.status(rs.id)
        if st["state"] != "exited":
            return
        await self._ingest(rs)  # tail after exit
        rs.exit_code = st["exit_code"]
        logs = await node.logs(rs.id, 80)
        if rs.kind == "verify":
            raw = await node.file(rs.id, "eval.json")
            if rs.exit_code == 0 and raw:
                rs.eval = json.loads(raw)
                self._finish(rs, "DONE")
            else:
                self._finish(rs, "FAILED", _last_line(logs))
            return
        sig = next((why for pat, why in CONTAINMENT_SIGNATURES if pat in logs), None)
        killed_by = st.get("killed_by")
        if st.get("oom_killed") or (rs.exit_code == 137 and killed_by is None):
            self._finish(rs, "CONTAINED", "killed at the memory cap (OOM)")
        elif killed_by == "watchdog":
            self._finish(rs, "CONTAINED", "killed by the wall-clock watchdog")
        elif rs.done_reason == "stop_flag" or killed_by in ("stop", "kill", "stop_timeout"):
            self._finish(rs, "STOPPED", rs.note)
        elif rs.exit_code == 0 and rs.done_reason == "budget":
            self._finish(rs, "DONE")
        elif sig and rs.code_files:
            self._finish(rs, "CONTAINED", f"{sig}: {_match_line(logs, sig)}")
        elif rs.done_reason == "nan":
            self._finish(rs, "FAILED", "NaN loss")
        else:
            self._finish(rs, "FAILED", _last_line(logs))

    # ------------------------------------------------------------ control actions
    async def stop(self, rs: RunState, why: str):
        rs.note = why
        await self.node_of(rs).stop(rs.id)


def _last_line(text: str) -> str:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1][:300] if lines else "no output"


def _match_line(text: str, why: str) -> str:
    for pat, w in CONTAINMENT_SIGNATURES:
        if w == why:
            for ln in text.splitlines():
                if pat in ln:
                    return ln.strip()[:300]
    return ""
