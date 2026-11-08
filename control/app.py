"""Control plane web app: FastAPI + SSE. Run: uvicorn control.app:app --port 8000"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from control.bus import bus
from control.engine import Engine
from control.features import RunState
from control.session import ENV_SHORT, SessionRunner
from control.settings import ROOT, settings

HERE = Path(__file__).parent
engine: Engine | None = None
SESSIONS: dict[str, SessionRunner] = {}
REDTEAM = ["egress", "membomb", "escape_write", "forkbomb", "gaming"]


@asynccontextmanager
async def lifespan(_app):
    global engine
    engine = Engine(settings())
    await engine.start()
    yield


app = FastAPI(title="Autoresearch", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=HERE / "templates")


def status() -> dict:
    nodes = [{"name": n.name, "healthy": n.healthy, "backend": n.backend, "sandboxed": n.sandboxed,
              "slots": n.slots, "running": engine.on_node.get(n.name, 0)} for n in engine.nodes]
    w = engine.cfg.get("worker", {})
    prov = getattr(engine, "provisioner", None)
    return {"llm": engine.llm.status, "nodes": nodes, "sandboxed": engine.sandboxed, "worker_mode": engine.worker_mode,
            "worker": {"plan": w.get("plan"), "region": w.get("region"), "price_per_hour": w.get("price_per_hour"),
                       **(prov.info() if prov else {})},
            "sps_estimate": round(engine.sps_estimate()), "slots": engine.total_slots}


@app.get("/api/status")
def api_status():
    return status()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    with engine.db_lock:
        rows = [dict(r) for r in engine.db.execute(
            "SELECT id, prompt, status, created_at, report_json IS NOT NULL AS has_report FROM sessions "
            "ORDER BY created_at DESC LIMIT 30")]
    for r in rows:
        if r["id"] in SESSIONS:
            r["status"] = SESSIONS[r["id"]].status
            r["phase"] = SESSIONS[r["id"]].phase
        r["created"] = time.strftime("%H:%M:%S", time.localtime(r["created_at"]))
    return templates.TemplateResponse(request, "index.html", {"sessions": rows, "status": status()})


@app.post("/sessions")
async def create_session(query: str = Form(...)):
    q = query.strip()
    if not q:
        raise HTTPException(400, "empty query")
    if any(s.status == "running" for s in SESSIONS.values()):
        raise HTTPException(429, "a session is already running; wait for it to finish")
    s = SessionRunner(engine, q[:2000]).start()
    SESSIONS[s.id] = s
    return RedirectResponse(f"/sessions/{s.id}", status_code=303)


@app.get("/sessions/{sid}", response_class=HTMLResponse)
def session_page(request: Request, sid: str):
    with engine.db_lock:
        row = engine.db.execute("SELECT id, prompt FROM sessions WHERE id=?", (sid,)).fetchone()
    if not row:
        raise HTTPException(404)
    return templates.TemplateResponse(request, "session.html", {"sid": sid, "prompt": row["prompt"], "status": status()})


def _json(s):
    return json.loads(s) if s else None


@app.get("/api/sessions/{sid}/state")
def state(sid: str):
    with engine.db_lock:
        row = engine.db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        if not row:
            raise HTTPException(404)
        decisions = [dict(r) for r in engine.db.execute(
            "SELECT ts, actor, run_id, action, reason FROM decisions WHERE session_id=? ORDER BY ts", (sid,))]
        events = [{"ts": r["ts"], "kind": r["kind"], "payload": json.loads(r["payload_json"])} for r in engine.db.execute(
            "SELECT ts, kind, payload_json FROM events WHERE session_id=? AND kind IN "
            "('phase','flag','containment','error','llm_call','redteam','worker') ORDER BY ts DESC LIMIT 400", (sid,))][::-1]
        plan_row = engine.db.execute("SELECT plan_json FROM plans WHERE session_id=? AND round=1", (sid,)).fetchone()
    runner = SESSIONS.get(sid)
    live = [r for r in engine.runs.values() if r.session_id == sid]
    if live:
        runs = [engine.run_view(r) for r in live]
        series = {r.id: [[u["step"], u["val_metric"]] for u in r.updates if u.get("val_metric") is not None] for r in live}
    else:  # finished session from an earlier server process: rebuild from the DB
        with engine.db_lock:
            runs = [{"id": r["id"], "wave": r["wave"], "segment": r["segment"], "env": r["dataset"],
                     "config": r["config_name"], "seed": r["seed"], "status": r["status"], "kind": r["kind"],
                     "step": r["budget_total"], "total": r["budget_total"], "start_at": r["start_at"],
                     "parent": r["parent_run_id"], "is_fork": "~" in (r["config_name"] or ""), "flags": [],
                     "ret_recent": None, "eval_mean": None, "note": None}
                    for r in engine.db.execute("SELECT * FROM runs WHERE session_id=?", (sid,))]
            series = {}
            for r in engine.db.execute("SELECT m.run_id, m.json FROM metrics m JOIN runs r ON r.id=m.run_id "
                                       "WHERE r.session_id=? AND m.type='update' ORDER BY m.step", (sid,)):
                d = json.loads(r["json"])
                if d.get("val_metric") is not None:
                    series.setdefault(r["run_id"], []).append([d["step"], d["val_metric"]])
    report = _json(row["report_json"])
    return {
        "id": sid, "prompt": row["prompt"],
        "status": runner.status if runner else row["status"], "phase": runner.phase if runner else row["status"],
        "created_at": row["created_at"], "brief": _json(row["brief_json"]),
        "plan": runner.plan.model_dump() if runner and runner.plan else _json(plan_row["plan_json"]) if plan_row else None,
        "budget": runner.budget if runner else (report or {}).get("budget"),
        "runs": runs, "series": series, "decisions": decisions, "events": events,
        "predictions": (report or {}).get("predictions"), "report_ready": report is not None,
        "system": status(), "redteam_presets": REDTEAM,
    }


@app.get("/sessions/{sid}/stream")
async def stream(request: Request, sid: str):
    q = bus.subscribe(sid)

    async def gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    yield {"data": json.dumps(ev, default=str)}
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "{}"}
        finally:
            bus.unsubscribe(sid, q)
    return EventSourceResponse(gen())


@app.post("/api/sessions/{sid}/redteam/{preset}")
async def redteam(sid: str, preset: str):
    if preset not in REDTEAM:
        raise HTTPException(404, "unknown preset")
    s = SESSIONS.get(sid)
    if not s or not s.brief or s.brief.problem_type != "rl":
        raise HTTPException(409, "red-team needs a live RL session")
    if not engine.sandboxed:
        raise HTTPException(409, "refusing to run red-team code: this node is NOT sandboxed (local dev backend). "
                                 "Run it against a gVisor worker.")
    src = (ROOT / "scripts" / "redteam" / f"{preset}.py").read_text()
    env = s.brief.data_source.ref
    rid = f"{sid}-rt-{ENV_SHORT[env]}-{preset}-{int(time.time()) % 100000}"
    rs = RunState(id=rid, session_id=sid, round=1, wave=0, segment=1, env=env, config=f"redteam:{preset}", seed=1,
                  params={"hidden": "64x64"}, start_at=0, total=20480, anneal_total=20480, kind="train",
                  code_files={"reward.py": src})
    engine.log_event(sid, "redteam", {"preset": preset, "run_id": rid,
                                      "note": "safety.py (layer 1) deliberately skipped so the sandbox must hold alone"})
    engine.submit(rs)
    return {"ok": True, "run_id": rid}


@app.post("/api/sessions/{sid}/cancel")
async def cancel(sid: str):
    s = SESSIONS.get(sid)
    if not s or not s.task:
        raise HTTPException(404)
    for r in engine.runs.values():
        if r.session_id == sid and r.status == "RUNNING" and engine.node_or_none(r):
            await engine.node_of(r).kill(r.id)
        elif r.session_id == sid and r.status == "PENDING":
            engine._finish(r, "STOPPED", "session cancelled")
    s.task.cancel()
    return {"ok": True}


@app.get("/sessions/{sid}/report", response_class=HTMLResponse)
def report_page(request: Request, sid: str):
    with engine.db_lock:
        row = engine.db.execute("SELECT report_json FROM sessions WHERE id=?", (sid,)).fetchone()
    if not row or not row["report_json"]:
        raise HTTPException(404, "report not ready")
    return templates.TemplateResponse(request, "report.html", {"r": json.loads(row["report_json"]), "sid": sid,
                                                               "status": status()})


@app.get("/api/sessions/{sid}/report.json")
def report_json(sid: str):
    with engine.db_lock:
        row = engine.db.execute("SELECT report_json FROM sessions WHERE id=?", (sid,)).fetchone()
    if not row or not row["report_json"]:
        raise HTTPException(404)
    return JSONResponse(json.loads(row["report_json"]))
