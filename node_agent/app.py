"""Node agent (trusted, runs on each worker VM, OUTSIDE the sandbox). RL spec §4.3 + main plan §10.5.

Backends:
  docker : docker run --runtime=runsc --network none ... (the real sandbox, Linux worker VMs)
  local  : plain subprocess (DEV ONLY - no sandbox; used on a laptop without Docker)

Run:  NODE_BACKEND=local uvicorn node_agent.app:app --host 127.0.0.1 --port 8100
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

REPO = Path(__file__).resolve().parents[1]
BACKEND = os.environ.get("NODE_BACKEND", "docker")
RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", "/srv/runs" if BACKEND == "docker" else REPO / "runs" / "node"))
DATASETS_ROOT = Path(os.environ.get("DATASETS_ROOT", "/srv/datasets" if BACKEND == "docker" else REPO / "runs" / "datasets"))
SEALED_ROOT = Path(os.environ.get("SEALED_ROOT", "/srv/sealed" if BACKEND == "docker" else REPO / "runs" / "sealed"))
SLOTS = int(os.environ.get("NODE_SLOTS", "15"))
TOKEN = os.environ.get("NODE_TOKEN", "")
IMAGE = os.environ.get("WORKER_IMAGE", "autoresearch-worker:latest")
RUNTIME = os.environ.get("DOCKER_RUNTIME", "runsc")
STOP_GRACE_S = 15

app = FastAPI(title="autoresearch node agent")


def auth(authorization: str = Header(default="")):
    if TOKEN and authorization != f"Bearer {TOKEN}":
        raise HTTPException(401, "bad token")


class StartRun(BaseModel):
    run_id: str = Field(pattern=r"^[A-Za-z0-9_\-]{1,96}$")
    kind: str = Field(default="train", pattern="^(train|verify|profile)$")
    config: dict
    code_files: dict[str, str] = {}      # filename -> source (agent-written reward/feature code)
    parent_run_id: str | None = None     # its out/ckpt is mounted read-only at /parent
    dataset: str | None = None           # sha256; train pool mounted at /data (and /sealed for verify)
    cpus: int = 2
    mem_gb: int = 3
    max_seconds: int = 3600


class Run:
    def __init__(self, req: StartRun):
        self.req = req
        self.dir = RUNS_ROOT / req.run_id
        self.job, self.out = self.dir / "job", self.dir / "out"
        self.proc: subprocess.Popen | None = None
        self.started_at = time.time()
        self.finished_at: float | None = None
        self.exit_code: int | None = None
        self.oom_killed = False
        self.killed_by: str | None = None   # watchdog / kill / stop


RUNS: dict[str, Run] = {}
LOCK = threading.Lock()


def _cmd_args(r: Run) -> list[str]:
    args = ["python", "-m", "worker.run", "--config", "/job/config.json"]
    if r.req.kind == "verify":
        args.append("--evaluate")
    elif r.req.kind == "profile":
        args = ["python", "-m", "worker.profile", "--config", "/job/config.json"]
    return args


def _parent_dir(r: Run) -> Path | None:
    return RUNS_ROOT / r.req.parent_run_id / "out" / "ckpt" if r.req.parent_run_id else None


def _start_docker(r: Run):
    q = r.req
    cmd = ["docker", "run", "-d", "--name", f"run-{q.run_id}", "--label", f"autoresearch.run={q.run_id}",
           f"--runtime={RUNTIME}", "--network", "none",
           "--cpus", str(q.cpus), "--memory", f"{q.mem_gb}g", "--memory-swap", f"{q.mem_gb}g",
           "--pids-limit", "256", "--read-only", "--tmpfs", "/tmp:size=256m",
           "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "1000:1000",
           "-e", f"OMP_NUM_THREADS={q.cpus}", "-e", f"MKL_NUM_THREADS={q.cpus}", "-e", "HOME=/tmp",
           "-e", "MUJOCO_GL=osmesa", "-e", "PYOPENGL_PLATFORM=osmesa", "-e", "MPLCONFIGDIR=/tmp",
           "-v", f"{r.job}:/job:ro", "-v", f"{r.out}:/out:rw"]
    if (pd := _parent_dir(r)) is not None:
        cmd += ["-v", f"{pd}:/parent:ro"]
    if q.dataset:
        cmd += ["-v", f"{DATASETS_ROOT / q.dataset}:/data:ro"]
        if q.kind == "verify":   # the sealed test set is mounted ONLY into verify jobs
            cmd += ["-v", f"{SEALED_ROOT / q.dataset}:/sealed:ro"]
    cmd += [IMAGE, *_cmd_args(r)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _start_local(r: Run):
    q = r.req
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/tmp"),
           "OMP_NUM_THREADS": str(q.cpus), "MKL_NUM_THREADS": str(q.cpus),
           "MUJOCO_GL": "disable" if platform.system() == "Darwin" else "osmesa",
           "JOB_DIR": str(r.job), "OUT_DIR": str(r.out), "PYTHONUNBUFFERED": "1"}
    if (pd := _parent_dir(r)) is not None:
        env["PARENT_DIR"] = str(pd)
    if q.dataset:
        env["DATA_DIR"] = str(DATASETS_ROOT / q.dataset)
        if q.kind == "verify":
            env["SEALED_DIR"] = str(SEALED_ROOT / q.dataset)
    args = [sys.executable, *_cmd_args(r)[1:]]
    args[args.index("/job/config.json")] = str(r.job / "config.json")
    log = open(r.dir / "logs.txt", "ab")
    r.proc = subprocess.Popen(args, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True)


def _docker_state(r: Run):
    p = subprocess.run(["docker", "inspect", f"run-{r.req.run_id}", "--format", "{{json .State}}"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return
    st = json.loads(p.stdout)
    if not st.get("Running") and r.exit_code is None:
        r.exit_code = st.get("ExitCode")
        r.oom_killed = bool(st.get("OOMKilled"))
        r.finished_at = time.time()


def _refresh(r: Run):
    if r.exit_code is not None:
        return
    if BACKEND == "docker":
        _docker_state(r)
    elif r.proc is not None and r.proc.poll() is not None:
        r.exit_code = r.proc.returncode
        r.finished_at = time.time()


def _kill(r: Run, why: str):
    r.killed_by = r.killed_by or why
    if BACKEND == "docker":
        subprocess.run(["docker", "kill", f"run-{r.req.run_id}"], capture_output=True)
    elif r.proc is not None and r.proc.poll() is None:
        try:
            os.killpg(r.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _watchdog():
    while True:
        time.sleep(2)
        with LOCK:
            runs = list(RUNS.values())
        for r in runs:
            _refresh(r)
            if r.exit_code is None and time.time() - r.started_at > r.req.max_seconds:
                _kill(r, "watchdog")


threading.Thread(target=_watchdog, daemon=True).start()


def _running() -> int:
    return sum(1 for r in RUNS.values() if r.exit_code is None)


@app.get("/health", dependencies=[Depends(auth)])
def health():
    with LOCK:
        for r in RUNS.values():
            _refresh(r)
        running = _running()
    return {"backend": BACKEND, "sandboxed": BACKEND == "docker", "slots": SLOTS,
            "free_slots": max(0, SLOTS - running), "running": running}


@app.post("/runs", dependencies=[Depends(auth)])
def start(req: StartRun):
    with LOCK:
        if req.run_id in RUNS:
            raise HTTPException(409, "run exists")
        if _running() >= SLOTS:
            raise HTTPException(429, "no free slots")
        r = Run(req)
        if r.dir.exists():
            shutil.rmtree(r.dir)
        r.job.mkdir(parents=True)
        r.out.mkdir(parents=True)
        os.chmod(r.out, 0o777)  # container runs as uid 1000
        (r.job / "config.json").write_text(json.dumps(req.config))
        for name, src in req.code_files.items():
            if "/" in name or name.startswith("."):
                raise HTTPException(400, "bad filename")
            (r.job / name).write_text(src)
        RUNS[req.run_id] = r
    try:
        (_start_docker if BACKEND == "docker" else _start_local)(r)
    except subprocess.CalledProcessError as e:
        r.exit_code, r.finished_at = 125, time.time()
        (r.dir / "logs.txt").write_text(e.stderr or str(e))
    return {"ok": True}


def _get(run_id: str) -> Run:
    r = RUNS.get(run_id)
    if r is None:
        raise HTTPException(404, "unknown run")
    return r


@app.post("/runs/{run_id}/stop", dependencies=[Depends(auth)])
def stop(run_id: str):
    r = _get(run_id)
    (r.job / "STOP").write_text("stop")
    r.killed_by = r.killed_by or "stop"

    def later():
        time.sleep(STOP_GRACE_S)
        _refresh(r)
        if r.exit_code is None:
            _kill(r, "stop_timeout")
    threading.Thread(target=later, daemon=True).start()
    return {"ok": True}


@app.post("/runs/{run_id}/kill", dependencies=[Depends(auth)])
def kill(run_id: str):
    _kill(_get(run_id), "kill")
    return {"ok": True}


@app.get("/runs/{run_id}/status", dependencies=[Depends(auth)])
def status(run_id: str):
    r = _get(run_id)
    _refresh(r)
    return {"state": "running" if r.exit_code is None else "exited", "exit_code": r.exit_code,
            "oom_killed": r.oom_killed, "killed_by": r.killed_by, "started_at": r.started_at,
            "finished_at": r.finished_at}


@app.get("/runs/{run_id}/metrics", dependencies=[Depends(auth)])
def metrics(run_id: str, offset: int = 0):
    path = _get(run_id).out / "metrics.jsonl"
    if not path.exists():
        return {"lines": [], "offset": offset}
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(4 << 20)
    cut = data.rfind(b"\n") + 1          # only complete lines
    lines = [ln for ln in data[:cut].decode(errors="replace").splitlines() if ln.strip()]
    return {"lines": lines, "offset": offset + cut}


@app.get("/runs/{run_id}/files/{path:path}", dependencies=[Depends(auth)])
def files(run_id: str, path: str):
    out = _get(run_id).out.resolve()
    p = (out / path).resolve()
    if not str(p).startswith(str(out)) or not p.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(p)


@app.get("/runs/{run_id}/logs", dependencies=[Depends(auth)])
def logs(run_id: str, tail: int = 200):
    r = _get(run_id)
    if BACKEND == "docker":
        p = subprocess.run(["docker", "logs", "--tail", str(tail), f"run-{run_id}"], capture_output=True, text=True)
        text = p.stdout + p.stderr
        if not text and (r.dir / "logs.txt").exists():
            text = (r.dir / "logs.txt").read_text()
    else:
        f = r.dir / "logs.txt"
        text = f.read_text(errors="replace") if f.exists() else ""
    return {"text": "\n".join(text.splitlines()[-tail:])}


@app.put("/datasets/{sha}", dependencies=[Depends(auth)])
async def put_dataset(sha: str, request: Request, part: str = "train"):
    if not sha.isalnum() or part not in ("train", "sealed"):
        raise HTTPException(400, "bad request")
    root = DATASETS_ROOT if part == "train" else SEALED_ROOT
    d = root / sha
    d.mkdir(parents=True, exist_ok=True)
    (d / ("train.parquet" if part == "train" else "test.parquet")).write_bytes(await request.body())
    return {"ok": True}
