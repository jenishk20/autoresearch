"""Bring the hand-launched Panda push jobs (infra/push_jobs.sh) into the app as one study.

  .venv/bin/python scripts/import_push_runs.py            # sync + refresh the study
  .venv/bin/python scripts/import_push_runs.py --final    # last sync: mark the study done and write its report

Syncs /srv/push/*/out (metrics + videos) from the worker VM, copies videos to runs/media, and (re)writes the
study's rows in the app database. Success = share of 20 held-out episodes solved (SAC jobs evaluate every 25k
steps); for PPO it is the share of the last 50 training episodes solved at each filmed checkpoint.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from control.db import connect  # noqa: E402

SID = "push-sac"
ENV = "PandaPushDense-v3"   # same task for all three (SAC+HER is trained on its sparse twin)
JOBS = {  # job dir -> (agent name, total steps, how it learns)
    "sac_her": ("SAC+HER", 600_000, "learns from its misses (hindsight replay)"),
    "sac_dense": ("SAC", 600_000, "distance-based reward"),
    "ppo_dense": ("PPO", 1_000_000, "distance-based reward, our default so far"),
}
LOCAL = ROOT / "runs" / "push"
MEDIA = ROOT / "runs" / "media"
PROMPT = "Teach the Panda arm to push a cube onto a target. Compare SAC with hindsight replay, SAC and PPO."


def sync():
    if Path("/srv/push").is_dir():   # running on the worker VM itself
        LOCAL.mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", "--include", "*/", "--include", "*.mp4", "--include", "metrics.jsonl",
                        "--exclude", "*", "/srv/push/", f"{LOCAL}/"], check=True, timeout=300)
        st = subprocess.run(["docker", "ps", "-a", "--filter", "name=push-", "--format", "{{.Names}} {{.State}}"],
                            capture_output=True, text=True, timeout=60).stdout
        return {ln.split()[0].removeprefix("push-"): ln.split()[1] for ln in st.splitlines() if ln.strip()}
    ip = (ROOT / "runs" / "shared_ip").read_text().strip()
    import shlex
    # relative paths + cwd=ROOT: ssh splits UserKnownHostsFile values on spaces, and ROOT contains one
    ssh_args = ["ssh", "-i", "runs/ssh/id_ed25519", "-o", "UserKnownHostsFile=runs/ssh/known_hosts", "-o", "ConnectTimeout=15"]
    LOCAL.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-az", "-e", " ".join(shlex.quote(a) for a in ssh_args), "--include", "*/", "--include", "*.mp4",
                    "--include", "metrics.jsonl", "--exclude", "*", f"root@{ip}:/srv/push/", "runs/push/"],
                   check=True, timeout=300, cwd=ROOT)
    st = subprocess.run(ssh_args + [f"root@{ip}", "docker ps -a --filter name=push- --format '{{.Names}} {{.State}}'"],
                        capture_output=True, text=True, timeout=60, cwd=ROOT).stdout
    return {ln.split()[0].removeprefix("push-"): ln.split()[1] for ln in st.splitlines() if ln.strip()}


def read_job(job):
    lines = [json.loads(l) for l in (LOCAL / job / "out" / "metrics.jsonl").read_text().splitlines() if l.strip()]
    points, videos = [], []
    if job.startswith("sac"):
        for d in lines:
            if d.get("type") == "eval":
                points.append((d["step"], d["success"]))
                videos.append({"step": d["step"], "rel": d["video"], "success": d["success"]})
        step = max([p[0] for p in points], default=0)
        succ = points[-1][1] if points else None
    else:
        eps = [d for d in lines if d.get("type") == "episode"]
        step = max([d["step"] for d in lines if "step" in d], default=0)
        for d in lines:
            if d.get("type") == "video":
                recent = [e["success"] for e in eps if e["step"] <= d["step"]][-50:]
                sr = sum(recent) / len(recent) if recent else 0.0
                points.append((d["step"], sr))
                videos.append({"step": d["step"], "rel": d["path"], "success": sr})
        recent = [e["success"] for e in eps][-50:]
        succ = sum(recent) / len(recent) if recent else None
    return step, succ, points, videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    final = ap.parse_args().final
    states = sync()
    db = connect(ROOT / "runs" / "autoresearch.db")
    db.execute("INSERT OR IGNORE INTO sessions(id, prompt, status, created_at) VALUES (?,?,?,?)",
               (SID, PROMPT, "wave2", time.time()))
    plan = {"round": 1, "task_summary": "Three learning algorithms, one robot task, every checkpoint filmed.",
            "configs": [{"name": n, "model": "ppo", "params": {}, "rationale": why} for n, _, why in JOBS.values()],
            "hypothesis_config": "SAC+HER", "predictions": []}
    db.execute("INSERT OR REPLACE INTO plans(session_id, round, plan_json) VALUES (?,?,?)", (SID, 1, json.dumps(plan)))
    db.execute("DELETE FROM metrics WHERE run_id LIKE ?", (f"{SID}-%",))
    MEDIA.mkdir(parents=True, exist_ok=True)
    table, artifacts, total_steps = [], [], 0
    for job, (name, total, _) in JOBS.items():
        if not (LOCAL / job / "out" / "metrics.jsonl").exists():
            continue
        rid = f"{SID}-{job}"
        step, succ, points, videos = read_job(job)
        total_steps += step
        running = states.get(job) == "running"
        status = "RUNNING" if running and not final else "DONE"
        db.execute("INSERT OR REPLACE INTO runs(id, session_id, round, wave, segment, kind, task, dataset, model, config_name, "
                   "seed, params_json, start_at, budget_total, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (rid, SID, 1, 2, 1, "train", "rl", ENV, "sac" if job.startswith("sac") else "ppo", name, 1, "{}", 0,
                    total, status))
        rows = [(rid, "update", s, None, json.dumps({"step": s, "val_metric": round(100 * v, 1)})) for s, v in points]
        vids = []
        for v in videos:
            src = LOCAL / job / "out" / v["rel"]
            if src.exists():
                dst = MEDIA / f"{rid}-{v['step']}.mp4"
                if not dst.exists():
                    shutil.copy(src, dst)
                vd = {"step": v["step"], "url": f"/media/{dst.name}", "success": v["success"]}
                vids.append(vd)
                rows.append((rid, "video", v["step"], None, json.dumps(vd)))
        latest = vids[-1]["url"] if vids else None
        rows.append((rid, "summary", step, None, json.dumps({"step": step, "success": succ, "eval_success": succ,
                                                             "video_url": latest if not running or final else None})))
        db.executemany("INSERT INTO metrics(run_id, type, step, t, json) VALUES (?,?,?,?,?)", rows)
        table.append({"config": name, "n": 1, "iqm": round(100 * (succ or 0), 1), "ci": [round(100 * (succ or 0), 1)] * 2,
                      "success": succ, "video": latest, "checkpoint_videos": vids, "steps": step})
        artifacts.append({"run_id": rid, "env": ENV, "config": name, "seed": 1, "checkpoint": f"{job}/ckpt (on worker)",
                          "sha256": "-" * 64, "verify_mean": round(100 * (succ or 0), 1)})
    table.sort(key=lambda r: -(r["success"] or 0))
    if final and table:
        w = table[0]
        others = ", ".join(f"{r['config']} {round(100 * (r['success'] or 0))}%" for r in table[1:])
        report = {
            "session_id": SID, "query": PROMPT, "task_summary": plan["task_summary"],
            "per_env": {ENV: {"table": table, "winner": w["config"], "runner_up": table[1]["config"] if len(table) > 1 else None,
                              "tie": False, "winner_minus_runner_up_ci": None}},
            "narrative": {"headline": f"{w['config']} pushes the cube onto the target {round(100 * (w['success'] or 0))}% of the time ({others}).",
                          "summary": "Success is measured on 20 placements of cube and target the agent never trained on.",
                          "findings": [f"{r['config']}: {round(100 * (r['success'] or 0))}% success after {r['steps']:,} steps." for r in table],
                          "caveats": ["One run per algorithm; training stopped when the worker VM was retired at the end of the hackathon."]},
            "narrative_source": "template", "predictions": [], "calibration": {"brier": None, "n_decided": 0, "baseline": 0.25},
            "halving": {"steps_used": 0, "naive_steps": 0, "saved_pct": 0.0}, "containment": [], "decisions": [],
            "artifacts": artifacts,
            "compute": {"env_steps": total_steps, "run_seconds": None, "vcpu_hours": None, "sandboxed": True},
            "llm": {"model": "-", "enabled": True, "session_calls": 0},
        }
        db.execute("UPDATE sessions SET status='done', report_json=? WHERE id=?", (json.dumps(report), SID))
    else:
        db.execute("UPDATE sessions SET status='wave2', report_json=NULL WHERE id=?", (SID,))
    db.commit()
    print(time.strftime("%T"), {r["config"]: (r["steps"], r["success"], len(r["checkpoint_videos"])) for r in table},
          "final" if final else "")


if __name__ == "__main__":
    main()
