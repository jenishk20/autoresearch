"""SQLite schema (plan §5.5). WAL mode so the poller and the web app can share the file."""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY, prompt TEXT, preset TEXT, status TEXT,
  plan_json TEXT, report_json TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, session_id TEXT, wave INTEGER, segment INTEGER, env_id TEXT,
  config_name TEXT, seed INTEGER, params_json TEXT, parent_run_id TEXT,
  start_step INTEGER, total_steps INTEGER, node TEXT, status TEXT,
  exit_code INTEGER, oom_killed INTEGER, started_at REAL, ended_at REAL
);
CREATE TABLE IF NOT EXISTS metrics (
  run_id TEXT, type TEXT, step INTEGER, t REAL, json TEXT
);
CREATE INDEX IF NOT EXISTS metrics_run_step ON metrics(run_id, step);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, ts REAL, actor TEXT,
  run_id TEXT, action TEXT, reason TEXT, evidence_json TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
  id TEXT, session_id TEXT, json TEXT, outcome TEXT, evidence_json TEXT,
  PRIMARY KEY (session_id, id)
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, ts REAL, kind TEXT, payload_json TEXT
);
CREATE TABLE IF NOT EXISTS artifacts (
  run_id TEXT, kind TEXT, path TEXT, sha256 TEXT
);
"""


def connect(path: str | Path = "rlagent.db") -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn
