from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@lru_cache
def settings() -> dict:
    s = yaml.safe_load((ROOT / "control" / "config.yaml").read_text())
    if os.environ.get("NODE_URL"):
        s["nodes"][0]["url"] = os.environ["NODE_URL"]
    if os.environ.get("WORKER_MODE"):
        s.setdefault("worker", {})["mode"] = os.environ["WORKER_MODE"]
    s["db_path"] = os.environ.get("DB_PATH", str(ROOT / "runs" / "autoresearch.db"))
    return s
