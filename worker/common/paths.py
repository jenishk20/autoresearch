"""Sandbox paths. Inside the container these are fixed mounts; in local dev the node agent
points them at per-run directories through env vars."""
import os
from pathlib import Path

JOB = Path(os.environ.get("JOB_DIR", "/job"))        # read-only: config.json, STOP, reward.py/features.py
OUT = Path(os.environ.get("OUT_DIR", "/out"))        # read-write: metrics.jsonl, ckpt/, eval.json, video.mp4
PARENT = Path(os.environ.get("PARENT_DIR", "/parent"))  # read-only: parent run's ckpt/ (resume / fork / verify)
DATA = Path(os.environ.get("DATA_DIR", "/data"))      # read-only: train pool (tabular)
SEALED = Path(os.environ.get("SEALED_DIR", "/sealed"))  # read-only: sealed test set (verify jobs only)


def stop_requested() -> bool:
    return (JOB / "STOP").exists()
