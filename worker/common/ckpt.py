import hashlib
from pathlib import Path


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ckpt_name(step: int, ext: str) -> str:
    return f"step_{step:09d}.{ext}"


def latest(dirpath: Path, ext: str) -> Path | None:
    files = sorted(Path(dirpath).glob(f"step_*.{ext}"))
    return files[-1] if files else None
