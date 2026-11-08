"""metrics.jsonl writer: one JSON object per line, flushed every line (main plan §10.2)."""
import json
import math
import time

from .paths import OUT


class MetricsWriter:
    def __init__(self, path=None):
        OUT.mkdir(parents=True, exist_ok=True)
        self.f = open(path or OUT / "metrics.jsonl", "a", buffering=1)
        self.t0 = time.time()

    def write(self, type_: str, **kw):
        kw = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in kw.items()}
        rec = {"type": type_, **kw}
        if "t" not in rec:
            rec["t"] = round(time.time() - self.t0, 3)
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()
