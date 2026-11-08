"""Worker entry point: python -m worker.run --config /job/config.json [--evaluate]

Runs INSIDE the sandbox. Dispatches on config.task. Exit codes: 0 ok, 2 bad config, 3 nan, 1 crash."""
from __future__ import annotations

import argparse
import json
import sys
import traceback

from worker.common.paths import OUT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--evaluate", action="store_true", help="verifier job")
    args = ap.parse_args()

    from control.models import RunConfig
    try:
        cfg = RunConfig.model_validate_json(open(args.config).read())
    except Exception as e:
        print(f"invalid config: {e}", file=sys.stderr)
        return 2

    OUT.mkdir(parents=True, exist_ok=True)
    try:
        if cfg.task == "rl":
            from worker.adapters import rl_ppo as adapter
        else:
            from worker.adapters import tabular as adapter
        if args.evaluate:
            result = adapter.evaluate(cfg)
            (OUT / "eval.json").write_text(json.dumps(result))
            return 0
        return adapter.train(cfg)
    except Exception:
        traceback.print_exc()
        try:
            from worker.common.jsonl import MetricsWriter
            MetricsWriter().write("done", step=0, reason="error")
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
