import glob

import pytest

from control.llm import extract_first_json_object
from control.planner import default_plan, size_budget, wave2_configs
from control.predictions import brier, score_predictions
from control.reporter import check_numbers
from control.safety import check
from control.stats import boot_ci, diff_ci, iqm
from control.understand import fallback_brief


def test_iqm_and_cis():
    assert iqm([1, 2, 3, 4, 100]) == pytest.approx(3.0)
    lo, hi = boot_ci([10, 12, 11, 13, 12])
    assert lo <= iqm([10, 12, 11, 13, 12]) <= hi
    lo, hi = diff_ci([5, 6, 7, 6, 5], [5, 6, 7, 6, 5])
    assert lo <= 0 <= hi


def test_safety_rejects_redteam_that_layer1_should_catch():
    for f in ("egress", "escape_write", "forkbomb"):
        assert check(open(f"scripts/redteam/{f}.py").read(), "shaped_reward")
    legit = ["import numpy as np\ndef shaped_reward(o, a, n, r, i):\n    return r - 0.001 * float(np.square(a).sum())\n",
             "def shaped_reward(o, a, n, r, i):\n    return r + 0.1 * float(n[0] > 1.0)\n",
             "import math\ndef shaped_reward(o, a, n, r, i):\n    return r + math.cos(float(n[1]))\n"]
    for src in legit:
        assert check(src, "shaped_reward") == []


def test_fallback_brief_and_plan():
    b = fallback_brief("Find a good PPO setup for Hopper and check whether it transfers to Walker2d. 20-minute budget.")
    assert b.data_source.ref == "Hopper-v4" and b.also_evaluate_on == ["Walker2d-v4"] and b.budget_minutes == 10   # capped: studies stay short
    plan = default_plan(b)
    assert plan.configs[0].resolved_params()["learning_rate"] == 3e-4
    assert "no_norm" in wave2_configs(default_plan(b), ["lr_high", "wide", "base", "no_norm", "lr_low", "explore"])
    assert not fallback_brief("predict churn").feasible


def test_budget_fits():
    cfg = {"seeds": 3, "verify_seconds": 15, "ladder": [[150000, 300000], [50000, 100000], [10000, 20480]]}
    b = size_budget(6, 2, 15, 1500, 15, cfg)
    assert b["est_minutes"] <= 15 and b["b1"] in (150000, 50000, 10000)


def test_prediction_scoring():
    plan = default_plan(fallback_brief("ppo hopper walker"))
    scores = {"Hopper-v4": {"base": [900, 950, 1000], "no_norm": [300, 320, 310]},
              "Walker2d-v4": {"base": [500, 520, 510], "no_norm": [100, 90, 95]}}
    res = {r["id"]: r["outcome"] for r in score_predictions(plan, scores, {"lr_high": {}}, {"Hopper-v4": "base"})}
    assert res == {"P1": "confirmed", "P2": "refuted", "P3": "confirmed", "P4": "confirmed"}
    assert brier([{"p": 0.8, "outcome": "confirmed"}, {"p": 0.6, "outcome": "refuted"}])["brier"] == pytest.approx(0.2)


def test_json_extraction_and_number_check():
    assert extract_first_json_object('<think>{"x":1}</think> ok ```json\n{"a": {"b": "}"}}\n```') == '{"a": {"b": "}"}}'
    rep = {"per_env": {"Hopper-v4": {"iqm": 445.42}}, "saved": 16.7}
    assert check_numbers("Hopper-v4 reached 445.4, saving 16.7%", rep) == []
    assert check_numbers("Hopper-v4 reached 512.3", rep) == ["512.3"]
