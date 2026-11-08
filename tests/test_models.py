import json

import pytest
from pydantic import ValidationError

from control.db import connect
from control.models import (MonitorDecision, Plan, PPOParams, RunConfig, UpdateLine,
                            bound_fork_overrides)

CONFIG_EXAMPLE = """{"run_id": "s12-w1-hc-c3-s1-seg1", "algo": "ppo", "env_id": "HalfCheetah-v4", "seed": 1,
 "total_steps": 150000, "start_step": 0, "anneal_total_steps": 300000,
 "params": {"hidden": [64, 64], "learning_rate": 3e-4, "ent_coef": 0.0, "norm_obs": true,
            "clip_coef": 0.2, "num_envs": 4, "num_steps": 512, "update_epochs": 10,
            "num_minibatches": 32, "gamma": 0.99, "gae_lambda": 0.95, "max_grad_norm": 0.5},
 "resume_from": null, "reward_fn": null, "ckpt_every": 100000}"""


def test_config_example_parses():
    cfg = RunConfig.model_validate_json(CONFIG_EXAMPLE)
    assert cfg.params.num_envs * cfg.params.num_steps == 2048


def test_worker_clamps_hostile_params():
    p = PPOParams(hidden=[4096, 4096], learning_rate=1.0, ent_coef=-5, clip_coef=9,
                  num_envs=999, update_epochs=1000).clamped()
    assert p.hidden == [64, 64] and p.learning_rate == 3e-3 and p.ent_coef == 0.0
    assert p.clip_coef == 0.3 and p.num_envs == 4 and p.update_epochs == 10


def test_metrics_line():
    UpdateLine.model_validate_json('{"type":"update","step":204800,"t":131.2,"sps":1561,"lr":2.1e-4,'
                                   '"pg_loss":-0.01,"v_loss":0.42,"entropy":4.1,"approx_kl":0.012,'
                                   '"clipfrac":0.11,"ev":0.83}')


def _plan(**over):
    cfg = lambda n, **k: {"name": n, "hidden": "64x64", "learning_rate": 0.0003, "ent_coef": 0.0,
                          "norm_obs": True, "rationale": "x", **k}
    pred = lambda i: {"id": f"P{i}", "statement": "s", "kind": "a_beats_b", "env": "Walker2d-v4",
                      "a": "c1", "b": "c2", "min_ratio": 1.3, "at_step": 500000, "p": 0.85}
    d = {"task_summary": "t", "envs": ["Walker2d-v4"], "configs": [cfg("c1"), cfg("c2", norm_obs=False)],
         "hypothesis_config": "c1", "predictions": [pred(i) for i in range(3)], **over}
    return json.dumps(d)


def test_plan_ok_and_rejects_out_of_space_lr():
    Plan.model_validate_json(_plan())
    bad = json.loads(_plan()); bad["configs"][0]["learning_rate"] = 0.05
    with pytest.raises(ValidationError):
        Plan.model_validate(bad)


def test_plan_rejects_overconfident_prediction():
    bad = json.loads(_plan()); bad["predictions"][0]["p"] = 0.99
    with pytest.raises(ValidationError):
        Plan.model_validate(bad)


def test_fork_bounds():
    parent = PPOParams(learning_rate=3e-4)
    out = bound_fork_overrides(parent, {"learning_rate": 1e-6, "ent_coef": 0.5, "hidden": 512})
    assert out == {"learning_rate": 1.5e-4, "ent_coef": 0.02}


def test_monitor_decision_max_actions():
    acts = [{"kind": "note", "reason": "r"}] * 7
    with pytest.raises(ValidationError):
        MonitorDecision.model_validate({"actions": acts, "summary": "s"})


def test_db_schema(tmp_path):
    conn = connect(tmp_path / "t.db")
    names = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    assert {"sessions", "runs", "metrics", "decisions", "predictions", "events", "artifacts"} <= names
