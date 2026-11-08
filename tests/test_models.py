import json

import pytest
from pydantic import ValidationError

from control.db import connect
from control.models import (Costs, MonitorDecision, Plan, Reflection, ResearchBrief, RunConfig,
                            UpdateLine, bound_fork_overrides, improvement_ratio, worker_clamp)

CHURN_CONFIG = """{"run_id": "s3-r1-w1-hgb2-s1-seg1", "task": "tabular", "dataset": "telco-3f9c", "model": "hgb",
 "params": {"learning_rate": 0.1, "max_leaf_nodes": 31, "l2": 0.0, "calibration": "none"},
 "preprocessing": {"drop": ["customerID", "contract_end_reason"]},
 "target": "Churn", "metric": "savings", "costs": {"offer_cost": 50, "churn_loss": 600, "save_rate": 0.3},
 "seed": 1, "budget": {"unit": "rounds", "total": 100, "start_at": 0, "anneal_total": 200, "chunk": 25},
 "resume_from": null, "feature_fn": null, "ckpt_every": 1}"""

PPO_CONFIG = {"run_id": "s4-r1-w1-c3-s1-seg1", "task": "rl", "dataset": "Hopper-v4", "model": "ppo",
              "params": {"hidden": "64x64", "learning_rate": 3e-4}, "metric": "return", "seed": 1,
              "budget": {"unit": "env_steps", "total": 150000, "anneal_total": 300000}}


# ---------------------------------------------------------------- config.json
def test_churn_config_parses():
    cfg = RunConfig.model_validate_json(CHURN_CONFIG)
    assert cfg.costs.breakeven_threshold == pytest.approx(50 / 180)


def test_ppo_config_parses_and_clamps():
    cfg = RunConfig.model_validate(PPO_CONFIG)
    p = worker_clamp(cfg.model, {**cfg.params, "learning_rate": 1.0, "num_envs": 999, "evil": 1})
    assert p["hidden"] == [64, 64] and p["learning_rate"] == 3e-3 and p["num_envs"] == 4
    assert "evil" not in p


@pytest.mark.parametrize("patch", [
    {"budget": {"unit": "epochs", "total": 10}},          # hgb budgets in rounds
    {"model": "ppo"},                                      # wrong task for model
    {"costs": None},                                       # savings without costs
    {"run_id": "../../etc"},                               # path-ish run id
])
def test_config_rejects_inconsistent(patch):
    d = {**json.loads(CHURN_CONFIG), **patch}
    with pytest.raises(ValidationError):
        RunConfig.model_validate(d)


def test_rl_rejects_unknown_env():
    with pytest.raises(ValidationError):
        RunConfig.model_validate({**PPO_CONFIG, "dataset": "Ant-v4"})


def test_worker_clamp_tabular():
    p = worker_clamp("mlp", {"lr": 5.0, "hidden": "4096x4096", "dropout": 0.9, "rm_rf": 1})
    assert p == {"hidden": "64x64", "lr": 3e-2, "weight_decay": 0.0, "dropout": 0.5, "calibration": "none"}


# ---------------------------------------------------------------- metrics
def test_metric_lines():
    UpdateLine.model_validate_json('{"type":"update","step":50,"unit":"rounds","t":3.1,"train_loss":0.41,'
                                   '"val_loss":0.43,"val_metric":11840.0,"extra":{"val_auc":0.846}}')


def test_improvement_ratio_respects_direction():
    assert improvement_ratio("rmse", 0.5, 1.0) == 2.0
    assert improvement_ratio("savings", 12000, 10000) == pytest.approx(1.2)
    assert improvement_ratio("savings", 5000, -100) is None


# ---------------------------------------------------------------- brief
def _brief(**over):
    d = {"problem_type": "binary", "data_source": {"kind": "upload", "ref": "u1"}, "target": "Churn",
         "metric": "savings", "costs": {"offer_cost": 50, "churn_loss": 600, "save_rate": 0.3},
         "questions": ["Who should we target?", "How much will it save?"], **over}
    return d


def test_brief_ok():
    assert ResearchBrief.model_validate(_brief()).higher_is_better


@pytest.mark.parametrize("over", [{"metric": "rmse"}, {"costs": None}, {"target": None},
                                  {"questions": []}])
def test_brief_rejects(over):
    with pytest.raises(ValidationError):
        ResearchBrief.model_validate(_brief(**over))


# ---------------------------------------------------------------- plan
def _plan(**over):
    cfgs = [
        {"name": "lr_base", "model": "logreg", "params": {"C": 1.0}, "rationale": "baseline"},
        {"name": "hgb_mid", "model": "hgb", "params": {"learning_rate": 0.1}, "rationale": "trees"},
    ]
    preds = [
        {"id": "P1", "statement": "s", "kind": "a_beats_b", "dataset": "telco", "a": "hgb_mid",
         "b": "lr_base", "min_ratio": 1.1, "p": 0.45},
        {"id": "P2", "statement": "s", "kind": "reaches_threshold", "dataset": "telco", "a": "hgb_mid",
         "threshold": 10000, "p": 0.6},
        {"id": "P3", "statement": "s", "kind": "top_feature", "dataset": "telco", "a": "hgb_mid",
         "feature": "tenure", "top_k": 2, "p": 0.75},
    ]
    return {"task_summary": "t", "configs": cfgs, "hypothesis_config": "hgb_mid", "predictions": preds, **over}


def test_plan_ok():
    plan = Plan.model_validate(_plan())
    assert plan.configs[1].resolved_params()["max_leaf_nodes"] == 15


def test_plan_rejects_param_outside_space():
    d = _plan(); d["configs"][1]["params"]["learning_rate"] = 0.05
    with pytest.raises(ValidationError, match="not allowed"):
        Plan.model_validate(d)


def test_plan_rejects_unknown_model_and_refs():
    d = _plan(); d["configs"][0]["model"] = "xgboost"
    with pytest.raises(ValidationError):
        Plan.model_validate(d)
    with pytest.raises(ValidationError):
        Plan.model_validate(_plan(hypothesis_config="nope"))
    d = _plan(); d["predictions"][0]["b"] = "ghost"
    with pytest.raises(ValidationError):
        Plan.model_validate(d)


def test_prediction_kind_requires_fields():
    d = _plan(); del d["predictions"][2]["feature"]
    with pytest.raises(ValidationError, match="needs"):
        Plan.model_validate(d)


def test_prediction_overconfident():
    d = _plan(); d["predictions"][0]["p"] = 0.99
    with pytest.raises(ValidationError):
        Plan.model_validate(d)


# ---------------------------------------------------------------- monitor / reflection
def test_fork_bounds():
    assert bound_fork_overrides("ppo", {"learning_rate": 3e-4},
                                {"learning_rate": 1e-6, "ent_coef": 0.5, "hidden": 512}) == \
        {"learning_rate": 1.5e-4, "ent_coef": 0.02}
    assert bound_fork_overrides("mlp", {"lr": 1e-3}, {"lr": 1.0, "dropout": 0.3}) == {"lr": 2e-3, "dropout": 0.3}
    assert bound_fork_overrides("logreg", {"C": 1.0}, {"C": 100}) == {}


def test_monitor_decision_limits():
    with pytest.raises(ValidationError):
        MonitorDecision.model_validate({"actions": [{"kind": "note", "reason": "r"}] * 7, "summary": "s"})
    with pytest.raises(ValidationError):
        MonitorDecision.model_validate({"actions": [{"kind": "fork", "run_id": "x", "reason": "r"}],
                                        "summary": "s"})


def test_reflection_exactly_one():
    Reflection.model_validate({"findings": ["f"], "stop_reason": "budget"})
    Reflection.model_validate({"findings": ["f"], "next_round": _plan(round=2)})
    with pytest.raises(ValidationError):
        Reflection.model_validate({"findings": ["f"]})


def test_costs_breakeven():
    assert Costs(offer_cost=50, churn_loss=600, save_rate=0.3).breakeven_threshold == pytest.approx(0.2778, abs=1e-4)


# ---------------------------------------------------------------- db
def test_db_schema(tmp_path):
    conn = connect(tmp_path / "t.db")
    names = {r[0] for r in conn.execute("select name from sqlite_master where type='table'")}
    assert {"sessions", "datasets", "plans", "runs", "metrics", "decisions", "predictions",
            "events", "artifacts"} <= names
