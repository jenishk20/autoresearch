"""Shared contracts (main plan §10) and per-model search spaces (§5.1, RL spec §6.2).

This file is the interface between Person A (infra/web) and Person B (agent/ML).
Change it deliberately and tell the other person.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Union

from pydantic import (BaseModel, ConfigDict, Field, confloat, conlist, field_validator,
                      model_validator)

# ================================================================ problem / metric vocabulary
ProblemType = Literal["binary", "multiclass", "regression", "rl"]
Task = Literal["tabular", "rl"]
Metric = Literal["savings", "auc", "logloss", "accuracy", "f1", "rmse", "mae", "return"]

# Direction is decided by code, never by the LLM.
HIGHER_IS_BETTER: dict[str, bool] = {
    "savings": True, "auc": True, "accuracy": True, "f1": True, "return": True,
    "logloss": False, "rmse": False, "mae": False,
}
METRICS_FOR_PROBLEM: dict[str, set[str]] = {
    "binary": {"savings", "auc", "logloss", "accuracy", "f1"},
    "multiclass": {"accuracy", "logloss", "f1"},
    "regression": {"rmse", "mae"},
    "rl": {"return"},
}
RL_ENVS = ("HalfCheetah-v4", "Walker2d-v4", "Hopper-v4", "InvertedDoublePendulum-v4")


def improvement_ratio(metric: str, a: float, b: float) -> float | None:
    """How many times better a is than b (>1 means a wins). None when the ratio is meaningless
    (non-positive values, e.g. negative savings): such predictions are scored inconclusive."""
    if a <= 0 or b <= 0:
        return None
    return a / b if HIGHER_IS_BETTER[metric] else b / a


# ================================================================ search spaces
# PLANNER_CHOICES: exact values the planner may pick. FORK_RULES: what a fork may change.
#   ("factor", factors, (lo, hi)) -> snap to parent * nearest factor, then clamp
#   ("range", (lo, hi))           -> clamp
MODELS_FOR_PROBLEM: dict[str, tuple[str, ...]] = {
    "binary": ("logreg", "random_forest", "hgb", "mlp"),
    "multiclass": ("logreg", "random_forest", "hgb", "mlp"),
    "regression": ("ridge", "random_forest", "hgb", "mlp"),
    "rl": ("ppo",),
}
TASK_FOR_MODEL = {"ppo": "rl"}  # everything else is tabular

BUDGET_UNIT: dict[str, str] = {
    "logreg": "fits", "ridge": "fits", "random_forest": "trees", "hgb": "rounds",
    "mlp": "epochs", "ppo": "env_steps",
}

_CAL = {"calibration": ("none", "isotonic")}
PLANNER_CHOICES: dict[str, dict[str, tuple]] = {
    "logreg": {"C": (0.01, 0.1, 1.0, 10.0), "class_weight": ("none", "balanced"), **_CAL},
    "ridge": {"alpha": (0.1, 1.0, 10.0, 100.0)},
    "random_forest": {"max_depth": (0, 8, 16), "min_samples_leaf": (1, 5, 20), **_CAL},  # 0 = None
    "hgb": {"learning_rate": (0.03, 0.1, 0.3), "max_leaf_nodes": (15, 31, 63), "l2": (0.0, 1.0), **_CAL},
    "mlp": {"hidden": ("64x64", "256x256"), "lr": (1e-4, 1e-3, 1e-2),
            "weight_decay": (0.0, 1e-4), "dropout": (0.0, 0.2), **_CAL},
    "ppo": {"hidden": ("64x64", "256x256"), "learning_rate": (1e-4, 3e-4, 1e-3),
            "ent_coef": (0.0, 0.01), "norm_obs": (True, False)},
}
FORK_RULES: dict[str, dict[str, tuple]] = {
    "hgb": {"l2": ("range", (0.0, 10.0)), "learning_rate": ("factor", (0.5,), (0.005, 0.3))},
    "mlp": {"lr": ("factor", (0.5, 2.0), (1e-5, 3e-2)), "weight_decay": ("range", (0.0, 1e-3)),
            "dropout": ("range", (0.0, 0.5))},
    "ppo": {"learning_rate": ("factor", (0.5, 2.0), (3e-5, 3e-3)), "ent_coef": ("range", (0.0, 0.02)),
            "clip_coef": ("range", (0.1, 0.3))},
}
HIDDEN_SIZES = {"64x64": [64, 64], "256x256": [256, 256]}


def default_params(model: str) -> dict[str, Any]:
    return {k: v[0] if k != "calibration" else "none" for k, v in PLANNER_CHOICES[model].items()}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _snap_factor(v: float, parent: float, factors: tuple[float, ...], lo: float, hi: float) -> float:
    f = min(factors, key=lambda c: abs(c - float(v) / parent))
    return _clamp(parent * f, lo, hi)


def bound_fork_overrides(model: str, parent: dict[str, Any], overrides: dict[str, float]) -> dict[str, float]:
    """Apply fork bounds. Keys the model doesn't allow to fork are dropped silently (and logged by caller)."""
    rules, out = FORK_RULES.get(model, {}), {}
    for k, v in overrides.items():
        rule = rules.get(k)
        if rule is None:
            continue
        if rule[0] == "factor":
            out[k] = _snap_factor(v, float(parent[k]), rule[1], *rule[2])
        else:
            out[k] = _clamp(v, *rule[1])
    return out


def worker_clamp(model: str, params: dict[str, Any]) -> dict[str, Any]:
    """Worker-side guard: never trust input. Unknown keys dropped; choice params outside the
    planner set fall back to the default unless a fork rule allows a range for them."""
    if model == "ppo":
        known = {k: v for k, v in params.items() if k in PPOParams.model_fields}
        return PPOParams.model_validate(known).clamped().model_dump()
    choices, rules, out = PLANNER_CHOICES[model], FORK_RULES.get(model, {}), default_params(model)
    for k, v in params.items():
        if k in rules:
            lo, hi = rules[k][-1]
            out[k] = _clamp(v, lo, hi)
        elif k in choices and v in choices[k]:
            out[k] = v
    return out


# ================================================================ §10.3 agent outputs
class DataSource(BaseModel):
    kind: Literal["upload", "catalog", "openml", "env"]
    ref: str  # upload id / catalog name / OpenML dataset id / env id


class Costs(BaseModel):
    offer_cost: float = Field(ge=0)
    churn_loss: float = Field(gt=0)
    save_rate: float = Field(gt=0, le=1)

    @property
    def breakeven_threshold(self) -> float:
        """Target if P(churn) > C / (s·L) for a calibrated model."""
        return min(1.0, self.offer_cost / (self.save_rate * self.churn_loss))


class ResearchBrief(BaseModel):
    problem_type: ProblemType
    data_source: DataSource
    target: str | None = None  # None for RL
    metric: Metric
    costs: Costs | None = None
    constraints: list[str] = []
    questions: conlist(str, min_length=1, max_length=4)
    feasible: bool = True
    scope_note: str | None = None

    @model_validator(mode="after")
    def _consistent(self):
        if self.metric not in METRICS_FOR_PROBLEM[self.problem_type]:
            raise ValueError(f"metric {self.metric!r} not valid for {self.problem_type}; "
                             f"allowed: {sorted(METRICS_FOR_PROBLEM[self.problem_type])}")
        if self.metric == "savings" and self.costs is None:
            raise ValueError("metric 'savings' requires costs")
        if self.problem_type != "rl" and not self.target:
            raise ValueError("tabular problems need a target column")
        return self

    @property
    def higher_is_better(self) -> bool:
        return HIGHER_IS_BETTER[self.metric]


class ColumnProfile(BaseModel):
    name: str
    dtype: Literal["numeric", "categorical", "boolean", "datetime", "text"]
    missing_frac: float
    cardinality: int
    single_feature_auc: float | None = None


class ColumnNote(BaseModel):
    column: str
    reason: str


class DataProfile(BaseModel):
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    target_stats: dict[str, float] = {}
    leakage_suspects: list[ColumnNote] = []
    id_like: list[str] = []
    time_column: str | None = None


class Preprocessing(BaseModel):
    drop: list[str] = []
    scale: bool | None = None  # None = adapter default (on for logreg/ridge/mlp)


class ConfigSpec(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9_\-]{1,32}$")
    model: str
    params: dict[str, Any] = {}
    preprocessing: Preprocessing = Preprocessing()
    answers_question: int | None = None  # index into ResearchBrief.questions
    rationale: str

    @model_validator(mode="after")
    def _in_space(self):
        if self.model not in PLANNER_CHOICES:
            raise ValueError(f"unknown model {self.model!r}; allowed: {sorted(PLANNER_CHOICES)}")
        space = PLANNER_CHOICES[self.model]
        for k, v in self.params.items():
            if k not in space:
                raise ValueError(f"{self.model}: param {k!r} not tunable; allowed: {sorted(space)}")
            if v not in space[k]:
                raise ValueError(f"{self.model}.{k}={v!r} not allowed; choose one of {list(space[k])}")
        return self

    def resolved_params(self) -> dict[str, Any]:
        return {**default_params(self.model), **self.params}


PredictionKind = Literal["a_beats_b", "reaches_threshold", "top_feature", "flag_count", "rank_transfer"]
_REQUIRED: dict[str, tuple[str, ...]] = {
    "a_beats_b": ("b", "min_ratio"),
    "reaches_threshold": ("threshold",),
    "top_feature": ("feature", "top_k"),
    "flag_count": ("flag", "min_count"),
    "rank_transfer": ("transfer_to", "top_k"),
}


class Prediction(BaseModel):
    id: str
    statement: str
    kind: PredictionKind
    dataset: str  # dataset sha/name or env id the prediction is about
    a: str        # config name
    b: str | None = None
    min_ratio: float | None = Field(default=None, gt=1.0)
    threshold: float | None = None
    feature: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=10)
    flag: Literal["nan", "diverging", "overfitting", "collapse", "stalled", "kl_high"] | None = None
    min_count: int | None = Field(default=None, ge=1)
    transfer_to: str | None = None
    at_budget: int | None = None
    p: confloat(ge=0.05, le=0.95)

    @model_validator(mode="after")
    def _fields_for_kind(self):
        missing = [f for f in _REQUIRED[self.kind] if getattr(self, f) is None]
        if missing:
            raise ValueError(f"prediction {self.id} of kind {self.kind} needs {missing}")
        return self


class Plan(BaseModel):
    round: int = Field(default=1, ge=1, le=2)
    task_summary: str
    drop_columns: list[ColumnNote] = []
    configs: conlist(ConfigSpec, min_length=2, max_length=8)
    hypothesis_config: str
    predictions: conlist(Prediction, min_length=3, max_length=5)
    feature_code: str | None = None  # Level-2 build_features(df) source, checked by safety.py

    @model_validator(mode="after")
    def _refs(self):
        names = [c.name for c in self.configs]
        if len(set(names)) != len(names):
            raise ValueError("config names must be unique")
        if self.hypothesis_config not in names:
            raise ValueError(f"hypothesis_config {self.hypothesis_config!r} is not one of {names}")
        for p in self.predictions:
            for ref in (p.a, p.b):
                if ref is not None and ref not in names:
                    raise ValueError(f"prediction {p.id} refers to unknown config {ref!r}")
        return self


class Action(BaseModel):
    kind: Literal["stop", "fork", "rescue", "note"]
    run_id: str | None = None
    reason: str
    overrides: dict[str, float] | None = None
    extra_budget: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _needs_run(self):
        if self.kind != "note" and not self.run_id:
            raise ValueError(f"{self.kind} needs run_id")
        if self.kind == "fork" and not self.extra_budget:
            raise ValueError("fork needs extra_budget")
        return self


class MonitorDecision(BaseModel):
    actions: conlist(Action, max_length=6)
    summary: str


class Reflection(BaseModel):
    findings: conlist(str, min_length=1, max_length=8)
    next_round: Plan | None = None
    stop_reason: str | None = None

    @model_validator(mode="after")
    def _one_of(self):
        if (self.next_round is None) == (self.stop_reason is None):
            raise ValueError("give exactly one of next_round or stop_reason")
        return self


# ================================================================ §10.1 config.json
class Budget(BaseModel):
    unit: Literal["fits", "trees", "rounds", "epochs", "env_steps"]
    total: int = Field(gt=0, le=5_000_000)
    start_at: int = Field(default=0, ge=0)
    anneal_total: int | None = None  # lineage-wide schedule length (lr annealing)
    chunk: int | None = Field(default=None, gt=0)  # log + checkpoint every chunk


class PPOParams(BaseModel):
    """RL adapter params (RL spec §5.1). Fixed values are pinned by clamped()."""
    model_config = ConfigDict(extra="forbid")
    hidden: list[int] = [64, 64]
    learning_rate: float = 3e-4
    ent_coef: float = 0.0
    norm_obs: bool = True
    clip_coef: float = 0.2
    num_envs: int = 4
    num_steps: int = 512
    update_epochs: int = 10
    num_minibatches: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5

    @field_validator("hidden", mode="before")
    @classmethod
    def _hidden_str(cls, v):
        return HIDDEN_SIZES.get(v, v) if isinstance(v, str) else v

    def clamped(self) -> "PPOParams":
        r = FORK_RULES["ppo"]
        return PPOParams(
            hidden=self.hidden if self.hidden in HIDDEN_SIZES.values() else [64, 64],
            learning_rate=_clamp(self.learning_rate, *r["learning_rate"][2]),
            ent_coef=_clamp(self.ent_coef, *r["ent_coef"][1]),
            norm_obs=bool(self.norm_obs),
            clip_coef=_clamp(self.clip_coef, *r["clip_coef"][1]),
        )


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(pattern=r"^[A-Za-z0-9_\-]{1,96}$")
    task: Task
    dataset: str  # dataset sha256 (tabular) or env id (rl)
    model: str
    params: dict[str, Any] = {}
    preprocessing: Preprocessing = Preprocessing()
    target: str | None = None
    metric: Metric
    costs: Costs | None = None
    seed: int = Field(ge=0)
    budget: Budget
    resume_from: str | None = None  # path under /parent
    feature_fn: str | None = None   # filename under /job (features.py or reward.py)
    ckpt_every: int = 1             # in chunks (tabular) or 100k steps (rl)

    @model_validator(mode="after")
    def _consistent(self):
        if TASK_FOR_MODEL.get(self.model, "tabular") != self.task:
            raise ValueError(f"model {self.model!r} does not belong to task {self.task!r}")
        if self.task == "rl" and self.dataset not in RL_ENVS:
            raise ValueError(f"unknown env {self.dataset!r}")
        if self.budget.unit != BUDGET_UNIT[self.model]:
            raise ValueError(f"{self.model} budgets in {BUDGET_UNIT[self.model]}, not {self.budget.unit}")
        if self.metric == "savings" and self.costs is None:
            raise ValueError("metric 'savings' requires costs")
        return self


RunKind = Literal["profile", "train", "verify"]  # only 'verify' gets /sealed mounted


# ================================================================ §10.2 metrics.jsonl
class UpdateLine(BaseModel):
    type: Literal["update"]
    step: int
    unit: str
    t: float
    train_loss: float | None = None
    val_loss: float | None = None
    val_metric: float | None = None
    extra: dict[str, float] = {}


class EpisodeLine(BaseModel):
    type: Literal["episode"]
    step: int
    t: float
    ret_true: float
    ret_shaped: float
    len: int


class CheckpointLine(BaseModel):
    type: Literal["checkpoint"]
    step: int
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DoneLine(BaseModel):
    type: Literal["done"]
    step: int
    reason: Literal["budget", "stop_flag", "nan", "error"]


MetricLine = Union[UpdateLine, EpisodeLine, CheckpointLine, DoneLine]


# ================================================================ run state machine
class RunStatus(str, Enum):
    PENDING = "PENDING"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    STOPPED = "STOPPED"
    FAILED = "FAILED"
    CONTAINED = "CONTAINED"


TERMINAL_STATUSES = {RunStatus.DONE, RunStatus.STOPPED, RunStatus.FAILED, RunStatus.CONTAINED}
