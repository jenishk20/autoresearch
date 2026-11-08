"""Shared contracts (plan §5) and search-space bounds (plan §6.2).

This file is the interface between Person A (infra/web) and Person B (RL/agent).
Change it deliberately and tell the other person.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, confloat, conlist

EnvId = Literal["HalfCheetah-v4", "Walker2d-v4", "Hopper-v4", "InvertedDoublePendulum-v4"]
HIDDEN_SIZES = {"64x64": [64, 64], "256x256": [256, 256]}

# ---------------------------------------------------------------- §6.2 bounds
LR_CHOICES = (1e-4, 3e-4, 1e-3)
ENT_CHOICES = (0.0, 0.01)
FORK_LR_RANGE = (3e-5, 3e-3)
FORK_LR_FACTORS = (0.5, 2.0)
FORK_ENT_RANGE = (0.0, 0.02)
FORK_CLIP_RANGE = (0.1, 0.3)
FORKABLE_KEYS = {"learning_rate", "ent_coef", "clip_coef"}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _nearest(x: float, choices: tuple[float, ...]) -> float:
    return min(choices, key=lambda c: abs(c - float(x)))


# ---------------------------------------------------------------- §5.1 config.json
class PPOParams(BaseModel):
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

    def clamped(self) -> "PPOParams":
        """Worker-side guard: never trust input. Pins everything §6.2 says is fixed."""
        hidden = self.hidden if self.hidden in HIDDEN_SIZES.values() else [64, 64]
        return PPOParams(
            hidden=hidden,
            learning_rate=_clamp(self.learning_rate, *FORK_LR_RANGE),
            ent_coef=_clamp(self.ent_coef, *FORK_ENT_RANGE),
            norm_obs=bool(self.norm_obs),
            clip_coef=_clamp(self.clip_coef, *FORK_CLIP_RANGE),
        )


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    algo: Literal["ppo", "sac"] = "ppo"
    env_id: EnvId
    seed: int
    total_steps: int = Field(gt=0, le=5_000_000)
    start_step: int = Field(default=0, ge=0)
    anneal_total_steps: int = Field(gt=0)
    params: PPOParams = PPOParams()
    resume_from: str | None = None  # path under /parent
    reward_fn: str | None = None    # filename under /job, e.g. "reward.py"
    ckpt_every: int = 100_000


# ---------------------------------------------------------------- §5.2 metrics.jsonl
class UpdateLine(BaseModel):
    type: Literal["update"]
    step: int; t: float; sps: float; lr: float
    pg_loss: float; v_loss: float; entropy: float
    approx_kl: float; clipfrac: float; ev: float


class EpisodeLine(BaseModel):
    type: Literal["episode"]
    step: int; t: float; ret_true: float; ret_shaped: float; len: int


class CheckpointLine(BaseModel):
    type: Literal["checkpoint"]
    step: int; path: str; sha256: str


class DoneLine(BaseModel):
    type: Literal["done"]
    step: int
    reason: Literal["budget", "stop_flag", "nan", "error"]


MetricLine = Union[UpdateLine, EpisodeLine, CheckpointLine, DoneLine]


# ---------------------------------------------------------------- §5.3 Plan
class ConfigSpec(BaseModel):
    name: str
    hidden: Literal["64x64", "256x256"]
    learning_rate: Literal[1e-4, 3e-4, 1e-3]
    ent_coef: Literal[0.0, 0.01]
    norm_obs: bool
    rationale: str

    def to_params(self) -> PPOParams:
        return PPOParams(hidden=HIDDEN_SIZES[self.hidden], learning_rate=self.learning_rate,
                         ent_coef=self.ent_coef, norm_obs=self.norm_obs)


class Prediction(BaseModel):
    id: str
    statement: str
    kind: Literal["a_beats_b", "reaches_threshold"]
    env: Literal["HalfCheetah-v4", "Walker2d-v4", "Hopper-v4", "all"]
    a: str
    b: str | None = None
    min_ratio: float | None = None
    threshold: float | None = None
    at_step: int
    p: confloat(ge=0.05, le=0.95)


class Plan(BaseModel):
    task_summary: str
    envs: list[str]
    configs: conlist(ConfigSpec, min_length=2, max_length=8)
    hypothesis_config: str  # carried to wave 2 to test one key prediction
    predictions: conlist(Prediction, min_length=3, max_length=5)
    reward_proposal: dict[str, str] | None = None  # env -> python source


# ---------------------------------------------------------------- §5.4 MonitorDecision
class Action(BaseModel):
    kind: Literal["stop", "fork", "rescue", "note"]
    run_id: str | None = None
    reason: str
    overrides: dict[str, float] | None = None
    extra_steps: int | None = None


class MonitorDecision(BaseModel):
    actions: conlist(Action, max_length=6)
    summary: str


def bound_fork_overrides(parent: PPOParams, overrides: dict[str, float]) -> dict[str, float]:
    """Apply §6.2 fork bounds. Unknown keys are dropped; lr snaps to ×0.5 / ×2 of the parent."""
    out: dict[str, float] = {}
    for k, v in overrides.items():
        if k not in FORKABLE_KEYS:
            continue
        if k == "learning_rate":
            factor = _nearest(v / parent.learning_rate, FORK_LR_FACTORS)
            out[k] = _clamp(parent.learning_rate * factor, *FORK_LR_RANGE)
        elif k == "ent_coef":
            out[k] = _clamp(v, *FORK_ENT_RANGE)
        elif k == "clip_coef":
            out[k] = _clamp(v, *FORK_CLIP_RANGE)
    return out


# ---------------------------------------------------------------- §4.4 run state machine
class RunStatus(str, Enum):
    PENDING = "PENDING"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    STOPPED = "STOPPED"
    FAILED = "FAILED"
    CONTAINED = "CONTAINED"


TERMINAL_STATUSES = {RunStatus.DONE, RunStatus.STOPPED, RunStatus.FAILED, RunStatus.CONTAINED}
