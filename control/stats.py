"""Statistics (RL spec §4.8, main plan §8.3). Unit of analysis = the seed."""
from __future__ import annotations

import numpy as np
from scipy.stats import trim_mean


def iqm(x) -> float:
    return float(trim_mean(np.asarray(x, dtype=float), 0.25))


def boot_ci(x, f=iqm, B=10_000, a=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return [float("nan"), float("nan")]
    idx = rng.integers(0, len(x), (B, len(x)))
    s = np.array([f(x[i]) for i in idx]) if f is not np.mean else x[idx].mean(1)
    return [float(v) for v in np.quantile(s, [a / 2, 1 - a / 2])]


def diff_ci(xa, xb, f=iqm, B=10_000, a=0.05, seed=0):
    rng = np.random.default_rng(seed)
    xa, xb = np.asarray(xa, dtype=float), np.asarray(xb, dtype=float)
    ia = rng.integers(0, len(xa), (B, len(xa)))
    ib = rng.integers(0, len(xb), (B, len(xb)))
    d = np.array([f(xa[i]) - f(xb[j]) for i, j in zip(ia, ib)])
    return [float(v) for v in np.quantile(d, [a / 2, 1 - a / 2])]


def paired_boot_ci(va, vb, B=10_000, a=0.05, seed=0):
    """Paired bootstrap over test rows: va/vb are per-row values (e.g. savings) of two models."""
    rng = np.random.default_rng(seed)
    d = np.asarray(va, dtype=float) - np.asarray(vb, dtype=float)
    idx = rng.integers(0, len(d), (B, len(d)))
    s = d[idx].sum(1)
    return [float(v) for v in np.quantile(s, [a / 2, 1 - a / 2])]
