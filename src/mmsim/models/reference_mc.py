"""Vectorised NumPy reference Monte Carlo of the A-S world (FR-19).

``as2008_mc`` follows Avellaneda and Stoikov (2008), section 3.3, step by step: at each
``t`` the quotes are computed from the state; with probability ``lambda_a(delta_a) dt`` the
inventory falls by one and cash rises by ``s + delta_a``; with probability
``lambda_b(delta_b) dt`` the inventory rises by one and cash falls by ``s - delta_b``; then
the mid moves by ``+-sigma sqrt(dt)``. Probabilities are clipped to [0, 1]. Profit is
``X_T + q_T S_T``.

``glft_mc`` runs the GLFT exact strategy the same way with inventory bounds, small ``dt``
and Gaussian mid increments, and returns terminal wealth for the CARA-utility check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from mmsim.models.glft import glft_exact

__all__ = ["MCResult", "as2008_avg_spread", "as2008_mc", "glft_mc"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class MCResult:
    """Per-path outcomes."""

    profit: FloatArray
    final_q: FloatArray
    avg_spread: float
    fills: FloatArray


def as2008_avg_spread(gamma: float, sigma: float, k: float, T: float, dt: float) -> float:
    """Mean over the decision times of eq. (30)'s spread (the symmetric benchmark's spread)."""
    n = round(T / dt)
    tau = T - np.arange(n) * dt
    spread = gamma * sigma**2 * tau + (2.0 / gamma) * math.log1p(gamma / k)
    return float(spread.mean())


def as2008_mc(
    gamma: float,
    *,
    strategy: Literal["inventory", "symmetric"],
    n_paths: int,
    seed: int,
    s0: float = 100.0,
    T: float = 1.0,
    sigma: float = 2.0,
    dt: float = 0.005,
    A: float = 140.0,
    k: float = 1.5,
    symmetric_spread: float | None = None,
) -> MCResult:
    """Simulate ``n_paths`` paths of the inventory or symmetric strategy."""
    rng = np.random.Generator(np.random.PCG64(seed))
    n = round(T / dt)
    s = np.full(n_paths, s0)
    q = np.zeros(n_paths)
    x = np.zeros(n_paths)
    fills = np.zeros(n_paths)
    spreads = np.empty(n)
    sym = symmetric_spread
    if strategy == "symmetric" and sym is None:
        sym = as2008_avg_spread(gamma, sigma, k, T, dt)
    for i in range(n):
        tau = T - i * dt
        if strategy == "inventory":
            spread = gamma * sigma**2 * tau + (2.0 / gamma) * math.log1p(gamma / k)
            r = s - q * gamma * sigma**2 * tau
            ask = r + spread / 2
            bid = r - spread / 2
        else:
            assert sym is not None
            spread = sym
            ask = s + sym / 2
            bid = s - sym / 2
        spreads[i] = spread
        da = ask - s
        db = s - bid
        pa = np.clip(A * np.exp(-k * da) * dt, 0.0, 1.0)
        pb = np.clip(A * np.exp(-k * db) * dt, 0.0, 1.0)
        hit_a = rng.random(n_paths) < pa
        hit_b = rng.random(n_paths) < pb
        x += np.where(hit_a, ask, 0.0) - np.where(hit_b, bid, 0.0)
        q += hit_b.astype(np.float64) - hit_a.astype(np.float64)
        fills += hit_a.astype(np.float64) + hit_b.astype(np.float64)
        s += np.where(rng.random(n_paths) < 0.5, 1.0, -1.0) * sigma * math.sqrt(dt)
    return MCResult(profit=x + q * s, final_q=q, avg_spread=float(spreads.mean()), fills=fills)


def glft_mc(
    gamma: float,
    sigma: float,
    A: float,
    k: float,
    q_max: int,
    T: float,
    dt: float,
    *,
    n_paths: int,
    seed: int,
    q0: int = 0,
    s0: float = 100.0,
) -> FloatArray:
    """Terminal wealth ``X_T + q_T S_T`` of the GLFT exact strategy (``x0 = 0``)."""
    n = round(T / dt)
    sol = glft_exact(gamma, sigma, A, k, q_max, T, n_t=n + 1)
    c1 = math.log1p(gamma / k) / gamma
    lv = sol.log_v  # (n + 1, 2Q + 1)
    # delta tables by (step, q index); closed sides get +inf distance.
    d_bid = np.full((n + 1, 2 * q_max + 1), np.inf)
    d_ask = np.full((n + 1, 2 * q_max + 1), np.inf)
    d_bid[:, :-1] = (lv[:, :-1] - lv[:, 1:]) / k + c1
    d_ask[:, 1:] = (lv[:, 1:] - lv[:, :-1]) / k + c1
    rng = np.random.Generator(np.random.PCG64(seed))
    s = np.full(n_paths, s0)
    q = np.full(n_paths, q0, dtype=np.int64)
    x = np.zeros(n_paths)
    for i in range(n):
        j = q + q_max
        db = d_bid[i, j]
        da = d_ask[i, j]
        with np.errstate(over="ignore"):
            pb = np.clip(A * np.exp(-k * db) * dt, 0.0, 1.0)
            pa = np.clip(A * np.exp(-k * da) * dt, 0.0, 1.0)
        hit_b = rng.random(n_paths) < pb
        hit_a = rng.random(n_paths) < pa
        x += np.where(hit_a, s + np.where(np.isfinite(da), da, 0.0), 0.0)
        x -= np.where(hit_b, s - np.where(np.isfinite(db), db, 0.0), 0.0)
        q += hit_b.astype(np.int64) - hit_a.astype(np.int64)
        s += sigma * math.sqrt(dt) * rng.standard_normal(n_paths)
    wealth: FloatArray = x + q * s
    return wealth
