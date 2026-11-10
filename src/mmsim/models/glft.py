"""Gueant, Lehalle and Fernandez-Tapia (2013) quotes with inventory bounds (FR-10, FR-13).

Gueant, O., Lehalle, C.-A. and Fernandez-Tapia, J. (2013). Dealing with the inventory risk:
a solution to the market making problem. Mathematics and Financial Economics 7(4),
477-507; arXiv:1105.3115. The arXiv text numbers results, not equations:

- Proposition 2: ``v(t) = exp(-M (T - t)) 1`` with ``M`` tridiagonal on
  ``q in [-Q, Q]``, diagonal ``alpha q**2``, off-diagonals ``-eta``,
  ``alpha = (k / 2) gamma sigma**2`` and ``eta = A (1 + gamma/k)**(-(1 + k/gamma))``.
- Theorem 1: ``delta_b(t, q) = (1/k) ln(v_q / v_{q+1}) + (1/gamma) ln(1 + gamma/k)`` for
  ``q != Q`` and ``delta_a(t, q) = (1/k) ln(v_q / v_{q-1}) + (1/gamma) ln(1 + gamma/k)`` for
  ``q != -Q``; value function ``u = -exp(-gamma (x + q s)) v_q(t)**(-gamma/k)``.
- Section 4, after Proposition 3: the closed-form approximation
  ``delta_b ~ c1 + (2q + 1)/2 * c2`` and ``delta_a ~ c1 - (2q - 1)/2 * c2`` with
  ``c1 = (1/gamma) ln(1 + gamma/k)`` and
  ``c2 = sqrt(sigma**2 gamma / (2 k A) * (1 + gamma/k)**(1 + k/gamma))``.

The fee shift (FR-13) is derived in docs/models.md: a maker fee ``c`` per unit turns
``A exp(-k delta)`` into ``A exp(-k c) exp(-k (delta - c))``, so the model is solved with
``A' = A exp(-k c)`` on the net distance and ``c`` is added back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "GLFTExact",
    "glft_asymptotic",
    "glft_c1",
    "glft_c2",
    "glft_certainty_equivalent",
    "glft_exact",
    "glft_matrix",
    "with_fee",
]

FloatArray = npt.NDArray[np.float64]


def _check(gamma: float, sigma: float, A: float, k: float) -> None:
    if gamma <= 0 or sigma <= 0 or A <= 0 or k <= 0:
        raise ValueError("need gamma, sigma, A and k all > 0")


def glft_c1(gamma: float, k: float) -> float:
    """``(1/gamma) ln(1 + gamma/k)``: the inventory-free half spread."""
    return math.log1p(gamma / k) / gamma


def glft_c2(gamma: float, sigma: float, A: float, k: float) -> float:
    """``sqrt(sigma^2 gamma / (2 k A) * (1 + gamma/k)^(1 + k/gamma))``: the skew per unit."""
    return math.sqrt(
        sigma**2 * gamma / (2.0 * k * A) * math.exp((1.0 + k / gamma) * math.log1p(gamma / k))
    )


def glft_asymptotic(
    q: float, gamma: float, sigma: float, A: float, k: float
) -> tuple[float, float]:
    """``(delta_bid, delta_ask)`` of the closed-form approximation (section 4)."""
    _check(gamma, sigma, A, k)
    c1, c2 = glft_c1(gamma, k), glft_c2(gamma, sigma, A, k)
    return c1 + (q + 0.5) * c2, c1 - (q - 0.5) * c2


def with_fee(A: float, k: float, fee_per_unit: float) -> float:
    """``A' = A exp(-k c)`` for a maker fee ``c`` per unit (negative = rebate)."""
    return A * math.exp(-k * fee_per_unit)


def glft_matrix(gamma: float, sigma: float, A: float, k: float, q_max: int) -> FloatArray:
    """The ``(2Q+1) x (2Q+1)`` matrix ``M`` of Proposition 2, rows ordered ``-Q..Q``."""
    _check(gamma, sigma, A, k)
    if q_max < 1:
        raise ValueError("q_max must be >= 1")
    alpha = 0.5 * k * gamma * sigma**2
    eta = A * math.exp(-(1.0 + k / gamma) * math.log1p(gamma / k))
    q = np.arange(-q_max, q_max + 1, dtype=np.float64)
    m = np.diag(alpha * q**2)
    off = np.full(2 * q_max, -eta)
    return m + np.diag(off, 1) + np.diag(off, -1)


@dataclass(frozen=True)
class GLFTExact:
    """Finite-horizon solution on a time grid.

    ``v[i, q + q_max]`` is ``v_q(times_s[i])``; it may overflow for long horizons, so the
    quotes and the certainty equivalent use ``log_v``.
    """

    q_max: int
    times_s: FloatArray
    v: FloatArray
    log_v: FloatArray
    gamma: float
    k: float
    horizon_s: float

    def _log_v_at(self, t_s: float) -> FloatArray:
        if not 0.0 <= t_s <= self.horizon_s:
            raise ValueError(f"t_s={t_s} outside [0, {self.horizon_s}]")
        i = int(np.searchsorted(self.times_s, t_s, side="right")) - 1
        i = min(max(i, 0), len(self.times_s) - 2)
        t0, t1 = self.times_s[i], self.times_s[i + 1]
        w = (t_s - t0) / (t1 - t0)
        row: FloatArray = (1.0 - w) * self.log_v[i] + w * self.log_v[i + 1]
        return row

    def deltas(self, t_s: float, q: int) -> tuple[float | None, float | None]:
        """``(delta_bid, delta_ask)`` at time ``t_s``; ``None`` on the side that is closed."""
        if abs(q) > self.q_max:
            raise ValueError(f"|q|={abs(q)} exceeds q_max={self.q_max}")
        lv = self._log_v_at(t_s)
        c1 = glft_c1(self.gamma, self.k)
        j = q + self.q_max
        bid = None if q == self.q_max else (lv[j] - lv[j + 1]) / self.k + c1
        ask = None if q == -self.q_max else (lv[j] - lv[j - 1]) / self.k + c1
        return bid, ask


def glft_exact(
    gamma: float,
    sigma: float,
    A: float,
    k: float,
    q_max: int,
    horizon_s: float,
    n_t: int = 1001,
) -> GLFTExact:
    """Solve Proposition 2 by eigendecomposition of the symmetric matrix ``M``.

    ``v(t) = V exp(-L (T - t)) V' 1`` is evaluated in logs, factoring out the smallest
    eigenvalue, so long horizons neither overflow nor underflow.
    """
    if horizon_s <= 0 or n_t < 2:
        raise ValueError("need horizon_s > 0 and n_t >= 2")
    m = glft_matrix(gamma, sigma, A, k, q_max)
    lam, vec = np.linalg.eigh(m)
    proj = vec.T @ np.ones(m.shape[0])
    times = np.linspace(0.0, horizon_s, n_t)
    tau = horizon_s - times
    shifted = np.exp(-np.outer(tau, lam - lam[0]))  # (n_t, n) in (0, 1]
    w = shifted @ (vec * proj).T  # w[i] = V diag(shifted[i]) V' 1
    if np.any(w <= 0):
        raise FloatingPointError("non-positive v from the eigen solution; reduce the horizon")
    log_v = np.log(w) - np.outer(tau, np.full(m.shape[0], lam[0]))
    with np.errstate(over="ignore", under="ignore"):
        v = np.exp(log_v)
    return GLFTExact(
        q_max=q_max,
        times_s=times,
        v=v,
        log_v=log_v,
        gamma=gamma,
        k=k,
        horizon_s=horizon_s,
    )


def glft_certainty_equivalent(sol: GLFTExact, q0: int, x0: float, s0: float) -> float:
    """Certainty equivalent at ``t = 0``: ``x0 + q0 s0 + (1/k) ln v_q0(0)``.

    It equals ``-(1/gamma) ln(-u)`` for Theorem 1's value function
    ``u = -exp(-gamma (x0 + q0 s0)) v_q0(0)**(-gamma/k)``.
    """
    if abs(q0) > sol.q_max:
        raise ValueError("|q0| exceeds q_max")
    return x0 + q0 * s0 + float(sol.log_v[0, q0 + sol.q_max]) / sol.k
