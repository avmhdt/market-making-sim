"""A-S and GLFT formulas (FR-09, FR-10, FR-13) and AC-W-14."""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.linalg

from mmsim.models.avellaneda_stoikov import as_half_spread, as_quotes, as_reservation_price
from mmsim.models.glft import (
    glft_asymptotic,
    glft_c1,
    glft_c2,
    glft_certainty_equivalent,
    glft_exact,
    glft_matrix,
    with_fee,
)
from mmsim.validate import ASYMPTOTIC_SETS, asymptotic_gap


def test_as_quotes_by_hand() -> None:
    # gamma = 0.1, sigma = 2, tau = 0.5, k = 1.5, q = 2, s = 100
    r = 100 - 2 * 0.1 * 4 * 0.5
    spread = 0.1 * 4 * 0.5 + 20 * math.log(1 + 0.1 / 1.5)
    bid, ask = as_quotes(100.0, 2, 0.5, 0.1, 2.0, 1.5)
    assert as_reservation_price(100.0, 2, 0.5, 0.1, 2.0) == pytest.approx(r)
    assert as_half_spread(0.5, 0.1, 2.0, 1.5) == pytest.approx(spread / 2)
    assert bid == pytest.approx(r - spread / 2)
    assert ask == pytest.approx(r + spread / 2)
    with pytest.raises(ValueError):
        as_quotes(100.0, 0, 1.0, 0.0, 2.0, 1.5)


def test_glft_asymptotic_by_hand() -> None:
    g, s, A, k = 0.1, 2.0, 140.0, 1.5
    c1 = math.log(1 + g / k) / g
    c2 = math.sqrt(s**2 * g / (2 * k * A) * (1 + g / k) ** (1 + k / g))
    assert glft_c1(g, k) == pytest.approx(c1)
    assert glft_c2(g, s, A, k) == pytest.approx(c2)
    bid, ask = glft_asymptotic(3, g, s, A, k)
    assert bid == pytest.approx(c1 + 3.5 * c2)
    assert ask == pytest.approx(c1 - 2.5 * c2)
    with pytest.raises(ValueError):
        glft_asymptotic(0, g, -1.0, A, k)


def test_glft_skew_is_monotone_in_q() -> None:
    bids = [glft_asymptotic(q, 0.05, 1.0, 10.0, 2.0)[0] for q in range(-5, 6)]
    asks = [glft_asymptotic(q, 0.05, 1.0, 10.0, 2.0)[1] for q in range(-5, 6)]
    assert all(np.diff(bids) > 0)
    assert all(np.diff(asks) < 0)


def test_fee_shift() -> None:
    assert with_fee(10.0, 2.0, 0.0) == 10.0
    assert with_fee(10.0, 2.0, 0.5) == pytest.approx(10.0 * math.exp(-1.0))
    assert with_fee(10.0, 2.0, -0.5) > 10.0


def test_exact_matches_matrix_exponential() -> None:
    g, s, A, k, Q, T = 0.1, 2.0, 140.0, 1.5, 4, 1.0
    sol = glft_exact(g, s, A, k, Q, T, n_t=5)
    m = glft_matrix(g, s, A, k, Q)
    for i, t in enumerate(sol.times_s):
        ref = scipy.linalg.expm(-m * (T - t)) @ np.ones(2 * Q + 1)
        assert np.allclose(np.log(ref), sol.log_v[i], atol=1e-10)
    assert np.allclose(sol.v, np.exp(sol.log_v))
    assert sol.log_v[-1] == pytest.approx(np.zeros(2 * Q + 1))


def test_exact_quotes_and_bounds() -> None:
    sol = glft_exact(0.1, 2.0, 140.0, 1.5, 3, 1.0, n_t=11)
    bid, ask = sol.deltas(0.25, 0)
    assert bid is not None and ask is not None
    assert bid == pytest.approx(ask)  # symmetric at q = 0
    assert sol.deltas(0.0, 3)[0] is None
    assert sol.deltas(0.0, -3)[1] is None
    # At T the quotes are the inventory-free c1.
    b_T, a_T = sol.deltas(1.0, 1)
    assert b_T == pytest.approx(glft_c1(0.1, 1.5))
    assert a_T == pytest.approx(glft_c1(0.1, 1.5))
    with pytest.raises(ValueError):
        sol.deltas(1.5, 0)
    with pytest.raises(ValueError):
        sol.deltas(0.0, 4)
    with pytest.raises(ValueError):
        glft_exact(0.1, 2.0, 140.0, 1.5, 3, 0.0)
    with pytest.raises(ValueError):
        glft_matrix(0.1, 2.0, 140.0, 1.5, 0)


def test_certainty_equivalent_matches_value_function() -> None:
    sol = glft_exact(0.1, 2.0, 140.0, 1.5, 3, 1.0)
    ce = glft_certainty_equivalent(sol, 1, 5.0, 100.0)
    u = -math.exp(-0.1 * (5.0 + 100.0)) * sol.v[0, 4] ** (-0.1 / 1.5)
    assert -math.log(-u) / 0.1 == pytest.approx(ce)
    with pytest.raises(ValueError):
        glft_certainty_equivalent(sol, 4, 0.0, 100.0)


def test_asymptotic_limit() -> None:
    """AC-W-14: at a converged horizon, within 5% for |q| <= Q/2 on >= 3 sets."""
    gaps = [asymptotic_gap(s) for s in ASYMPTOTIC_SETS]
    inner = [g[1] for g in gaps]
    assert sum(x <= 0.05 for x in inner) >= 3
    # The gap near the bounds is larger and reported, not hidden.
    assert all(g[2] >= 0 for g in gaps)
