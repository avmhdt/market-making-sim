"""Avellaneda and Stoikov (2008) quotes (FR-09).

Avellaneda, M. and Stoikov, S. (2008). High-frequency trading in a limit order book.
Quantitative Finance 8(3), 217-224. Section 3.2:

- reservation price, eq. (29): ``r(s, t) = s - q * gamma * sigma**2 * (T - t)``
- total spread, eq. (30): ``delta_a + delta_b = gamma * sigma**2 * (T - t)
  + (2 / gamma) * ln(1 + gamma / k)``, quoted symmetrically around ``r``.

``tau_s`` is ``T - t`` for a finite horizon or a constant for the rolling variant used on
24/7 markets. Units: ``sigma`` in price per square-root second, ``k`` in 1/price.
"""

from __future__ import annotations

import math

__all__ = ["as_half_spread", "as_quotes", "as_reservation_price"]


def as_reservation_price(mid: float, q: float, tau_s: float, gamma: float, sigma: float) -> float:
    """Eq. (29): ``s - q * gamma * sigma**2 * tau``."""
    return mid - q * gamma * sigma**2 * tau_s


def as_half_spread(tau_s: float, gamma: float, sigma: float, k: float) -> float:
    """Half of eq. (30)'s total spread."""
    return 0.5 * (gamma * sigma**2 * tau_s + (2.0 / gamma) * math.log1p(gamma / k))


def as_quotes(
    mid: float, q: float, tau_s: float, gamma: float, sigma: float, k: float
) -> tuple[float, float]:
    """``(bid_price, ask_price)`` from eqs. (29) and (30)."""
    if gamma <= 0 or k <= 0 or sigma < 0 or tau_s < 0:
        raise ValueError("need gamma > 0, k > 0, sigma >= 0, tau >= 0")
    r = as_reservation_price(mid, q, tau_s, gamma, sigma)
    h = as_half_spread(tau_s, gamma, sigma, k)
    return r - h, r + h
