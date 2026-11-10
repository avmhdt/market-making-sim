"""Intensity A, k and volatility sigma estimation (FR-12).

The estimators arrive in M3; the parameter record is part of the interface (section 10)
because strategies receive it from M2 on.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CalibratedParams"]


@dataclass(frozen=True, slots=True)
class CalibratedParams:
    """One fit: intensity ``A * exp(-k * delta)`` and mid volatility ``sigma``.

    ``A`` is in fills per second at distance 0, ``k`` in 1 / quote-currency units and
    ``sigma`` in quote currency per square-root second. Only data before ``ts_fit`` was used.
    """

    ts_fit: int
    A: float
    k: float
    sigma: float
    r2: float
    n_orders: int
