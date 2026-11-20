"""Signal-skewed quotes (FR-16).

``y`` is the predicted mid change in ticks over the signal horizon ``h`` (seconds).

- ``reservation_shift``: both quotes move by ``alpha * y`` ticks (``alpha`` in [0, 1],
  tuned on the tuning days only).
- ``glft_drift``: GLFT with the drift ``mu = y * tick / h`` (price per second; Gueant,
  Lehalle and Fernandez-Tapia 2013, Proposition 4 and section 5.1). ``mu = 0`` reproduces
  the plain model exactly.
- Optional one-sided pull: when ``|y|`` exceeds ``pull_threshold_ticks``, the side the
  move would run over (the ask for an expected rise, the bid for a fall) is not quoted.

With ``y = 0`` (or no signal) every mode returns the base quotes unchanged.
"""

from __future__ import annotations

__all__ = ["drift_per_second", "pull_sides", "shift_distances"]


def shift_distances(
    d_bid: float | None, d_ask: float | None, y_ticks: float, alpha: float, tick: float
) -> tuple[float | None, float | None]:
    """Move both quotes up by ``alpha * y`` ticks: the bid comes closer, the ask moves out."""
    shift = alpha * y_ticks * tick
    return (
        None if d_bid is None else d_bid - shift,
        None if d_ask is None else d_ask + shift,
    )


def drift_per_second(y_ticks: float, tick: float, horizon_s: float) -> float:
    """``mu = y * tick / h`` in price per second."""
    if horizon_s <= 0:
        raise ValueError("signal horizon must be > 0")
    return y_ticks * tick / horizon_s


def pull_sides(y_ticks: float, threshold: float | None) -> tuple[bool, bool]:
    """``(quote_bid, quote_ask)`` after the optional one-sided pull."""
    if threshold is None or abs(y_ticks) <= threshold:
        return True, True
    return (y_ticks > 0, y_ticks < 0)
