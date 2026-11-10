"""Policy interface: what a quoting strategy sees and returns (spec section 10)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mmsim.models.calibration import CalibratedParams
from mmsim.sim.types import BookView

__all__ = ["QuoteDecision", "QuotePolicy", "StrategyState"]


@dataclass(frozen=True, slots=True)
class QuoteDecision:
    """Desired quotes; ``None`` means do not quote that side."""

    bid_ticks: int | None
    ask_ticks: int | None
    bid_lots: int
    ask_lots: int


@dataclass(frozen=True, slots=True)
class StrategyState:
    """Everything a policy may use at ``ts_local`` (no look-ahead, NFR-13)."""

    ts_local: int
    book: BookView
    inventory_lots: int
    q_max_lots: int
    params: CalibratedParams
    signal_ticks: float | None
    t_remaining_s: float | None


class QuotePolicy(Protocol):
    """Maps a state to a quote decision."""

    name: str

    def decide(self, state: StrategyState) -> QuoteDecision:
        """The quotes to hold now."""
        ...
