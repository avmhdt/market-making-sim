"""Queue models from L2 data (FR-02, DQ-12; semantics in docs/simulator-contract.md).

- ``front``: we are always first in the queue.
- ``conservative``: queue ahead is the displayed size at entry and shrinks only with
  trades at our price.
- ``prob``: as conservative, and a cancellation of ``c`` lots at our level is ahead of us
  with probability ``f(ahead) / (f(ahead) + f(behind))``, ``f(x) = x**n`` (default n = 2),
  in which case ``ahead`` falls by ``c`` (floored at 0).
"""

from __future__ import annotations

import numpy as np

from mmsim.config import QueueModelConfig
from mmsim.sim.protocol import QueueModel

__all__ = ["ConservativeQueue", "FrontQueue", "ProbQueue", "make_queue"]


class FrontQueue:
    """Always first: trades at our price fill us at once."""

    def on_entry(self, level_lots: int) -> int:
        """Nothing ahead."""
        return 0

    def on_trade(self, ahead: int, traded_lots: int) -> tuple[int, int]:
        """All traded volume reaches us."""
        return 0, traded_lots

    def on_decrease(
        self, ahead: int, behind: int, cancelled_lots: int, rng: np.random.Generator
    ) -> int:
        """Cancellations never matter."""
        return 0


class ConservativeQueue:
    """Behind the whole displayed level; only trades move us forward."""

    def on_entry(self, level_lots: int) -> int:
        """Everything displayed is ahead."""
        return max(level_lots, 0)

    def on_trade(self, ahead: int, traded_lots: int) -> tuple[int, int]:
        """Trades consume the queue ahead first, then fill us."""
        consumed = min(ahead, traded_lots)
        return ahead - consumed, traded_lots - consumed

    def on_decrease(
        self, ahead: int, behind: int, cancelled_lots: int, rng: np.random.Generator
    ) -> int:
        """Cancellations are assumed to be behind us."""
        return ahead


class ProbQueue(ConservativeQueue):
    """Cancellations are ahead with probability ``f(ahead) / (f(ahead) + f(behind))``."""

    def __init__(self, power: float = 2.0) -> None:
        if power <= 0:
            raise ValueError("power must be > 0")
        self.power = power

    def on_decrease(
        self, ahead: int, behind: int, cancelled_lots: int, rng: np.random.Generator
    ) -> int:
        """Draw whether the cancellation was ahead of us."""
        if ahead <= 0 or cancelled_lots <= 0:
            return max(ahead, 0)
        fa = float(ahead) ** self.power
        fb = float(max(behind, 0)) ** self.power
        if rng.random() < fa / (fa + fb):
            return max(ahead - cancelled_lots, 0)
        return ahead


def make_queue(cfg: QueueModelConfig) -> QueueModel:
    """The queue model named by the config."""
    if cfg.kind == "front":
        return FrontQueue()
    if cfg.kind == "conservative":
        return ConservativeQueue()
    return ProbQueue(cfg.power)
