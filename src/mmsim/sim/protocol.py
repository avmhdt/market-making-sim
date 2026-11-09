"""The simulator contract (spec section 10, FR-01).

Order, timer and query methods use the names and signatures of edbt API 1.0.0
(event-driven-backtester section 10, ``edbt.Context``); stepping wraps ``edbt.Backtest``.
Message types are this project's (:mod:`mmsim.sim.types`); the edbt adapter converts.
Semantics are documented in ``docs/simulator-contract.md``. Implementations that lack a
feature declare it through :meth:`MarketSimulator.capabilities` instead of changing this
Protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from mmsim.sim.types import BookView, InstrumentSpec, OpenOrder, Side, StepResult, TimeInForce

__all__ = ["CAPABILITIES", "LatencyModel", "MarketSimulator", "QueueModel"]

CAPABILITIES: frozenset[str] = frozenset({"l2", "queue_position", "latency", "fees", "timers"})
"""Every capability a full implementation declares."""


@runtime_checkable
class MarketSimulator(Protocol):
    """An event-driven exchange simulator driven one step at a time by the runner."""

    @property
    def instrument(self) -> InstrumentSpec:
        """The quoted instrument and fee tier."""
        ...

    def capabilities(self) -> frozenset[str]:
        """Subset of :data:`CAPABILITIES` this implementation supports."""
        ...

    def reset(self, *, seed: int) -> None:
        """Rewind to the start of the data and reseed every random stream."""
        ...

    def now(self) -> int:
        """Strategy-local time in ns (edbt ``Context.now``)."""
        ...

    def step(self) -> StepResult | None:
        """Advance to the next strategy-visible event; ``None`` once the data is exhausted."""
        ...

    def set_timer(self, *, at_ns: int | None = None, every_ns: int | None = None) -> int:
        """Schedule a one-off (``at_ns``) or periodic (``every_ns``) wake-up; returns its id."""
        ...

    def cancel_timer(self, timer_id: int) -> None:
        """Remove a timer."""
        ...

    def submit_limit_ticks(
        self,
        symbol: str,
        side: Side,
        price_ticks: int,
        qty_lots: int,
        *,
        tif: TimeInForce = TimeInForce.GTX,
        tag: str | None = None,
    ) -> int:
        """Send a limit order; returns its client order id. It reaches the exchange later."""
        ...

    def cancel(self, client_order_id: int) -> None:
        """Request a cancel; it reaches the exchange after the order latency."""
        ...

    def cancel_all(self, symbol: str | None = None) -> None:
        """Request a cancel of every working order (of ``symbol`` if given)."""
        ...

    def open_orders(self, symbol: str | None = None) -> list[OpenOrder]:
        """Orders the strategy believes are working, from the acks it has seen."""
        ...

    def position_lots(self, symbol: str) -> int:
        """Signed position in lots from the fills the strategy has seen."""
        ...

    def book(self, symbol: str) -> BookView:
        """The book as visible to the strategy now."""
        ...


class QueueModel(Protocol):
    """Queue position from L2 data (FR-02; semantics in docs/simulator-contract.md)."""

    def on_entry(self, level_lots: int) -> int:
        """Lots ahead of a new order joining a level that shows ``level_lots``."""
        ...

    def on_trade(self, ahead: int, traded_lots: int) -> tuple[int, int]:
        """Apply a trade at our level; returns ``(new ahead, lots filled to us)``."""
        ...

    def on_decrease(
        self, ahead: int, behind: int, cancelled_lots: int, rng: np.random.Generator
    ) -> int:
        """Apply a cancellation of ``cancelled_lots`` at our level; returns the new ahead."""
        ...


class LatencyModel(Protocol):
    """Feed, order and acknowledgement delays in ns (DQ-02)."""

    def feed_delay_ns(self, ts_exchange: int, ts_recv: int) -> int:
        """Delay between an exchange event and the moment the strategy sees it."""
        ...

    def order_delay_ns(self, ts_local: int) -> int:
        """Delay between a strategy decision and its arrival at the exchange."""
        ...

    def ack_delay_ns(self, ts_exchange: int) -> int:
        """Delay between an exchange-side order event and the strategy seeing it."""
        ...
