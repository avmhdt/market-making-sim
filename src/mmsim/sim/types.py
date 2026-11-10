"""Message types of the simulator contract (spec section 10, FR-01).

Every type here is a frozen, slotted dataclass. Prices are integer ticks and quantities
integer lots; timestamps are int64 nanoseconds UTC. ``ts_exchange`` is the simulated
exchange (matching) clock and ``ts_local`` the strategy's clock, which lags the exchange
by the feed latency (DQ-02). Strategies, the runner, analytics and RL code import only
this module and :mod:`mmsim.sim.protocol`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Literal

__all__ = [
    "BookView",
    "Fill",
    "InstrumentSpec",
    "OpenOrder",
    "OrderUpdate",
    "Side",
    "StepResult",
    "TimeInForce",
    "TradeEvent",
]


class Side(IntEnum):
    """Order side; same names and values as ``edbt.Side``."""

    BUY = 1
    """Bid side."""
    SELL = -1
    """Ask side."""


class TimeInForce(StrEnum):
    """Time in force; same names and values as ``edbt.TimeInForce``."""

    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    GTX = "gtx"
    """Post-only, the default for quotes."""


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """Static description of the quoted instrument, with the fee tier of the run.

    Fees are in basis points of notional; a negative ``maker_bps`` is a rebate.
    """

    venue: str
    symbol: str
    tick_size: float
    lot_size: float
    contract_type: Literal["perp", "spot", "equity"]
    maker_bps: float
    taker_bps: float
    fee_tier: str


@dataclass(frozen=True, slots=True)
class BookView:
    """Top-N book as visible to the strategy at ``ts_local``.

    Level 0 is the touch. Bid prices fall and ask prices rise with the level index.
    """

    ts_local: int
    bid_ticks: tuple[int, ...]
    bid_lots: tuple[int, ...]
    ask_ticks: tuple[int, ...]
    ask_lots: tuple[int, ...]

    @property
    def mid_x2(self) -> int:
        """Twice the mid price in ticks: ``bid_ticks[0] + ask_ticks[0]``.

        Raises:
            IndexError: If either side of the book is empty.
        """
        return self.bid_ticks[0] + self.ask_ticks[0]


@dataclass(frozen=True, slots=True)
class TradeEvent:
    """A public trade. ``side`` is +1 buyer-initiated, -1 seller-initiated, 0 unknown."""

    ts_exchange: int
    ts_local: int
    price_ticks: int
    qty_lots: int
    side: int


@dataclass(frozen=True, slots=True)
class Fill:
    """One (partial) fill of one of our orders. ``side`` is BUY when we bought.

    ``fee_quote`` is in quote currency: positive is a cost, negative a rebate.
    ``mid_x2_before`` is the exchange-time mid (bid + ask ticks) just before the event
    that triggered the fill (DQ-09).
    """

    ts_exchange: int
    ts_local: int
    order_id: int
    side: Side
    price_ticks: int
    qty_lots: int
    liquidity: Literal["maker", "taker"]
    fee_quote: float
    queue_ahead_lots_at_entry: int
    order_latency_ns: int
    mid_x2_before: int


@dataclass(frozen=True, slots=True)
class OrderUpdate:
    """A change in an order's status, as seen by the strategy at ``ts_local``."""

    ts_local: int
    order_id: int
    status: Literal["accepted", "rejected", "cancelled", "partially_filled", "filled"]
    reason: str | None


@dataclass(frozen=True, slots=True)
class OpenOrder:
    """An order the strategy believes is working."""

    order_id: int
    side: Side
    price_ticks: int
    remaining_lots: int
    status: str


@dataclass(frozen=True, slots=True)
class StepResult:
    """Everything that became visible to the strategy in one simulator step."""

    ts_local: int
    book: BookView | None
    """Set when the visible book changed."""
    trades: tuple[TradeEvent, ...]
    fills: tuple[Fill, ...]
    order_updates: tuple[OrderUpdate, ...]
    timer: bool
    """A scheduled wake-up fired."""
