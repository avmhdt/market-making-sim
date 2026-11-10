"""Turn a :class:`QuoteDecision` into submits and cancels (FR-11).

Per side: a working order whose price is within ``requote_threshold_ticks`` of the target
(``|delta| < threshold``; 0 or 1 both mean "only an exact match is kept") keeps its queue
position, unless the inventory limit no longer allows its size. Everything else on the side
is cancelled and, if the limit leaves room, one new order is sent. Orders whose cancel is
in flight still count against the limit (worst case).
"""

from __future__ import annotations

from dataclasses import dataclass

from mmsim.sim.protocol import MarketSimulator
from mmsim.sim.types import OpenOrder, Side
from mmsim.strategy.base import QuoteDecision
from mmsim.strategy.limits import ask_room, bid_room

__all__ = ["QuoteActions", "sync_quotes"]


@dataclass(frozen=True, slots=True)
class QuoteActions:
    """What one sync did."""

    submitted: tuple[int, ...]
    cancelled: tuple[int, ...]
    bid_lots: int
    ask_lots: int


def _side(
    sim: MarketSimulator,
    symbol: str,
    side: Side,
    target: int | None,
    lots: int,
    orders: list[OpenOrder],
    position: int,
    q_max_lots: int,
    threshold: int,
    submitted: list[int],
    cancelled: list[int],
) -> int:
    mine = [o for o in orders if o.side == side]
    live = [o for o in mine if o.status != "pending_cancel"]
    keep: OpenOrder | None = None
    if target is not None and lots > 0:
        near = [o for o in live if abs(o.price_ticks - target) < max(threshold, 1)]
        if len(near) == 1:
            keep = near[0]
    for order in live:
        if order is not keep:
            sim.cancel(order.order_id)
            cancelled.append(order.order_id)
    resting = sum(o.remaining_lots for o in mine)  # includes cancels in flight
    room_fn = bid_room if side == Side.BUY else ask_room
    if keep is not None:
        room_without = room_fn(position, resting - keep.remaining_lots, q_max_lots)
        if keep.remaining_lots <= room_without:
            return keep.remaining_lots
        sim.cancel(keep.order_id)
        cancelled.append(keep.order_id)
    if target is None or lots <= 0:
        return 0
    size = min(lots, room_fn(position, resting, q_max_lots))
    if size > 0:
        submitted.append(sim.submit_limit_ticks(symbol, side, target, size))
    return size


def sync_quotes(
    sim: MarketSimulator,
    decision: QuoteDecision,
    *,
    q_max_lots: int,
    requote_threshold_ticks: int = 1,
) -> QuoteActions:
    """Bring the working orders in line with ``decision`` under the inventory limit."""
    symbol = sim.instrument.symbol
    orders = sim.open_orders(symbol)
    position = sim.position_lots(symbol)
    submitted: list[int] = []
    cancelled: list[int] = []
    bid = _side(
        sim, symbol, Side.BUY, decision.bid_ticks, decision.bid_lots, orders, position,
        q_max_lots, requote_threshold_ticks, submitted, cancelled,
    )  # fmt: skip
    ask = _side(
        sim, symbol, Side.SELL, decision.ask_ticks, decision.ask_lots, orders, position,
        q_max_lots, requote_threshold_ticks, submitted, cancelled,
    )  # fmt: skip
    return QuoteActions(tuple(submitted), tuple(cancelled), bid, ask)
