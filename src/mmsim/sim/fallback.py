"""Pure-Python fallback simulator implementing :class:`MarketSimulator` (FR-02).

One timeline carries every event. Market events happen at their matching time (the
exchange clock); the strategy sees them after the feed delay, its orders reach the
exchange after the order delay, and exchange-side order events (acks, fills) reach it
after the ack delay. At equal times the order is: market event, order or cancel arrival,
strategy-visible delivery, timer. All strategy-visible items with the same time form one
:class:`StepResult`.

Matching (docs/simulator-contract.md): resting quotes fill only against public trades.
A trade priced through our price fills the whole remaining quantity; a trade at our price
goes through the queue model. Depth decreases at our level are netted against volume
already traded there before counting as cancellations (DQ-12). Post-only orders that would
cross the exchange book on arrival are rejected (DQ-11).
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from mmsim.config import SimConfig
from mmsim.data.replay import BookBatchEvent, MarketEvent, TradeRowEvent
from mmsim.sim.book import L2Book
from mmsim.sim.fees import fill_fee_quote
from mmsim.sim.protocol import CAPABILITIES, LatencyModel, QueueModel
from mmsim.sim.queue import make_queue
from mmsim.sim.types import (
    BookView,
    Fill,
    InstrumentSpec,
    OpenOrder,
    OrderUpdate,
    Side,
    StepResult,
    TimeInForce,
    TradeEvent,
)

__all__ = ["EventSource", "FallbackSimulator", "SimStats"]

EventSource = Callable[[], Iterator[MarketEvent]]
"""Returns a fresh iterator over the merged event stream (called on every reset)."""

MidObserver = Callable[[int, int, int], None]
"""Called with ``(ts_exchange, mid_x2, spread_ticks)`` on every top-of-book change."""

_P_EXCH, _P_VISIBLE, _P_TIMER = 1, 2, 3


@dataclass(slots=True)
class SimStats:
    """Counters reported in ``metrics.json``."""

    market_events: int = 0
    crossed_book_events: int = 0
    side0_trades: int = 0
    orders_submitted: int = 0
    orders_rejected: int = 0
    cancels_sent: int = 0
    fills: int = 0


@dataclass(slots=True)
class _ExchOrder:
    order_id: int
    side: int
    price: int
    remaining: int
    submit_local: int
    arrive_ts: int = 0
    ahead: int = 0
    ahead_at_entry: int = 0


@dataclass(slots=True)
class _StratOrder:
    order_id: int
    side: int
    price: int
    remaining: int
    status: str


@dataclass(slots=True)
class _Pending:
    book: BookView | None = None
    trades: list[TradeEvent] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    updates: list[OrderUpdate] = field(default_factory=list)
    timer: bool = False


class FallbackSimulator:
    """The reference implementation of the simulator contract."""

    def __init__(
        self,
        events: EventSource,
        instrument: InstrumentSpec,
        cfg: SimConfig,
        latency: LatencyModel,
        *,
        book_depth: int = 10,
        start_ns: int | None = None,
        on_mid: MidObserver | None = None,
        queue: QueueModel | None = None,
        seed: int = 0,
    ) -> None:
        if not cfg.post_only:
            raise NotImplementedError("the fallback simulates post-only quotes only")
        self._events_fn = events
        self._instrument = instrument
        self._cfg = cfg
        self._latency = latency
        self._queue: QueueModel = queue if queue is not None else make_queue(cfg.queue_model)
        self._depth = book_depth
        self._start_ns = start_ns
        self._on_mid = on_mid
        self.reset(seed=seed)

    # ---- contract -----------------------------------------------------------------------

    @property
    def instrument(self) -> InstrumentSpec:
        """The quoted instrument and fee tier."""
        return self._instrument

    def capabilities(self) -> frozenset[str]:
        """All capabilities."""
        return CAPABILITIES

    def reset(self, *, seed: int) -> None:
        """Rewind the data and reseed the queue model's random stream."""
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._market: Iterator[MarketEvent] = self._events_fn()
        self._peek: MarketEvent | None = next(self._market, None)
        self._heap: list[tuple[int, int, int, str, Any]] = []
        self._seq = itertools.count()
        self._nontimer = 0
        self._book = L2Book()
        self._exch_orders: dict[int, _ExchOrder] = {}
        self._strat_orders: dict[int, _StratOrder] = {}
        self._traded: dict[tuple[int, int], int] = {}
        self._timers: dict[int, int | None] = {}
        self._timer_ids = itertools.count(1)
        self._order_ids = itertools.count(1)
        self._visible_book = BookView(0, (), (), (), ())
        self._position_seen = 0
        self._position_exch = 0
        self._mid_x2: int | None = None
        self._spread: int | None = None
        self.stats = SimStats()
        first = self._peek.ts_match if self._peek is not None else 0
        self._now = self._start_ns if self._start_ns is not None else first

    def now(self) -> int:
        """Strategy-local time of the last step."""
        return self._now

    def step(self) -> StepResult | None:
        """Process events until something becomes visible to the strategy."""
        while True:
            if self._peek is None and self._nontimer == 0:
                return None
            top = self._heap[0] if self._heap else None
            t_market = self._peek.ts_match if self._peek is not None else None
            if top is not None and (t_market is None or (top[0], top[1]) < (t_market, 0)):
                if top[1] == _P_EXCH:
                    heapq.heappop(self._heap)
                    self._nontimer -= 1
                    self._exchange_item(top[0], top[3], top[4])
                    continue
                result = self._deliver(top[0])
                if result is not None:
                    return result
                continue
            assert self._peek is not None
            event = self._peek
            self._peek = next(self._market, None)
            self._market_event(event)

    def set_timer(self, *, at_ns: int | None = None, every_ns: int | None = None) -> int:
        """One-off at ``at_ns`` or periodic every ``every_ns`` (first at ``at_ns`` if given)."""
        if at_ns is None and every_ns is None:
            raise ValueError("need at_ns or every_ns")
        if every_ns is not None and every_ns <= 0:
            raise ValueError("every_ns must be > 0")
        first = at_ns if at_ns is not None else self._now + (every_ns or 0)
        if first < self._now:
            raise ValueError("timer in the past")
        timer_id = next(self._timer_ids)
        self._timers[timer_id] = every_ns
        heapq.heappush(self._heap, (first, _P_TIMER, next(self._seq), "timer", timer_id))
        return timer_id

    def cancel_timer(self, timer_id: int) -> None:
        """Remove a timer (its queued wake-up is dropped)."""
        self._timers.pop(timer_id, None)

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
        """Send a post-only limit order; it arrives after the order delay."""
        self._check_symbol(symbol)
        if tif != TimeInForce.GTX:
            raise NotImplementedError("the fallback supports post-only (GTX) quotes only")
        if qty_lots <= 0:
            raise ValueError("qty_lots must be > 0")
        order_id = next(self._order_ids)
        side_i = int(side)
        self._strat_orders[order_id] = _StratOrder(
            order_id, side_i, price_ticks, qty_lots, "pending_new"
        )
        arrive = self._now + self._latency.order_delay_ns(self._now)
        order = _ExchOrder(order_id, side_i, price_ticks, qty_lots, self._now)
        self._push(arrive, _P_EXCH, "arrive", order)
        self.stats.orders_submitted += 1
        return order_id

    def cancel(self, client_order_id: int) -> None:
        """Request a cancel; it arrives after the order delay."""
        order = self._strat_orders.get(client_order_id)
        if order is None or order.status == "pending_cancel":
            return
        order.status = "pending_cancel"
        arrive = self._now + self._latency.order_delay_ns(self._now)
        self._push(arrive, _P_EXCH, "cancel", client_order_id)
        self.stats.cancels_sent += 1

    def cancel_all(self, symbol: str | None = None) -> None:
        """Cancel every working order."""
        if symbol is not None:
            self._check_symbol(symbol)
        for order_id in list(self._strat_orders):
            self.cancel(order_id)

    def open_orders(self, symbol: str | None = None) -> list[OpenOrder]:
        """Working orders as the strategy knows them, including pending ones."""
        if symbol is not None:
            self._check_symbol(symbol)
        return [
            OpenOrder(o.order_id, Side(o.side), o.price, o.remaining, o.status)
            for o in self._strat_orders.values()
        ]

    def position_lots(self, symbol: str) -> int:
        """Position from the fills the strategy has seen."""
        self._check_symbol(symbol)
        return self._position_seen

    def book(self, symbol: str) -> BookView:
        """The last book the strategy has seen."""
        self._check_symbol(symbol)
        return self._visible_book

    # ---- extras (not part of the Protocol) ----------------------------------------------

    @property
    def exchange_mid_x2(self) -> int | None:
        """Current exchange-side mid (for end-of-run marking)."""
        return self._mid_x2

    @property
    def exchange_position_lots(self) -> int:
        """Position at the exchange, including fills not yet seen by the strategy."""
        return self._position_exch

    # ---- internals ----------------------------------------------------------------------

    def _check_symbol(self, symbol: str) -> None:
        if symbol != self._instrument.symbol:
            raise ValueError(f"unknown symbol {symbol!r}")

    def _push(self, t: int, prio: int, kind: str, payload: Any) -> None:
        heapq.heappush(self._heap, (t, prio, next(self._seq), kind, payload))
        if prio != _P_TIMER:
            self._nontimer += 1

    def _visible_at(self, t_match: int, ts_exchange: int, ts_recv: int) -> int:
        return max(t_match, ts_exchange + self._latency.feed_delay_ns(ts_exchange, ts_recv))

    def _market_event(self, event: MarketEvent) -> None:
        self.stats.market_events += 1
        t = event.ts_match
        if isinstance(event, BookBatchEvent):
            self._book_batch(event, t)
        else:
            self._trade(event, t)

    def _book_batch(self, event: BookBatchEvent, t: int) -> None:
        changes = self._book.apply(event.side, event.price_ticks, event.qty_lots, event.is_snapshot)
        for side, price, old, new in changes:
            traded = self._traded.pop((side, price), 0)
            if new >= old:
                continue
            cancelled = max(old - new - traded, 0)
            for order in self._exch_orders.values():
                if order.side != side or order.price != price:
                    continue
                if cancelled:
                    behind = max(old - order.ahead, 0)
                    order.ahead = self._queue.on_decrease(order.ahead, behind, cancelled, self._rng)
                order.ahead = min(order.ahead, new)
        mid = self._book.mid_x2()
        if mid is not None and (mid != self._mid_x2 or self._book_spread() != self._spread):
            self._mid_x2 = mid
            self._spread = self._book_spread()
            if self._on_mid is not None:
                self._on_mid(t, mid, self._spread)
        if self._book.crossed():
            self.stats.crossed_book_events += 1
            return
        visible = self._visible_at(t, event.ts_exchange, event.ts_recv)
        self._push(visible, _P_VISIBLE, "book", self._book.view(visible, self._depth))

    def _book_spread(self) -> int:
        bid, ask = self._book.best_bid, self._book.best_ask
        assert bid is not None and ask is not None
        return ask - bid

    def _trade(self, event: TradeRowEvent, t: int) -> None:
        visible = self._visible_at(t, event.ts_exchange, event.ts_recv)
        trade = TradeEvent(
            event.ts_exchange, visible, event.price_ticks, event.qty_lots, event.side
        )
        if event.side == 0:
            self.stats.side0_trades += 1
            self._push(visible, _P_VISIBLE, "trade", trade)
            return
        mid_before = self._book.mid_x2()
        if mid_before is None:
            mid_before = self._mid_x2 if self._mid_x2 is not None else 0
        passive = -event.side
        price = event.price_ticks
        hits = sorted(
            (o for o in self._exch_orders.values() if o.side == passive),
            key=lambda o: (-o.price * passive, o.arrive_ts, o.order_id),
        )
        for order in hits:
            through = order.price > price if passive == 1 else order.price < price
            if through:
                filled = order.remaining
            elif order.price == price:
                order.ahead, reached = self._queue.on_trade(order.ahead, event.qty_lots)
                filled = min(reached, order.remaining)
            else:
                continue
            if filled > 0:
                self._fill(order, filled, t, mid_before)
        key = (passive, price)
        self._traded[key] = self._traded.get(key, 0) + event.qty_lots
        self._push(visible, _P_VISIBLE, "trade", trade)

    def _fill(self, order: _ExchOrder, qty: int, t: int, mid_before: int) -> None:
        order.remaining -= qty
        self._position_exch += order.side * qty
        self.stats.fills += 1
        visible = t + self._latency.ack_delay_ns(t)
        fill = Fill(
            ts_exchange=t,
            ts_local=visible,
            order_id=order.order_id,
            side=Side(order.side),
            price_ticks=order.price,
            qty_lots=qty,
            liquidity="maker",
            fee_quote=fill_fee_quote(self._instrument, order.price, qty),
            queue_ahead_lots_at_entry=order.ahead_at_entry,
            order_latency_ns=order.arrive_ts - order.submit_local,
            mid_x2_before=mid_before,
        )
        self._push(visible, _P_VISIBLE, "fill", fill)
        status: Literal["partially_filled", "filled"] = (
            "filled" if order.remaining == 0 else "partially_filled"
        )
        self._push(
            visible, _P_VISIBLE, "update", OrderUpdate(visible, order.order_id, status, None)
        )
        if order.remaining == 0:
            del self._exch_orders[order.order_id]

    def _exchange_item(self, t: int, kind: str, payload: Any) -> None:
        visible = t + self._latency.ack_delay_ns(t)
        if kind == "arrive":
            order: _ExchOrder = payload
            order.arrive_ts = t
            if order.side == 1:
                best = self._book.best_ask
                crosses = best is not None and order.price >= best
            else:
                best = self._book.best_bid
                crosses = best is not None and order.price <= best
            if crosses:
                self.stats.orders_rejected += 1
                update = OrderUpdate(visible, order.order_id, "rejected", "post_only_would_cross")
            else:
                order.ahead = self._queue.on_entry(self._book.size(order.side, order.price))
                order.ahead_at_entry = order.ahead
                self._exch_orders[order.order_id] = order
                update = OrderUpdate(visible, order.order_id, "accepted", None)
            self._push(visible, _P_VISIBLE, "update", update)
            return
        order_id: int = payload
        if self._exch_orders.pop(order_id, None) is not None:
            self._push(
                visible, _P_VISIBLE, "update", OrderUpdate(visible, order_id, "cancelled", None)
            )

    def _deliver(self, t: int) -> StepResult | None:
        pending = _Pending()
        while self._heap and self._heap[0][0] == t and self._heap[0][1] >= _P_VISIBLE:
            _, prio, _, kind, payload = heapq.heappop(self._heap)
            if prio == _P_TIMER:
                if payload not in self._timers:
                    continue
                pending.timer = True
                every = self._timers[payload]
                if every is None:
                    del self._timers[payload]
                else:
                    heapq.heappush(
                        self._heap, (t + every, _P_TIMER, next(self._seq), "timer", payload)
                    )
                continue
            self._nontimer -= 1
            if kind == "book":
                pending.book = payload
                self._visible_book = payload
            elif kind == "trade":
                pending.trades.append(payload)
            elif kind == "fill":
                self._apply_fill_seen(payload)
                pending.fills.append(payload)
            else:
                self._apply_update_seen(payload)
                pending.updates.append(payload)
        self._now = t
        if not (
            pending.book or pending.trades or pending.fills or pending.updates or pending.timer
        ):
            return None
        return StepResult(
            ts_local=t,
            book=pending.book,
            trades=tuple(pending.trades),
            fills=tuple(pending.fills),
            order_updates=tuple(pending.updates),
            timer=pending.timer,
        )

    def _apply_fill_seen(self, fill: Fill) -> None:
        self._position_seen += int(fill.side) * fill.qty_lots
        order = self._strat_orders.get(fill.order_id)
        if order is not None:
            order.remaining -= fill.qty_lots
            if order.remaining <= 0:
                del self._strat_orders[fill.order_id]

    def _apply_update_seen(self, update: OrderUpdate) -> None:
        order = self._strat_orders.get(update.order_id)
        if order is None:
            return
        if update.status in ("rejected", "cancelled"):
            del self._strat_orders[update.order_id]
        elif update.status == "accepted" and order.status == "pending_new":
            order.status = "accepted"
