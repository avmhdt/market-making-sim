"""The simulator contract (AC-W-08): run against every available implementation."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Sequence

import numpy as np
import pytest

from mmsim.config import SimConfig
from mmsim.data.replay import BookBatchEvent, MarketEvent, TradeRowEvent
from mmsim.sim.fallback import FallbackSimulator
from mmsim.sim.latency import ConstantLatency
from mmsim.sim.protocol import CAPABILITIES, MarketSimulator
from mmsim.sim.registry import make_simulator
from mmsim.sim.types import InstrumentSpec, Side, StepResult, TimeInForce

INST = InstrumentSpec("synthetic", "SYN", 0.1, 1.0, "perp", 1.0, 5.0, "test")
_batch = itertools.count(1)


def book(
    t: int, rows: Sequence[tuple[int, int, int]], snap: bool = False, recv: int | None = None
) -> BookBatchEvent:
    """rows: (side, price_ticks, lots)."""
    b = next(_batch)
    return BookBatchEvent(
        ts_match=t, ts_exchange=t, ts_recv=t if recv is None else recv, batch_id=b, seq=b,
        is_snapshot=snap,
        side=np.array([r[0] for r in rows], dtype=np.int8),
        price_ticks=np.array([r[1] for r in rows], dtype=np.int64),
        qty_lots=np.array([r[2] for r in rows], dtype=np.int64),
    )  # fmt: skip


def trade(t: int, price: int, qty: int, side: int, recv: int | None = None) -> TradeRowEvent:
    return TradeRowEvent(t, t, t if recv is None else recv, str(t), price, qty, side)


SNAP = [(1, 99, 10), (1, 98, 10), (-1, 101, 10), (-1, 102, 10)]

Factory = Callable[..., MarketSimulator]


def fallback_factory(
    events: list[MarketEvent],
    *,
    queue: str = "front",
    order_ns: int = 0,
    ack_ns: int = 0,
    feed_ns: int | None = 0,
) -> MarketSimulator:
    cfg = SimConfig.model_validate({"queue_model": {"kind": queue}})
    lat = ConstantLatency(order_ns=order_ns, ack_ns=ack_ns, feed_ns=feed_ns)

    def source() -> Iterator[MarketEvent]:
        return iter(events)

    return FallbackSimulator(source, INST, cfg, lat, book_depth=5)


IMPLEMENTATIONS: list[Factory] = [fallback_factory]


def drain(sim: MarketSimulator) -> list[StepResult]:
    out = []
    while (step := sim.step()) is not None:
        out.append(step)
    return out


def step_until(sim: MarketSimulator, t: int) -> list[StepResult]:
    """Steps while the next visible time is <= t (consumes one step past it)."""
    out = []
    while (step := sim.step()) is not None:
        out.append(step)
        if step.ts_local >= t:
            break
    return out


@pytest.fixture(params=IMPLEMENTATIONS, ids=["fallback"])
def factory(request: pytest.FixtureRequest) -> Factory:
    f: Factory = request.param
    return f


def test_capabilities_and_protocol(factory: Factory) -> None:
    sim = factory([book(0, SNAP, True)])
    assert isinstance(sim, MarketSimulator)
    assert sim.capabilities() <= CAPABILITIES
    assert sim.instrument == INST
    with pytest.raises(ValueError):
        sim.position_lots("OTHER")


def test_post_only_reject_and_accept(factory: Factory) -> None:
    sim = factory([book(0, SNAP, True), book(50, [(1, 99, 11)])])
    first = sim.step()
    assert first is not None and first.book is not None
    bad = sim.submit_limit_ticks("SYN", Side.BUY, 101, 1)  # would cross the ask
    good = sim.submit_limit_ticks("SYN", Side.SELL, 103, 2)
    assert {o.order_id for o in sim.open_orders("SYN")} == {bad, good}
    updates = [u for s in drain(sim) for u in s.order_updates]
    statuses = {u.order_id: u.status for u in updates}
    assert statuses == {bad: "rejected", good: "accepted"}
    assert [u.reason for u in updates if u.order_id == bad] == ["post_only_would_cross"]
    assert [o.order_id for o in sim.open_orders()] == [good]
    with pytest.raises(NotImplementedError):
        sim.submit_limit_ticks("SYN", Side.BUY, 90, 1, tif=TimeInForce.GTC)
    with pytest.raises(ValueError):
        sim.submit_limit_ticks("SYN", Side.BUY, 90, 0)


def test_fill_fields_latency_and_fees(factory: Factory) -> None:
    events = [book(0, SNAP, True), trade(5, 99, 3, -1), trade(30, 99, 3, -1)]
    sim = factory(events, order_ns=10, ack_ns=4)
    sim.step()
    oid = sim.submit_limit_ticks("SYN", Side.BUY, 99, 2)
    steps = drain(sim)
    fills = [f for s in steps for f in s.fills]
    # The trade at 5 happened before the order arrived at 10: no fill from it.
    assert len(fills) == 1
    f = fills[0]
    assert (f.order_id, f.side, f.price_ticks, f.qty_lots) == (oid, Side.BUY, 99, 2)
    assert f.ts_exchange == 30 and f.ts_local == 34
    assert f.order_latency_ns == 10
    assert f.mid_x2_before == 200
    assert f.queue_ahead_lots_at_entry == 0
    assert f.liquidity == "maker"
    assert f.fee_quote == pytest.approx(99 * 0.1 * 2 * 1.0 * 1.0 / 1e4)
    accepted = [u for s in steps for u in s.order_updates if u.status == "accepted"]
    assert accepted[0].ts_local == 14
    assert sim.position_lots("SYN") == 2
    assert sim.open_orders() == []


def test_cancel_prevents_later_fill(factory: Factory) -> None:
    sim = factory([book(0, SNAP, True), trade(20, 99, 1, -1)], order_ns=5)
    sim.step()
    oid = sim.submit_limit_ticks("SYN", Side.BUY, 99, 1)
    sim.cancel(oid)
    sim.cancel(oid)  # a second cancel is ignored
    steps = drain(sim)
    assert [f for s in steps for f in s.fills] == []
    assert [u.status for s in steps for u in s.order_updates] == ["accepted", "cancelled"]
    assert sim.open_orders() == []


def test_trade_through_fills_whole_order_and_queue_counts(factory: Factory) -> None:
    events = [book(0, SNAP, True), trade(10, 99, 4, -1), trade(20, 98, 1, -1)]
    sim = factory(events, queue="conservative")
    sim.step()
    oid = sim.submit_limit_ticks("SYN", Side.BUY, 99, 5)
    fills = [f for s in drain(sim) for f in s.fills]
    # 10 displayed ahead; 4 traded at our level leaves 6 ahead, no fill; the trade
    # through at 98 fills all 5.
    assert [(f.order_id, f.qty_lots, f.queue_ahead_lots_at_entry) for f in fills] == [(oid, 5, 10)]


def test_traded_volume_is_netted_before_cancellations(factory: Factory) -> None:
    """DQ-12: a depth decrease explained by trades is not a cancellation."""
    events = [
        book(0, SNAP, True),
        trade(10, 99, 4, -1),
        book(11, [(1, 99, 6)]),  # the 4 traded lots, reported late: ahead stays 6
        book(12, [(1, 99, 2)]),  # 4 cancelled, all ahead (nothing behind): ahead -> 2
        trade(20, 99, 3, -1),  # 2 to the queue, 1 to us
    ]
    sim = factory(events, queue="prob")
    sim.step()
    sim.submit_limit_ticks("SYN", Side.BUY, 99, 5)
    fills = [f for s in drain(sim) for f in s.fills]
    assert [f.qty_lots for f in fills] == [1]


def test_side_zero_trades_never_fill(factory: Factory) -> None:
    sim = factory([book(0, SNAP, True), trade(10, 98, 5, 0)])
    sim.step()
    sim.submit_limit_ticks("SYN", Side.BUY, 99, 1)
    steps = drain(sim)
    assert [f for s in steps for f in s.fills] == []
    assert [t.side for s in steps for t in s.trades] == [0]


def test_timers(factory: Factory) -> None:
    sim = factory([book(0, SNAP, True), book(100, [(1, 97, 1)])])
    sim.step()
    sim.set_timer(every_ns=30)
    sim.set_timer(at_ns=45)
    fired = [s.ts_local for s in drain(sim) if s.timer]
    assert fired == [30, 45, 60, 90]
    sim.reset(seed=0)
    sim.step()
    tid = sim.set_timer(every_ns=30)
    sim.cancel_timer(tid)
    assert [s.ts_local for s in drain(sim) if s.timer] == []
    with pytest.raises(ValueError):
        sim.set_timer()
    with pytest.raises(ValueError):
        sim.set_timer(every_ns=0)
    with pytest.raises(ValueError):
        sim.set_timer(at_ns=-1)


def test_determinism(factory: Factory) -> None:
    events = [book(0, SNAP, True)]
    events += [book(10 * i, [(1, 99, 10 - i % 3)]) for i in range(1, 30)]
    events += [trade(10 * i + 5, 99, 2, -1) for i in range(1, 30)]
    events.sort(key=lambda e: (e.ts_match, isinstance(e, BookBatchEvent)))

    def run(seed: int) -> list[tuple[int, int, int]]:
        sim = factory(events, queue="prob", order_ns=3, ack_ns=3)
        sim.reset(seed=seed)
        sim.step()
        sim.submit_limit_ticks("SYN", Side.BUY, 99, 50)
        return [(f.ts_exchange, f.qty_lots, f.order_id) for s in drain(sim) for f in s.fills]

    assert run(1) == run(1)


def test_no_look_ahead_sentinel(factory: Factory) -> None:
    """NFR-13: an event is never visible before ts_exchange + feed delay."""
    sentinel_price = 123
    events = [
        book(0, SNAP, True, recv=7),
        trade(50, sentinel_price, 777, 1, recv=57),
        book(60, [(-1, 101, 1)], recv=67),
    ]
    sim = factory(events, feed_ns=None)  # recorded feed latency: ts_recv - ts_exchange
    seen_at = None
    while (step := sim.step()) is not None:
        for t in step.trades:
            if t.qty_lots == 777:
                seen_at = step.ts_local
        if step.ts_local < 57:
            assert all(t.qty_lots != 777 for t in step.trades)
        assert sim.now() == step.ts_local
    assert seen_at == 57
    assert sim.book("SYN").ts_local == 67


def test_registry_builds_fallback_and_refuses_edbt() -> None:
    cfg = SimConfig.model_validate(
        {"latency": {"order": {"kind": "constant", "ns": 1}, "ack": {"kind": "constant", "ns": 1}}}
    )
    sim = make_simulator(cfg, lambda: iter([book(0, SNAP, True)]), INST)
    assert isinstance(sim, FallbackSimulator)
    with pytest.raises(NotImplementedError):
        make_simulator(cfg, lambda: iter([]), INST, "edbt")
    with pytest.raises(NotImplementedError):
        FallbackSimulator(
            lambda: iter([]), INST, SimConfig.model_validate({"post_only": False}),
            ConstantLatency(0, 0),
        )  # fmt: skip
    assert sim.exchange_mid_x2 is None
    sim.step()
    assert sim.exchange_mid_x2 == 200
    assert sim.exchange_position_lots == 0
    sim.cancel_all("SYN")
