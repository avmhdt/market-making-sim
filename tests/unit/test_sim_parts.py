"""Book, queue models, latency and fees on scripted inputs."""

from __future__ import annotations

import numpy as np
import pytest

from mmsim.config import LatencyConfig, QueueModelConfig
from mmsim.sim.book import L2Book
from mmsim.sim.fees import fill_fee_quote
from mmsim.sim.latency import ConstantLatency, make_latency, median_feed_latency_ns
from mmsim.sim.queue import ConservativeQueue, FrontQueue, ProbQueue, make_queue
from mmsim.sim.types import InstrumentSpec


def arr(*v: int) -> np.ndarray:
    return np.array(v, dtype=np.int64)


def test_book_apply_snapshot_and_diffs() -> None:
    book = L2Book()
    changes = book.apply(np.array([1, 1, -1], np.int8), arr(99, 98, 101), arr(5, 7, 3), True)
    assert sorted(changes) == [(-1, 101, 0, 3), (1, 98, 0, 7), (1, 99, 0, 5)]
    assert (book.best_bid, book.best_ask, book.mid_x2()) == (99, 101, 200)
    changes = book.apply(np.array([1, -1], np.int8), arr(99, 100), arr(0, 2), False)
    assert changes == [(1, 99, 5, 0), (-1, 100, 0, 2)]
    assert book.best_bid == 98
    assert book.size(-1, 100) == 2
    view = book.view(7, 1)
    assert view.bid_ticks == (98,) and view.ask_ticks == (100,) and view.ts_local == 7
    changes = book.apply(np.array([1], np.int8), arr(97), arr(1), True)
    assert (1, 98, 7, 0) in changes and (-1, 100, 2, 0) in changes
    assert book.best_ask is None and book.mid_x2() is None
    assert not book.crossed()
    book.set_level(-1, 97, 1)
    assert book.crossed()
    assert book.view(0, 0).bid_ticks == ()


def test_queue_models() -> None:
    rng = np.random.Generator(np.random.PCG64(0))
    front, cons, prob = FrontQueue(), ConservativeQueue(), ProbQueue(2.0)
    assert front.on_entry(50) == 0
    assert front.on_trade(0, 3) == (0, 3)
    assert front.on_decrease(0, 10, 5, rng) == 0
    assert cons.on_entry(50) == 50
    assert cons.on_trade(50, 20) == (30, 0)
    assert cons.on_trade(10, 25) == (0, 15)
    assert cons.on_decrease(10, 40, 5, rng) == 10
    # All ahead, nothing behind: the cancellation must be ahead of us.
    assert prob.on_decrease(10, 0, 4, rng) == 6
    # Nothing ahead: nothing changes.
    assert prob.on_decrease(0, 10, 4, rng) == 0
    draws = [prob.on_decrease(10, 10, 4, rng) for _ in range(2000)]
    assert 0.4 < np.mean([d == 6 for d in draws]) < 0.6
    with pytest.raises(ValueError):
        ProbQueue(0.0)
    assert isinstance(make_queue(QueueModelConfig(kind="front")), FrontQueue)
    assert isinstance(make_queue(QueueModelConfig(kind="conservative")), ConservativeQueue)
    assert isinstance(make_queue(QueueModelConfig(kind="prob", power=3.0)), ProbQueue)


def test_latency_models() -> None:
    lat = ConstantLatency(order_ns=10, ack_ns=20, feed_ns=None, feed_extra_ns=5)
    assert lat.feed_delay_ns(100, 130) == 35
    assert lat.feed_delay_ns(100, 90) == 5  # negative recorded latency clamps to 0
    assert lat.order_delay_ns(0) == 10 and lat.ack_delay_ns(0) == 20
    assert median_feed_latency_ns([0, 0, 0], [5, 7, 100]) == 7
    assert median_feed_latency_ns([], []) == 0
    cfg = LatencyConfig.model_validate(
        {"order": {"kind": "median_feed"}, "ack": {"kind": "constant", "ns": 3},
         "added_rtt_ns": 11}
    )  # fmt: skip
    built = make_latency(cfg, median_feed_ns=100)
    assert (built.order_ns, built.ack_ns) == (106, 8)
    with pytest.raises(ValueError):
        make_latency(cfg)
    with pytest.raises(ValueError):
        make_latency(LatencyConfig.model_validate({"order": {"kind": "constant"}}), 1)
    with pytest.raises(NotImplementedError):
        make_latency(LatencyConfig.model_validate({"order": {"kind": "empirical"}}), 1)
    with pytest.raises(ValueError):
        make_latency(LatencyConfig.model_validate({"feed": {"kind": "constant"}}), 1)
    constant = make_latency(
        LatencyConfig.model_validate({"feed": {"kind": "constant", "ns": 9}}), 1
    )
    assert constant.feed_delay_ns(0, 1000) == 9


def test_fee_sign() -> None:
    inst = InstrumentSpec("v", "s", 0.1, 0.001, "perp", -0.5, 5.0, "rebate")
    rebate = fill_fee_quote(inst, 700_000, 10)  # 70,000 x 0.01 = 700 quote notional
    assert rebate == pytest.approx(-0.035)
    assert fill_fee_quote(inst, 700_000, 10, maker=False) == pytest.approx(0.35)
