"""AC-W-05: inventory never exceeds Q_max under random policies and fills."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import read_fixture
from mmsim.config import SimConfig
from mmsim.data.replay import replay
from mmsim.models.calibration import CalibratedParams
from mmsim.sim.fallback import FallbackSimulator
from mmsim.sim.latency import ConstantLatency
from mmsim.sim.types import BookView, InstrumentSpec
from mmsim.strategy.analytic import distances_to_ticks
from mmsim.strategy.base import QuoteDecision, StrategyState
from mmsim.strategy.limits import ask_room, bid_room
from mmsim.strategy.quote_manager import sync_quotes

BOOK = read_fixture("book_l2")
TRADES = read_fixture("trades")
INST = InstrumentSpec("synthetic", "SYNUSDT", 0.1, 0.001, "perp", 0.0, 0.0, "retail")


def _events():  # type: ignore[no-untyped-def]
    return replay(BOOK.to_arrow().to_batches(), TRADES.to_arrow().to_batches(), 0.1, 0.001)


@settings(max_examples=15, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    q_max=st.integers(1, 30),
    latency_ns=st.sampled_from([0, 1_000_000, 20_000_000]),
    queue=st.sampled_from(["front", "conservative", "prob"]),
    offsets=st.lists(
        st.tuples(st.integers(-3, 3), st.integers(-3, 3), st.integers(0, 40)),
        min_size=5,
        max_size=50,
    ),
)
def test_position_stays_within_limit(
    seed: int, q_max: int, latency_ns: int, queue: str, offsets: list[tuple[int, int, int]]
) -> None:
    cfg = SimConfig.model_validate({"queue_model": {"kind": queue}})
    sim = FallbackSimulator(_events, INST, cfg, ConstantLatency(latency_ns, latency_ns), seed=seed)
    i = 0
    worst = 0
    while (step := sim.step()) is not None:
        worst = max(worst, abs(sim.exchange_position_lots))
        book = sim.book("SYNUSDT")
        # Decide on book changes only (like the runner's timer), so acks cannot loop.
        if step.book is None or not (book.bid_ticks and book.ask_ticks):
            continue
        bid_off, ask_off, lots = offsets[i % len(offsets)]
        i += 1
        decision = QuoteDecision(
            book.bid_ticks[0] + bid_off, book.ask_ticks[0] + ask_off, lots, lots
        )
        sync_quotes(sim, decision, q_max_lots=q_max, requote_threshold_ticks=1)
    assert worst <= q_max


def test_rooms() -> None:
    assert bid_room(5, 3, 10) == 2
    assert bid_room(9, 3, 10) == 0
    assert ask_room(-5, 3, 10) == 2
    assert ask_room(10, 0, 10) == 20


@given(
    bid=st.integers(-5, 5), ask=st.integers(-5, 5),
    d_bid=st.floats(-2.0, 2.0), d_ask=st.floats(-2.0, 2.0),
)  # fmt: skip
def test_quotes_never_cross_the_visible_book(
    bid: int, ask: int, d_bid: float, d_ask: float
) -> None:
    view = BookView(0, (1000 + min(bid, ask - 1),), (1,), (1001 + max(ask, bid + 1),), (1,))
    state = StrategyState(0, view, 0, 10, CalibratedParams(0, 1, 1, 1, 1, 0), None, None)
    b, a = distances_to_ticks(state, 0.1, d_bid, d_ask)
    assert b is not None and a is not None
    assert b < view.ask_ticks[0]
    assert a > view.bid_ticks[0]
