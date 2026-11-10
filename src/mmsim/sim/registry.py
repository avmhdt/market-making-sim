"""Backend selection (spec section 10)."""

from __future__ import annotations

from typing import Literal

from mmsim.config import SimConfig
from mmsim.sim.fallback import EventSource, FallbackSimulator, MidObserver
from mmsim.sim.latency import make_latency
from mmsim.sim.protocol import MarketSimulator
from mmsim.sim.types import InstrumentSpec

__all__ = ["make_simulator"]


def make_simulator(
    cfg: SimConfig,
    events: EventSource,
    instrument: InstrumentSpec,
    backend: Literal["fallback", "edbt"] = "fallback",
    *,
    median_feed_ns: int | None = None,
    book_depth: int = 10,
    start_ns: int | None = None,
    on_mid: MidObserver | None = None,
    seed: int = 0,
) -> MarketSimulator:
    """Build a simulator; the edbt backend arrives with M6 (D-02)."""
    if backend == "edbt":
        raise NotImplementedError("the edbt backend arrives in M6 (P-08, D-02)")
    return FallbackSimulator(
        events,
        instrument,
        cfg,
        make_latency(cfg.latency, median_feed_ns),
        book_depth=book_depth,
        start_ns=start_ns,
        on_mid=on_mid,
        seed=seed,
    )
