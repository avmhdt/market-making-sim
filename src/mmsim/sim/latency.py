"""Feed, order and acknowledgement latency (FR-02, DQ-02, D-09).

The feed delay is the recorded ``ts_recv - ts_exchange`` (clamped at 0) or a constant,
plus ``extra_ns``. Order and ack delays are constants: D-09's default sets both to the
day's median recorded feed latency, which the caller measures and passes in. The latency
sweep's ``added_rtt_ns`` is split evenly between the order and the ack leg.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from mmsim.config import LatencyConfig, OrderLatencyConfig

__all__ = ["ConstantLatency", "make_latency", "median_feed_latency_ns"]


@dataclass(frozen=True, slots=True)
class ConstantLatency:
    """Constant order and ack delays; feed delay recorded or constant."""

    order_ns: int
    ack_ns: int
    feed_ns: int | None = None
    """``None`` uses the recorded ``ts_recv - ts_exchange``."""
    feed_extra_ns: int = 0

    def feed_delay_ns(self, ts_exchange: int, ts_recv: int) -> int:
        """Recorded (or constant) feed latency plus the extra."""
        base = max(ts_recv - ts_exchange, 0) if self.feed_ns is None else self.feed_ns
        return base + self.feed_extra_ns

    def order_delay_ns(self, ts_local: int) -> int:
        """Decision to exchange arrival."""
        return self.order_ns

    def ack_delay_ns(self, ts_exchange: int) -> int:
        """Exchange event to the strategy seeing it."""
        return self.ack_ns


def median_feed_latency_ns(ts_exchange: npt.ArrayLike, ts_recv: npt.ArrayLike) -> int:
    """Median of ``ts_recv - ts_exchange`` over a day, clamped at 0 (D-09)."""
    lat = np.asarray(ts_recv, dtype=np.int64) - np.asarray(ts_exchange, dtype=np.int64)
    if lat.size == 0:
        return 0
    return max(int(np.median(lat)), 0)


def _leg(cfg: OrderLatencyConfig, median_feed_ns: int | None) -> int:
    if cfg.kind == "constant":
        if cfg.ns is None:
            raise ValueError("constant latency needs ns")
        return cfg.ns
    if cfg.kind == "median_feed":
        if median_feed_ns is None:
            raise ValueError("median_feed latency needs the measured median feed latency")
        return median_feed_ns
    raise NotImplementedError("empirical order latency is not implemented in the fallback")


def make_latency(cfg: LatencyConfig, median_feed_ns: int | None = None) -> ConstantLatency:
    """The latency model of a config; ``median_feed_ns`` is needed by ``median_feed`` legs."""
    half, odd = divmod(cfg.added_rtt_ns, 2)
    if cfg.feed.kind == "constant" and cfg.feed.ns is None:
        raise ValueError("constant feed latency needs ns")
    return ConstantLatency(
        order_ns=_leg(cfg.order, median_feed_ns) + half + odd,
        ack_ns=_leg(cfg.ack, median_feed_ns) + half,
        feed_ns=cfg.feed.ns if cfg.feed.kind == "constant" else None,
        feed_extra_ns=cfg.feed.extra_ns,
    )
