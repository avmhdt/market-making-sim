"""Signal providers: the stub and the artifact-backed model (FR-15).

A provider sees every strategy-visible step and returns, at a local time, the latest
prediction ``y`` (mid change in ticks over ``horizon_s``) that is already usable, i.e.
whose sample time plus the compute latency has passed; ``None`` while warming up.

- :class:`StubSignal`: micro-price minus mid in ticks, the size-weighted mid's offset
  ``(I - 1/2) * spread`` with ``I = bid_1 / (bid_1 + ask_1)``. It needs no artifact and
  keeps M4's plumbing testable before P-06.
- :class:`ModelSignal`: the online feature engine feeding a :class:`SignalModel`.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol

from mmsim.signal.artifact import SignalModel
from mmsim.signal.online import OnlineFeatureEngine
from mmsim.sim.types import StepResult

__all__ = ["ModelSignal", "SignalProvider", "StubSignal"]


class SignalProvider(Protocol):
    """Short-horizon mid-change prediction, in ticks."""

    horizon_s: float

    def update(self, step: StepResult) -> None:
        """See one step's visible events."""
        ...

    def signal(self, now_ns: int) -> float | None:
        """The latest usable prediction at ``now_ns``."""
        ...


class StubSignal:
    """Micro-price minus mid, in ticks, from the visible top of book."""

    def __init__(self, horizon_s: float = 1.0) -> None:
        self.horizon_s = horizon_s
        self._value: float | None = None

    def update(self, step: StepResult) -> None:
        """Recompute on every visible book."""
        book = step.book
        if book is None or not book.bid_ticks or not book.ask_ticks:
            return
        b, a = float(book.bid_lots[0]), float(book.ask_lots[0])
        imbalance = b / (b + a) if b + a > 0 else 0.5
        self._value = (imbalance - 0.5) * (book.ask_ticks[0] - book.bid_ticks[0])

    def signal(self, now_ns: int) -> float | None:
        """The current value (the stub has no compute latency)."""
        return self._value


class ModelSignal:
    """Features computed online, predicted by the artifact, usable after the latency."""

    def __init__(self, model: SignalModel, tick_size: float, compute_latency_ns: int = 0) -> None:
        self.model = model
        self.engine = OnlineFeatureEngine(model.spec, tick_size)
        self.latency_ns = compute_latency_ns
        self.horizon_s = model.horizon_ns / 1e9 if model.horizon_ns else 1.0
        self._pending: deque[tuple[int, float]] = deque()
        self._last_sample: int | None = None
        self._current: float | None = None
        self.predictions = 0

    def update(self, step: StepResult) -> None:
        """Feed the engine; predict once per new sample."""
        self.engine.update(step)
        ts = self.engine.last_ts
        if ts is None or ts == self._last_sample:
            return
        self._last_sample = ts
        x = self.engine.vector()
        if x is None:
            return
        self._pending.append((ts + self.latency_ns, self.model.predict(x)))
        self.predictions += 1

    def signal(self, now_ns: int) -> float | None:
        """The newest prediction whose ready time has passed."""
        while self._pending and self._pending[0][0] <= now_ns:
            self._current = self._pending.popleft()[1]
        return self._current
