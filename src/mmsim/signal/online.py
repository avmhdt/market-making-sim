"""Online feature engine over strategy-visible events (FR-15, D-03: re-implemented here).

crypto-lob-alpha computes its features offline (LazyFrame in, LazyFrame out) and offers no
incremental API, so each feature kind is re-implemented here from the definitions in its
spec (section 10, ``FR-17``). The parity test (AC-W-09) decides whether they match.

Sampling. With ``clock = "time"`` a row is produced at every multiple of ``sample_grid_ms``
of the timestamp the strategy sees (``ts_local``, which equals ``ts_recv`` under recorded
feed latency). The row at grid time ``t`` reflects every event with timestamp ``<= t``;
time windows cover ``(t - w, t]``. With ``clock = "event"`` a row follows every book update.
A feature is not ready until its window is filled; :meth:`OnlineFeatureEngine.vector`
returns ``None`` until every feature is ready.

Kinds (quantities in base units, prices in ticks; definitions in docs/signal-artifact.md):
``ofi``, ``mlofi``, ``integrated_ofi``, ``queue_imbalance``, ``microprice``,
``weighted_mid``, ``trade_sign_imbalance``, ``spread``, ``realized_vol``, ``mid_changes``.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from mmsim.signal.artifact import ArtifactError, FeatureDef, FeatureSpec
from mmsim.sim.types import BookView, StepResult

__all__ = ["SUPPORTED_KINDS", "OnlineFeatureEngine"]

MS = 1_000_000
SUPPORTED_KINDS = frozenset(
    {
        "ofi",
        "mlofi",
        "integrated_ofi",
        "queue_imbalance",
        "microprice",
        "weighted_mid",
        "trade_sign_imbalance",
        "spread",
        "realized_vol",
        "mid_changes",
    }
)


class _Window:
    """Sum of integer values (lots, counts) over the last ``w`` ms or the last ``n`` events.

    Values are Python ints, so the running sum is exact: a window that empties sums to exactly
    0, and no floating-point drift builds up over a day. Callers scale by the lot size after
    summing.
    """

    def __init__(self, size: int, unit: str) -> None:
        self.size = size
        self.unit = unit
        self.items: deque[tuple[int, int]] = deque()
        self.total = 0
        self.first_ts: int | None = None

    def start(self, ts: int) -> None:
        """The engine's first event: time windows count their warm-up from here."""
        if self.first_ts is None:
            self.first_ts = ts

    def add(self, ts: int, value: int) -> None:
        if self.first_ts is None:
            self.first_ts = ts
        self.items.append((ts, value))
        self.total += value
        if self.unit == "events" and len(self.items) > self.size:
            _, v = self.items.popleft()
            self.total -= v

    def value_at(self, t: int) -> int | None:
        if self.unit == "ms":
            lo = t - self.size * MS
            while self.items and self.items[0][0] <= lo:
                _, v = self.items.popleft()
                self.total -= v
            if self.first_ts is None or t - self.first_ts < self.size * MS:
                return None
            return self.total
        if len(self.items) < self.size:
            return None
        return self.total


def _level(prices: tuple[int, ...], lots: tuple[int, ...], level: int) -> tuple[int, int]:
    """Price and size at a level; a missing level is price -1 and size 0 (as the producer)."""
    if len(prices) <= level:
        return -1, 0
    return prices[level], lots[level]


def _ofi_level(prev: BookView, cur: BookView, level: int) -> int:
    """Cont, Kukanov and Stoikov (2014) ``e_n`` at a book level (0 = touch), in lots."""
    pb, qb = _level(prev.bid_ticks, prev.bid_lots, level)
    cb, cqb = _level(cur.bid_ticks, cur.bid_lots, level)
    pa, qa = _level(prev.ask_ticks, prev.ask_lots, level)
    ca, cqa = _level(cur.ask_ticks, cur.ask_lots, level)
    e = 0
    if cb >= pb:
        e += cqb
    if cb <= pb:
        e -= qb
    if ca <= pa:
        e -= cqa
    if ca >= pa:
        e += qa
    return e


class _Feature:
    """Base: book and trade hooks, and the raw value at a grid time."""

    def __init__(self, fdef: FeatureDef, tick: float, lot: float) -> None:
        self.fdef = fdef
        self.tick = tick
        self.lot = lot

    def on_start(self, ts: int) -> None:
        """The engine's first event."""
        for win in self.__dict__.values():
            if isinstance(win, _Window):
                win.start(ts)

    def on_book(self, ts: int, prev: BookView | None, cur: BookView) -> None:
        """A new visible book."""

    def on_trade(self, ts: int, price_ticks: int, qty_lots: int, side: int) -> None:
        """A visible trade."""

    def value(self, t: int, book: BookView | None) -> float | None:
        raise NotImplementedError


def _window(fdef: FeatureDef) -> _Window:
    if fdef.window is None or fdef.window_unit is None:
        raise ArtifactError(f"{fdef.name}: kind {fdef.kind} needs window and window_unit")
    return _Window(fdef.window, fdef.window_unit)


class _OFI(_Feature):
    """OFI at one level (``ofi``: level 1; ``mlofi``: its ``levels[0]``), summed over the window
    and divided by ``params.depth_norm`` when given."""

    def __init__(self, fdef: FeatureDef, tick: float, lot: float) -> None:
        super().__init__(fdef, tick, lot)
        self.level = (fdef.levels or [1])[0] - 1
        self.win = _window(fdef)
        self.norm = float(fdef.params.get("depth_norm", 1.0)) or 1.0

    def on_book(self, ts: int, prev: BookView | None, cur: BookView) -> None:
        if prev is None:
            return
        self.win.add(ts, int(_ofi_level(prev, cur, self.level)))

    def value(self, t: int, book: BookView | None) -> float | None:
        v = self.win.value_at(t)
        return None if v is None else v * self.lot / self.norm


class _IntegratedOFI(_Feature):
    """``sum_l w_l (level-l OFI over the window)``, ``params.weights`` per level (first PC),
    divided by ``params.depth_norm`` when given."""

    def __init__(self, fdef: FeatureDef, tick: float, lot: float) -> None:
        super().__init__(fdef, tick, lot)
        weights = fdef.params.get("weights")
        if not weights:
            raise ArtifactError(f"{fdef.name}: integrated_ofi needs params.weights")
        self.weights = [float(w) for w in weights]
        self.wins = [_window(fdef) for _ in self.weights]
        self.norm = float(fdef.params.get("depth_norm", 1.0)) or 1.0

    def on_start(self, ts: int) -> None:
        for win in self.wins:
            win.start(ts)

    def on_book(self, ts: int, prev: BookView | None, cur: BookView) -> None:
        if prev is None:
            return
        for level, win in enumerate(self.wins):
            win.add(ts, int(_ofi_level(prev, cur, level)))

    def value(self, t: int, book: BookView | None) -> float | None:
        total = 0.0
        for w, win in zip(self.weights, self.wins, strict=True):
            s = win.value_at(t)
            if s is None:
                return None
            total += w * (s * self.lot)
        return total / self.norm


class _QueueImbalance(_Feature):
    """``(sum bid_qty_1..L - sum ask_qty_1..L) / (sum both)``, ``L = max(levels)``; missing
    (NaN) when both sums are 0."""

    def value(self, t: int, book: BookView | None) -> float | None:
        if book is None or not book.bid_ticks or not book.ask_ticks:
            return None
        n = max(self.fdef.levels or [1])
        b = float(sum(book.bid_lots[:n]))
        a = float(sum(book.ask_lots[:n]))
        return (b - a) / (b + a) if b + a > 0 else math.nan


def _imbalance(book: BookView) -> float:
    b, a = float(book.bid_lots[0]), float(book.ask_lots[0])
    return b / (b + a) if b + a > 0 else 0.5


class _WeightedMid(_Feature):
    """Size-weighted mid minus the mid, in ticks: ``(I - 1/2) * spread``, computed as the
    producer pins it: ``float((qb - qa) * S) / float(2 (qb + qa))`` in integer lots and ticks."""

    def value(self, t: int, book: BookView | None) -> float | None:
        if book is None or not book.bid_ticks or not book.ask_ticks:
            return None
        spread = int(book.ask_ticks[0]) - int(book.bid_ticks[0])
        qb, qa = int(book.bid_lots[0]), int(book.ask_lots[0])
        if qb + qa == 0:
            return 0.0
        return float((qb - qa) * spread) / float(2 * (qb + qa))


class _Microprice(_Feature):
    """Stoikov (2018) adjustment ``g(I, S)`` in ticks from ``params.table``.

    Rows ``[imbalance_lower_edge, adjustment]`` (one spread) or
    ``[spread_ticks, imbalance_lower_edge, adjustment]``; ``I = bid_1 / (bid_1 + ask_1)``
    falls in the last row whose edge is ``<= I``. Spreads above ``max_spread_ticks`` use
    the largest spread in the table.
    """

    def __init__(self, fdef: FeatureDef, tick: float, lot: float) -> None:
        super().__init__(fdef, tick, lot)
        table = fdef.params.get("table")
        if not table:
            raise ArtifactError(f"{fdef.name}: microprice needs params.table")
        rows = [[float(x) for x in r] for r in table]
        self.by_spread: dict[int, list[tuple[float, float]]] = {}
        for r in rows:
            if len(r) == 2:
                self.by_spread.setdefault(0, []).append((r[0], r[1]))
            elif len(r) == 3:
                self.by_spread.setdefault(int(r[0]), []).append((r[1], r[2]))
            else:
                raise ArtifactError(f"{fdef.name}: microprice rows need 2 or 3 numbers")
        for v in self.by_spread.values():
            v.sort()
        self.max_spread = int(fdef.params.get("max_spread_ticks", max(self.by_spread)))

    def value(self, t: int, book: BookView | None) -> float | None:
        if book is None or not book.bid_ticks or not book.ask_ticks:
            return None
        if 0 in self.by_spread:
            rows = self.by_spread[0]
        else:
            s = min(book.ask_ticks[0] - book.bid_ticks[0], self.max_spread)
            below = [k for k in self.by_spread if k <= s]
            rows = self.by_spread[max(below) if below else min(self.by_spread)]
        imb = _imbalance(book)
        adj = rows[0][1]
        for edge, a in rows:
            if edge <= imb:
                adj = a
            else:
                break
        return adj


class _TradeSign(_Feature):
    """``sum side * w / sum w`` over the window, ``w`` = volume (``params.weight``) or 1."""

    def __init__(self, fdef: FeatureDef, tick: float, lot: float) -> None:
        super().__init__(fdef, tick, lot)
        self.signed = _window(fdef)
        self.total = _window(fdef)
        self.volume = str(fdef.params.get("weight", "volume")) == "volume"

    def on_trade(self, ts: int, price_ticks: int, qty_lots: int, side: int) -> None:
        if side == 0:
            return
        w = int(qty_lots) if self.volume else 1  # lots: the lot size cancels in the ratio
        self.signed.add(ts, side * w)
        self.total.add(ts, w)

    def value(self, t: int, book: BookView | None) -> float | None:
        s, w = self.signed.value_at(t), self.total.value_at(t)
        if s is None or w is None:
            return None
        return s / w if w > 0 else 0.0


class _Spread(_Feature):
    """Best ask minus best bid, in ticks."""

    def value(self, t: int, book: BookView | None) -> float | None:
        if book is None or not book.bid_ticks or not book.ask_ticks:
            return None
        return float(book.ask_ticks[0] - book.bid_ticks[0])


class _MidHistory(_Feature):
    """Mid (ticks) changes: count (``mid_changes``) or realised volatility (``realized_vol``):
    the square root of the summed squared changes of the mid sampled every
    ``params.sample_ms`` (default 100) over the window, in ticks."""

    def __init__(self, fdef: FeatureDef, tick: float, lot: float) -> None:
        super().__init__(fdef, tick, lot)
        if fdef.window_unit != "ms" or fdef.window is None:
            raise ArtifactError(f"{fdef.name}: {fdef.kind} needs a window in ms")
        self.window_ns = fdef.window * MS
        self.sample_ns = int(fdef.params.get("sample_ms", 100)) * MS
        self.mids: deque[tuple[int, int]] = deque()  # (ts, bid_1 + ask_1 in ticks)
        self.first: int | None = None

    def on_start(self, ts: int) -> None:
        """Warm-up counts from the engine's first event."""
        self.first = ts

    def on_book(self, ts: int, prev: BookView | None, cur: BookView) -> None:
        if not cur.bid_ticks or not cur.ask_ticks:
            return
        mid = int(cur.mid_x2)
        if not self.mids or self.mids[-1][1] != mid:
            self.mids.append((ts, mid))

    def value(self, t: int, book: BookView | None) -> float | None:
        if self.first is None or t - self.first < self.window_ns:
            return None
        lo = t - self.window_ns
        while len(self.mids) > 1 and self.mids[1][0] <= lo:
            self.mids.popleft()
        if self.fdef.kind == "mid_changes":
            return float(sum(1 for ts, _ in self.mids if lo < ts <= t))
        grid = np.arange(lo, t + 1, self.sample_ns, dtype=np.int64)
        times = np.array([m[0] for m in self.mids], dtype=np.int64)
        vals = np.array([m[1] for m in self.mids], dtype=np.int64)
        idx = np.searchsorted(times, grid, side="right") - 1
        sampled = vals[np.clip(idx, 0, len(vals) - 1)]
        # sqrt(float(sum d^2)) / 2 with d the integer change of bid_1 + ask_1 (the producer's form)
        return math.sqrt(float(int(np.sum(np.diff(sampled) ** 2)))) / 2


_KINDS: dict[str, Callable[[FeatureDef, float, float], _Feature]] = {
    "ofi": _OFI,
    "mlofi": _OFI,
    "integrated_ofi": _IntegratedOFI,
    "queue_imbalance": _QueueImbalance,
    "microprice": _Microprice,
    "weighted_mid": _WeightedMid,
    "trade_sign_imbalance": _TradeSign,
    "spread": lambda f, t, lot: _Spread(f, t, lot),
    "realized_vol": _MidHistory,
    "mid_changes": _MidHistory,
}


class OnlineFeatureEngine:
    """Computes the spec's raw features from :class:`StepResult` events."""

    def __init__(self, spec: FeatureSpec, tick_size: float, lot_size: float | None = None) -> None:
        unknown = sorted({f.kind for f in spec.features} - SUPPORTED_KINDS)
        if unknown:
            raise ArtifactError(f"feature kinds not implemented online: {', '.join(unknown)}")
        self.spec = spec
        lot = spec.lot_size if lot_size is None else lot_size
        self.features = [_KINDS[f.kind](f, tick_size, lot) for f in spec.features]
        self.grid_ns = (spec.sample_grid_ms or 0) * MS
        if spec.clock == "time" and self.grid_ns <= 0:
            raise ArtifactError("clock 'time' needs sample_grid_ms")
        self._book: BookView | None = None
        self._next: int | None = None
        self._started: int | None = None
        self.last_ts: int | None = None
        self.last_row: npt.NDArray[np.float64] | None = None
        self.rows: list[tuple[int, list[float | None]]] | None = None

    def record(self) -> None:
        """Keep every sampled row (for the parity test)."""
        self.rows = []

    def _sample(self, t: int) -> None:
        values = [f.value(t, self._book) for f in self.features]
        if self.rows is not None:
            self.rows.append((t, values))
        self.last_ts = t
        if any(v is None for v in values):
            self.last_row = None
        else:
            self.last_row = np.array(values, dtype=np.float64)

    def update(self, step: StepResult) -> None:
        """Apply one step's visible book and trades at ``step.ts_local``."""
        t = step.ts_local
        if self._started is None:
            self._started = t
            for f in self.features:
                f.on_start(t)
        if self.spec.clock == "time":
            if self._next is None:
                self._next = (t // self.grid_ns + 1) * self.grid_ns
            while t > self._next:
                self._sample(self._next)
                self._next += self.grid_ns
        for tr in step.trades:
            for f in self.features:
                f.on_trade(t, tr.price_ticks, tr.qty_lots, tr.side)
        if step.book is not None:
            prev = self._book
            self._book = step.book
            for f in self.features:
                f.on_book(t, prev, step.book)
            if self.spec.clock == "event":
                self._sample(t)

    def vector(self) -> npt.NDArray[np.float64] | None:
        """Raw features of the latest completed sample, or ``None`` while warming up."""
        return self.last_row

    def snapshot(self) -> dict[str, Any]:
        """Diagnostics for logs."""
        return {"last_ts": self.last_ts, "ready": self.last_row is not None}
