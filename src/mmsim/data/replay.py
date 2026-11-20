"""Deterministic merged event stream in integer ticks and lots (FR-04, DQ-01, DQ-03).

Each stream is read in a stable sort by ``ts_recv``, then ``batch_id`` (book_l2) or the
numeric trade id (trades), then original row order. Each row's matching time is its
``ts_exchange`` made non-decreasing within its stream. The two streams are merged by
``(matching time, stream priority, ts_recv, row index)`` with trades before book_l2 at
equal matching time; a book_l2 batch is never split. Replay starts at the first snapshot
batch: earlier events are dropped and counted.

Files are hourly by ``ts_recv``, so sorting each row group and checking the order across
row-group boundaries gives the global order without loading a day at once.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc

__all__ = [
    "BOOK_PRIORITY",
    "TRADE_PRIORITY",
    "BookBatchEvent",
    "MarketEvent",
    "ReplayOrderError",
    "ReplayStats",
    "TickGridError",
    "TradeRowEvent",
    "book_batches",
    "merge_streams",
    "replay",
    "to_lots",
    "to_ticks",
    "trade_rows",
]

TRADE_PRIORITY = 0
BOOK_PRIORITY = 1
GRID_TOLERANCE = 1e-6
OFF_GRID_MAX_SHARE = 1e-3
"""Book rows off the tick grid are dropped and counted up to this share of a chunk."""

Int64Array = npt.NDArray[np.int64]
Int8Array = npt.NDArray[np.int8]


class ReplayOrderError(ValueError):
    """Rows arrive out of ``ts_recv`` order across a row-group or file boundary."""


class TickGridError(ValueError):
    """A price or quantity is off the instrument's tick or lot grid."""


@dataclass(frozen=True, slots=True)
class BookBatchEvent:
    """One venue message: all book_l2 rows sharing a ``batch_id``, applied atomically."""

    ts_match: int
    ts_exchange: int
    ts_recv: int
    batch_id: int
    seq: int
    is_snapshot: bool
    side: Int8Array
    price_ticks: Int64Array
    qty_lots: Int64Array
    """New absolute size at the level; 0 removes it."""


@dataclass(frozen=True, slots=True)
class TradeRowEvent:
    """One public trade."""

    ts_match: int
    ts_exchange: int
    ts_recv: int
    trade_id: str
    price_ticks: int
    qty_lots: int
    side: int


MarketEvent = BookBatchEvent | TradeRowEvent


@dataclass(slots=True)
class ReplayStats:
    """Counters written to ``metrics.json``."""

    book_batches: int = 0
    trades: int = 0
    side0_trades: int = 0
    dropped_before_snapshot: int = 0
    off_grid_book_rows: int = 0
    """Book rows whose price is off the tick grid (dropped; seen on SOLUSDT, M3-11)."""
    exchange_time_clamped: int = field(default=0)
    """Rows whose ``ts_exchange`` went backwards and were clamped to the running maximum."""


def _to_grid(values: npt.NDArray[np.float64], step: float, what: str) -> Int64Array:
    scaled = values / step
    rounded = np.rint(scaled)
    residual = np.abs(scaled - rounded)
    if residual.size and float(residual.max()) >= GRID_TOLERANCE:
        worst = int(np.argmax(residual))
        raise TickGridError(
            f"{what} {values[worst]!r} is {residual[worst]:.3g} {what}-units off the grid {step}"
        )
    return rounded.astype(np.int64)


def to_ticks(prices: npt.NDArray[np.float64], tick_size: float) -> Int64Array:
    """Prices to integer ticks; the residual must be below 1e-6 tick."""
    return _to_grid(np.asarray(prices, dtype=np.float64), tick_size, "tick")


def to_lots(qty: npt.NDArray[np.float64], lot_size: float) -> Int64Array:
    """Quantities to integer lots; the residual must be below 1e-6 lot."""
    return _to_grid(np.asarray(qty, dtype=np.float64), lot_size, "lot")


_BOOK_COLS = ("ts_exchange", "ts_recv", "seq", "side", "price", "qty", "is_snapshot", "batch_id")


def _col(batch: pa.RecordBatch, name: str) -> npt.NDArray[Any]:
    return np.asarray(batch.column(name).to_numpy(zero_copy_only=False))


def _is_sorted(primary: Int64Array, secondary: Int64Array) -> bool:
    if primary.size < 2:
        return True
    dp = np.diff(primary)
    ds = np.diff(secondary)
    return bool(np.all((dp > 0) | ((dp == 0) & (ds >= 0))))


def book_batches(
    chunks: Iterable[pa.RecordBatch],
    tick_size: float,
    lot_size: float,
    stats: ReplayStats | None = None,
) -> Iterator[BookBatchEvent]:
    """Group book_l2 rows into batches with non-decreasing matching time.

    Works on NumPy arrays per row group (no dataframe per chunk), so it is cheap and
    single-threaded.
    """
    stats = stats if stats is not None else ReplayStats()
    last_key: tuple[int, int] | None = None
    ts_match = np.iinfo(np.int64).min
    carry: dict[str, npt.NDArray[Any]] | None = None
    for chunk in chunks:
        if chunk.num_rows == 0:
            continue
        cols = {name: _col(chunk, name) for name in _BOOK_COLS}
        recv = cols["ts_recv"].astype(np.int64)
        bid = cols["batch_id"].astype(np.int64)
        if not _is_sorted(recv, bid):
            order = np.lexsort((bid, recv))
            cols = {k: v[order] for k, v in cols.items()}
            recv, bid = recv[order], bid[order]
        first = (int(recv[0]), int(bid[0]))
        if last_key is not None and first < last_key:
            raise ReplayOrderError(f"book_l2 key {first} after {last_key}")
        last_key = (int(recv[-1]), int(bid[-1]))
        price = cols.pop("price").astype(np.float64)
        scaled = price / tick_size
        off = np.abs(scaled - np.rint(scaled)) >= GRID_TOLERANCE
        if off.any():
            n_off = int(off.sum())
            if n_off > OFF_GRID_MAX_SHARE * off.size:
                to_ticks(price, tick_size)  # raises with the worst row
            stats.off_grid_book_rows += n_off
            keep = ~off
            cols = {k: v[keep] for k, v in cols.items()}
            price = price[keep]
            recv, bid = recv[keep], bid[keep]
            if price.size == 0:
                continue
        cols["price_ticks"] = to_ticks(price, tick_size)
        cols["qty_lots"] = to_lots(cols.pop("qty").astype(np.float64), lot_size)
        if carry is not None:
            cols = {k: np.concatenate([carry[k], v]) for k, v in cols.items()}
            carry = None
        ids = cols["batch_id"]
        starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        ends = np.r_[starts[1:], ids.size]
        ts_ex_all = cols["ts_exchange"].astype(np.int64)
        seq_all = cols["seq"].astype(np.int64)
        side_all = cols["side"].astype(np.int8)
        snap_all = cols["is_snapshot"].astype(bool)
        recv_all = cols["ts_recv"].astype(np.int64)
        px_all = cols["price_ticks"]
        qty_all = cols["qty_lots"]
        # Per-batch maxima in one pass; the last batch may continue in the next chunk.
        ts_max = np.maximum.reduceat(ts_ex_all, starts).tolist()
        seq_max = np.maximum.reduceat(seq_all, starts).tolist()
        recv_first = recv_all[starts].tolist()
        id_first = ids[starts].tolist()
        snap_first = snap_all[starts].tolist()
        for j, (lo, hi) in enumerate(zip(starts[:-1].tolist(), ends[:-1].tolist(), strict=True)):
            ts_ex = ts_max[j]
            if ts_ex < ts_match:
                stats.exchange_time_clamped += 1
            else:
                ts_match = ts_ex
            stats.book_batches += 1
            yield BookBatchEvent(
                ts_match=ts_match,
                ts_exchange=ts_ex,
                ts_recv=recv_first[j],
                batch_id=id_first[j],
                seq=seq_max[j],
                is_snapshot=bool(snap_first[j]),
                side=side_all[lo:hi],
                price_ticks=px_all[lo:hi],
                qty_lots=qty_all[lo:hi],
            )
        tail = int(starts[-1])
        carry = {k: v[tail:] for k, v in cols.items()}
    if carry is not None and carry["batch_id"].size:
        ts_ex = int(carry["ts_exchange"].astype(np.int64).max())
        if ts_ex < ts_match:
            stats.exchange_time_clamped += 1
        ts_match = max(ts_match, ts_ex)
        stats.book_batches += 1
        yield BookBatchEvent(
            ts_match=ts_match,
            ts_exchange=ts_ex,
            ts_recv=int(carry["ts_recv"][0]),
            batch_id=int(carry["batch_id"][0]),
            seq=int(carry["seq"].astype(np.int64).max()),
            is_snapshot=bool(carry["is_snapshot"][0]),
            side=carry["side"].astype(np.int8),
            price_ticks=carry["price_ticks"],
            qty_lots=carry["qty_lots"],
        )


def _trade_id_key(ids: pa.Array | pa.ChunkedArray) -> Int64Array:
    try:
        return np.asarray(pc.cast(ids, pa.int64()).to_numpy(zero_copy_only=False), dtype=np.int64)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return np.zeros(len(ids), dtype=np.int64)


def trade_rows(
    chunks: Iterable[pa.RecordBatch],
    tick_size: float,
    lot_size: float,
    stats: ReplayStats | None = None,
) -> Iterator[TradeRowEvent]:
    """Trades in replay order with non-decreasing matching time."""
    stats = stats if stats is not None else ReplayStats()
    last_recv: int | None = None
    ts_match = np.iinfo(np.int64).min
    for chunk in chunks:
        if chunk.num_rows == 0:
            continue
        ids_arr = chunk.column("trade_id")
        if not pa.types.is_string(ids_arr.type) and not pa.types.is_large_string(ids_arr.type):
            ids_arr = pc.cast(ids_arr, pa.string())
        recv = _col(chunk, "ts_recv").astype(np.int64)
        key = _trade_id_key(ids_arr)
        ids = ids_arr.to_pylist()
        cols = {n: _col(chunk, n) for n in ("ts_exchange", "price", "qty", "side")}
        if not _is_sorted(recv, key):
            order = np.lexsort((key, recv))
            recv = recv[order]
            cols = {k: v[order] for k, v in cols.items()}
            ids = [ids[i] for i in order.tolist()]
        first = int(recv[0])
        if last_recv is not None and first < last_recv:
            raise ReplayOrderError(f"trades ts_recv {first} after {last_recv}")
        last_recv = int(recv[-1])
        prices = to_ticks(cols["price"].astype(np.float64), tick_size).tolist()
        lots = to_lots(cols["qty"].astype(np.float64), lot_size).tolist()
        ts_ex = cols["ts_exchange"].astype(np.int64).tolist()
        sides = cols["side"].astype(np.int64).tolist()
        recv_l = recv.tolist()
        for i, ex in enumerate(ts_ex):
            if ex < ts_match:
                stats.exchange_time_clamped += 1
            else:
                ts_match = ex
            side = sides[i]
            stats.trades += 1
            if side == 0:
                stats.side0_trades += 1
            yield TradeRowEvent(
                ts_match=ts_match,
                ts_exchange=ex,
                ts_recv=recv_l[i],
                trade_id=str(ids[i]),
                price_ticks=prices[i],
                qty_lots=lots[i],
                side=side,
            )


def merge_streams(
    books: Iterable[BookBatchEvent], trades: Iterable[TradeRowEvent]
) -> Iterator[MarketEvent]:
    """Merge by (matching time, priority, ts_recv, row index): trades first on ties."""

    def keyed_trades() -> Iterator[tuple[int, int, int, int, MarketEvent]]:
        for index, event in enumerate(trades):
            yield (event.ts_match, TRADE_PRIORITY, event.ts_recv, index, event)

    def keyed_books() -> Iterator[tuple[int, int, int, int, MarketEvent]]:
        for index, event in enumerate(books):
            yield (event.ts_match, BOOK_PRIORITY, event.ts_recv, index, event)

    for item in heapq.merge(keyed_trades(), keyed_books(), key=lambda t: t[:4]):
        yield item[4]


def replay(
    book_chunks: Iterable[pa.RecordBatch],
    trade_chunks: Iterable[pa.RecordBatch],
    tick_size: float,
    lot_size: float,
    stats: ReplayStats | None = None,
    *,
    start: BookBatchEvent | None = None,
) -> Iterator[MarketEvent]:
    """The merged stream, starting at the first snapshot batch (DQ-03).

    With ``start`` (a book checkpoint as a snapshot batch, see ``load_checkpoint``) the
    stream begins with it instead: book batches up to its ``seq`` and trades received
    before it are dropped and counted, and earlier matching times are lifted to it.
    """
    stats = stats if stats is not None else ReplayStats()
    books: Iterable[BookBatchEvent] = book_batches(book_chunks, tick_size, lot_size, stats)
    trades: Iterable[TradeRowEvent] = trade_rows(trade_chunks, tick_size, lot_size, stats)
    if start is not None:
        floor = start.ts_match

        def after_books(src: Iterable[BookBatchEvent]) -> Iterator[BookBatchEvent]:
            yield start
            for b in src:
                if b.seq <= start.seq:
                    stats.dropped_before_snapshot += 1
                    continue
                yield b if b.ts_match >= floor else replace(b, ts_match=floor)

        def after_trades(src: Iterable[TradeRowEvent]) -> Iterator[TradeRowEvent]:
            for t in src:
                if t.ts_recv < start.ts_recv:
                    stats.dropped_before_snapshot += 1
                    continue
                yield t if t.ts_match >= floor else replace(t, ts_match=floor)

        books, trades = after_books(books), after_trades(trades)
    started = False
    for event in merge_streams(books, trades):
        if not started:
            if isinstance(event, BookBatchEvent) and event.is_snapshot:
                started = True
            else:
                stats.dropped_before_snapshot += 1
                continue
        yield event
