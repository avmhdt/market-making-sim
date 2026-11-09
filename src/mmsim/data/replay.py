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
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import polars as pl
import pyarrow as pa

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


def _trade_id_key(ids: pl.Series) -> pl.Series:
    numeric = ids.cast(pl.Int64, strict=False)
    if numeric.null_count() == 0:
        return numeric
    return pl.Series(ids.name, np.zeros(len(ids), dtype=np.int64))


def _sorted_chunk(batch: pa.RecordBatch, keys: list[str]) -> pl.DataFrame:
    frame = pl.from_arrow(batch)
    assert isinstance(frame, pl.DataFrame)
    return frame.sort(keys, maintain_order=True)


def book_batches(
    chunks: Iterable[pa.RecordBatch],
    tick_size: float,
    lot_size: float,
    stats: ReplayStats | None = None,
) -> Iterator[BookBatchEvent]:
    """Group book_l2 rows into batches with non-decreasing matching time."""
    stats = stats if stats is not None else ReplayStats()
    last_key: tuple[int, int] | None = None
    ts_match = np.iinfo(np.int64).min
    carry: pl.DataFrame | None = None

    def emit(rows: pl.DataFrame) -> BookBatchEvent:
        nonlocal ts_match
        ts_ex = int(rows.get_column("ts_exchange").max())  # type: ignore[arg-type]
        if ts_ex < ts_match:
            stats.exchange_time_clamped += 1
        ts_match = max(ts_match, ts_ex)
        stats.book_batches += 1
        snap = rows.get_column("is_snapshot")
        return BookBatchEvent(
            ts_match=ts_match,
            ts_exchange=ts_ex,
            ts_recv=int(rows.get_column("ts_recv")[0]),
            batch_id=int(rows.get_column("batch_id")[0]),
            seq=int(rows.get_column("seq").max()),  # type: ignore[arg-type]
            is_snapshot=bool(snap[0]),
            side=rows.get_column("side").to_numpy().astype(np.int8),
            price_ticks=to_ticks(rows.get_column("price").to_numpy(), tick_size),
            qty_lots=to_lots(rows.get_column("qty").to_numpy(), lot_size),
        )

    for chunk in chunks:
        if chunk.num_rows == 0:
            continue
        frame = _sorted_chunk(chunk, ["ts_recv", "batch_id"])
        first = (int(frame["ts_recv"][0]), int(frame["batch_id"][0]))
        if last_key is not None and first < last_key:
            raise ReplayOrderError(f"book_l2 key {first} after {last_key}")
        last_key = (int(frame["ts_recv"][-1]), int(frame["batch_id"][-1]))
        if carry is not None:
            frame = pl.concat([carry, frame], how="vertical_relaxed")
            carry = None
        ids = frame.get_column("batch_id").to_numpy()
        starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
        ends = np.r_[starts[1:], len(ids)]
        # The last batch may continue in the next chunk: hold it back.
        for lo, hi in zip(starts[:-1], ends[:-1], strict=True):
            yield emit(frame.slice(int(lo), int(hi - lo)))
        carry = frame.slice(int(starts[-1]))
    if carry is not None and carry.height:
        yield emit(carry)


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
        frame = pl.from_arrow(chunk)
        assert isinstance(frame, pl.DataFrame)
        frame = (
            frame.with_columns(_trade_id_key(frame.get_column("trade_id").cast(pl.String)).alias("_k"))
            .sort(["ts_recv", "_k"], maintain_order=True)
        )
        first = int(frame["ts_recv"][0])
        if last_recv is not None and first < last_recv:
            raise ReplayOrderError(f"trades ts_recv {first} after {last_recv}")
        last_recv = int(frame["ts_recv"][-1])
        prices = to_ticks(frame.get_column("price").to_numpy(), tick_size)
        lots = to_lots(frame.get_column("qty").to_numpy(), lot_size)
        ts_ex = frame.get_column("ts_exchange").to_numpy()
        ts_recv = frame.get_column("ts_recv").to_numpy()
        sides = frame.get_column("side").to_numpy()
        ids = frame.get_column("trade_id").cast(pl.String).to_list()
        for i in range(frame.height):
            ex = int(ts_ex[i])
            if ex < ts_match:
                stats.exchange_time_clamped += 1
            ts_match = max(ts_match, ex)
            side = int(sides[i])
            stats.trades += 1
            if side == 0:
                stats.side0_trades += 1
            yield TradeRowEvent(
                ts_match=ts_match,
                ts_exchange=ex,
                ts_recv=int(ts_recv[i]),
                trade_id=str(ids[i]),
                price_ticks=int(prices[i]),
                qty_lots=int(lots[i]),
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
) -> Iterator[MarketEvent]:
    """The merged stream, starting at the first snapshot batch (DQ-03)."""
    stats = stats if stats is not None else ReplayStats()
    started = False
    for event in merge_streams(
        book_batches(book_chunks, tick_size, lot_size, stats),
        trade_rows(trade_chunks, tick_size, lot_size, stats),
    ):
        if not started:
            if isinstance(event, BookBatchEvent) and event.is_snapshot:
                started = True
            else:
                stats.dropped_before_snapshot += 1
                continue
        yield event
