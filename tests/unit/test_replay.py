from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from conftest import BOOK_DTYPES, TRADE_DTYPES
from mmsim.data.loaders import day_files, iter_row_groups, list_days, read_day, read_files
from mmsim.data.replay import (
    BookBatchEvent,
    ReplayOrderError,
    ReplayStats,
    TickGridError,
    TradeRowEvent,
    book_batches,
    merge_streams,
    replay,
    to_lots,
    to_ticks,
    trade_rows,
)

BookRow = tuple[int, int, int, float, float, bool, int]
TradeRow = tuple[int, int, str, float, float, int]


def make_book(rows: list[BookRow]) -> pl.DataFrame:
    """rows: (ts_exchange, ts_recv, side, price, qty, is_snapshot, batch_id)."""
    return pl.DataFrame(
        {
            "ts_exchange": [r[0] for r in rows],
            "ts_recv": [r[1] for r in rows],
            "venue": ["v"] * len(rows),
            "symbol": ["s"] * len(rows),
            "seq": [r[6] for r in rows],
            "side": [r[2] for r in rows],
            "price": [r[3] for r in rows],
            "qty": [r[4] for r in rows],
            "is_snapshot": [r[5] for r in rows],
            "batch_id": [r[6] for r in rows],
        },
        schema=BOOK_DTYPES,
    )


def make_trades(rows: list[TradeRow]) -> pl.DataFrame:
    """rows: (ts_exchange, ts_recv, trade_id, price, qty, side)."""
    return pl.DataFrame(
        {
            "ts_exchange": [r[0] for r in rows],
            "ts_recv": [r[1] for r in rows],
            "venue": ["v"] * len(rows),
            "symbol": ["s"] * len(rows),
            "trade_id": [r[2] for r in rows],
            "price": [r[3] for r in rows],
            "qty": [r[4] for r in rows],
            "side": [r[5] for r in rows],
        },
        schema=TRADE_DTYPES,
    )


def test_grid_conversion() -> None:
    assert to_ticks(np.array([70123.4, 0.3]), 0.1).tolist() == [701234, 3]
    assert to_lots(np.array([0.005]), 0.001).tolist() == [5]
    with pytest.raises(TickGridError):
        to_ticks(np.array([1.05]), 0.1)


def test_batches_are_never_split_across_row_groups() -> None:
    book = make_book(
        [
            (10, 11, 1, 1.0, 1.0, True, 1),
            (10, 11, -1, 1.2, 1.0, True, 1),
            (20, 21, 1, 1.1, 2.0, False, 2),
            (20, 21, -1, 1.2, 0.0, False, 2),
        ]
    )
    chunks = book.to_arrow().to_batches(max_chunksize=1)
    events = list(book_batches(chunks, 0.1, 1.0))
    assert [e.batch_id for e in events] == [1, 2]
    assert [len(e.price_ticks) for e in events] == [2, 2]
    assert events[0].is_snapshot
    assert not events[1].is_snapshot
    assert events[1].qty_lots.tolist() == [2, 0]


def test_matching_time_is_monotone_and_clamps_are_counted() -> None:
    book = make_book([(30, 31, 1, 1.0, 1.0, True, 1), (20, 32, 1, 1.0, 2.0, False, 2)])
    trades = make_trades([(25, 26, "2", 1.0, 1.0, 1), (24, 27, "10", 1.0, 1.0, 0)])
    stats = ReplayStats()
    b = list(book_batches(book.to_arrow().to_batches(), 0.1, 1.0, stats))
    t = list(trade_rows(trades.to_arrow().to_batches(), 0.1, 1.0, stats))
    assert [e.ts_match for e in b] == [30, 30]
    assert [e.ts_match for e in t] == [25, 25]
    assert stats.exchange_time_clamped == 2
    assert stats.side0_trades == 1


def test_trades_sort_by_numeric_id_within_equal_recv() -> None:
    trades = make_trades([(5, 9, "10", 1.0, 1.0, 1), (5, 9, "9", 1.0, 1.0, 1)])
    ids = [e.trade_id for e in trade_rows(trades.to_arrow().to_batches(), 0.1, 1.0)]
    assert ids == ["9", "10"]
    text = make_trades([(5, 9, "b", 1.0, 1.0, 1), (5, 9, "a", 1.0, 1.0, 1)])
    ids = [e.trade_id for e in trade_rows(text.to_arrow().to_batches(), 0.1, 1.0)]
    assert ids == ["b", "a"]


def test_merge_puts_trades_first_at_equal_matching_time() -> None:
    book = make_book([(10, 12, 1, 1.0, 1.0, True, 1), (20, 21, 1, 1.0, 2.0, False, 2)])
    trades = make_trades([(5, 6, "1", 1.0, 1.0, 1), (20, 25, "2", 1.0, 1.0, 1)])
    stats = ReplayStats()
    events = list(
        replay(book.to_arrow().to_batches(), trades.to_arrow().to_batches(), 0.1, 1.0, stats)
    )
    kinds = [type(e).__name__ for e in events]
    # The trade at 5 comes before the first snapshot and is dropped (DQ-03).
    assert kinds == ["BookBatchEvent", "TradeRowEvent", "BookBatchEvent"]
    assert stats.dropped_before_snapshot == 1


def test_out_of_order_chunks_raise() -> None:
    book = make_book([(10, 20, 1, 1.0, 1.0, True, 1), (10, 10, 1, 1.0, 1.0, False, 2)])
    with pytest.raises(ReplayOrderError):
        list(book_batches(book.to_arrow().to_batches(max_chunksize=1), 0.1, 1.0))
    trades = make_trades([(5, 9, "1", 1.0, 1.0, 1), (5, 8, "2", 1.0, 1.0, 1)])
    with pytest.raises(ReplayOrderError):
        list(trade_rows(trades.to_arrow().to_batches(max_chunksize=1), 0.1, 1.0))


def test_empty_chunks_are_skipped() -> None:
    book_schema = make_book([(1, 1, 1, 1.0, 1.0, True, 1)]).to_arrow().schema
    trade_schema = make_trades([(1, 1, "1", 1.0, 1.0, 1)]).to_arrow().schema
    assert list(book_batches([pa.RecordBatch.from_pylist([], schema=book_schema)], 0.1, 1.0)) == []
    assert list(trade_rows([pa.RecordBatch.from_pylist([], schema=trade_schema)], 0.1, 1.0)) == []


def test_fixture_replays_from_partitions(mini_day_root: Path) -> None:
    (day,) = list_days(mini_day_root, "book_l2", "synthetic", "SYNUSDT")
    later = day + timedelta(days=1)
    assert list_days(mini_day_root, "book_l2", "synthetic", "SYNUSDT", start=later) == []
    assert list_days(mini_day_root, "book_l2", "synthetic", "NOPE") == []
    assert day_files(mini_day_root, "book_l2", "synthetic", "SYNUSDT", later) == []
    assert read_files([]).is_empty()
    book_files = day_files(mini_day_root, "book_l2", "synthetic", "SYNUSDT", day)
    trade_files = day_files(mini_day_root, "trades", "synthetic", "SYNUSDT", day)
    stats = ReplayStats()
    events = list(
        replay(iter_row_groups(book_files), iter_row_groups(trade_files), 0.1, 0.001, stats)
    )
    trades = read_day(mini_day_root, "trades", "synthetic", "SYNUSDT", day)
    assert stats.trades == trades.height
    assert sum(isinstance(e, TradeRowEvent) for e in events) == trades.height
    keys = [(e.ts_match, 0 if isinstance(e, TradeRowEvent) else 1) for e in events]
    assert keys == sorted(keys)
    first = events[0]
    assert isinstance(first, BookBatchEvent)
    assert first.is_snapshot


def test_merge_streams_is_stable_for_equal_keys() -> None:
    a = TradeRowEvent(1, 1, 1, "1", 1, 1, 1)
    b = TradeRowEvent(1, 1, 1, "2", 1, 1, 1)
    assert list(merge_streams([], [a, b])) == [a, b]
