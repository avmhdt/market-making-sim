"""Replay order is a deterministic total order (DQ-01)."""

from __future__ import annotations

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import BOOK_DTYPES, TRADE_DTYPES
from mmsim.data.replay import BookBatchEvent, TradeRowEvent, replay

times = st.lists(st.integers(min_value=0, max_value=50), min_size=1, max_size=30)


def _book(ex: list[int]) -> pl.DataFrame:
    n = len(ex)
    recv = sorted(e + 5 for e in ex)
    return pl.DataFrame(
        {
            "ts_exchange": ex,
            "ts_recv": recv,
            "venue": ["v"] * n,
            "symbol": ["s"] * n,
            "seq": list(range(1, n + 1)),
            "side": [1] * n,
            "price": [1.0] * n,
            "qty": [1.0] * n,
            "is_snapshot": [i == 0 for i in range(n)],
            "batch_id": list(range(1, n + 1)),
        },
        schema=BOOK_DTYPES,
    )


def _trades(ex: list[int]) -> pl.DataFrame:
    n = len(ex)
    return pl.DataFrame(
        {
            "ts_exchange": ex,
            "ts_recv": sorted(e + 3 for e in ex),
            "venue": ["v"] * n,
            "symbol": ["s"] * n,
            "trade_id": [str(i) for i in range(n)],
            "price": [1.0] * n,
            "qty": [1.0] * n,
            "side": [1] * n,
        },
        schema=TRADE_DTYPES,
    )


@settings(max_examples=200, deadline=None)
@given(book_ex=times, trade_ex=times, chunk=st.integers(min_value=1, max_value=7))
def test_replay_is_sorted_complete_and_deterministic(
    book_ex: list[int], trade_ex: list[int], chunk: int
) -> None:
    book, trades = _book(book_ex), _trades(trade_ex)

    def run() -> list[tuple[int, int, int]]:
        events = replay(
            book.to_arrow().to_batches(max_chunksize=chunk),
            trades.to_arrow().to_batches(max_chunksize=chunk),
            1.0,
            1.0,
        )
        return [(e.ts_match, 0 if isinstance(e, TradeRowEvent) else 1, e.ts_recv) for e in events]

    first = run()
    assert first == run()
    assert first == sorted(first)
    n_book = sum(1 for k in first if k[1] == 1)
    assert n_book == book.height
    assert first[0][1] == 1  # starts at the snapshot
    assert isinstance(BookBatchEvent, type)
