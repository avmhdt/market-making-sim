"""Shared fixtures: the mini_day CSVs converted to canonical Parquet in a temp folder."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from mmsim.io.parquet import write_canonical

FIXTURES = Path(__file__).parent / "fixtures"
REPO = Path(__file__).parents[1]

BOOK_DTYPES = {
    "ts_exchange": pl.Int64,
    "ts_recv": pl.Int64,
    "venue": pl.String,
    "symbol": pl.String,
    "seq": pl.Int64,
    "side": pl.Int8,
    "price": pl.Float64,
    "qty": pl.Float64,
    "is_snapshot": pl.Boolean,
    "batch_id": pl.Int64,
}
TRADE_DTYPES = {
    "ts_exchange": pl.Int64,
    "ts_recv": pl.Int64,
    "venue": pl.String,
    "symbol": pl.String,
    "trade_id": pl.String,
    "price": pl.Float64,
    "qty": pl.Float64,
    "side": pl.Int8,
}


def read_fixture(dataset: str) -> pl.DataFrame:
    """One mini_day CSV with canonical dtypes."""
    dtypes = BOOK_DTYPES if dataset == "book_l2" else TRADE_DTYPES
    return pl.read_csv(FIXTURES / "mini_day" / f"{dataset}.csv", schema_overrides=dtypes)


@pytest.fixture
def mini_day_root(tmp_path: Path) -> Path:
    """A canonical root holding the mini_day fixture as hourly Parquet partitions."""
    root = tmp_path / "canonical"
    write_canonical(read_fixture("book_l2"), root, "book_l2")
    write_canonical(read_fixture("trades"), root, "trades")
    return root


@pytest.fixture
def mini_day_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    """(book_l2, trades) of the mini_day fixture."""
    return read_fixture("book_l2"), read_fixture("trades")
