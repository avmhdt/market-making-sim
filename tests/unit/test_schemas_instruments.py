from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pyarrow as pa
import pytest

from conftest import REPO
from mmsim.data.instruments import (
    FeeCoverageError,
    load_fee_rows,
    load_instrument_rows,
    resolve_instrument,
)
from mmsim.data.schemas import check_schema, validate_book_l2, validate_trades

INSTR = REPO / "configs" / "instruments.yaml"
FEES = REPO / "configs" / "fees.yaml"
Frames = tuple[pl.DataFrame, pl.DataFrame]


def test_fixture_frames_are_valid(mini_day_frames: Frames) -> None:
    book, trades = mini_day_frames
    assert validate_book_l2(book) == []
    assert validate_trades(trades) == []


def test_book_validator_catches_problems(mini_day_frames: Frames) -> None:
    book, _ = mini_day_frames
    assert "missing column batch_id" in validate_book_l2(book.drop("batch_id"))[0]
    assert any("non-decreasing" in p for p in validate_book_l2(book.reverse()))
    zero_side = (
        pl.when(pl.int_range(pl.len()) == 3)
        .then(0)
        .otherwise(pl.col("side"))
        .cast(pl.Int8)
        .alias("side")
    )
    assert any("side not in" in p for p in validate_book_l2(book.with_columns(zero_side)))
    neg = book.with_columns(pl.lit(-1.0).alias("qty"))
    assert any("negative qty" in p for p in validate_book_l2(neg))
    px = book.with_columns(pl.lit(0.0).alias("price"))
    assert any("non-positive price" in p for p in validate_book_l2(px))
    mixed = book.with_columns(pl.lit(1, dtype=pl.Int64).alias("batch_id"))
    assert any("mix snapshot" in p for p in validate_book_l2(mixed))
    nulls = book.with_columns(pl.lit(None, dtype=pl.Int64).alias("seq"))
    assert any("null values in seq" in p for p in validate_book_l2(nulls))
    assert validate_book_l2(book.head(0)) == []


def test_trade_validator_catches_problems(mini_day_frames: Frames) -> None:
    _, trades = mini_day_frames
    assert any("non-decreasing" in p for p in validate_trades(trades.reverse()))
    two = trades.with_columns(pl.lit(2, dtype=pl.Int8).alias("side"))
    assert any("side not in" in p for p in validate_trades(two))
    zero = trades.with_columns(pl.lit(0.0).alias("qty"))
    assert any("non-positive" in p for p in validate_trades(zero))
    text = trades.with_columns(pl.col("price").cast(pl.String))
    assert "expected double" in validate_trades(text)[0]


def test_schema_accepts_dictionary_strings_and_extra_columns() -> None:
    schema = pa.schema(
        [
            ("ts_exchange", pa.int64()),
            ("ts_recv", pa.int64()),
            ("venue", pa.dictionary(pa.int32(), pa.string())),
            ("symbol", pa.large_string()),
            ("trade_id", pa.string()),
            ("price", pa.float64()),
            ("qty", pa.float64()),
            ("side", pa.int32()),
            ("is_aggregated", pa.bool_()),
        ]
    )
    assert check_schema(schema, "trades") == []


def test_instruments_and_fee_resolution() -> None:
    rows = load_instrument_rows(INSTR)
    assert {r.symbol for r in rows} >= {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    spec = resolve_instrument(INSTR, FEES, "binance-usdm", "BTCUSDT", date(2026, 11, 3), "rebate")
    assert spec.tick_size == 0.1
    assert spec.maker_bps == -0.5
    assert spec.fee_tier == "rebate"
    assert all(r.hypothetical for r in load_fee_rows(FEES) if r.tier != "retail")
    with pytest.raises(FeeCoverageError):
        resolve_instrument(INSTR, FEES, "binance-usdm", "BTCUSDT", date(2026, 10, 31), "retail")
    with pytest.raises(KeyError):
        resolve_instrument(INSTR, FEES, "binance-usdm", "DOGEUSDT", date(2026, 11, 3), "retail")


def test_fee_rows_effective_to_and_bad_contract(tmp_path: Path) -> None:
    fees = tmp_path / "fees.yaml"
    fees.write_text(
        "schedules:\n"
        "  - {venue: v, tier: t, maker_bps: 1, taker_bps: 2,"
        " effective_from: 2026-01-01, effective_to: 2026-01-31}\n",
        encoding="utf-8",
    )
    (row,) = load_fee_rows(fees)
    assert row.covers(date(2026, 1, 31))
    assert not row.covers(date(2026, 2, 1))
    inst = tmp_path / "inst.yaml"
    inst.write_text(
        "instruments:\n"
        "  - {venue: v, symbol: s, tick_size: 1, lot_size: 1, contract_type: option}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_instrument_rows(inst)
