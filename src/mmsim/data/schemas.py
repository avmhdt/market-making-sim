"""Canonical dataset schemas and validators (spec section 7(b), DQ-13).

The canonical columns are shared with crypto-lob-alpha, tick-data-platform and
event-driven-backtester and are never renamed. Producers may add columns after them
(crypto-lob-alpha adds ``seq_first``, ``sync_epoch`` and others); validators check the
canonical ones and ignore the rest.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl
import pyarrow as pa

__all__ = [
    "BOOK_L2",
    "CANONICAL",
    "FUNDING",
    "GAPS",
    "INSTRUMENTS",
    "TRADES",
    "check_schema",
    "validate_book_l2",
    "validate_trades",
]

BOOK_L2 = pa.schema(
    [
        pa.field("ts_exchange", pa.int64(), nullable=False),
        pa.field("ts_recv", pa.int64(), nullable=False),
        pa.field("venue", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("seq", pa.int64(), nullable=False),
        pa.field("side", pa.int8(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
        pa.field("qty", pa.float64(), nullable=False),
        pa.field("is_snapshot", pa.bool_(), nullable=False),
        pa.field("batch_id", pa.int64(), nullable=False),
    ]
)

TRADES = pa.schema(
    [
        pa.field("ts_exchange", pa.int64(), nullable=False),
        pa.field("ts_recv", pa.int64(), nullable=False),
        pa.field("venue", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("trade_id", pa.string(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
        pa.field("qty", pa.float64(), nullable=False),
        pa.field("side", pa.int8(), nullable=False),
    ]
)

INSTRUMENTS = pa.schema(
    [
        pa.field("venue", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("tick_size", pa.float64(), nullable=False),
        pa.field("lot_size", pa.float64(), nullable=False),
        pa.field("contract_type", pa.string(), nullable=False),
        pa.field("maker_bps", pa.float64(), nullable=False),
        pa.field("taker_bps", pa.float64(), nullable=False),
        pa.field("effective_from", pa.date32(), nullable=False),
        pa.field("effective_to", pa.date32()),
        pa.field("fee_tier", pa.string(), nullable=False),
    ]
)

FUNDING = pa.schema(
    [
        pa.field("ts", pa.int64(), nullable=False),
        pa.field("venue", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("funding_rate", pa.float64(), nullable=False),
        pa.field("mark_price", pa.float64()),
    ]
)

GAPS = pa.schema(
    [
        pa.field("venue", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("stream", pa.string(), nullable=False),
        pa.field("start_ns", pa.int64(), nullable=False),
        pa.field("end_ns", pa.int64(), nullable=False),
        pa.field("reason", pa.string(), nullable=False),
    ]
)
"""The columns of crypto-lob-alpha's gap log that this project reads."""

CANONICAL: Mapping[str, pa.Schema] = {
    "book_l2": BOOK_L2,
    "trades": TRADES,
    "instruments": INSTRUMENTS,
    "funding": FUNDING,
    "gaps": GAPS,
}

_INTS = frozenset({pa.int8(), pa.int16(), pa.int32(), pa.int64()})


def _compatible(want: pa.DataType, got: pa.DataType) -> bool:
    if want == got:
        return True
    if pa.types.is_string(want):
        # Parquet readers may return dictionary-encoded or large strings.
        if pa.types.is_dictionary(got):
            value = got.value_type
            return bool(pa.types.is_string(value) or pa.types.is_large_string(value))
        return bool(pa.types.is_large_string(got))
    if want in _INTS:
        return got in _INTS
    return False


def check_schema(schema: pa.Schema, dataset: str) -> list[str]:
    """Problems with ``schema`` against the canonical columns of ``dataset`` (extras allowed)."""
    want = CANONICAL[dataset]
    problems: list[str] = []
    for field in want:
        index = schema.get_field_index(field.name)
        if index < 0:
            problems.append(f"{dataset}: missing column {field.name}")
            continue
        got = schema.field(index).type
        if not _compatible(field.type, got):
            problems.append(f"{dataset}: column {field.name} is {got}, expected {field.type}")
    return problems


def _nulls(df: pl.DataFrame, dataset: str) -> list[str]:
    problems: list[str] = []
    for field in CANONICAL[dataset]:
        if not field.nullable and field.name in df.columns:
            count = df.get_column(field.name).null_count()
            if count:
                problems.append(f"{dataset}: {count} null values in {field.name}")
    return problems


def _non_decreasing(df: pl.DataFrame, keys: list[str]) -> bool:
    """True if rows are non-decreasing in the lexicographic order of ``keys``."""
    if df.height < 2:
        return True
    ordered = df.select(keys).with_row_index("_row").sort(keys, maintain_order=True)
    return bool((ordered.get_column("_row").diff().drop_nulls() > 0).all())


def validate_book_l2(df: pl.DataFrame) -> list[str]:
    """DQ-13 checks for a book_l2 partition; an empty list means valid."""
    problems = check_schema(df.to_arrow().schema, "book_l2")
    if problems:
        return problems
    problems += _nulls(df, "book_l2")
    if problems or df.is_empty():
        return problems
    if not _non_decreasing(df, ["ts_recv", "batch_id"]):
        problems.append("book_l2: rows not non-decreasing in (ts_recv, batch_id)")
    bad_side = df.filter(~pl.col("side").is_in([1, -1])).height
    if bad_side:
        problems.append(f"book_l2: {bad_side} rows with side not in (+1, -1)")
    if df.filter(pl.col("price") <= 0).height:
        problems.append("book_l2: non-positive price")
    if df.filter(pl.col("qty") < 0).height:
        problems.append("book_l2: negative qty (0 removes a level)")
    mixed = (
        df.group_by("batch_id")
        .agg(pl.col("is_snapshot").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
        .height
    )
    if mixed:
        problems.append(f"book_l2: {mixed} batches mix snapshot and incremental rows")
    return problems


def validate_trades(df: pl.DataFrame) -> list[str]:
    """DQ-13 checks for a trades partition; an empty list means valid."""
    problems = check_schema(df.to_arrow().schema, "trades")
    if problems:
        return problems
    problems += _nulls(df, "trades")
    if problems or df.is_empty():
        return problems
    if not _non_decreasing(df, ["ts_recv"]):
        problems.append("trades: rows not non-decreasing in ts_recv")
    bad_side = df.filter(~pl.col("side").is_in([1, -1, 0])).height
    if bad_side:
        problems.append(f"trades: {bad_side} rows with side not in (+1, -1, 0)")
    if df.filter((pl.col("price") <= 0) | (pl.col("qty") <= 0)).height:
        problems.append("trades: non-positive price or qty")
    return problems
