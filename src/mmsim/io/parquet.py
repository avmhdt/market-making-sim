"""Canonical partition layout and Parquet helpers (spec section 7(d)).

Layout: ``<root>/<dataset>/venue=<venue>/symbol=<symbol>/date=YYYY-MM-DD/part-HH.parquet``
where the date and the hour ``HH`` are those of ``ts_recv`` in UTC and the symbol is
URI-percent-encoded (``BTC/USD`` becomes ``BTC%2FUSD``).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from urllib.parse import quote, unquote

import polars as pl

__all__ = [
    "decode_symbol",
    "encode_symbol",
    "partition_dir",
    "symbol_dir",
    "write_canonical",
    "write_parquet",
]

NS_PER_HOUR = 3_600_000_000_000


def encode_symbol(symbol: str) -> str:
    """Percent-encode a venue-native symbol for a partition directory name."""
    return quote(symbol, safe="")


def decode_symbol(encoded: str) -> str:
    """Inverse of :func:`encode_symbol`."""
    return unquote(encoded)


def symbol_dir(root: Path, dataset: str, venue: str, symbol: str) -> Path:
    """``<root>/<dataset>/venue=<venue>/symbol=<encoded symbol>``."""
    return root / dataset / f"venue={venue}" / f"symbol={encode_symbol(symbol)}"


def partition_dir(root: Path, dataset: str, venue: str, symbol: str, day: date) -> Path:
    """The directory holding one symbol-day of ``dataset``."""
    return symbol_dir(root, dataset, venue, symbol) / f"date={day.isoformat()}"


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write ``df`` as zstd Parquet, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd", statistics=True)


def write_canonical(df: pl.DataFrame, root: Path, dataset: str) -> list[Path]:
    """Write a canonical frame into the partition layout, one file per UTC hour of ``ts_recv``.

    Rows keep their order within each file. Every row needs ``venue``, ``symbol`` and
    ``ts_recv``. Returns the written paths, sorted.
    """
    if df.is_empty():
        return []
    keyed = df.with_columns(
        (pl.col("ts_recv") // NS_PER_HOUR).alias("_hour"),
        pl.from_epoch(pl.col("ts_recv"), time_unit="ns").dt.date().alias("_date"),
    )
    written: list[Path] = []
    for (venue, symbol, day, hour), part in keyed.group_by(
        ["venue", "symbol", "_date", "_hour"], maintain_order=True
    ):
        assert isinstance(venue, str) and isinstance(symbol, str) and isinstance(day, date)
        assert isinstance(hour, int)
        path = partition_dir(root, dataset, venue, symbol, day) / f"part-{hour % 24:02d}.parquet"
        if path.exists():
            raise FileExistsError(f"{path} exists; canonical partitions are immutable")
        write_parquet(part.drop("_hour", "_date"), path)
        written.append(path)
    return sorted(written)
