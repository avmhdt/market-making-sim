"""Partition-pruned access to canonical datasets (FR-04).

Readers never list more than the requested symbol and days, and stream one Parquet row
group at a time so that a symbol-day never has to fit in memory. Nothing here writes to
the source root: crypto-lob-alpha's capture is read in place, read-only (P-05).
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from mmsim.io.parquet import partition_dir, symbol_dir

__all__ = [
    "day_files",
    "iter_row_groups",
    "list_days",
    "load_gaps",
    "read_day",
    "read_files",
]

_DATE_DIR = re.compile(r"^date=(\d{4}-\d{2}-\d{2})$")


def list_days(
    root: Path,
    dataset: str,
    venue: str,
    symbol: str,
    start: date | None = None,
    end: date | None = None,
) -> list[date]:
    """Sorted UTC days with a partition directory for ``dataset``, within ``[start, end]``."""
    base = symbol_dir(root, dataset, venue, symbol)
    if not base.is_dir():
        return []
    days: list[date] = []
    for child in base.iterdir():
        match = _DATE_DIR.match(child.name)
        if not match or not child.is_dir():
            continue
        day = date.fromisoformat(match.group(1))
        if (start is None or day >= start) and (end is None or day <= end):
            days.append(day)
    return sorted(days)


def day_files(root: Path, dataset: str, venue: str, symbol: str, day: date) -> list[Path]:
    """The Parquet parts of one symbol-day, in name (hour) order."""
    directory = partition_dir(root, dataset, venue, symbol, day)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix == ".parquet" and p.is_file())


def iter_row_groups(
    files: Sequence[Path], columns: Sequence[str] | None = None
) -> Iterator[pa.RecordBatch]:
    """Yield the files' contents one row group at a time, in file order."""
    for path in files:
        parquet = pq.ParquetFile(path)
        try:
            for index in range(parquet.num_row_groups):
                table = parquet.read_row_group(index, columns=list(columns) if columns else None)
                yield from table.combine_chunks().to_batches()
        finally:
            parquet.close()


def read_files(files: Sequence[Path], columns: Sequence[str] | None = None) -> pl.DataFrame:
    """Read whole files into one frame (fixtures, deep checks, small datasets)."""
    if not files:
        return pl.DataFrame()
    tables = [pq.read_table(path, columns=list(columns) if columns else None) for path in files]
    frame = pl.from_arrow(pa.concat_tables(tables, promote_options="default"))
    assert isinstance(frame, pl.DataFrame)
    return frame


def read_day(
    root: Path,
    dataset: str,
    venue: str,
    symbol: str,
    day: date,
    columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """One whole symbol-day of ``dataset``."""
    return read_files(day_files(root, dataset, venue, symbol, day), columns)


def load_gaps(root: Path, venue: str, symbol: str, start: date, end: date) -> pl.DataFrame:
    """Gap-log rows of ``symbol`` and of all symbols (``*``) that overlap ``[start, end]``.

    The gap log is partitioned by the day a gap started, so the day before ``start`` is read
    too. Returns an empty frame if there is no gap log.
    """
    frames: list[pl.DataFrame] = []
    first = start - timedelta(days=1)
    for name in (symbol, "*"):
        for day in list_days(root, "gaps", venue, name, first, end):
            frame = read_day(root, "gaps", venue, name, day)
            if not frame.is_empty():
                frames.append(frame)
    if not frames:
        return pl.DataFrame()
    gaps = pl.concat(frames, how="diagonal_relaxed")
    lo = _day_start_ns(start)
    hi = _day_start_ns(end + timedelta(days=1))
    return gaps.filter((pl.col("end_ns") > lo) & (pl.col("start_ns") < hi)).sort("start_ns")


def _day_start_ns(day: date) -> int:
    return (day - date(1970, 1, 1)).days * 86_400 * 1_000_000_000
