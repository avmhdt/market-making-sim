"""Partition-pruned access to canonical datasets (FR-04).

Readers never list more than the requested symbol and days, and stream one Parquet row
group at a time so that a symbol-day never has to fit in memory. Nothing here writes to
the source root: crypto-lob-alpha's capture is read in place, read-only (P-05).
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from mmsim.io.parquet import partition_dir, symbol_dir

__all__ = [
    "Checkpoint",
    "day_files",
    "gap_intervals",
    "iter_row_groups",
    "list_days",
    "load_checkpoint",
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


_PART = re.compile(r"^part-(\d+)(?:-r(\d+))?\.parquet$")


def _part_key(path: Path) -> tuple[int, int, str]:
    """Hour, then restart number: ``part-21.parquet`` before ``part-21-r1.parquet``."""
    m = _PART.match(path.name)
    if not m:
        return (10**9, 0, path.name)
    return (int(m.group(1)), int(m.group(2) or 0), path.name)


def day_files(root: Path, dataset: str, venue: str, symbol: str, day: date) -> list[Path]:
    """The closed Parquet parts of one symbol-day, in hour then restart order.

    The recorder writes ``part-HH.parquet`` per UTC hour of ``ts_recv`` and, after a restart
    within the hour, ``part-HH-r1.parquet`` and so on; files still being written end in
    ``.tmp`` and are skipped.
    """
    directory = partition_dir(root, dataset, venue, symbol, day)
    if not directory.is_dir():
        return []
    files = [p for p in directory.iterdir() if p.suffix == ".parquet" and p.is_file()]
    return sorted(files, key=_part_key)


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


@dataclass(frozen=True)
class Checkpoint:
    """A full book captured by the recorder and checked against its live book."""

    checkpoint_id: str
    ts_recv: int
    seq: int
    batch_id: int
    levels: pl.DataFrame
    """``side``, ``price``, ``qty`` of every level."""


def load_checkpoint(root: Path, venue: str, symbol: str, day: date) -> Checkpoint | None:
    """The first live-verified (``ok_live``) checkpoint of ``day``, or ``None``.

    crypto-lob-alpha writes a REST L2 snapshot about every 600 s; incremental days contain
    no snapshot batch unless the feed reconnected, so a day's replay starts from this.
    """
    meta = read_day(root, "checkpoints", venue, symbol, day)
    if meta.is_empty():
        return None
    meta = meta.filter(pl.col("ok_live")).sort("ts_recv")
    if meta.is_empty():
        return None
    first = meta.row(0, named=True)
    levels = read_day(root, "checkpoint_levels", venue, symbol, day)
    if levels.is_empty():
        return None
    levels = levels.filter(pl.col("checkpoint_id") == first["checkpoint_id"])
    if levels.is_empty():
        return None
    return Checkpoint(
        checkpoint_id=str(first["checkpoint_id"]),
        ts_recv=int(first["ts_recv"]),
        seq=int(first["seq"]),
        batch_id=int(first["batch_id"]),
        levels=levels.select("side", "price", "qty"),
    )


GAP_STREAMS = ("book", "trades")
"""Gap-log streams that interrupt replay (``bbo`` is not replayed)."""


def gap_intervals(
    gaps: pl.DataFrame, *, exclude_reasons: Sequence[str] = ("clock",)
) -> list[tuple[int, int]]:
    """Merged ``[start, end)`` intervals of book or trades gaps, excluding ``clock`` rows."""
    if gaps.is_empty():
        return []
    rows = (
        gaps.filter(
            pl.col("stream").is_in(list(GAP_STREAMS))
            & ~pl.col("reason").is_in(list(exclude_reasons))
        )
        .select("start_ns", "end_ns")
        .sort("start_ns")
        .rows()
    )
    merged: list[tuple[int, int]] = []
    for s, e in rows:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((int(s), int(e)))
    return merged
