"""``mms data check``: data-access smoke test (AC-W-11, spec section 7(e)).

Checks that the paths exist, that every partition has the canonical schema, and reports
row counts and time coverage per symbol-day, the gap summary from the gap log, and
instrument and fee coverage (DQ-08). By default it reads only Parquet footers (schema,
row counts, ``ts_recv`` statistics), which keeps it light on the HDD the live recorder
writes to; ``deep`` also reads the rows and runs the DQ-13 content checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from mmsim.config import DataConfig, FeesConfig
from mmsim.data.instruments import FeeCoverageError, resolve_instrument
from mmsim.data.loaders import day_files, list_days, load_gaps, read_files
from mmsim.data.schemas import check_schema, validate_book_l2, validate_trades

__all__ = ["CheckReport", "DayCoverage", "run_check"]

NS_PER_S = 1_000_000_000
NOT_DOWNTIME = frozenset({"clock"})
"""Gap reasons that crypto-lob-alpha logs but does not count as downtime."""


@dataclass
class DayCoverage:
    """What one symbol-day holds."""

    symbol: str
    day: date
    book_rows: int = 0
    trade_rows: int = 0
    book_files: int = 0
    trade_files: int = 0
    first_ns: int | None = None
    last_ns: int | None = None
    gaps: int = 0
    gap_s: float = 0.0
    clock_gap_s: float = 0.0
    fee_ok: bool = False

    @property
    def span_h(self) -> float:
        """Hours between the first and last ``ts_recv`` seen."""
        if self.first_ns is None or self.last_ns is None:
            return 0.0
        return (self.last_ns - self.first_ns) / NS_PER_S / 3600


@dataclass
class CheckReport:
    """The check's findings; ``problems`` empty means it passed."""

    root: Path
    days: list[DayCoverage] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no problem was found."""
        return not self.problems

    def lines(self) -> list[str]:
        """A printable summary."""
        out = [f"root: {self.root}"]
        out.append(
            f"{'symbol':<10} {'date':<10} {'book_rows':>11} {'trades':>9} {'files':>5} "
            f"{'span_h':>6} {'gaps':>4} {'gap_s':>8} {'clock_s':>8} fees"
        )
        for d in self.days:
            out.append(
                f"{d.symbol:<10} {d.day.isoformat():<10} {d.book_rows:>11,} {d.trade_rows:>9,} "
                f"{d.book_files:>5} {d.span_h:>6.2f} {d.gaps:>4} {d.gap_s:>8.1f} "
                f"{d.clock_gap_s:>8.1f} {'ok' if d.fee_ok else 'MISSING'}"
            )
        out.append(f"{len(self.days)} symbol-days, {len(self.problems)} problems")
        out.extend(f"PROBLEM: {p}" for p in self.problems)
        return out


def _footer(
    files: list[Path], dataset: str, problems: list[str]
) -> tuple[int, int | None, int | None]:
    rows = 0
    lo: int | None = None
    hi: int | None = None
    for path in files:
        meta = pq.ParquetFile(path)
        try:
            for issue in check_schema(meta.schema_arrow, dataset):
                problems.append(f"{path}: {issue}")
            rows += meta.metadata.num_rows
            index = meta.schema_arrow.get_field_index("ts_recv")
            if index < 0:
                continue
            for group in range(meta.metadata.num_row_groups):
                col = meta.metadata.row_group(group).column(index)
                stats = col.statistics
                if stats is None or not stats.has_min_max:
                    problems.append(f"{path}: no ts_recv statistics")
                    continue
                lo = int(stats.min) if lo is None else min(lo, int(stats.min))
                hi = int(stats.max) if hi is None else max(hi, int(stats.max))
        finally:
            meta.close()
    return rows, lo, hi


def run_check(
    root: Path,
    data: DataConfig,
    fees: FeesConfig,
    *,
    symbols: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    deep: bool = False,
    repo_root: Path = Path(),
) -> CheckReport:
    """Run the check over ``symbols`` (default: the config's) and ``[start, end]``."""
    report = CheckReport(root=root)
    start = start or data.start
    end = end or data.end
    instruments_path = repo_root / data.instruments
    fees_path = repo_root / fees.schedule
    for path, what in ((root, "data root"), (instruments_path, "instruments"), (fees_path, "fees")):
        if not path.exists():
            report.problems.append(f"{what} {path} does not exist")
    if report.problems:
        return report
    for symbol in symbols or data.symbols:
        days = sorted(
            set(list_days(root, "book_l2", data.venue, symbol, start, end))
            | set(list_days(root, "trades", data.venue, symbol, start, end))
        )
        if not days:
            report.problems.append(f"{data.venue}/{symbol}: no book_l2 or trades days in window")
            continue
        gaps = load_gaps(root, data.venue, symbol, days[0], days[-1])
        for day in days:
            cov = DayCoverage(symbol=symbol, day=day)
            book = day_files(root, "book_l2", data.venue, symbol, day)
            trades = day_files(root, "trades", data.venue, symbol, day)
            cov.book_files, cov.trade_files = len(book), len(trades)
            if not book:
                report.problems.append(f"{symbol} {day}: no book_l2 files")
            if not trades:
                report.problems.append(f"{symbol} {day}: no trades files")
            cov.book_rows, b_lo, b_hi = _footer(book, "book_l2", report.problems)
            cov.trade_rows, t_lo, t_hi = _footer(trades, "trades", report.problems)
            los = [x for x in (b_lo, t_lo) if x is not None]
            his = [x for x in (b_hi, t_hi) if x is not None]
            cov.first_ns = min(los) if los else None
            cov.last_ns = max(his) if his else None
            if deep:
                for issue in validate_book_l2(read_files(book)) if book else []:
                    report.problems.append(f"{symbol} {day}: {issue}")
                for issue in validate_trades(read_files(trades)) if trades else []:
                    report.problems.append(f"{symbol} {day}: {issue}")
            if not gaps.is_empty():
                lo = (day - date(1970, 1, 1)).days * 86_400 * NS_PER_S
                hi = lo + 86_400 * NS_PER_S
                today = gaps.filter((pl.col("end_ns") > lo) & (pl.col("start_ns") < hi))
                clipped = today.with_columns(
                    (
                        (
                            pl.min_horizontal(pl.col("end_ns"), pl.lit(hi))
                            - pl.max_horizontal(pl.col("start_ns"), pl.lit(lo))
                        )
                        / NS_PER_S
                    ).alias("_s")
                )
                is_clock = pl.col("reason").is_in(list(NOT_DOWNTIME))
                cov.gaps = clipped.filter(~is_clock).height
                cov.gap_s = float(clipped.filter(~is_clock).get_column("_s").sum())
                cov.clock_gap_s = float(clipped.filter(is_clock).get_column("_s").sum())
            try:
                resolve_instrument(instruments_path, fees_path, data.venue, symbol, day, fees.tier)
                cov.fee_ok = True
            except (KeyError, FeeCoverageError) as exc:
                report.problems.append(f"{symbol} {day}: {exc}")
            report.days.append(cov)
    return report
