"""Run lifecycle: run directory, outputs and the experiment-log row (FR-25).

The event loop and quote manager arrive in M2. What exists from M1 is the part every run
shares: a run either finishes, writing its returns and metrics and appending a trial row,
or fails, appending a row with ``n_obs = 0``. No code path skips the log.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from mmsim.explog import append_failed, append_trial
from mmsim.io.runlog import RunDir

__all__ = ["fail_run", "finish_run"]


def finish_run(
    run: RunDir,
    *,
    explog_path: Path,
    strategy_family: str,
    params: dict[str, object],
    dates: list[date],
    pnl_quote: list[float],
    capital: float,
    metrics: dict[str, Any],
    selected: bool = False,
) -> dict[str, str]:
    """Write ``returns.parquet`` and ``metrics.json``, then append the trial row."""
    if not dates:
        raise ValueError("a finished run needs at least one daily return")
    run.write_returns(dates, pnl_quote, capital)
    run.write_metrics(metrics)
    return append_trial(
        explog_path,
        run_id=run.run_id,
        strategy_family=strategy_family,
        params=params,
        start=dates[0],
        end=dates[-1],
        returns_path=run.returns_path,
        selected=selected,
    )


def fail_run(
    run: RunDir,
    *,
    explog_path: Path,
    strategy_family: str,
    params: dict[str, object],
    error: str,
    start: date | None = None,
    end: date | None = None,
) -> dict[str, str]:
    """Record a failed run: ``metrics.json`` with the error, and an ``n_obs = 0`` row."""
    run.write_metrics({"status": "failed", "error": error})
    return append_failed(
        explog_path,
        run_id=run.run_id,
        strategy_family=strategy_family,
        params=params,
        start=start,
        end=end,
    )
