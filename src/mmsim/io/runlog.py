"""Per-run output directory ``results/runs/<run_id>/`` (spec section 7(b)).

A run directory holds the resolved config (secrets redacted), ``metrics.json``
(schema ``mms.metrics/1``), the daily ``returns.parquet`` the trial log points at, and
the event logs written by the runner from M2 on.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from mmsim.explog import new_run_id
from mmsim.io.parquet import write_parquet

__all__ = ["LF", "METRICS_SCHEMA", "RUNTIME_FIELDS", "RunDir", "normalized_metrics"]

LF = "\n"
"""Line ending for every text output, so files match across operating systems."""
METRICS_SCHEMA = "mms.metrics/1"
RUNTIME_FIELDS = frozenset({"runtime_s", "events_per_s", "finished_utc", "package_versions"})
"""Fields excluded from determinism comparisons (NFR-03)."""


@dataclass(frozen=True)
class RunDir:
    """One run's output folder."""

    run_id: str
    path: Path

    @classmethod
    def create(cls, results_dir: Path, run_id: str | None = None) -> RunDir:
        """Create ``<results_dir>/runs/<run_id>/``; the id is a new UUID v4 by default."""
        run_id = run_id or new_run_id()
        path = results_dir / "runs" / run_id
        path.mkdir(parents=True, exist_ok=False)
        return cls(run_id, path)

    @property
    def returns_path(self) -> Path:
        """The daily series referenced by the trial log."""
        return self.path / "returns.parquet"

    def write_config(self, resolved: dict[str, Any], settings: dict[str, str | None]) -> Path:
        """Write ``config.resolved.yaml`` with the env-derived settings (already redacted)."""
        target = self.path / "config.resolved.yaml"
        payload = {"config": resolved, "settings": settings}
        target.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8", newline=LF)
        return target

    def write_metrics(self, metrics: dict[str, Any]) -> Path:
        """Write ``metrics.json`` with the schema tag; non-finite floats are rejected."""
        target = self.path / "metrics.json"
        body = {"schema": METRICS_SCHEMA, **metrics}
        target.write_text(
            json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + LF,
            encoding="utf-8",
            newline=LF,
        )
        return target

    def write_returns(self, dates: list[date], pnl_quote: list[float], capital: float) -> Path:
        """Write the daily ``returns.parquet``: ``date``, ``ret = pnl / capital``, ``pnl_quote``."""
        if len(dates) != len(pnl_quote):
            raise ValueError("dates and pnl_quote differ in length")
        if any(b <= a for a, b in itertools.pairwise(dates)):
            raise ValueError("dates must be strictly increasing")
        if not all(math.isfinite(x) for x in pnl_quote):
            raise ValueError("non-finite P&L")
        frame = pl.DataFrame(
            {"date": dates, "ret": [x / capital for x in pnl_quote], "pnl_quote": pnl_quote},
            schema={"date": pl.Date, "ret": pl.Float64, "pnl_quote": pl.Float64},
        )
        write_parquet(frame, self.returns_path)
        return self.returns_path


def normalized_metrics(path: Path) -> dict[str, Any]:
    """``metrics.json`` without runtime fields, for determinism checks."""
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return {key: value for key, value in data.items() if key not in RUNTIME_FIELDS}
