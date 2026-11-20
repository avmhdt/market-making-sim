"""Record NFR-01, NFR-02 and NFR-12 in reports/tables/benchmark.csv (AC-W-15).

Usage (WSL, from the repo): python scripts/benchmark.py [--run-id ID ...]

- NFR-01: events per second of the fallback with GLFT on a ~1M-event synthetic stream.
- NFR-02 and NFR-12: wall time and peak memory of real symbol-day replays, read from the
  given runs' ``metrics.json`` and ``sessions.parquet``.

Published benchmark runs wait until no multi-week run is active on the shared PC (after
about 2026-11-29); rows measured before then are labelled provisional, with the 1-minute
load average at the time.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tests"))

from bench.test_throughput import TARGET_EVENTS_PER_S, benchmark_stream  # noqa: E402
from mmsim.reporting.report import write_table  # noqa: E402

PUBLISH_AFTER = datetime(2026, 11, 29, 21, 0, tzinfo=UTC)


def main() -> int:
    """Measure and write the table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--results", type=Path, default=REPO / "results")
    args = parser.parse_args()
    now = datetime.now(UTC)
    load = os.getloadavg()[0]
    provisional = now < PUBLISH_AFTER
    rows: list[dict[str, object]] = []
    result = benchmark_stream()
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    common = {"measured_utc": now.strftime("%Y-%m-%dT%H:%MZ"), "load_1m": round(load, 1),
              "provisional": provisional, "python": platform.python_version()}  # fmt: skip
    rows.append({"metric": "NFR-01 events/s, synthetic 1M-event stream, GLFT",
                 "target": f">= {TARGET_EVENTS_PER_S}", "value": round(result["events_per_s"]),
                 "met": result["events_per_s"] >= TARGET_EVENTS_PER_S, **common})  # fmt: skip
    rows.append({"metric": "NFR-12 peak RSS MB, synthetic stream", "target": "<= 4096",
                 "value": round(rss_mb), "met": rss_mb <= 4096, **common})  # fmt: skip
    for run_id in args.run_id:
        run = args.results / "runs" / run_id
        m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        days = pl.read_parquet(run / "sessions.parquet").height
        per_day_min = m["runtime_s"] / days / 60
        label = f"{m['symbol']} {days} day(s), run {run_id[:8]}"
        rows.append({"metric": f"NFR-02 minutes per symbol-day, {label}",
                     "target": "<= 15", "value": round(per_day_min, 1),
                     "met": per_day_min <= 15, **common})  # fmt: skip
        rss = m.get("peak_rss_mb")
        if rss is not None:
            rows.append({"metric": f"NFR-12 peak RSS MB, {label}", "target": "<= 4096",
                         "value": round(rss), "met": rss <= 4096, **common})  # fmt: skip
        rows.append({"metric": f"events/s real replay, {label}", "target": "reported",
                     "value": round(m["events_per_s"] or 0), "met": True, **common})  # fmt: skip
    table = pl.DataFrame(rows)
    write_table(table, REPO / "reports", "benchmark")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
