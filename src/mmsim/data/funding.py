"""Binance USD-M funding-rate history (FR-08, spec section 7(a)).

``GET https://fapi.binance.com/fapi/v1/fundingRate`` is public: no key. Pages of up to
1,000 rows are fetched with a pause between requests. The raw JSON is kept under
``raw/funding`` and the canonical ``funding`` dataset is written under ``processed``:
``ts`` (settlement, ns UTC), ``venue``, ``symbol``, ``funding_rate`` and ``mark_price``.
Longs pay shorts when the rate is positive.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import polars as pl

from mmsim.io.parquet import symbol_dir, write_parquet

__all__ = ["FUNDING_URL", "download_funding", "funding_frame", "load_funding"]

FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
VENUE = "binance-usdm"
PAGE = 1000
PAUSE_S = 0.5
SCHEMA = {
    "ts": pl.Int64,
    "venue": pl.String,
    "symbol": pl.String,
    "funding_rate": pl.Float64,
    "mark_price": pl.Float64,
}


def _ms(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)


def funding_frame(rows: list[dict[str, object]], symbol: str) -> pl.DataFrame:
    """Canonical rows from the endpoint's JSON objects (empty ``markPrice`` becomes null)."""

    def mark(row: dict[str, object]) -> float | None:
        value = row.get("markPrice")
        return float(str(value)) if value not in (None, "") else None

    return (
        pl.DataFrame(
            {
                "ts": [int(str(r["fundingTime"])) * 1_000_000 for r in rows],
                "venue": [VENUE] * len(rows),
                "symbol": [symbol] * len(rows),
                "funding_rate": [float(str(r["fundingRate"])) for r in rows],
                "mark_price": [mark(r) for r in rows],
            },
            schema=SCHEMA,
        )
        .unique("ts", keep="first")
        .sort("ts")
    )


def download_funding(
    symbol: str,
    start: date,
    end: date,
    *,
    raw_root: Path,
    processed_root: Path,
    client: httpx.Client | None = None,
    pause_s: float = PAUSE_S,
) -> pl.DataFrame:
    """Fetch ``[start, end]`` (inclusive UTC days), save raw and canonical files."""
    own = client is None
    http = client or httpx.Client(timeout=30.0, headers={"User-Agent": "market-making-sim"})
    rows: list[dict[str, object]] = []
    begin, stop = _ms(start), _ms(end + timedelta(days=1)) - 1
    try:
        while begin <= stop:
            resp = http.get(
                FUNDING_URL,
                params={"symbol": symbol, "startTime": begin, "endTime": stop, "limit": PAGE},
            )
            resp.raise_for_status()
            page = resp.json()
            if not isinstance(page, list) or not page:
                break
            rows.extend(page)
            begin = int(page[-1]["fundingTime"]) + 1
            if len(page) < PAGE:
                break
            time.sleep(pause_s)
    finally:
        if own:
            http.close()
    raw = raw_root / "funding" / f"{symbol}_{start.isoformat()}_{end.isoformat()}.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(rows), encoding="utf-8")
    frame = funding_frame(rows, symbol)
    target = symbol_dir(processed_root, "funding", VENUE, symbol) / (
        f"part-{start.isoformat()}_{end.isoformat()}.parquet"
    )
    write_parquet(frame, target)
    return frame


def load_funding(processed_root: Path, symbol: str, start_ns: int, end_ns: int) -> pl.DataFrame:
    """Settlements in ``[start_ns, end_ns)``; empty if nothing was downloaded."""
    base = symbol_dir(processed_root, "funding", VENUE, symbol)
    files = sorted(base.glob("*.parquet")) if base.is_dir() else []
    if not files:
        return pl.DataFrame(schema=SCHEMA)
    frame = pl.concat([pl.read_parquet(f) for f in files]).unique("ts", keep="last").sort("ts")
    return frame.filter((pl.col("ts") >= start_ns) & (pl.col("ts") < end_ns))
