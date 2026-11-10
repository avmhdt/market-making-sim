"""Markouts from mid and from fill price (FR-18, DQ-09).

``mo_mid_bps(i, h) = sign(q_i) (m(t_i + h) - m_i) / m_i * 1e4`` and
``mo_fill_bps(i, h) = sign(q_i) (m(t_i + h) - p_i) / p_i * 1e4``, with ``m_i`` the mid just
before the fill's triggering event and ``m(t)`` the as-of exchange-time mid. A markout is
invalid when ``t_i + h`` falls after the data end or inside a gap. Day-block bootstrap
confidence intervals arrive with the real-data curves (M3).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from mmsim.analytics.pnl import NS_PER_MS, mid_asof

__all__ = ["markout_summary", "markout_table"]


def markout_table(
    fills: pl.DataFrame,
    mids: pl.DataFrame,
    horizons_ms: Sequence[int],
    *,
    end_ns: int | None = None,
    gaps: Sequence[tuple[int, int]] = (),
) -> pl.DataFrame:
    """One row per fill and horizon."""
    schema = {
        "order_id": pl.Int64,
        "ts_exchange": pl.Int64,
        "side": pl.Int8,
        "qty_lots": pl.Int64,
        "horizon_ms": pl.Int32,
        "mo_mid_bps": pl.Float64,
        "mo_fill_bps": pl.Float64,
        "valid": pl.Boolean,
    }
    if fills.is_empty() or mids.is_empty():
        return pl.DataFrame(schema=schema)
    mid_ts = mids.get_column("ts_exchange").to_numpy().astype(np.int64)
    mid_x2 = mids.get_column("mid_x2").to_numpy().astype(np.int64)
    end = int(mid_ts[-1]) if end_ns is None else end_ns
    t = fills.get_column("ts_exchange").to_numpy().astype(np.int64)
    side = fills.get_column("side").to_numpy().astype(np.int64)
    m_i = fills.get_column("mid_x2_before").to_numpy().astype(np.float64) / 2.0
    p = fills.get_column("price_ticks").to_numpy().astype(np.float64)
    frames: list[pl.DataFrame] = []
    for h in horizons_ms:
        target = t + h * NS_PER_MS
        m_h = mid_asof(mid_ts, mid_x2, target).astype(np.float64) / 2.0
        valid = target <= end
        for lo, hi in gaps:
            valid &= ~((target >= lo) & (target < hi))
        frames.append(
            pl.DataFrame(
                {
                    "order_id": fills.get_column("order_id"),
                    "ts_exchange": t,
                    "side": side.astype(np.int8),
                    "qty_lots": fills.get_column("qty_lots"),
                    "horizon_ms": np.full(t.size, h, dtype=np.int32),
                    "mo_mid_bps": side * (m_h - m_i) / m_i * 1e4,
                    "mo_fill_bps": side * (m_h - p) / p * 1e4,
                    "valid": valid,
                },
                schema=schema,
            )
        )
    return pl.concat(frames)


def markout_summary(table: pl.DataFrame) -> pl.DataFrame:
    """Volume-weighted mean markouts per horizon over valid rows."""
    return (
        table.filter(pl.col("valid"))
        .group_by("horizon_ms")
        .agg(
            ((pl.col("mo_mid_bps") * pl.col("qty_lots")).sum() / pl.col("qty_lots").sum()).alias(
                "mo_mid_bps"
            ),
            ((pl.col("mo_fill_bps") * pl.col("qty_lots")).sum() / pl.col("qty_lots").sum()).alias(
                "mo_fill_bps"
            ),
            pl.len().alias("n_fills"),
        )
        .sort("horizon_ms")
    )
