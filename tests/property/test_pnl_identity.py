"""AC-W-06: the decomposition identity holds exactly for random fills and mids."""

from __future__ import annotations

import math

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mmsim.analytics.markouts import markout_summary, markout_table
from mmsim.analytics.pnl import (
    ReconciliationError,
    attribution_table,
    decompose_pnl,
)
from mmsim.sim.types import InstrumentSpec

INST = InstrumentSpec("v", "s", 0.1, 0.001, "perp", 2.0, 5.0, "retail")

fill_st = st.tuples(
    st.integers(0, 10_000),  # time
    st.sampled_from([1, -1]),
    st.integers(690_000, 710_000),  # price ticks
    st.integers(1, 50),  # lots
    st.floats(-1.0, 1.0, allow_nan=False),  # fee
)


def frames(
    fills: list[tuple[int, int, int, int, float]], mids: list[tuple[int, int]]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    mids = sorted(mids)
    mid_df = pl.DataFrame(
        {"ts_exchange": [m[0] for m in mids], "mid_x2": [m[1] for m in mids]},
        schema={"ts_exchange": pl.Int64, "mid_x2": pl.Int64},
    )
    fills = sorted(fills)
    fill_df = pl.DataFrame(
        {
            "order_id": list(range(len(fills))),
            "ts_exchange": [f[0] for f in fills],
            "side": [f[1] for f in fills],
            "price_ticks": [f[2] for f in fills],
            "qty_lots": [f[3] for f in fills],
            "fee_quote": [f[4] for f in fills],
            "mid_x2_before": [2 * f[2] + f[1] * 3 for f in fills],
        },
        schema={
            "order_id": pl.Int64,
            "ts_exchange": pl.Int64,
            "side": pl.Int8,
            "price_ticks": pl.Int64,
            "qty_lots": pl.Int64,
            "fee_quote": pl.Float64,
            "mid_x2_before": pl.Int64,
        },
    )
    return fill_df, mid_df


@settings(max_examples=200, deadline=None)
@given(
    fills=st.lists(fill_st, max_size=60),
    mids=st.lists(
        st.tuples(st.integers(0, 12_000), st.integers(1_380_000, 1_420_000)),
        min_size=1,
        max_size=40,
    ),
    q0=st.integers(-100, 100),
    horizon=st.sampled_from([0, 1, 100, 5000]),
    funding=st.lists(
        st.tuples(st.integers(0, 12_000), st.floats(-1e-3, 1e-3), st.floats(60_000, 80_000)),
        max_size=4,
    ),
)
def test_identity_is_exact(
    fills: list[tuple[int, int, int, int, float]],
    mids: list[tuple[int, int]],
    q0: int,
    horizon: int,
    funding: list[tuple[int, float, float]],
) -> None:
    fill_df, mid_df = frames(fills, mids)
    f_df = pl.DataFrame(
        {"ts": [f[0] for f in funding], "funding_rate": [f[1] for f in funding],
         "mark_price": [f[2] for f in funding]},
        schema={"ts": pl.Int64, "funding_rate": pl.Float64, "mark_price": pl.Float64},
    )  # fmt: skip
    bd = decompose_pnl(
        fill_df, mid_df, horizon_ms=horizon, instrument=INST, funding=f_df,
        start_position_lots=q0, end_ns=12_000,
    )  # fmt: skip
    parts = math.fsum([bd.spread, bd.adverse_selection, bd.inventory_residual, bd.fees, bd.funding])
    assert abs(parts - bd.total) <= 1e-9
    assert abs(bd.residual) <= 1e-9
    attr = attribution_table(fill_df, mid_df, horizons_ms=[horizon, 100], end_ns=12_000)
    assert attr.height == fill_df.height
    unit = INST.tick_size / 2 * INST.lot_size
    assert int(attr["spread_x2"].sum()) * unit == pytest.approx(bd.spread, abs=1e-9)


def test_reconciliation_failure_is_an_error() -> None:
    fill_df, mid_df = frames([(5, 1, 700_000, 1, 0.0)], [(0, 1_400_000)])
    bd = decompose_pnl(fill_df, mid_df, horizon_ms=10, instrument=INST)
    assert bd.total == pytest.approx(0.0)
    bad = fill_df.with_columns(pl.lit(float("nan")).alias("fee_quote"))
    with pytest.raises(ReconciliationError):
        decompose_pnl(bad, mid_df, horizon_ms=10, instrument=INST)
    with pytest.raises(ValueError):
        decompose_pnl(fill_df, mid_df.head(0), horizon_ms=10, instrument=INST)
    with pytest.raises(OverflowError):
        decompose_pnl(fill_df.with_columns(pl.lit(2**62).alias("price_ticks")), mid_df,
                      horizon_ms=10, instrument=INST)  # fmt: skip


def test_markouts_by_hand() -> None:
    fill_df, mid_df = frames(
        [(1_000_000, 1, 700_000, 2, 0.0)], [(0, 1_400_006), (1_000_000_000, 1_400_206)]
    )
    table = markout_table(fill_df, mid_df, [100, 1000, 5000], end_ns=2_500_000_000)
    rows = {r["horizon_ms"]: r for r in table.iter_rows(named=True)}
    m_i = (2 * 700_000 + 3) / 2  # mid_x2_before / 2
    assert rows[100]["mo_mid_bps"] == pytest.approx((700_003 - m_i) / m_i * 1e4)
    assert rows[1000]["mo_mid_bps"] == pytest.approx((700_103 - m_i) / m_i * 1e4)
    assert rows[1000]["mo_fill_bps"] == pytest.approx((700_103 - 700_000) / 700_000 * 1e4)
    assert rows[5000]["valid"] is False
    gapped = markout_table(fill_df, mid_df, [100], gaps=[(0, 2_000_000_000)])
    assert gapped["valid"].to_list() == [False]
    summary = markout_summary(table)
    assert summary["horizon_ms"].to_list() == [100, 1000]
    assert markout_table(fill_df.head(0), mid_df, [100]).is_empty()
