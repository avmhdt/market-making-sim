"""Exact P&L decomposition (FR-17, NFR-04; definitions in docs/pnl-decomposition.md).

For fills ``i`` with signed quantity ``q_i`` (+ buy), price ``p_i``, pre-fill mid ``m_i``,
exchange-time mid ``m(t)``, run end ``T``, start position ``Q0`` and ``t_i+ = min(t_i + h, T)``::

    spread_capture     = sum q_i (m_i - p_i)
    adverse_selection  = sum q_i (m(t_i+) - m_i)
    inventory_residual = sum q_i (m_T - m(t_i+)) + Q0 (m_T - m_0)
    fees               = - sum fee_i
    funding            = - sum_f Q(t_f) mark_f rate_f
    total              = cash_T + Q_T m_T - (cash_0 + Q0 m_0)

Price terms are int64 in half-tick x lot units (``mid_x2`` is bid + ask ticks), so the
identity holds exactly; ``total``'s price part comes independently from the integer cash
ledger. Fees and funding are summed with ``math.fsum``. A residual above 1e-9 quote
currency is an error, not a warning.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from mmsim.sim.types import InstrumentSpec

__all__ = [
    "MAX_ABS",
    "RECONCILE_TOL",
    "PnLBreakdown",
    "ReconciliationError",
    "attribution_table",
    "decompose_pnl",
    "mid_asof",
]

RECONCILE_TOL = 1e-9
MAX_ABS = 2**62
NS_PER_MS = 1_000_000

Int64Array = npt.NDArray[np.int64]


class ReconciliationError(AssertionError):
    """The components do not sum to the independently computed total."""


@dataclass(frozen=True, slots=True)
class PnLBreakdown:
    """Components in quote currency."""

    spread: float
    adverse_selection: float
    inventory_residual: float
    fees: float
    funding: float
    total: float
    residual: float
    horizon_ms: int


def mid_asof(mid_ts: Int64Array, mid_x2: Int64Array, t: Int64Array) -> Int64Array:
    """The last mid with timestamp ``<= t`` (the first mid before the series starts)."""
    idx = np.searchsorted(mid_ts, t, side="right") - 1
    out: Int64Array = mid_x2[np.clip(idx, 0, len(mid_x2) - 1)]
    return out


def _guard(*arrays: Int64Array | int) -> None:
    for a in arrays:
        if np.any(np.abs(np.asarray(a, dtype=np.int64)) >= MAX_ABS):
            raise OverflowError("price term exceeds the 2**62 guard")


def _components(
    fills: pl.DataFrame, mids: pl.DataFrame, horizon_ms: int, end_ns: int | None
) -> dict[str, Int64Array | int]:
    if mids.is_empty():
        raise ValueError("no mids")
    mid_ts = mids.get_column("ts_exchange").to_numpy().astype(np.int64)
    mid_x2 = mids.get_column("mid_x2").to_numpy().astype(np.int64)
    end = int(mid_ts[-1]) if end_ns is None else end_ns
    m_T = int(mid_asof(mid_ts, mid_x2, np.array([end], dtype=np.int64))[0])
    m_0 = int(mid_x2[0])
    if fills.is_empty():
        z = np.zeros(0, dtype=np.int64)
        return {"q": z, "p2": z, "m_i": z, "m_plus": z, "m_T": m_T, "m_0": m_0}
    t = fills.get_column("ts_exchange").to_numpy().astype(np.int64)
    q = fills.get_column("side").to_numpy().astype(np.int64) * fills.get_column(
        "qty_lots"
    ).to_numpy().astype(np.int64)
    p2 = 2 * fills.get_column("price_ticks").to_numpy().astype(np.int64)
    m_i = fills.get_column("mid_x2_before").to_numpy().astype(np.int64)
    t_plus = np.minimum(t + horizon_ms * NS_PER_MS, end)
    m_plus = mid_asof(mid_ts, mid_x2, t_plus)
    return {"q": q, "p2": p2, "m_i": m_i, "m_plus": m_plus, "m_T": m_T, "m_0": m_0}


def attribution_table(
    fills: pl.DataFrame,
    mids: pl.DataFrame,
    *,
    horizons_ms: Sequence[int],
    end_ns: int | None = None,
) -> pl.DataFrame:
    """Per-fill integer terms: ``spread_x2``, ``as_x2_<h>`` per horizon, ``inv_resid_x2``.

    ``inv_resid_x2`` uses the first horizon; callers pass the headline horizon first.
    """
    first = _components(fills, mids, horizons_ms[0], end_ns)
    q, m_i, m_T = first["q"], first["m_i"], first["m_T"]
    assert isinstance(q, np.ndarray) and isinstance(m_i, np.ndarray)
    p2, m_plus = first["p2"], first["m_plus"]
    out: dict[str, object] = {
        "order_id": fills.get_column("order_id") if not fills.is_empty() else [],
        "ts_exchange": fills.get_column("ts_exchange") if not fills.is_empty() else [],
        "spread_x2": q * (m_i - p2),
        "inv_resid_x2": q * (m_T - m_plus),
        "fee_quote": fills.get_column("fee_quote") if not fills.is_empty() else [],
    }
    for h in horizons_ms:
        comp = _components(fills, mids, h, end_ns)
        mp = comp["m_plus"]
        assert isinstance(mp, np.ndarray)
        out[f"as_x2_{h}"] = q * (mp - m_i)
    return pl.DataFrame(out, schema_overrides={"spread_x2": pl.Int64, "inv_resid_x2": pl.Int64})


def decompose_pnl(
    fills: pl.DataFrame,
    mids: pl.DataFrame,
    *,
    horizon_ms: int,
    instrument: InstrumentSpec,
    funding: pl.DataFrame | None = None,
    start_position_lots: int = 0,
    end_ns: int | None = None,
) -> PnLBreakdown:
    """Decompose one session's P&L and reconcile it against the cash ledger.

    Raises:
        ReconciliationError: If the residual exceeds 1e-9 quote currency.
    """
    c = _components(fills, mids, horizon_ms, end_ns)
    q, p2, m_i, m_plus = c["q"], c["p2"], c["m_i"], c["m_plus"]
    m_T, m_0 = c["m_T"], c["m_0"]
    assert isinstance(q, np.ndarray) and isinstance(p2, np.ndarray)
    assert isinstance(m_i, np.ndarray) and isinstance(m_plus, np.ndarray)
    assert isinstance(m_T, int) and isinstance(m_0, int)
    _guard(q, p2, m_i, m_plus, m_T, m_0)
    q0 = start_position_lots
    spread_x2 = int(np.sum(q * (m_i - p2)))
    as_x2 = int(np.sum(q * (m_plus - m_i)))
    inv_x2 = int(np.sum(q * (m_T - m_plus))) + q0 * (m_T - m_0)
    # Independent ledger: cash in half-tick lots, then mark the end position to mid.
    q_T = q0 + int(q.sum())
    cash_x2 = -int(np.sum(q * p2))
    total_x2 = cash_x2 + q_T * m_T - q0 * m_0
    _guard(spread_x2, as_x2, inv_x2, total_x2)
    unit = instrument.tick_size / 2.0 * instrument.lot_size
    fee_values = [] if fills.is_empty() else fills.get_column("fee_quote").to_list()
    fees = -math.fsum(fee_values)
    funding_quote = 0.0
    if funding is not None and not funding.is_empty():
        f_ts = funding.get_column("ts").to_numpy().astype(np.int64)
        t_fill = (
            np.zeros(0, dtype=np.int64)
            if fills.is_empty()
            else fills.get_column("ts_exchange").to_numpy().astype(np.int64)
        )
        order = np.argsort(t_fill, kind="stable")
        cum = np.concatenate([[0], np.cumsum(q[order])])
        idx = np.searchsorted(t_fill[order], f_ts, side="right")
        pos = q0 + cum[idx]
        mark = funding.get_column("mark_price").to_numpy().astype(np.float64)
        rate = funding.get_column("funding_rate").to_numpy().astype(np.float64)
        funding_quote = -math.fsum((pos * instrument.lot_size * mark * rate).tolist())
    total = total_x2 * unit + fees + funding_quote
    parts = math.fsum([spread_x2 * unit, as_x2 * unit, inv_x2 * unit, fees, funding_quote])
    if spread_x2 + as_x2 + inv_x2 != total_x2:
        raise ReconciliationError("integer price terms do not sum to the ledger total")
    residual = parts - total
    if not math.isfinite(residual) or abs(residual) > RECONCILE_TOL:
        raise ReconciliationError(f"residual {residual} exceeds {RECONCILE_TOL}")
    return PnLBreakdown(
        spread=spread_x2 * unit,
        adverse_selection=as_x2 * unit,
        inventory_residual=inv_x2 * unit,
        fees=fees,
        funding=funding_quote,
        total=total,
        residual=residual,
        horizon_ms=horizon_ms,
    )
