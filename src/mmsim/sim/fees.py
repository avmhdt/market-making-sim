"""Fee per fill in quote currency (FR-08): positive is a cost, negative a rebate."""

from __future__ import annotations

from mmsim.sim.types import InstrumentSpec

__all__ = ["fill_fee_quote"]


def fill_fee_quote(
    inst: InstrumentSpec, price_ticks: int, qty_lots: int, *, maker: bool = True
) -> float:
    """``notional * bps / 1e4`` with the maker or taker rate of the run's tier."""
    bps = inst.maker_bps if maker else inst.taker_bps
    notional = price_ticks * inst.tick_size * qty_lots * inst.lot_size
    return notional * bps / 1e4
