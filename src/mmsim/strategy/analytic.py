"""Analytic quoting policies over the :class:`QuotePolicy` protocol (FR-09, FR-10, FR-13).

Model distances are in price units around the visible mid and are turned into ticks by
rounding away from the mid (bid down, ask up, DQ-11). A quote that would cross the visible
book is pulled back to one tick inside it, so post-only orders are not rejected for that
reason; the exchange still rejects on its own, later book (docs/decisions.md, M2-04).

Inventory ``q`` in the formulas is the position in quote units (lots / quote lots). A
maker fee ``c`` per unit is handled exactly by solving with ``A' = A exp(-k c)`` and adding
``c`` to both distances (FR-13); ``c`` is the maker rate times the mid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from mmsim.models.avellaneda_stoikov import as_quotes
from mmsim.models.glft import GLFTExact, glft_asymptotic, glft_exact, with_fee
from mmsim.strategy.base import QuoteDecision, StrategyState

__all__ = ["AnalyticPolicy", "SymmetricPolicy", "distances_to_ticks"]


def distances_to_ticks(
    state: StrategyState, tick_size: float, delta_bid: float | None, delta_ask: float | None
) -> tuple[int | None, int | None]:
    """Quote prices in ticks for distances from the visible mid, rounded away from it."""
    mid = state.book.mid_x2 / 2.0
    bid = None if delta_bid is None else math.floor(mid - delta_bid / tick_size + 1e-9)
    ask = None if delta_ask is None else math.ceil(mid + delta_ask / tick_size - 1e-9)
    book = state.book
    if bid is not None and book.ask_ticks and bid >= book.ask_ticks[0]:
        bid = book.ask_ticks[0] - 1
    if ask is not None and book.bid_ticks and ask <= book.bid_ticks[0]:
        ask = book.bid_ticks[0] + 1
    return bid, ask


@dataclass
class AnalyticPolicy:
    """A-S (2008), GLFT asymptotic or GLFT exact quotes."""

    family: str
    gamma: float
    tick_size: float
    quote_lots: int
    q_max_units: int
    horizon_s: float
    tau_mode: str = "rolling"
    maker_bps: float = 0.0
    name: str = ""
    _exact_cache: dict[tuple[float, float, float], GLFTExact] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in ("as", "glft", "glft_exact"):
            raise ValueError(f"unknown analytic family {self.family!r}")
        self.name = self.name or self.family

    def _tau(self, state: StrategyState) -> float:
        if self.tau_mode == "finite":
            if state.t_remaining_s is None:
                raise ValueError("finite tau needs t_remaining_s")
            return max(state.t_remaining_s, 0.0)
        return self.horizon_s

    def _exact(self, A: float, k: float, sigma: float) -> GLFTExact:
        key = (A, k, sigma)
        sol = self._exact_cache.get(key)
        if sol is None:
            sol = glft_exact(self.gamma, sigma, A, k, self.q_max_units, self.horizon_s, n_t=1001)
            self._exact_cache.clear()
            self._exact_cache[key] = sol
        return sol

    def decide(self, state: StrategyState) -> QuoteDecision:
        """Quotes for the current state."""
        p = state.params
        mid = state.book.mid_x2 * self.tick_size / 2.0
        q = state.inventory_lots / self.quote_lots
        fee = self.maker_bps / 1e4 * mid
        if self.family == "as":
            bid_px, ask_px = as_quotes(mid, q, self._tau(state), self.gamma, p.sigma, p.k)
            d_bid: float | None = mid - bid_px + fee
            d_ask: float | None = ask_px - mid + fee
        elif self.family == "glft":
            d_bid, d_ask = glft_asymptotic(q, self.gamma, p.sigma, with_fee(p.A, p.k, fee), p.k)
            d_bid += fee
            d_ask += fee
        else:
            sol = self._exact(with_fee(p.A, p.k, fee), p.k, p.sigma)
            t = 0.0 if self.tau_mode == "rolling" else self.horizon_s - self._tau(state)
            q_int = max(-self.q_max_units, min(self.q_max_units, round(q)))
            d_bid, d_ask = sol.deltas(min(max(t, 0.0), self.horizon_s), q_int)
            d_bid = None if d_bid is None else d_bid + fee
            d_ask = None if d_ask is None else d_ask + fee
        bid, ask = distances_to_ticks(state, self.tick_size, d_bid, d_ask)
        return QuoteDecision(bid, ask, self.quote_lots, self.quote_lots)


@dataclass
class SymmetricPolicy:
    """A-S 2008's benchmark: a fixed spread centred on the mid, ignoring inventory."""

    half_spread: float
    tick_size: float
    quote_lots: int
    name: str = "symmetric"

    def decide(self, state: StrategyState) -> QuoteDecision:
        """Quotes at ``mid +- half_spread``."""
        bid, ask = distances_to_ticks(state, self.tick_size, self.half_spread, self.half_spread)
        return QuoteDecision(bid, ask, self.quote_lots, self.quote_lots)
