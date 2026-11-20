"""Intensity ``A``, ``k`` and volatility ``sigma`` from data (FR-12, DQ-05).

Each market order is the set of trades with the same exchange time and side (a sweep).
Its depth is ``side * (worst trade price - pre-trade mid)``, with the mid the last visible
mid strictly before it. For the grid ``delta_j = j * tick``, ``j = 1..J``, the rate
``lambda_j`` is the number of orders with depth ``>= delta_j`` per second and side, and
``ln lambda_j = ln A - k delta_j`` is fitted by count-weighted least squares. ``sigma`` is
the standard deviation of 1 s mid changes (price per square-root second). Fits use only
data before the fit time; a fit with fewer than ``min_trades`` orders or ``R^2 < min_r2``
is rejected and the previous one kept.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from mmsim.config import CalibrationConfig
from mmsim.sim.types import StepResult

__all__ = [
    "CalibratedParams",
    "IntensityFit",
    "RollingCalibrator",
    "estimate_intensity",
    "fit_intensity",
    "market_order_depths",
    "sigma_from_mids",
]

NS = 1_000_000_000
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class CalibratedParams:
    """One fit: intensity ``A * exp(-k * delta)`` and mid volatility ``sigma``.

    ``A`` is in fills per second per side at distance 0, ``k`` in 1 / quote-currency units
    and ``sigma`` in quote currency per square-root second. Only data before ``ts_fit`` was
    used.
    """

    ts_fit: int
    A: float
    k: float
    sigma: float
    r2: float
    n_orders: int


@dataclass(frozen=True, slots=True)
class IntensityFit:
    """Fit diagnostics, pooled and per side (for ``calib.parquet``)."""

    A: float
    k: float
    r2: float
    n_orders: int
    A_bid: float
    k_bid: float
    A_ask: float
    k_ask: float
    max_depth_ticks: int
    rates: tuple[float, ...]
    """Pooled per-side rate at each grid depth (fills per second)."""


def market_order_depths(
    ts: IntArray, price_ticks: IntArray, side: IntArray, mid_ts: IntArray, mid_x2: IntArray
) -> tuple[IntArray, IntArray]:
    """``(side, depth_x2)`` per market order: depth in half ticks from the pre-trade mid.

    Trades with side 0 or no earlier mid are dropped.
    """
    if ts.size == 0 or mid_ts.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    keep = side != 0
    ts, price_ticks, side = ts[keep], price_ticks[keep], side[keep]
    if ts.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    order = np.lexsort((side, ts))
    ts, price_ticks, side = ts[order], price_ticks[order], side[order]
    new = np.r_[True, (ts[1:] != ts[:-1]) | (side[1:] != side[:-1])]
    starts = np.flatnonzero(new)
    # Worst price of each sweep: highest for buys, lowest for sells.
    worst = np.where(
        side[starts] > 0,
        np.maximum.reduceat(price_ticks, starts),
        np.minimum.reduceat(price_ticks, starts),
    )
    t0 = ts[starts]
    idx = np.searchsorted(mid_ts, t0, side="left") - 1  # strictly before
    ok = idx >= 0
    s = side[starts][ok]
    depth_x2 = s * (2 * worst[ok] - mid_x2[idx[ok]])
    return s.astype(np.int64), depth_x2.astype(np.int64)


def _wls(x: FloatArray, y: FloatArray, w: FloatArray) -> tuple[float, float, float]:
    """Weighted least squares ``y = a + b x``; returns ``(a, b, r2)``."""
    sw = w.sum()
    xm, ym = (w * x).sum() / sw, (w * y).sum() / sw
    sxx = (w * (x - xm) ** 2).sum()
    if sxx <= 0:
        return math.nan, math.nan, 0.0
    b = (w * (x - xm) * (y - ym)).sum() / sxx
    a = ym - b * xm
    ss_res = (w * (y - a - b * x) ** 2).sum()
    ss_tot = (w * (y - ym) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(a), float(b), float(r2)


def _fit_side(
    depth_x2: IntArray, grid: IntArray, tick: float, seconds: float, sides: int
) -> tuple[float, float, float, FloatArray]:
    counts = np.array([np.count_nonzero(depth_x2 >= 2 * j) for j in grid], dtype=np.float64)
    rates = counts / (seconds * sides)
    use = counts > 0
    if use.sum() < 2:
        return math.nan, math.nan, 0.0, rates
    x = (grid[use] * tick).astype(np.float64)
    a, b, r2 = _wls(x, np.log(rates[use]).astype(np.float64), counts[use])
    return math.exp(a), -b, r2, rates


def fit_intensity(
    side: IntArray, depth_x2: IntArray, *, tick_size: float, seconds: float, min_ticks: int,
    max_ticks: int,
) -> IntensityFit:  # fmt: skip
    """Pooled and per-side fits on the grid ``min_ticks..max_ticks``."""
    grid = np.arange(min_ticks, max_ticks + 1, dtype=np.int64)
    A, k, r2, rates = _fit_side(depth_x2, grid, tick_size, seconds, 2)
    a_b, k_b, _, _ = _fit_side(depth_x2[side < 0], grid, tick_size, seconds, 1)
    a_a, k_a, _, _ = _fit_side(depth_x2[side > 0], grid, tick_size, seconds, 1)
    return IntensityFit(
        A=A, k=k, r2=r2, n_orders=int(depth_x2.size), A_bid=a_b, k_bid=k_b, A_ask=a_a, k_ask=k_a,
        max_depth_ticks=int(depth_x2.max() // 2) if depth_x2.size else 0,
        rates=tuple(float(r) for r in rates),
    )  # fmt: skip


def sigma_from_mids(
    mid_ts: IntArray, mid_x2: IntArray, tick_size: float, start: int, end: int, sample_ns: int
) -> float:
    """Standard deviation of mid changes on a ``sample_ns`` grid, per square-root second."""
    grid = np.arange(start + sample_ns, end + 1, sample_ns, dtype=np.int64)
    if grid.size < 3 or mid_ts.size == 0:
        return math.nan
    idx = np.searchsorted(mid_ts, grid, side="right") - 1
    idx = idx[idx >= 0]
    if idx.size < 3:
        return math.nan
    mids = mid_x2[idx].astype(np.float64) * tick_size / 2.0
    dm = np.diff(mids)
    return float(np.std(dm, ddof=1) / math.sqrt(sample_ns / NS))


def estimate_intensity(
    trades: pl.DataFrame,
    mids: pl.DataFrame,
    *,
    tick_size: float,
    max_depth_ticks: int,
    window_s: float,
    min_depth_ticks: int = 1,
    sigma_sample_ms: int = 1000,
    ts_fit: int | None = None,
) -> CalibratedParams:
    """One fit over ``window_s`` seconds ending at ``ts_fit`` (default: the last trade).

    ``trades``: ``ts``, ``price_ticks``, ``side``; ``mids``: ``ts``, ``mid_x2``, sorted.
    """
    t = trades["ts"].to_numpy().astype(np.int64)
    end = int(ts_fit if ts_fit is not None else (t.max() if t.size else 0))
    start = end - int(window_s * NS)
    sel = (t >= start) & (t < end)
    m_ts = mids["ts"].to_numpy().astype(np.int64)
    m_x2 = mids["mid_x2"].to_numpy().astype(np.int64)
    s, d = market_order_depths(
        t[sel], trades["price_ticks"].to_numpy().astype(np.int64)[sel],
        trades["side"].to_numpy().astype(np.int64)[sel], m_ts, m_x2,
    )  # fmt: skip
    fit = fit_intensity(s, d, tick_size=tick_size, seconds=window_s, min_ticks=min_depth_ticks,
                        max_ticks=max_depth_ticks)  # fmt: skip
    sigma = sigma_from_mids(m_ts, m_x2, tick_size, start, end, sigma_sample_ms * 1_000_000)
    return CalibratedParams(end, fit.A, fit.k, sigma, fit.r2, fit.n_orders)


class RollingCalibrator:
    """Point-in-time calibration from strategy-visible events (DQ-05)."""

    def __init__(self, cfg: CalibrationConfig, tick_size: float) -> None:
        self.cfg = cfg
        self.tick_size = tick_size
        self._trades: deque[tuple[int, int, int]] = deque()
        self._mids: deque[tuple[int, int]] = deque()
        self._first: int | None = None
        self._next_fit: int | None = None
        self._current: CalibratedParams | None = None
        self.fits: list[dict[str, object]] = []

    def update(self, step: StepResult) -> None:
        """Record visible trades and mids."""
        if self._first is None:
            self._first = step.ts_local
            self._next_fit = step.ts_local + int(self.cfg.window_s * NS)
        for tr in step.trades:
            self._trades.append((tr.ts_local, tr.price_ticks, tr.side))
        book = step.book
        if book is not None and book.bid_ticks and book.ask_ticks:
            mid = book.mid_x2
            if not self._mids or self._mids[-1][1] != mid:
                self._mids.append((step.ts_local, mid))

    def _refit(self, now: int) -> None:
        start = now - int(self.cfg.window_s * NS)
        while self._trades and self._trades[0][0] < start:
            self._trades.popleft()
        # Keep one mid before the window as the pre-trade mid of its first orders.
        while len(self._mids) > 1 and self._mids[1][0] < start:
            self._mids.popleft()
        if not self._trades or not self._mids:
            return
        tr = np.array(self._trades, dtype=np.int64)
        md = np.array(self._mids, dtype=np.int64)
        s, d = market_order_depths(tr[:, 0], tr[:, 1], tr[:, 2], md[:, 0], md[:, 1])
        grid = self.cfg.depth_grid_ticks
        fit = fit_intensity(s, d, tick_size=self.tick_size, seconds=self.cfg.window_s,
                            min_ticks=grid.min, max_ticks=grid.max)  # fmt: skip
        sigma = sigma_from_mids(md[:, 0], md[:, 1], self.tick_size, start, now,
                                self.cfg.sigma_sample_ms * 1_000_000)  # fmt: skip
        ok = (
            fit.n_orders >= self.cfg.min_trades
            and fit.r2 >= self.cfg.min_r2
            and all(math.isfinite(v) and v > 0 for v in (fit.A, fit.k, sigma))
        )
        row = asdict(fit)
        row["rates"] = list(fit.rates)
        self.fits.append({"ts_fit": now, **row, "sigma": sigma, "accepted": ok})
        if ok:
            self._current = CalibratedParams(now, fit.A, fit.k, sigma, fit.r2, fit.n_orders)

    def params(self, now_ns: int) -> CalibratedParams | None:
        """The current parameters; ``None`` until the first accepted fit."""
        if self._next_fit is None or now_ns < self._next_fit:
            return self._current
        while self._next_fit <= now_ns:
            self._next_fit += int(self.cfg.refit_every_s * NS)
        self._refit(now_ns)
        return self._current

    def fits_frame(self) -> pl.DataFrame:
        """Every fit, accepted or not."""
        return pl.DataFrame(self.fits) if self.fits else pl.DataFrame()
