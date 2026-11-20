"""Calibration pieces on hand-built inputs (FR-12, DQ-05)."""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from mmsim.config import CalibrationConfig
from mmsim.models.calibration import (
    RollingCalibrator,
    estimate_intensity,
    fit_intensity,
    market_order_depths,
    sigma_from_mids,
)
from mmsim.sim.types import BookView, StepResult, TradeEvent

NS = 1_000_000_000


def a(*v: int) -> np.ndarray:
    return np.array(v, dtype=np.int64)


def test_sweeps_are_aggregated_and_measured_from_the_pre_trade_mid() -> None:
    # Mid 100.5 ticks (mid_x2 201) before t = 10; a buy sweep at 10 prints 101 and 103,
    # a sell at 10 prints 99, a buy at 20 prints 102 after the mid moved to 101.5 at 15.
    ts = a(10, 10, 10, 20, 30)
    px = a(101, 103, 99, 102, 100)
    side = a(1, 1, -1, 1, 0)  # the side-0 trade is dropped
    s, d = market_order_depths(ts, px, side, a(0, 15), a(201, 203))
    assert s.tolist() == [-1, 1, 1]
    assert d.tolist() == [3, 5, 1]  # half ticks: 201-198, 206-201, 204-203
    # No mid before the order: dropped.
    s2, _ = market_order_depths(a(5), a(101), a(1), a(10), a(201))
    assert s2.size == 0
    assert market_order_depths(a(), a(), a(), a(0), a(1))[0].size == 0


def test_fit_recovers_an_exact_exponential() -> None:
    k, A, tick, seconds = 0.5, 2.0, 1.0, 1000.0
    grid = np.arange(1, 11)
    counts = np.rint(A * np.exp(-k * grid * tick) * seconds * 2).astype(int)
    # Depths whose survival counts match: count(depth >= j) = counts[j - 1].
    depths: list[int] = []
    for j, c in enumerate(counts, start=1):
        nxt = counts[j] if j < len(counts) else 0
        depths += [2 * j] * int(c - nxt)
    d = np.array(depths, dtype=np.int64)
    side = np.where(np.arange(d.size) % 2 == 0, 1, -1).astype(np.int64)
    fit = fit_intensity(side, d, tick_size=tick, seconds=seconds, min_ticks=1, max_ticks=10)
    assert abs(fit.k / k - 1) < 0.02
    assert abs(fit.A / A - 1) < 0.03
    assert fit.r2 > 0.99
    empty = fit_intensity(a(), a(), tick_size=1.0, seconds=1.0, min_ticks=1, max_ticks=5)
    assert math.isnan(empty.k)


def test_sigma_from_one_second_mids() -> None:
    mids = a(*([200, 202] * 50))  # mid alternates by one tick each second
    ts = np.arange(100, dtype=np.int64) * NS
    sigma = sigma_from_mids(ts, mids, 0.1, 0, 99 * NS, NS)
    assert abs(sigma - 0.1 * 2 * 0.5 * 1.0) < 0.02  # |dm| = 0.1 every second
    assert math.isnan(sigma_from_mids(ts[:2], mids[:2], 0.1, 0, NS, NS))


def test_estimate_intensity_frame_api() -> None:
    trades = pl.DataFrame({"ts": [1 * NS, 2 * NS], "price_ticks": [102, 98], "side": [1, -1]})
    mids = pl.DataFrame({"ts": [0], "mid_x2": [200]})
    p = estimate_intensity(
        trades, mids, tick_size=1.0, max_depth_ticks=3, window_s=10, ts_fit=5 * NS
    )
    assert p.n_orders == 2
    assert p.ts_fit == 5 * NS


def step(t: int, trades: tuple[TradeEvent, ...] = (), mid: int | None = None) -> StepResult:
    book = None if mid is None else BookView(t, (mid // 2,), (1,), (mid - mid // 2,), (1,))
    return StepResult(t, book, trades, (), (), False)


def test_rolling_calibrator_warm_up_refit_and_rejection() -> None:
    cfg = CalibrationConfig(window_s=10, refit_every_s=5, min_trades=5, min_r2=0.5,
                            depth_grid_ticks={"min": 1, "max": 4})  # fmt: skip
    cal = RollingCalibrator(cfg, tick_size=1.0)
    rng = np.random.default_rng(0)
    cal.update(step(0, mid=200))
    for i in range(1, 400):
        t = i * NS // 20
        depth = int(rng.exponential(1.5)) + 1
        side = 1 if i % 2 else -1
        cal.update(
            step(t, (TradeEvent(t, t, 100 + side * depth, 1, side),), mid=200 + (i // 50) % 2 * 2)
        )
        if t < 10 * NS:
            assert cal.params(t) is None  # warm-up (DQ-05)
    p = cal.params(20 * NS)
    assert p is not None and p.k > 0 and p.n_orders >= 5
    assert cal.fits and cal.fits[-1]["accepted"]
    # A window without orders leaves the previous fit in force.
    assert cal.params(60 * NS) == p
    assert not cal.fits_frame().is_empty()
    # A fit below min_trades is rejected and logged; with no earlier fit, no parameters.
    strict = RollingCalibrator(cfg.model_copy(update={"min_trades": 10**6}), tick_size=1.0)
    strict.update(step(0, mid=200))
    strict.update(step(15 * NS, (TradeEvent(15 * NS, 15 * NS, 102, 1, 1),), mid=200))
    assert strict.params(20 * NS) is None
    assert strict.fits[-1]["accepted"] is False
