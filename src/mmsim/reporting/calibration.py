"""``mms calibrate``: rolling calibration of one symbol-day without quoting (FR-12)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from mmsim.config import load_config
from mmsim.io.parquet import partition_dir, write_parquet
from mmsim.models.calibration import RollingCalibrator
from mmsim.runner import build_sessions, run_session
from mmsim.settings import Settings, check_data_location
from mmsim.sim.registry import make_simulator
from mmsim.strategy.base import QuoteDecision, QuotePolicy, StrategyState

__all__ = ["CalibrationOutcome", "calibrate_day", "intensity_figure"]


class _NoQuotes:
    """A policy that never quotes: the session only feeds the calibrator."""

    name = "none"

    def decide(self, state: StrategyState) -> QuoteDecision:
        """Quote nothing."""
        return QuoteDecision(None, None, 0, 0)


@dataclass(frozen=True)
class CalibrationOutcome:
    """Summary of one calibrated day."""

    path: Path
    fits: int
    accepted: int
    A: float
    k: float
    sigma: float


def calibrate_day(
    config: Path, symbol: str, day: date, *, settings: Settings, repo_root: Path = Path()
) -> CalibrationOutcome:
    """Replay ``symbol`` on ``day`` through the simulator and log every rolling fit."""
    overrides = [f"data.symbols=[{symbol}]", f"data.start={day}", f"data.end={day}"]
    cfg = load_config(config, overrides, settings=settings)
    (session,) = build_sessions(cfg, settings, repo_root)
    inst = session.instrument
    mids: list[tuple[int, int, int]] = []

    def on_mid(t: int, m: int, s: int) -> None:
        mids.append((t, m, s))

    sim = make_simulator(cfg.sim, session.events, inst, cfg.run.sim,
                         median_feed_ns=session.median_feed_ns, on_mid=on_mid)  # fmt: skip
    cal = RollingCalibrator(cfg.calibration, inst.tick_size)

    def factory(lots: int) -> QuotePolicy:
        return _NoQuotes()

    run_session(sim, session, factory, cal, q_max_units=1, quote_lots=1,
                decision_interval_ns=cfg.strategy.decision_interval_ms * 1_000_000,
                requote_threshold_ticks=1, mids=mids)  # fmt: skip
    fits = cal.fits_frame().with_columns(pl.lit(day).alias("day"))
    root = check_data_location(settings.project_data_root) / "calibration"
    path = partition_dir(root, "calib", inst.venue, symbol, day) / "calib.parquet"
    path.unlink(missing_ok=True)
    write_parquet(fits, path)
    ok = fits.filter(pl.col("accepted")) if fits.height else fits
    med = {
        c: float(np.median(ok[c].to_numpy())) if ok.height else math.nan
        for c in ("A", "k", "sigma")
    }
    if ok.height:
        figure = settings.mms_reports_dir / "figures" / f"intensity_fit_{symbol}.png"
        grid_min = cfg.calibration.depth_grid_ticks.min
        intensity_figure(ok, symbol, day, inst.tick_size, grid_min, figure)
    return CalibrationOutcome(path, fits.height, ok.height, med["A"], med["k"], med["sigma"])


def intensity_figure(
    fits: pl.DataFrame, symbol: str, day: date, tick: float, min_ticks: int, path: Path
) -> None:
    """Median empirical rate per depth (across accepted windows) against the median fit."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = np.array(fits["rates"].to_list(), dtype=np.float64)
    depth = (np.arange(rates.shape[1]) + min_ticks) * tick
    median = np.nanmedian(np.where(rates > 0, rates, np.nan), axis=0)
    A, k = float(np.median(fits["A"].to_numpy())), float(np.median(fits["k"].to_numpy()))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1600 / 150, 4.5), dpi=150)
    ax.semilogy(depth, median, "o", label=f"empirical, median of {len(fits)} 600 s windows")
    ax.semilogy(
        depth,
        A * np.exp(-k * depth),
        "-",
        label=f"median fit A exp(-k delta), A={A:.3g}, k={k:.3g}",
    )
    ax.set_xlabel("depth beyond the pre-trade mid (price)")
    ax.set_ylabel("market orders per second and side")
    ax.set_title(f"{symbol} {day}: order arrival intensity by depth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
