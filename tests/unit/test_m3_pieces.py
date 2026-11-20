"""Bootstrap CIs, unit-scaled policies, reference Monte Carlo and ``mms calibrate``."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from conftest import FIXTURES, REPO
from mmsim.analytics.stats import day_block_bootstrap_ci, weighted_mean
from mmsim.cli import app
from mmsim.models.calibration import CalibratedParams
from mmsim.models.glft import glft_asymptotic
from mmsim.models.reference_mc import as2008_avg_spread, as2008_mc, glft_mc
from mmsim.sim.types import BookView
from mmsim.strategy.analytic import AnalyticPolicy
from mmsim.strategy.base import StrategyState


def test_bootstrap_ci_brackets_the_mean() -> None:
    rng = np.random.default_rng(0)
    days = [rng.normal(1.0, 1.0, 50) for _ in range(20)]
    lo, hi = day_block_bootstrap_ci(days, lambda x: float(np.mean(x)), n=500)
    assert lo < float(np.mean(np.concatenate(days))) < hi
    assert all(math.isnan(v) for v in day_block_bootstrap_ci([np.ones(3)], np.mean))
    assert weighted_mean(np.array([[1.0, 1.0], [3.0, 3.0]])) == pytest.approx(2.5)
    assert math.isnan(weighted_mean(np.array([[1.0, 0.0]])))
    nan_days = [np.array([np.nan]), np.array([np.nan])]
    assert all(math.isnan(v) for v in day_block_bootstrap_ci(nan_days, lambda x: float(x[0])))


def state(q_lots: int) -> StrategyState:
    book = BookView(0, (700_000,), (5,), (700_001,), (5,))  # BTC-like, tick 0.1
    params = CalibratedParams(0, A=5.0, k=0.3, sigma=5.0, r2=0.99, n_orders=1000)
    return StrategyState(0, book, q_lots, 140, params, None, 100.0)


@pytest.mark.parametrize("family", ["as", "glft", "glft_exact"])
def test_policies_scale_to_quote_units(family: str) -> None:
    pol = AnalyticPolicy(family, 0.05, 0.1, 14, 10, 3600.0, maker_bps=2.0, lot_size=0.001)
    flat = pol.decide(state(0))
    long = pol.decide(state(42))  # three quote units long
    assert flat.bid_ticks is not None and flat.ask_ticks is not None
    assert flat.bid_ticks < 700_000 and flat.ask_ticks > 700_001
    assert long.bid_ticks is not None and long.ask_ticks is not None
    assert long.bid_ticks <= flat.bid_ticks  # skewed away from more inventory
    assert long.ask_ticks <= flat.ask_ticks
    assert flat.bid_lots == 14


def test_unit_scaling_matches_the_formula() -> None:
    m = 14 * 0.001
    pol = AnalyticPolicy("glft", 0.05, 0.1, 14, 10, 3600.0, lot_size=0.001)
    d_bid_u, _ = glft_asymptotic(0, 0.05, 5.0 * m, 5.0, 0.3 / m)
    bid = pol.decide(state(0)).bid_ticks
    assert bid == math.floor(700_000.5 - d_bid_u / m / 0.1 + 1e-9)
    with pytest.raises(ValueError):
        AnalyticPolicy("rl", 0.05, 0.1, 1, 1, 1.0)
    finite = AnalyticPolicy("as", 0.05, 0.1, 14, 10, 3600.0, tau_mode="finite", lot_size=0.001)
    no_t = StrategyState(0, state(0).book, 0, 140, state(0).params, None, None)
    with pytest.raises(ValueError):
        finite.decide(no_t)


def test_reference_monte_carlo_runs() -> None:
    inv = as2008_mc(0.1, strategy="inventory", n_paths=200, seed=1)
    sym = as2008_mc(0.1, strategy="symmetric", n_paths=200, seed=1)
    assert inv.profit.size == 200 and inv.final_q.std() < sym.final_q.std()
    assert inv.avg_spread == pytest.approx(as2008_avg_spread(0.1, 2.0, 1.5, 1.0, 0.005))
    w = glft_mc(0.1, 2.0, 140.0, 1.5, 3, 0.2, 0.001, n_paths=500, seed=2)
    assert w.shape == (500,) and np.isfinite(w).all()


def test_calibrate_cli_on_fixture(
    mini_day_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    processed = mini_day_root.parent / "processed"
    mini_day_root.rename(processed)
    monkeypatch.setenv("MMS_DATA_ROOT", str(processed.parent))
    monkeypatch.setenv("MMS_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("MMS_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.chdir(REPO)
    # pytest's temp folder is on C: on Windows; the guard is tested elsewhere.
    monkeypatch.setattr("mmsim.reporting.calibration.check_data_location", lambda p: p)
    cfg = tmp_path / "cal.yaml"
    cfg.write_text(
        (FIXTURES / "configs" / "mini_day_glft.yaml").read_text(encoding="utf-8")
        + "calibration: {window_s: 30, refit_every_s: 10, min_trades: 20, min_r2: 0.5}\n",
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["calibrate", "--config", str(cfg), "--symbol", "SYNUSDT",
                                      "--date", "2026-01-05"])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "accepted" in result.output
    assert list((processed.parent / "calibration").rglob("calib.parquet"))
    assert (tmp_path / "reports" / "figures" / "intensity_fit_SYNUSDT.png").exists()
