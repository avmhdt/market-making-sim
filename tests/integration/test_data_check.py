"""AC-W-11 on fixtures: ``mms data check`` exits 0 and reports coverage and gaps."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from conftest import REPO
from mmsim.cli import app
from mmsim.config import DataConfig, FeesConfig
from mmsim.data.check import run_check
from mmsim.io.parquet import partition_dir, write_parquet

FIXTURE_DATA = DataConfig(source="synthetic", venue="synthetic", symbols=["SYNUSDT"])
FEES = FeesConfig(tier="retail")


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "check.yaml"
    path.write_text(
        "data: {source: cla, venue: synthetic, symbols: [SYNUSDT]}\nfees: {tier: retail}\n",
        encoding="utf-8",
    )
    return path


def test_cli_check_passes_on_fixture(
    mini_day_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MMS_CLA_DATA_ROOT", str(mini_day_root))
    monkeypatch.chdir(REPO)
    result = CliRunner().invoke(
        app, ["data", "check", "--config", str(_config(tmp_path)), "--deep"]
    )
    assert result.exit_code == 0, result.output
    assert "SYNUSDT" in result.output
    assert "1 symbol-days, 0 problems" in result.output


def test_report_counts_rows_and_span(mini_day_root: Path) -> None:
    report = run_check(mini_day_root, FIXTURE_DATA, FEES, deep=True, repo_root=REPO)
    assert report.ok, report.problems
    (day,) = report.days
    assert day.book_rows > 0
    assert day.trade_rows > 0
    assert 0.03 < day.span_h < 0.034  # two minutes
    assert day.fee_ok


def test_gap_summary_splits_clock_rows(mini_day_root: Path) -> None:
    day = date(2026, 1, 5)
    lo = (day - date(1970, 1, 1)).days * 86_400 * 10**9
    gaps = pl.DataFrame(
        {
            "venue": ["synthetic"] * 3,
            "symbol": ["SYNUSDT", "*", "SYNUSDT"],
            "stream": ["book", "book", "trades"],
            "start_ns": [lo + 10**9, lo + 5 * 10**9, lo - 10 * 10**9],
            "end_ns": [lo + 3 * 10**9, lo + 305 * 10**9, lo + 10**9],
            "reason": ["reconnect", "clock", "reconnect"],
        }
    )
    write_parquet(
        gaps.filter(pl.col("symbol") != "*"),
        partition_dir(mini_day_root, "gaps", "synthetic", "SYNUSDT", day) / "part-00.parquet",
    )
    write_parquet(
        gaps.filter(pl.col("symbol") == "*"),
        partition_dir(mini_day_root, "gaps", "synthetic", "*", day) / "part-00.parquet",
    )
    report = run_check(mini_day_root, FIXTURE_DATA, FEES, repo_root=REPO)
    (cov,) = report.days
    assert cov.gaps == 2
    assert cov.gap_s == pytest.approx(3.0)  # 2 s + the 1 s inside the day
    assert cov.clock_gap_s == pytest.approx(300.0)


def test_check_fails_on_missing_pieces(mini_day_root: Path, tmp_path: Path) -> None:
    missing_root = run_check(tmp_path / "nope", FIXTURE_DATA, FEES, repo_root=REPO)
    assert not missing_root.ok
    no_symbol = run_check(
        mini_day_root, DataConfig(venue="synthetic", symbols=["OTHER"]), FEES, repo_root=REPO
    )
    assert any("no book_l2 or trades days" in p for p in no_symbol.problems)
    no_fee = run_check(mini_day_root, FIXTURE_DATA, FeesConfig(tier="rebate"), repo_root=REPO)
    assert any("fee rows" in p for p in no_fee.problems)
    shutil.rmtree(mini_day_root / "trades")
    no_trades = run_check(mini_day_root, FIXTURE_DATA, FEES, repo_root=REPO)
    assert any("no trades files" in p for p in no_trades.problems)
    assert any(line.startswith("PROBLEM") for line in no_trades.lines())


def test_cli_check_exit_code_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMS_CLA_DATA_ROOT", str(tmp_path / "absent"))
    monkeypatch.chdir(REPO)
    result = CliRunner().invoke(
        app,
        [
            "data",
            "check",
            "--config",
            str(_config(tmp_path)),
            "--start",
            "2026-01-05",
            "--end",
            "2026-01-05",
        ],
    )
    assert result.exit_code == 1
