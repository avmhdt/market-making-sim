"""AC-W-09 on the fixture artifact (CI); the real artifact runs locally (marker ``signal``)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import FIXTURES, REPO
from mmsim.cli import app
from mmsim.config import load_config
from mmsim.runner import execute_run
from mmsim.settings import Settings
from mmsim.signal.parity import check_parity

ART = FIXTURES / "signal_artifact" / "lob-synthetic-SYNUSDT-dmid_ticks_time_1000" / "1.0.0"


@pytest.fixture
def data_root(mini_day_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    processed = mini_day_root.parent / "processed"
    mini_day_root.rename(processed)
    monkeypatch.setenv("MMS_DATA_ROOT", str(processed.parent))
    monkeypatch.chdir(REPO)
    return processed.parent


def test_fixture_parity(data_root: Path) -> None:
    result = check_parity(ART, settings=Settings(_env_file=None, mms_data_root=data_root))  # type: ignore[call-arg]
    assert result.rows == 1000
    assert result.passed, result


def test_parity_cli(data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMS_REPORTS_DIR", str(tmp_path / "reports"))
    result = CliRunner().invoke(app, ["validate", "signal-parity", "--model-dir", str(ART)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "reports" / "tables" / "signal_parity.csv").exists()


@pytest.mark.parametrize("mode", ["stub", "reservation_shift", "glft_drift"])
def test_signal_runs_end_to_end(
    data_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    monkeypatch.setenv("MMS_RESULTS_DIR", str(tmp_path / "results"))
    settings = Settings(
        _env_file=None, mms_data_root=data_root, mms_results_dir=tmp_path / "results"
    )  # type: ignore[call-arg]
    overrides = ["strategy.signal.enabled=true", f"strategy.signal.mode={mode}",
                 f"strategy.signal.model_dir={ART.as_posix()}", "strategy.signal.alpha=1.0"]  # fmt: skip
    cfg = load_config(FIXTURES / "configs" / "mini_day_glft.yaml", overrides, settings=settings)
    out = execute_run(cfg, settings=settings, repo_root=REPO)
    assert out.row["strategy_family"] == "glft_signal"
    import polars as pl

    quotes = pl.read_parquet(out.run.path / "quotes.parquet")
    assert quotes["signal"].drop_nulls().len() > 0


@pytest.mark.signal
def test_real_artifact_parity() -> None:
    """Point MMS_SIGNAL_PARITY_DIR at a crypto-lob-alpha artifact version folder."""
    target = os.environ.get("MMS_SIGNAL_PARITY_DIR")
    if not target:
        pytest.skip("MMS_SIGNAL_PARITY_DIR not set")
    result = check_parity(Path(target), settings=Settings(), repo_root=REPO)
    assert result.passed, result
