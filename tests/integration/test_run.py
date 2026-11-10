"""End-to-end ``mms run`` on the mini_day fixture: outputs, log row, determinism (AC-W-07)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from conftest import FIXTURES, REPO
from mmsim.cli import app
from mmsim.explog import read_log
from mmsim.io.runlog import LF, normalized_metrics

CONFIG = FIXTURES / "configs" / "mini_day_glft.yaml"
GOLDEN = REPO / "tests" / "golden" / "mini_day_glft.json"
OUTPUTS = [
    "orders.parquet", "fills.parquet", "mids.parquet", "quotes.parquet",
    "pnl_attribution.parquet", "markouts.parquet", "returns.parquet",
    "returns_hourly.parquet", "metrics.json", "config.resolved.yaml",
]  # fmt: skip


def run_fixture(root: Path, results: Path, monkeypatch: pytest.MonkeyPatch, *args: str) -> Path:
    monkeypatch.setenv("MMS_DATA_ROOT", str(root.parent))
    monkeypatch.setenv("MMS_RESULTS_DIR", str(results))
    monkeypatch.chdir(REPO)
    result = CliRunner().invoke(app, ["run", "--config", str(CONFIG), *args])
    assert result.exit_code == 0, result.output
    run_id = result.output.split()[1].rstrip(":")
    return results / "runs" / run_id


@pytest.fixture
def processed_root(mini_day_root: Path) -> Path:
    target = mini_day_root.parent / "processed"
    mini_day_root.rename(target)
    return target


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_run_writes_every_output(
    processed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = run_fixture(processed_root, tmp_path / "results", monkeypatch)
    for name in OUTPUTS:
        assert (run_dir / name).exists(), name
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["reconciliation_residual"] <= 1e-9
    assert metrics["max_abs_inventory_lots"] <= metrics["q_max_lots"]
    assert metrics["fills"] > 0
    (row,) = read_log(tmp_path / "results" / "trial_log.csv")
    assert row["strategy_family"] == "glft"
    assert row["run_id"] == run_dir.name
    fills = pl.read_parquet(run_dir / "fills.parquet")
    assert fills["position_after_lots"].abs().max() <= metrics["q_max_lots"]
    resolved = (run_dir / "config.resolved.yaml").read_text()
    assert "SYNUSDT" in resolved


def test_determinism(processed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = run_fixture(processed_root, tmp_path / "a", monkeypatch)
    b = run_fixture(processed_root, tmp_path / "b", monkeypatch)
    for name in ("orders.parquet", "fills.parquet"):
        assert sha(a / name) == sha(b / name)
    assert normalized_metrics(a / "metrics.json") == normalized_metrics(b / "metrics.json")


def test_golden(
    processed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:  # fmt: skip
    run_dir = run_fixture(processed_root, tmp_path / "results", monkeypatch)
    current = {
        "metrics": normalized_metrics(run_dir / "metrics.json"),
        "fills_sha256": sha(run_dir / "fills.parquet"),
    }
    if request.config.getoption("--update-golden"):
        GOLDEN.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline=LF
        )
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert current["metrics"] == golden["metrics"]
    assert current["fills_sha256"] == golden["fills_sha256"]


def test_failed_run_is_logged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMS_DATA_ROOT", str(tmp_path / "empty"))
    monkeypatch.setenv("MMS_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.chdir(REPO)
    result = CliRunner().invoke(app, ["run", "--config", str(CONFIG)])
    assert result.exit_code != 0
    (row,) = read_log(tmp_path / "results" / "trial_log.csv")
    assert row["n_obs"] == "0"
    metrics = json.loads(
        (tmp_path / "results" / "runs" / row["run_id"] / "metrics.json").read_text()
    )
    assert metrics["status"] == "failed"
