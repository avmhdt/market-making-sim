"""AC-W-07 on one real day: two runs give identical orders, fills and metrics (``data``)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from conftest import REPO
from mmsim.config import load_config
from mmsim.io.runlog import normalized_metrics
from mmsim.runner import execute_run
from mmsim.settings import Settings

pytestmark = pytest.mark.data

DAY = os.environ.get("MMS_DETERMINISM_DAY", "2026-11-03")
SYMBOL = os.environ.get("MMS_DETERMINISM_SYMBOL", "SOLUSDT")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_day_is_deterministic(tmp_path: Path) -> None:
    settings = Settings()
    if not settings.mms_cla_data_root.is_dir():
        pytest.skip("no recorded data on this machine")
    overrides = [f"data.symbols=[{SYMBOL}]", f"data.start={DAY}", f"data.end={DAY}"]
    cfg = load_config(REPO / "configs" / "strategies" / "glft.yaml", overrides, settings=settings)
    a = execute_run(cfg, settings=settings, results_dir=tmp_path / "a", repo_root=REPO)
    b = execute_run(cfg, settings=settings, results_dir=tmp_path / "b", repo_root=REPO)
    for name in ("orders.parquet", "fills.parquet"):
        assert sha(a.run.path / name) == sha(b.run.path / name)
    assert normalized_metrics(a.run.path / "metrics.json") == normalized_metrics(
        b.run.path / "metrics.json"
    )
    assert a.metrics["fills"] > 0
