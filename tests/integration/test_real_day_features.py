"""Checkpoint start, gap pauses (DQ-03), day quality and the report pipeline on fixtures."""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from conftest import FIXTURES, REPO, read_fixture
from mmsim.config import load_config
from mmsim.data.loaders import gap_intervals, iter_row_groups, load_checkpoint
from mmsim.data.replay import BookBatchEvent, ReplayStats, TradeRowEvent, replay
from mmsim.io.parquet import partition_dir, write_parquet
from mmsim.reporting.report import ensure_runs, markouts_report, pnl_report
from mmsim.runner import execute_run
from mmsim.settings import Settings

DAY = date(2026, 1, 5)
CONFIG = FIXTURES / "configs" / "mini_day_glft.yaml"


def _write_checkpoint(root: Path, ts: int, seq: int) -> None:
    book = read_fixture("book_l2")
    snap = book.filter(pl.col("is_snapshot"))
    meta = pl.DataFrame({
        "checkpoint_id": ["c1", "c0"], "ts_recv": [ts, ts - 10], "venue": ["synthetic"] * 2,
        "symbol": ["SYNUSDT"] * 2, "kind": ["rest_l2"] * 2, "seq": [seq, 0], "batch_id": [seq, 0],
        "expected": [None, None], "computed_live": [None, None], "ok_live": [True, False],
        "n_levels": [snap.height, 0],
    })  # fmt: skip
    write_parquet(
        meta, partition_dir(root, "checkpoints", "synthetic", "SYNUSDT", DAY) / "p.parquet"
    )
    levels = snap.select(pl.lit("c1").alias("checkpoint_id"), "side", "price", "qty")
    write_parquet(
        levels, partition_dir(root, "checkpoint_levels", "synthetic", "SYNUSDT", DAY) / "p.parquet"
    )


def test_checkpoint_start_drops_earlier_events(mini_day_root: Path) -> None:
    trades = read_fixture("trades")
    ts = int(trades["ts_recv"][trades.height // 2])
    _write_checkpoint(mini_day_root, ts, seq=10**12)
    ck = load_checkpoint(mini_day_root, "synthetic", "SYNUSDT", DAY)
    assert ck is not None and ck.checkpoint_id == "c1"  # the not-ok_live row is skipped
    lv = ck.levels
    start = BookBatchEvent(
        ts, ts, ts, ck.batch_id, ck.seq, True,
        lv["side"].to_numpy().astype(np.int8),
        np.rint(lv["price"].to_numpy() / 0.1).astype(np.int64),
        np.rint(lv["qty"].to_numpy() / 0.001).astype(np.int64),
    )  # fmt: skip
    files = sorted(mini_day_root.rglob("*.parquet"))
    book = [f for f in files if "book_l2" in f.parts]
    trd = [f for f in files if "trades" in f.parts]
    stats = ReplayStats()
    events = list(
        replay(iter_row_groups(book), iter_row_groups(trd), 0.1, 0.001, stats, start=start)
    )
    assert events[0] is start
    assert all(e.ts_match >= ts for e in events)
    # Every book batch has seq below the checkpoint's: all dropped; later trades remain.
    assert not any(isinstance(e, BookBatchEvent) for e in events[1:])
    assert all(isinstance(e, TradeRowEvent) and e.ts_recv >= ts for e in events[1:])
    assert stats.dropped_before_snapshot > 0
    assert load_checkpoint(mini_day_root, "synthetic", "OTHER", DAY) is None


def test_gap_intervals_merge_and_skip_clock() -> None:
    gaps = pl.DataFrame({
        "stream": ["book", "trades", "bbo", "book", "book"],
        "reason": ["disconnect", "disconnect", "disconnect", "clock", "stale"],
        "start_ns": [10, 15, 0, 100, 50], "end_ns": [20, 30, 1000, 200, 60],
    })  # fmt: skip
    assert gap_intervals(gaps) == [(10, 30), (50, 60)]
    assert gap_intervals(pl.DataFrame()) == []


def _settings(tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("MMS_DATA_ROOT", str(root.parent))
    monkeypatch.setenv("MMS_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("MMS_REPORTS_DIR", str(tmp_path / "reports"))
    return Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def processed(mini_day_root: Path) -> Path:
    target = mini_day_root.parent / "processed"
    mini_day_root.rename(target)
    return target


def test_gap_pauses_quoting_and_counts(
    processed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trades = read_fixture("trades")
    lo, hi = int(trades["ts_exchange"][100]), int(trades["ts_exchange"][200])
    gaps = pl.DataFrame({"venue": ["synthetic"], "symbol": ["SYNUSDT"], "stream": ["book"],
                         "start_ns": [lo], "end_ns": [hi], "reason": ["disconnect"]})  # fmt: skip
    write_parquet(gaps, partition_dir(processed, "gaps", "synthetic", "SYNUSDT", DAY) / "p.parquet")
    settings = _settings(tmp_path, processed, monkeypatch)
    monkeypatch.chdir(REPO)
    out = execute_run(load_config(CONFIG, settings=settings), settings=settings)
    m = out.metrics
    assert m["gaps_skipped"] == 1 and m["gap_pauses"] == 1
    assert m["gap_s"] == pytest.approx((hi - lo) / 1e9)
    fills = pl.read_parquet(out.run.path / "fills.parquet")
    # No quote rests through the gap: nothing fills inside it after the pause.
    inside = fills.filter((pl.col("ts_exchange") > lo + 50_000_000) & (pl.col("ts_exchange") < hi))
    assert inside.height == 0
    mo = pl.read_parquet(out.run.path / "markouts.parquet")
    landed = mo.filter((pl.col("ts_exchange") + pl.col("horizon_ms").cast(pl.Int64) * 1_000_000)
                        .is_between(lo, hi - 1))  # fmt: skip
    assert not landed["valid"].any()
    sess = pl.read_parquet(out.run.path / "sessions.parquet")
    assert sess["valid"].to_list() == [True]  # a 2-minute gap is far below 5% of the day


def test_report_pipeline(processed: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path, processed, monkeypatch)
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True)
    shutil.copy(CONFIG, repo / "configs" / "fixture.yaml")
    for name in ("instruments.yaml", "fees.yaml"):
        shutil.copy(REPO / "configs" / name, repo / "configs" / name)
    (repo / "configs" / "reports.yaml").write_text(
        "first_decomposition:\n  configs: [configs/fixture.yaml]\n  symbols: [SYNUSDT]\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)
    refs = ensure_runs("first_decomposition", settings=settings, repo_root=repo)
    again = ensure_runs("first_decomposition", settings=settings, repo_root=repo, run_missing=False)
    assert again == refs  # found in the trial log, not rerun
    pnl = pnl_report(refs, settings.mms_reports_dir)
    assert pnl.height == 1 and pnl["days"].to_list() == [1]
    mo = markouts_report(refs, settings.mms_reports_dir)
    assert set(mo["side"].to_list()) == {"all", "buy", "sell"}
    for name in ("pnl_decomposition.png", "markouts_SYNUSDT.png"):
        assert (settings.mms_reports_dir / "figures" / name).exists()
    assert (settings.mms_reports_dir / "tables" / "excluded_days.csv").exists()
    with pytest.raises(FileNotFoundError):
        (repo / "configs" / "fixture.yaml").write_text(
            (repo / "configs" / "fixture.yaml")
            .read_text(encoding="utf-8")
            .replace("seed: 3", "seed: 4"),
            encoding="utf-8",
        )
        ensure_runs("first_decomposition", settings=settings, repo_root=repo, run_missing=False)
