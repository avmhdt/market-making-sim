"""AC-W-12 on a dummy run: the trial log is well-formed and validates with overfitlab."""

from __future__ import annotations

import csv
import json
import threading
from datetime import date, timedelta
from pathlib import Path

import pytest
from overfitlab.cli import app as overfitlab_app
from typer.testing import CliRunner

from conftest import REPO
from mmsim.explog import (
    COLUMNS,
    append_failed,
    append_trial,
    canonical_json,
    config_digest,
    git_sha,
    new_run_id,
    periods_per_year,
    read_log,
)
from mmsim.io.runlog import RunDir, normalized_metrics
from mmsim.runner import fail_run, finish_run

DAYS = [date(2026, 11, 2) + timedelta(days=i) for i in range(5)]
PNL = [12.5, -3.0, 4.25, 0.0, 7.75]


def _dummy_run(results: Path, family: str = "glft", **params: object) -> dict[str, str]:
    run = RunDir.create(results)
    return finish_run(
        run,
        explog_path=results / "trial_log.csv",
        strategy_family=family,
        params={"symbol": "BTCUSDT", "seed": 1, **params},
        dates=DAYS,
        pnl_quote=PNL,
        capital=10_000.0,
        metrics={"total_pnl": sum(PNL), "max_abs_inventory_lots": 3, "q_max_lots": 10},
    )


def test_dummy_run_appends_one_contract_row(tmp_path: Path) -> None:
    results = tmp_path / "results"
    row = _dummy_run(results)
    with (results / "trial_log.csv").open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert tuple(header) == COLUMNS
    (logged,) = read_log(results / "trial_log.csv")
    assert logged == row
    assert logged["returns_path"] == f"runs/{logged['run_id']}/returns.parquet"
    assert logged["n_obs"] == "5"
    assert logged["x_periods_per_year"] == "365"
    assert logged["start"] == "2026-11-02"
    assert logged["end"] == "2026-11-06"
    assert json.loads(logged["params_json"]) == {"seed": 1, "symbol": "BTCUSDT"}
    metrics = json.loads((results / "runs" / logged["run_id"] / "metrics.json").read_text())
    assert metrics["schema"] == "mms.metrics/1"


def test_overfitlab_validates_the_log(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _dummy_run(results)
    _dummy_run(results, gamma=0.1)
    fail_run(
        RunDir.create(results),
        explog_path=results / "trial_log.csv",
        strategy_family="glft",
        params={"symbol": "BTCUSDT", "seed": 2},
        error="dummy failure",
    )
    _dummy_run(results, family="glft_lobster")
    result = CliRunner().invoke(
        overfitlab_app, ["trials", "validate", str(results / "trial_log.csv")]
    )
    assert result.exit_code == 0, result.output
    rows = read_log(results / "trial_log.csv")
    assert len(rows) == 4
    assert len(rows) >= len(list((results / "runs").iterdir()))
    assert rows[2]["n_obs"] == "0"
    assert rows[2]["sharpe_annual"] == ""
    assert rows[3]["x_periods_per_year"] == "252"


def test_committed_trial_log_validates() -> None:
    result = CliRunner().invoke(
        overfitlab_app,
        [
            "trials",
            "validate",
            str(REPO / "results" / "trial_log.csv"),
        ],
    )
    assert result.exit_code == 0, result.output


def test_parallel_appends_never_interleave(tmp_path: Path) -> None:
    results = tmp_path / "results"
    threads = [
        threading.Thread(target=_dummy_run, args=(results,), kwargs={"i": i}) for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = read_log(results / "trial_log.csv")
    assert len(rows) == 8
    assert len({r["run_id"] for r in rows}) == 8


def test_bad_inputs_are_rejected(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run = RunDir.create(results)
    log = results / "trial_log.csv"
    with pytest.raises(ValueError, match="unknown strategy_family"):
        append_failed(log, run_id=run.run_id, strategy_family="momentum", params={})
    with pytest.raises(ValueError):
        append_failed(log, run_id="not-a-uuid", strategy_family="glft", params={})
    run.write_returns(DAYS, PNL, 10_000.0)
    with pytest.raises(ValueError, match="does not match"):
        append_trial(
            log,
            run_id=run.run_id,
            strategy_family="glft",
            params={},
            start=DAYS[1],
            end=DAYS[-1],
            returns_path=run.returns_path,
        )
    other = RunDir.create(results)
    with pytest.raises(ValueError, match="returns_path must be"):
        append_trial(
            log,
            run_id=other.run_id,
            strategy_family="glft",
            params={},
            start=DAYS[0],
            end=DAYS[-1],
            returns_path=run.returns_path,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        run.write_returns([DAYS[1], DAYS[0]], [1.0, 2.0], 1.0)
    with pytest.raises(ValueError, match="length"):
        run.write_returns(DAYS, [1.0], 1.0)
    with pytest.raises(ValueError, match="non-finite"):
        run.write_returns(DAYS[:1], [float("nan")], 1.0)
    with pytest.raises(ValueError, match="at least one"):
        finish_run(
            other,
            explog_path=log,
            strategy_family="glft",
            params={},
            dates=[],
            pnl_quote=[],
            capital=1.0,
            metrics={},
        )
    with pytest.raises(FileExistsError):
        RunDir.create(results, run.run_id)


def test_wrong_header_is_refused(tmp_path: Path) -> None:
    log = tmp_path / "trial_log.csv"
    log.write_text("run_id,timestamp\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contract header"):
        append_failed(log, run_id=new_run_id(), strategy_family="glft", params={})
    with pytest.raises(ValueError, match="contract header"):
        read_log(log)
    assert read_log(tmp_path / "missing.csv") == []


def test_single_day_run_logs_zero_sharpe(tmp_path: Path) -> None:
    results = tmp_path / "results"
    run = RunDir.create(results)
    row = finish_run(
        run,
        explog_path=results / "trial_log.csv",
        strategy_family="synthetic_validation",
        params={},
        dates=DAYS[:1],
        pnl_quote=[1.0],
        capital=1.0,
        metrics={},
    )
    assert row["sharpe_annual"] == "0"
    assert row["n_obs"] == "1"


def test_helpers(tmp_path: Path) -> None:
    assert canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'
    assert periods_per_year("as_lobster") == 252
    assert periods_per_year("glft") == 365
    d = config_digest("{}", DAYS[0], DAYS[1], "0" * 40)
    assert len(d) == 8
    assert d != config_digest("{}", DAYS[0], DAYS[2], "0" * 40)
    assert git_sha(tmp_path / "outside" / "log.csv").endswith("-dirty")
    sha = git_sha(REPO / "results" / "trial_log.csv")
    assert len(sha.removesuffix("-dirty")) == 40


def test_normalized_metrics_drop_runtime_fields(tmp_path: Path) -> None:
    run = RunDir.create(tmp_path)
    path = run.write_metrics({"total": 1.5, "runtime_s": 3.2, "events_per_s": 10.0})
    assert normalized_metrics(path) == {"schema": "mms.metrics/1", "total": 1.5}
    cfg_path = run.write_config({"run": {"seed": 1}}, {"TARDIS_API_KEY": "**redacted**"})
    assert "redacted" in cfg_path.read_text(encoding="utf-8")


def test_log_changes_do_not_mark_later_rows_dirty(tmp_path: Path) -> None:
    import subprocess

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "results").mkdir()
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "init")
    log = tmp_path / "results" / "trial_log.csv"
    first = append_failed(log, run_id=new_run_id(), strategy_family="glft", params={})
    second = append_failed(log, run_id=new_run_id(), strategy_family="glft", params={})
    assert not first["git_sha"].endswith("-dirty")
    assert not second["git_sha"].endswith("-dirty")
    (tmp_path / "code.py").write_text("x = 2\n", encoding="utf-8")
    third = append_failed(log, run_id=new_run_id(), strategy_family="glft", params={})
    assert third["git_sha"].endswith("-dirty")
