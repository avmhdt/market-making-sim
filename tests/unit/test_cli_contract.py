from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import FIXTURES
from mmsim.cli import app
from mmsim.explog import new_run_id
from mmsim.io.runlog import RunDir
from mmsim.runner import finish_run
from mmsim.sim.protocol import CAPABILITIES, LatencyModel, MarketSimulator, QueueModel
from mmsim.sim.types import BookView, Side, TimeInForce

COMMANDS = [
    [],
    ["data"],
    ["data", "check"],
    ["data", "synth"],
    ["data", "funding"],
    ["data", "ingest-lobster"],
    ["data", "ingest-tardis"],
    ["calibrate"],
    ["run"],
    ["sweep"],
    ["validate"],
    ["validate", "synthetic"],
    ["validate", "signal-parity"],
    ["validate", "sim-parity"],
    ["rl"],
    ["rl", "train"],
    ["rl", "eval"],
    ["report"],
    ["explog"],
    ["explog", "summary"],
]


@pytest.mark.parametrize("command", COMMANDS, ids=lambda c: " ".join(c) or "mms")
def test_every_command_has_help(command: list[str]) -> None:
    result = CliRunner().invoke(app, [*command, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output


@pytest.mark.parametrize(
    "args",
    [
        ["data", "funding", "--symbol", "BTCUSDT", "--start", "2026-11-01", "--end", "2026-11-02"],
        ["data", "ingest-lobster", "--src", "x", "--ticker", "AAPL", "--date", "2012-06-21"],
        [
            "data",
            "ingest-tardis",
            "--exchange",
            "binance-futures",
            "--symbol",
            "BTCUSDT",
            "--start",
            "a",
            "--end",
            "b",
        ],
        ["calibrate", "--config", "x", "--symbol", "BTCUSDT", "--date", "2026-11-02"],
        ["sweep", "--config", "x"],
        ["validate", "signal-parity", "--model-dir", "x"],
        ["validate", "sim-parity", "--config", "x"],
        ["rl", "train", "--algo", "ppo", "--config", "x", "--seed", "1"],
        ["rl", "eval", "--run-id", "x", "--split", "val"],
        ["report", "all"],
    ],
)
def test_future_commands_exit_2(args: list[str]) -> None:
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2
    assert "arrives in M" in result.output


def test_data_synth_writes_canonical(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "data",
            "synth",
            "--config",
            str(FIXTURES / "mini_day" / "config.yaml"),
            "--out",
            str(tmp_path),
            "--seed",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert list((tmp_path / "book_l2").rglob("*.parquet"))


def test_explog_summary_and_limits(tmp_path: Path) -> None:
    results = tmp_path / "results"
    for breach in (False, True):
        finish_run(
            RunDir.create(results),
            explog_path=results / "trial_log.csv",
            strategy_family="glft",
            params={"b": breach},
            dates=[date(2026, 11, 2), date(2026, 11, 3)],
            pnl_quote=[1.0, 2.0],
            capital=100.0,
            metrics={"max_abs_inventory_lots": 11 if breach else 3, "q_max_lots": 10},
        )
    log = str(results / "trial_log.csv")
    result = CliRunner().invoke(app, ["explog", "summary", "--path", log])
    assert result.exit_code == 0
    assert "total trials: 2" in result.output
    result = CliRunner().invoke(app, ["explog", "summary", "--path", log, "--check-limits"])
    assert result.exit_code == 1
    result = CliRunner().invoke(
        app, ["explog", "summary", "--path", log, "--family", "as", "--check-limits"]
    )
    assert result.exit_code == 0
    assert "no breach" in result.output


def test_types_and_protocol_shapes() -> None:
    view = BookView(0, (99, 98), (1, 1), (101, 102), (2, 2))
    assert view.mid_x2 == 200
    assert Side.BUY == 1 and Side.SELL == -1
    assert TimeInForce.GTX == "gtx"
    assert {"l2", "queue_position", "latency", "fees", "timers"} == CAPABILITIES
    assert not isinstance(object(), MarketSimulator)
    for proto, members in (
        (QueueModel, {"on_entry", "on_trade", "on_decrease"}),
        (LatencyModel, {"feed_delay_ns", "order_delay_ns", "ack_delay_ns"}),
    ):
        assert members <= set(dir(proto))
    assert len(new_run_id()) == 36


def test_synth_default_root_on_c_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", "/mnt/c/data")
    monkeypatch.setenv("MMS_DATA_ROOT", "")
    result = CliRunner().invoke(
        app, ["data", "synth", "--config", str(FIXTURES / "mini_day" / "config.yaml")]
    )
    assert result.exit_code == 1


def test_strategy_state_holds_calibrated_params() -> None:
    from mmsim.models.calibration import CalibratedParams
    from mmsim.strategy.base import QuoteDecision, StrategyState

    params = CalibratedParams(ts_fit=1, A=1.0, k=2.0, sigma=0.5, r2=0.9, n_orders=300)
    state = StrategyState(
        ts_local=2,
        book=BookView(2, (99,), (1,), (101,), (1,)),
        inventory_lots=0,
        q_max_lots=10,
        params=params,
        signal_ticks=None,
        t_remaining_s=None,
    )
    assert state.params.k == 2.0
    assert QuoteDecision(None, 101, 0, 1).ask_ticks == 101
