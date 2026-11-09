from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from conftest import REPO
from mmsim.config import (
    apply_overrides,
    deep_merge,
    find_base_config,
    load_config,
    load_synth_config,
    load_yaml,
)
from mmsim.settings import Settings


def test_settings_defaults_and_expansion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", "~/somewhere")
    monkeypatch.setenv("MMS_DATA_ROOT", "")
    monkeypatch.setenv("TARDIS_API_KEY", "")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.data_root == Path("~/somewhere").expanduser()
    assert s.mms_data_root is None
    assert s.tardis_api_key is None
    assert s.project_data_root == s.data_root
    assert s.processed_root == s.data_root / "processed"
    assert s.raw_root == s.data_root / "raw"
    assert s.mms_sim_backend == "fallback"


def test_settings_redacts_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TARDIS_API_KEY", "not-a-real-key")
    monkeypatch.setenv("MMS_DATA_ROOT", "/data/x")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    red = s.redacted()
    assert red["TARDIS_API_KEY"] == "**redacted**"
    assert "not-a-real-key" not in str(red)
    assert s.project_data_root == Path("/data/x")


def test_env_example_has_exactly_the_spec_keys() -> None:
    keys = [
        line.split("=", 1)[0]
        for line in (REPO / ".env.example").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert keys == [
        "DATA_ROOT",
        "MMS_DATA_ROOT",
        "MMS_CLA_DATA_ROOT",
        "MMS_SIGNAL_MODEL_DIR",
        "MMS_RESULTS_DIR",
        "MMS_REPORTS_DIR",
        "MMS_SIM_BACKEND",
        "MMS_EDBT_PATH",
        "MMS_LOG_LEVEL",
        "MMS_N_JOBS",
        "MMS_TORCH_THREADS",
        "TARDIS_API_KEY",
    ]
    assert set(k.lower() for k in keys) == set(Settings.model_fields)


def test_base_config_loads_with_interpolation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMS_RESULTS_DIR", "./elsewhere")
    cfg = load_config(REPO / "configs" / "base.yaml", settings=Settings(_env_file=None))  # type: ignore[call-arg]
    assert cfg.run.results_dir == "elsewhere" or cfg.run.results_dir.endswith("elsewhere")
    assert cfg.data.symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert cfg.analysis.headline_horizon_ms == 5000
    assert cfg.sim.queue_model.kind == "prob"


def test_strategy_config_merges_over_base(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    (configs / "strategies").mkdir(parents=True)
    (configs / "base.yaml").write_text(
        "run: {name: base, results_dir: x}\nstrategy: {gamma: 0.05, family: glft}\n",
        encoding="utf-8",
    )
    child = configs / "strategies" / "as.yaml"
    child.write_text("strategy: {family: as}\n", encoding="utf-8")
    assert find_base_config(child) == (configs / "base.yaml").resolve()
    cfg = load_config(
        child,
        ["strategy.gamma=0.1", "data.symbols=[ETHUSDT]", "data.start=2026-11-02"],
        settings=Settings(_env_file=None),  # type: ignore[call-arg]
    )
    assert cfg.strategy.family == "as"
    assert cfg.strategy.gamma == 0.1
    assert cfg.data.symbols == ["ETHUSDT"]
    assert cfg.data.start == date(2026, 11, 2)


def test_unknown_keys_and_bad_windows_fail(tmp_path: Path) -> None:
    path = tmp_path / "x.yaml"
    path.write_text("strategy: {gama: 0.1}\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path, settings=Settings(_env_file=None))  # type: ignore[call-arg]
    path.write_text("data: {start: 2026-11-05, end: 2026-11-01}\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path, settings=Settings(_env_file=None))  # type: ignore[call-arg]
    path.write_text("run: {results_dir: '${NOPE_NOT_SET}'}\n", encoding="utf-8")
    with pytest.raises(KeyError):
        load_config(path, settings=Settings(_env_file=None))  # type: ignore[call-arg]


def test_overrides_and_merge_helpers(tmp_path: Path) -> None:
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}
    assert apply_overrides({}, ["a.b.c=2.5e+3"]) == {"a": {"b": {"c": 2500.0}}}
    with pytest.raises(ValueError):
        apply_overrides({}, ["novalue"])
    with pytest.raises(ValueError):
        apply_overrides({}, ["=1"])
    empty = tmp_path / "e.yaml"
    empty.write_text("# only a comment\n", encoding="utf-8")
    assert load_yaml(empty) == {}
    listy = tmp_path / "l.yaml"
    listy.write_text("- 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_yaml(listy)


def test_synth_config_validation(tmp_path: Path) -> None:
    cfg = load_synth_config(REPO / "configs" / "synthetic" / "as2008.yaml", ["paths=3"])
    assert cfg.paths == 3
    assert cfg.market_orders.A == 140.0
    bad = tmp_path / "s.yaml"
    bad.write_text("start_utc: 2026-01-01T00:00:00\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_synth_config(bad)
    bad.write_text("horizon: 1.0\ndt: 2.0\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_synth_config(bad)


def test_data_location_guard() -> None:
    from mmsim.settings import DataLocationError, check_data_location

    assert check_data_location(Path("/home/ze/data/x"), windows=False) == Path("/home/ze/data/x")
    with pytest.raises(DataLocationError):
        check_data_location(Path("/mnt/d/data"), windows=False)
    with pytest.raises(DataLocationError):
        check_data_location(Path("C:/Users/x/data"), windows=True)
    assert check_data_location(Path("D:/data"), windows=True) == Path("D:/data")
