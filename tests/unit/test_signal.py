"""Signal artifact loader, online features, providers and skewed quotes (FR-13 to FR-16)."""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from conftest import FIXTURES
from mmsim.models.calibration import CalibratedParams
from mmsim.signal.artifact import ArtifactError, FeatureSpec, SignalModel, sha256_file
from mmsim.signal.online import OnlineFeatureEngine
from mmsim.signal.stub import ModelSignal, StubSignal
from mmsim.sim.types import BookView, StepResult, TradeEvent
from mmsim.strategy.analytic import AnalyticPolicy
from mmsim.strategy.base import StrategyState
from mmsim.strategy.signal_skew import drift_per_second, pull_sides, shift_distances

ART = FIXTURES / "signal_artifact" / "lob-synthetic-SYNUSDT-dmid_ticks_time_1000" / "1.0.0"
MS = 1_000_000


def copy_art(tmp_path: Path) -> Path:
    dst = tmp_path / "m" / "1.0.0"
    shutil.copytree(ART, dst)
    return dst


def rewrite_meta(path: Path, **changes: object) -> None:
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    meta.update(changes)
    (path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")


def test_fixture_loads_and_predicts() -> None:
    m = SignalModel.load(ART, eval_start=date(2026, 1, 5), symbol="SYNUSDT")
    assert m.horizon_ns == 1000 * MS
    assert len(m.feature_names) == 6
    x = np.zeros(6)
    assert np.isfinite(m.predict(x))
    assert m.normalize(np.full(6, 1e9)).max() == 5.0  # clipped


def test_loader_rejects_contract_breaks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ArtifactError, match="DQ-06"):
        SignalModel.load(ART, eval_start=date(2026, 1, 4))
    with pytest.raises(ArtifactError, match="symbol"):
        SignalModel.load(ART, symbol="BTCUSDT")
    p = copy_art(tmp_path)
    (p / "model.txt").write_bytes((p / "model.txt").read_bytes() + b"\n")
    with pytest.raises(ArtifactError, match=r"model\.txt SHA-256"):
        SignalModel.load(p)
    shutil.rmtree(p)
    p = copy_art(tmp_path)
    (p / "parity_sample.csv").write_text("x\n", encoding="utf-8")
    with pytest.raises(ArtifactError, match=r"parity_sample\.csv SHA-256"):
        SignalModel.load(p)
    shutil.rmtree(p)
    p = copy_art(tmp_path)
    rewrite_meta(p, lightgbm_version="3.3.0")
    with pytest.raises(ArtifactError, match=r"major\.minor"):
        SignalModel.load(p)
    shutil.rmtree(p)
    p = copy_art(tmp_path)
    spec = json.loads((p / "feature_spec.json").read_text(encoding="utf-8"))
    spec["spec_version"] = "2.0.0"
    (p / "feature_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    rewrite_meta(p, files_sha256={"feature_spec.json": sha256_file(p / "feature_spec.json"),
                                  "parity_sample.csv": sha256_file(p / "parity_sample.csv")})  # fmt: skip
    with pytest.raises(ArtifactError, match="spec_version"):
        SignalModel.load(p)
    spec["spec_version"] = "1.0.0"
    spec["features"][0]["kind"] = "hawkes_intensity"
    (p / "feature_spec.json").write_text(json.dumps(spec), encoding="utf-8")
    rewrite_meta(p, files_sha256={"feature_spec.json": sha256_file(p / "feature_spec.json"),
                                  "parity_sample.csv": sha256_file(p / "parity_sample.csv")})  # fmt: skip
    with pytest.raises(ArtifactError, match="hawkes_intensity"):
        SignalModel.load(p)
    (p / "metadata.json").unlink()
    with pytest.raises(ArtifactError, match="missing"):
        SignalModel.load(p)


def test_schema_file_is_applied_when_vendored(tmp_path: Path) -> None:
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"type": "object", "required": ["not_there"]}), encoding="utf-8")
    with pytest.raises(ArtifactError, match="JSON Schema"):
        SignalModel.load(ART, schema_path=schema)
    schema.write_text(json.dumps({"type": "object"}), encoding="utf-8")
    SignalModel.load(ART, schema_path=schema)


def spec_with(*features: dict[str, object], clock: str = "time") -> FeatureSpec:
    base = json.loads((ART / "feature_spec.json").read_text(encoding="utf-8"))
    base["features"] = [{"dtype": "float64", "normalization": {"method": "none"},
                         "levels": None, "window": None, "window_unit": None, "params": {},
                         **f} for f in features]  # fmt: skip
    base["clock"] = clock
    return FeatureSpec.model_validate(base)


def book(t: int, bid: int, bq: int, ask: int, aq: int, depth: int = 1) -> BookView:
    return BookView(t, tuple(bid - i for i in range(depth)), (bq,) * depth,
                    tuple(ask + i for i in range(depth)), (aq,) * depth)  # fmt: skip


def step(t: int, b: BookView | None = None, trades: tuple[TradeEvent, ...] = ()) -> StepResult:
    return StepResult(t, b, trades, (), (), False)


def test_ofi_matches_cks_by_hand() -> None:
    spec = spec_with({"name": "ofi", "kind": "ofi", "levels": [1], "window": 1000,
                      "window_unit": "ms", "params": {"depth_norm": 2.0}})  # fmt: skip
    eng = OnlineFeatureEngine(spec, 0.1, 1.0)
    eng.update(step(0, book(0, 100, 5, 101, 5)))
    eng.update(step(200 * MS, book(200 * MS, 100, 8, 101, 5)))  # bid size +3: e = +3
    eng.update(step(400 * MS, book(400 * MS, 101, 2, 102, 4)))  # bid up: +2; ask up: +5
    eng.update(step(1100 * MS))  # samples 100..1000 ms
    assert eng.vector() is not None
    assert eng.vector()[0] == pytest.approx((3 + 2 + 5) / 2.0)  # type: ignore[index]
    eng.update(step(1300 * MS))  # the +3 at 200 ms left the (t-1000, t] window at 1200
    assert eng.vector()[0] == pytest.approx(7 / 2.0)  # type: ignore[index]


def test_book_and_trade_kinds() -> None:
    spec = spec_with(
        {"name": "qi1", "kind": "queue_imbalance", "levels": [1]},
        {"name": "qi3", "kind": "queue_imbalance", "levels": [3]},
        {"name": "sp", "kind": "spread"},
        {"name": "wm", "kind": "weighted_mid"},
        {"name": "mp", "kind": "microprice",
         "params": {"max_spread_ticks": 2, "table": [[1, 0.0, -1.0], [1, 0.5, 1.0], [2, 0.0, -2.0]]}},
        {"name": "tsi", "kind": "trade_sign_imbalance", "window": 2, "window_unit": "events",
         "params": {"weight": "count"}},
        {"name": "mc", "kind": "mid_changes", "window": 500, "window_unit": "ms"},
        {"name": "rv", "kind": "realized_vol", "window": 500, "window_unit": "ms",
         "params": {"sample_ms": 100}},
        {"name": "iofi", "kind": "integrated_ofi", "window": 2, "window_unit": "events",
         "params": {"weights": [1.0, 0.5]}},
    )  # fmt: skip
    eng = OnlineFeatureEngine(spec, 0.1, 1.0)
    eng.update(step(0, book(0, 100, 3, 101, 1, depth=3)))
    eng.update(step(100 * MS, book(100 * MS, 101, 3, 102, 1, depth=3),
                    (TradeEvent(0, 100 * MS, 102, 1, 1), TradeEvent(0, 100 * MS, 100, 1, -1))))  # fmt: skip
    eng.update(step(300 * MS, book(300 * MS, 101, 3, 103, 1, depth=3),
                    (TradeEvent(0, 300 * MS, 103, 1, 1),)))  # fmt: skip
    eng.update(step(700 * MS))
    v = dict(zip([f.name for f in spec.features], eng.vector().tolist(), strict=True))  # type: ignore[union-attr]
    assert v["qi1"] == pytest.approx(0.5) and v["qi3"] == pytest.approx(0.5)
    assert v["sp"] == 2.0
    assert v["wm"] == pytest.approx((0.75 - 0.5) * 2)
    assert v["mp"] == -2.0  # spread 2 row
    assert v["tsi"] == 0.0  # last two trades: -1, +1
    assert v["mc"] == 1.0  # of the changes at 100 and 300 ms only 300 is in (200, 700]
    assert v["rv"] == pytest.approx(0.5)  # 100 ms samples on (200, 700]: one 0.5-tick step
    assert np.isfinite(v["iofi"])


def test_engine_event_clock_and_warm_up() -> None:
    spec = spec_with({"name": "ofi", "kind": "ofi", "levels": [1], "window": 2,
                      "window_unit": "events"}, clock="event")  # fmt: skip
    eng = OnlineFeatureEngine(spec, 0.1, 1.0)
    eng.update(step(0, book(0, 100, 1, 101, 1)))
    eng.update(step(1, book(1, 100, 2, 101, 1)))
    assert eng.vector() is None  # one e_n so far
    eng.update(step(2, book(2, 100, 3, 101, 1)))
    assert eng.vector()[0] == 2.0  # type: ignore[index]
    assert eng.snapshot()["ready"]
    with pytest.raises(ArtifactError):
        OnlineFeatureEngine(spec_with({"name": "x", "kind": "ofi"}), 0.1, 1.0)


def test_model_signal_respects_compute_latency() -> None:
    model = SignalModel.load(ART)
    sig = ModelSignal(model, 0.1, compute_latency_ns=50 * MS)
    rng = np.random.default_rng(0)
    for i in range(30):
        t = i * 100 * MS
        sig.update(step(t + 1, book(t, 700_000 + i % 3, int(rng.integers(1, 9)), 700_002, 4, 5),
                        (TradeEvent(t, t, 700_002, 2, 1),)))  # fmt: skip
    assert sig.predictions > 0
    pending = list(sig._pending)
    last_ready, last_value = pending[-1]
    # Not usable before its ready time: the previous prediction is served instead.
    before = sig.signal(last_ready - 1)
    assert before == (pending[-2][1] if len(pending) > 1 else None)
    assert sig.signal(last_ready) == last_value


def state(signal: float | None) -> StrategyState:
    b = BookView(0, (700_000,), (5,), (700_001,), (5,))
    p = CalibratedParams(0, A=5.0, k=0.3, sigma=5.0, r2=0.99, n_orders=1000)
    return StrategyState(0, b, 0, 140, p, signal, 100.0)


@settings(max_examples=30, deadline=None)
@given(family=st.sampled_from(["as", "glft", "glft_exact"]),
       mode=st.sampled_from(["reservation_shift", "glft_drift"]), alpha=st.floats(0, 1))  # fmt: skip
def test_zero_signal_reproduces_the_base_policy(family: str, mode: str, alpha: float) -> None:
    if family == "as" and mode == "glft_drift":
        return
    base = AnalyticPolicy(family, 0.05, 0.1, 14, 10, 3600.0, maker_bps=2.0, lot_size=0.001)
    skew = AnalyticPolicy(family, 0.05, 0.1, 14, 10, 3600.0, maker_bps=2.0, lot_size=0.001,
                          signal_mode=mode, signal_alpha=alpha)  # fmt: skip
    assert skew.decide(state(0.0)) == base.decide(state(None))
    assert skew.decide(state(None)) == base.decide(state(None))


def test_skew_modes_move_quotes_the_right_way() -> None:
    base = AnalyticPolicy("glft", 0.05, 0.1, 14, 10, 3600.0, lot_size=0.001).decide(state(None))
    up = AnalyticPolicy("glft", 0.05, 0.1, 14, 10, 3600.0, lot_size=0.001,
                        signal_mode="reservation_shift", signal_alpha=1.0).decide(state(3.0))  # fmt: skip
    assert up.bid_ticks == base.bid_ticks + 3 and up.ask_ticks == base.ask_ticks + 3  # type: ignore[operator]
    drift = AnalyticPolicy("glft", 0.05, 0.1, 14, 10, 3600.0, lot_size=0.001,
                           signal_mode="glft_drift").decide(state(50.0))  # fmt: skip
    assert drift.bid_ticks >= base.bid_ticks and drift.ask_ticks >= base.ask_ticks  # type: ignore[operator]
    pulled = AnalyticPolicy("glft", 0.05, 0.1, 14, 10, 3600.0, lot_size=0.001,
                            signal_mode="reservation_shift",
                            pull_threshold_ticks=1.0).decide(state(2.0))  # fmt: skip
    assert pulled.ask_ticks is None and pulled.bid_ticks is not None
    with pytest.raises(ValueError):
        AnalyticPolicy("as", 0.05, 0.1, 1, 1, 1.0, signal_mode="glft_drift")
    with pytest.raises(ValueError):
        AnalyticPolicy("glft", 0.05, 0.1, 1, 1, 1.0, signal_mode="sideways")


def test_skew_helpers() -> None:
    assert shift_distances(1.0, 1.0, 2.0, 0.5, 0.1) == pytest.approx((0.9, 1.1))
    assert shift_distances(None, 1.0, 2.0, 0.5, 0.1)[0] is None
    assert drift_per_second(5.0, 0.1, 2.0) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        drift_per_second(1.0, 0.1, 0.0)
    assert pull_sides(3.0, None) == (True, True)
    assert pull_sides(-3.0, 1.0) == (False, True)  # expected fall: the bid is pulled


def test_stub_signal() -> None:
    sig = StubSignal()
    assert sig.signal(0) is None
    sig.update(step(0, book(0, 100, 3, 102, 1)))
    assert sig.signal(0) == pytest.approx((0.75 - 0.5) * 2)
