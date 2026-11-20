"""Signal parity (AC-W-09): the online engine and loader against ``parity_sample.csv``.

Three comparisons, each within 1e-9:

1. **Loader.** ``predict`` on the sample's normalised features (``<name>__norm``) against
   its ``prediction`` column (absolute).
2. **Normalisation.** The loader's normalisation of the raw features against ``__norm``.
3. **Online.** The engine replays the sample's day through the simulator (no quotes) and
   must produce rows at the identical timestamps, every feature within 1e-9 absolute or
   relative, and predictions within 1e-9 absolute.

The replay uses recorded feed latency when the spec's timestamp is ``ts_recv`` (the
strategy then sees each event at its ``ts_recv``) and zero feed latency for ``ts_exchange``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from mmsim.config import MmsConfig
from mmsim.runner import build_sessions
from mmsim.settings import Settings
from mmsim.signal.artifact import FeatureSpec, SignalModel
from mmsim.signal.online import OnlineFeatureEngine
from mmsim.sim.registry import make_simulator

__all__ = ["ParityResult", "check_parity", "replay_features"]

TOL = 1e-9


def replay_features(
    spec: FeatureSpec, symbol: str, day: date, *, settings: Settings, repo_root: Path = Path()
) -> dict[int, list[float | None]]:
    """Every sampled row of the engine over one recorded (or synthetic) day."""
    source = "synthetic" if spec.venue == "synthetic" else "cla"
    feed = {"kind": "recorded"} if spec.timestamp == "ts_recv" else {"kind": "constant", "ns": 0}
    zero = {"kind": "constant", "ns": 0}
    cfg = MmsConfig.model_validate({
        "data": {"source": source, "venue": spec.venue, "symbols": [symbol], "start": day,
                 "end": day},
        "fees": {"tier": "retail"},
        "sim": {"latency": {"feed": feed, "order": zero, "ack": zero}},
    })  # fmt: skip
    (session,) = build_sessions(cfg, settings, repo_root)
    sim = make_simulator(cfg.sim, session.events, session.instrument,
                         book_depth=max(spec.book_depth, 10))  # fmt: skip
    sim.reset(seed=0)
    engine = OnlineFeatureEngine(spec, session.instrument.tick_size, session.instrument.lot_size)
    engine.record()
    while (step := sim.step()) is not None:
        engine.update(step)
    assert engine.rows is not None
    return dict(engine.rows)


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= TOL or abs(a - b) <= TOL * max(abs(a), abs(b))


@dataclass(frozen=True)
class ParityResult:
    """Worst differences and counts per comparison."""

    rows: int
    loader_max_abs: float
    norm_max_abs: float
    online_missing_rows: int
    online_worst_feature: str
    online_feature_max_diff: float
    online_feature_failures: int
    online_prediction_max_abs: float

    @property
    def passed(self) -> bool:
        """All three comparisons within tolerance."""
        return (
            self.loader_max_abs <= TOL
            and self.norm_max_abs <= TOL
            and self.online_missing_rows == 0
            and self.online_feature_failures == 0
            and self.online_prediction_max_abs <= TOL
        )


def check_parity(
    model_dir: Path, *, settings: Settings, repo_root: Path = Path(), online: bool = True
) -> ParityResult:
    """Run the three comparisons for one artifact version folder."""
    model = SignalModel.load(model_dir, schema_path=repo_root / "docs" / "feature_spec.schema.json")
    names = model.feature_names
    sample = pl.read_csv(model_dir / "parity_sample.csv", infer_schema_length=0).with_columns(
        [pl.col(c).cast(pl.Float64) for c in [*names, *[f"{n}__norm" for n in names], "prediction"]]
        + [pl.col("ts_recv").cast(pl.Int64), pl.col("ts_exchange").cast(pl.Int64)]
    )
    raw = sample.select(names).to_numpy()
    norm = sample.select([f"{n}__norm" for n in names]).to_numpy()
    pred = sample["prediction"].to_numpy()
    loader = float(np.max(np.abs(model.predict_normalized(norm) - pred)))
    norm_diff = float(np.max(np.abs(model.normalize(raw) - norm)))
    missing = failures = 0
    worst_name, worst, pred_diff = "", 0.0, 0.0
    if online:
        ts_col = model.spec.timestamp
        ts = sample[ts_col].to_numpy()
        day = datetime.fromtimestamp(int(ts[0]) / 1e9, tz=UTC).date()
        symbol = str(sample["symbol"][0])
        rows = replay_features(model.spec, symbol, day, settings=settings, repo_root=repo_root)
        for i, t in enumerate(ts.tolist()):
            got = rows.get(int(t))
            if got is None or any(v is None for v in got):
                missing += 1
                continue
            for j, name in enumerate(names):
                g, w = float(got[j]), float(raw[i, j])  # type: ignore[arg-type]
                d = abs(g - w)
                if not _close(g, w):
                    failures += 1
                if d > worst:
                    worst, worst_name = d, name
            p = model.predict(np.array(got, dtype=np.float64))
            pred_diff = max(pred_diff, abs(p - float(pred[i])))
    return ParityResult(
        rows=sample.height, loader_max_abs=loader, norm_max_abs=norm_diff,
        online_missing_rows=missing, online_worst_feature=worst_name,
        online_feature_max_diff=worst, online_feature_failures=failures,
        online_prediction_max_abs=pred_diff,
    )  # fmt: skip
