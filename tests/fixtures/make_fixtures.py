"""Regenerate the committed synthetic fixtures (CSV only; Parquet is built in tmp_path).

Run from the repo root: ``uv run python tests/fixtures/make_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path

from mmsim.config import load_synth_config
from mmsim.data.synth import generate_path

HERE = Path(__file__).parent


def make_mini_day() -> None:
    """tests/fixtures/mini_day/{book_l2,trades}.csv from mini_day/config.yaml."""
    cfg = load_synth_config(HERE / "mini_day" / "config.yaml")
    world = generate_path(cfg, 0)
    world.book_l2.write_csv(HERE / "mini_day" / "book_l2.csv", line_terminator="\n")
    world.trades.write_csv(HERE / "mini_day" / "trades.csv", line_terminator="\n")


SIGNAL_FEATURES = [
    {"name": "ofi_l1_ms1000", "kind": "ofi", "levels": [1], "window": 1000, "window_unit": "ms",
     "params": {"depth_norm": 1.0}, "definition": "sum of CKS e_n at level 1 over (t-1000ms, t]"},
    {"name": "qi_l1", "kind": "queue_imbalance", "levels": [1], "window": None,
     "window_unit": None, "params": {}, "definition": "(bid_1 - ask_1) / (bid_1 + ask_1)"},
    {"name": "qi_l5", "kind": "queue_imbalance", "levels": [5], "window": None,
     "window_unit": None, "params": {}, "definition": "top-5 queue imbalance"},
    {"name": "microprice_adj_ticks", "kind": "microprice", "levels": [1], "window": None,
     "window_unit": None,
     "params": {"imbalance_buckets": 10, "max_spread_ticks": 4,
                "table": [[i / 10, round((i / 10 + 0.05 - 0.5) * 0.8, 6)] for i in range(10)]},
     "definition": "g(I) lookup, ticks"},
    {"name": "tsi_vol_ms1000", "kind": "trade_sign_imbalance", "levels": [], "window": 1000,
     "window_unit": "ms", "params": {"weight": "volume"},
     "definition": "signed volume / volume over (t-1000ms, t]"},
    {"name": "rv_ms1000", "kind": "realized_vol", "levels": [1], "window": 1000,
     "window_unit": "ms", "params": {"sample_ms": 100},
     "definition": "sqrt of summed squared 100 ms mid changes over the window, ticks"},
]  # fmt: skip


def make_signal_artifact() -> None:
    """tests/fixtures/signal_artifact/lob-synthetic-SYNUSDT-dmid_ticks_time_1000/1.0.0/.

    A tiny LightGBM model trained on the mini_day's own features (a fixture, not research:
    its training and validation dates in metadata.json are placeholders before the day).
    """
    import hashlib
    import json
    import tempfile

    import lightgbm
    import numpy as np

    from conftest import read_fixture
    from mmsim.data.synth import generate_path
    from mmsim.io.parquet import write_canonical
    from mmsim.settings import Settings
    from mmsim.signal.artifact import FeatureSpec
    from mmsim.signal.parity import replay_features

    name = "lob-synthetic-SYNUSDT-dmid_ticks_time_1000"
    out = HERE / "signal_artifact" / name / "1.0.0"
    out.mkdir(parents=True, exist_ok=True)
    spec_raw = {
        "spec_version": "1.0.0", "model_name": name, "model_version": "1.0.0",
        "venue": "synthetic", "symbols": ["SYNUSDT"], "tick_size": 0.1, "lot_size": 0.001,
        "book_depth": 10, "timestamp": "ts_recv", "clock": "time", "sample_grid_ms": 100,
        "target": {"name": "dmid_ticks_time_1000", "unit": "ticks",
                   "horizon": {"kind": "clock", "value": 1000, "unit": "ms"}, "latency_ms": 0},
        "features": [{**f, "dtype": "float64", "normalization": {"method": "none"}}
                     for f in SIGNAL_FEATURES],
        "compatibility": "consumers MUST reject a different spec_version major; MUST apply "
                         "normalization before predict",
    }  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_canonical(read_fixture("book_l2"), root / "processed", "book_l2")
        write_canonical(read_fixture("trades"), root / "processed", "trades")
        settings = Settings(_env_file=None, mms_data_root=root)  # type: ignore[call-arg]
        from datetime import date

        spec = FeatureSpec.model_validate(spec_raw)
        rows = replay_features(spec, "SYNUSDT", date(2026, 1, 5), settings=settings,
                               repo_root=HERE.parents[1])  # fmt: skip
    ready = sorted((t, v) for t, v in rows.items() if all(x is not None for x in v))
    ts = np.array([t for t, _ in ready], dtype=np.int64)
    x = np.array([v for _, v in ready], dtype=np.float64)
    world = generate_path(load_synth_config(HERE / "mini_day" / "config.yaml"), 0)
    mid_at = lambda t: world.mid_ticks[np.searchsorted(world.mid_times_ns, t, side="right") - 1]  # noqa: E731
    y = (mid_at(ts + 1_000_000_000) - mid_at(ts)).astype(np.float64)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1.0
    for i, f in enumerate(spec_raw["features"]):  # type: ignore[attr-defined]
        f["normalization"] = {"method": "zscore_clip", "mean": float(mean[i]),
                              "std": float(std[i]), "clip": 5.0}  # fmt: skip
    xn = np.clip((x - mean) / std, -5.0, 5.0)
    params = {"objective": "regression", "num_leaves": 4, "learning_rate": 0.1,
              "min_data_in_leaf": 20, "deterministic": True, "num_threads": 1, "seed": 0,
              "verbose": -1, "force_row_wise": True}  # fmt: skip
    names = [str(f["name"]) for f in SIGNAL_FEATURES]
    data = lightgbm.Dataset(xn, y, feature_name=names)
    booster = lightgbm.train(params, data, num_boost_round=30)
    booster.save_model(str(out / "model.txt"))
    (out / "feature_spec.json").write_text(
        json.dumps(spec_raw, indent=2) + "\n", encoding="utf-8", newline=chr(10)
    )
    pick = slice(0, 1000)
    pred = booster.predict(xn[pick])
    cols = {"ts_exchange": ts[pick], "ts_recv": ts[pick], "symbol": ["SYNUSDT"] * 1000}
    lines = [",".join([*cols, *names, *[f"{n}__norm" for n in names], "prediction"])]
    for i in range(1000):
        vals = [*(f"{v:.17g}" for v in x[i]), *(f"{v:.17g}" for v in xn[i]), f"{pred[i]:.17g}"]
        lines.append(",".join([str(ts[i]), str(ts[i]), "SYNUSDT", *vals]))
    (out / "parity_sample.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline=chr(10)
    )

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    meta = {
        "git_sha": "0" * 40, "created_utc": "2026-01-05T00:00:00Z",
        "lightgbm_version": lightgbm.__version__, "model_sha256": sha(out / "model.txt"),
        "files_sha256": {"feature_spec.json": sha(out / "feature_spec.json"),
                         "parity_sample.csv": sha(out / "parity_sample.csv")},
        "fold": 0, "train_start": "2026-01-03", "train_end": "2026-01-03",
        "valid_start": "2026-01-04", "valid_end": "2026-01-04", "train_days": 1,
        "validation_days": 1, "target_unit": "ticks",
        "metrics": {"oos_r2": None, "ic_spearman": None, "n_test_days": 0,
                    "note": "synthetic test fixture; dates are placeholders"},
    }  # fmt: skip
    (out / "metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline=chr(10)
    )


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(HERE.parent))
    make_mini_day()
    make_signal_artifact()
