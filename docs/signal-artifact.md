# Signal artifact (consumer side)

Contract v1.0.0, shared with crypto-lob-alpha (its spec section 10, FR-28 `lob export-model`). Producer: `G:\trading-strategy-brainstorm\crypto-lob-alpha\artifacts\models\<model_name>\<model_version>\`, read here as `$MMS_SIGNAL_MODEL_DIR/<model_name>/<model_version>/` with `MMS_SIGNAL_MODEL_DIR=../crypto-lob-alpha/artifacts/models`. A version folder is immutable once written.

| File | What this repo checks |
|---|---|
| `model.txt` | LightGBM text booster; SHA-256 equals `metadata.model_sha256`; its feature count and order (names, if it has them) match `feature_spec.json` |
| `feature_spec.json` | Parses into the contract's structure (`src/mmsim/signal/artifact.py`); `spec_version` major is 1; validated against `docs/feature_spec.schema.json` (vendored from crypto-lob-alpha at fb0cce3); SHA-256 equals `metadata.files_sha256["feature_spec.json"]`; every `kind` is implemented online |
| `metadata.json` | Installed LightGBM `major.minor` equals `lightgbm_version`'s; `test_days` and this repo's evaluation days are strictly after `valid_end` (DQ-06); the folder is `<model_name>/<model_version>`; `provisional: true` marks a pre-release that runs refuse unless `strategy.signal.allow_provisional=true` (docs/decisions.md M4-05) |
| `parity_sample.csv` | SHA-256 equals `metadata.files_sha256["parity_sample.csv"]`; the parity test below |

`tests/contract/test_artifact_contract.py` is crypto-lob-alpha's consumer contract test, copied unchanged except for where the folders come from: the fixture always, and every version under `MMS_SIGNAL_MODEL_DIR` when it exists.

Code: `SignalModel.load` (loader), `OnlineFeatureEngine` (features from strategy-visible events), `ModelSignal` (prediction usable after `compute_latency_us`), `StubSignal` (micro-price minus mid), `check_parity` and `mms validate signal-parity --model-dir <folder>`.

## Parity test (AC-W-09)

Each comparison is within 1e-9:

1. **Loader:** `predict` on the sample's `<name>__norm` columns against `prediction` (absolute).
2. **Normalisation:** the loader's normalisation of the raw columns against `__norm`.
3. **Online:** the engine replays the sample's day through the simulator with no quotes. It must produce rows at identical timestamps (the spec's `timestamp`: `ts_recv` replays with recorded feed latency, `ts_exchange` with none), every feature within 1e-9 absolute or relative, and every prediction within 1e-9 absolute.

A recorded day replays from its first live-verified checkpoint (docs/decisions.md M3-04), so the parity rows must come from after it, and after the longest feature window has filled. The replay decodes the recordings directly and writes no event cache, and it stops after the last parity row.

Run it with `mms validate signal-parity --model-dir <folder>`, or with `MMS_SIGNAL_PARITY_DIR=<folder> pytest -m signal`. The results on the provisional artifact are in docs/decisions.md M4-08.

## Feature kinds as implemented here (D-03)

crypto-lob-alpha computes its features offline and has no incremental API, so this repo re-implements each kind from the spec's definitions. Where its definitions are not yet published, the choices below are this repo's; the parity test will show any difference, which is then written up for crypto-lob-alpha (section 16: no edits to that repo from here).

Quantities are in base units (lots × lot size), prices in ticks, times in the strategy's clock. Window sums are exact integers in lots, scaled after summing (M4-06). The strategy sees each event at its known-at time: with recorded feed latency, the running maximum of `ts_recv` per stream (M4-07). With `clock = "time"` a row is sampled at every multiple of `sample_grid_ms`; the row at `t` reflects every event with timestamp `<= t`, and time windows cover `(t − w, t]`. With `clock = "event"` a row follows every book update. Windows count their warm-up from the engine's first event; a row is ready once every feature is.

| Kind | Definition |
|---|---|
| `ofi` | Cont, Kukanov and Stoikov (2014) `e_n` at the touch, summed over the window, divided by `params.depth_norm`; a missing level is price −1 and size 0 |
| `mlofi` | The same at book level `levels[0]` |
| `integrated_ofi` | `Σ_l params.weights[l] · (Σ_window e_n(l))`, divided by `params.depth_norm` |
| `queue_imbalance` | `(Σ bid_1..L − Σ ask_1..L) / (Σ bid_1..L + Σ ask_1..L)`, `L = max(levels)`; NaN when both sums are 0 |
| `microprice` | Stoikov (2018) adjustment from `params.table`: rows `[imbalance_edge, adj]`, or `[spread_ticks, imbalance_edge, adj]`, with `I = bid_1 / (bid_1 + ask_1)` |
| `weighted_mid` | `(I − 1/2) × spread` ticks |
| `trade_sign_imbalance` | `Σ side · w / Σ w` over the window, `w` = volume or 1 (`params.weight`) |
| `spread` | Best ask minus best bid, ticks |
| `realized_vol` | `sqrt(Σ Δmid²)` of the mid sampled every `params.sample_ms` (default 100) over the window, ticks |
| `mid_changes` | Number of mid changes in the window |

## Fixture

`tests/fixtures/signal_artifact/lob-synthetic-SYNUSDT-dmid_ticks_time_1000/1.0.0/` is a small LightGBM model trained on the mini_day fixture's own features by `tests/fixtures/make_fixtures.py`. Its dates in `metadata.json` are placeholders before that day. It exercises the loader, the parity test and the signal-skew runs in CI; it is not a research result. The folder is stored byte for byte (`-text` in `.gitattributes`) so its SHA-256 values hold on every platform.

## Known differences from the producer

- crypto-lob-alpha's windows stop at a book reset and at 00:00 UTC. This engine's windows warm up from its first event and do not reset, so values agree once the last reset and the last midnight are at least the longest window in the past. Parity rows satisfy that (they start at or after 12:00 UTC and contain no gap).
- crypto-lob-alpha produces grid rows only while the stream is live (no logged gap). Here quoting pauses inside a gap (M3-05), but the engine keeps sampling.
