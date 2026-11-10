# Decisions

Owner decisions copied from the portfolio's `DECISIONS.md` (2026-10-31 onwards), then the decisions made during the build. `DECISIONS.md` overrides `SPEC.html`, which overrides `README.md`.

## Owner decisions that apply here

| Date | Item | Decision |
|---|---|---|
| 2026-10-31 | Scope and pace | All 15 portfolio projects, built a few at a time. This is project 12, started 2026-11-09 with #6, #13, #14 and #15 |
| 2026-10-31 | Merging | Feature branch per milestone, one PR each, merged with `gh pr merge --auto --squash` as `avmhdt` once CI is green. Branch protection once CI exists (M1). If the permission check refuses the merge, the PR waits for the owner |
| 2026-10-31 | GitHub | `gh` account `avmhdt`, admin on the repo; Actions enabled (P-03) |
| 2026-10-31 | Country (P-12) | Cayman Islands (`OWNER_COUNTRY=KY`). Binance spot and USD-M market data allowed; OKX not allowed and not used |
| 2026-10-31 | Money | Nothing is spent. Tardis.dev spend (D-01) is not approved: free first-day-of-month samples and crypto-lob-alpha's recordings only |
| 2026-10-31 | Licence (D-13) | MIT for the code; licensed or recorded data is never redistributed |
| 2026-10-31 | Spec defaults | Every other section 3 decision with a default (D-02 to D-12) stands as written |
| 2026-10-31 | Storage (P-04) | One shared crypto capture on D:. This repo reads crypto-lob-alpha's recordings in place under `~/data/crypto-lob-alpha`, read-only, and never records or copies them. Data, run outputs and RL artifacts live in WSL ext4 under `DATA_ROOT`, never on C: and never in git |
| 2026-11-03 | Memory | Parallel runs scale to free memory and stop below 30 GB free |
| 2026-11-07 | crypto-lob-alpha AC-W-01 window | Its 14-day window was moved to 2026-11-08 to 11-21 after two ISP outages (11-04 22:05–23:51 and 11-07 13:31–16:24 UTC). The recorder keeps running; those outages show as gaps in the shared capture |
| 2026-11-09 | Windows time sync | `MaxPollInterval` set to 10 (1,024 s) after the host clock drifted up to about 55 ms fast between 9.1 h syncs. `ts_recv` on earlier days carries that offset, which enters the recorded feed latency (`ts_recv − ts_exchange`, D-09). crypto-lob-alpha's `clock` table has the per-minute offset |

### Brief for this build (2026-11-09)

- **Recordings (P-05):** crypto-lob-alpha's existing recordings under `~/data/crypto-lob-alpha`, read-only, light reads only (the live recorder writes to the same HDD). Never start a second recorder.
- **Signal artifact (P-06):** comes from crypto-lob-alpha, which is still capturing. If M4 arrives first, report and wait.
- **Simulator (P-08):** the `edbt` package from event-driven-backtester; the fallback simulator if the swap isn't ready.
- **Money:** none spent unless D-01 changes.
- **Shared PC:** at most 8 threads at `nice 10` for heavy runs; downloads capped at 10 MB/s; keep D: above 150 GB free; never restart WSL. Published benchmark runs wait until no multi-week run is active (after about 2026-11-29).

## Build decisions

| ID | Date | Decision | Why |
|---|---|---|---|
| M1-01 | 2026-11-09 | Dependencies join `pyproject.toml` with the milestone that first imports them: NumPy, SciPy, Polars, PyArrow, Typer, pydantic, PyYAML and structlog now; Numba, DuckDB, matplotlib and httpx from M2/M3; LightGBM at M4; gymnasium, stable-baselines3 and CPU PyTorch at M6 | Keeps the lockfile and CI to what the code uses; the versions in section 4 still apply when each arrives |
| M1-02 | 2026-11-09 | CI runs lint, mypy, tests on ubuntu-latest and windows-latest (Python 3.12) and a trial-log job. The `validate-fast` job arrives with M2 (`mms validate synthetic`) and `rl-smoke` with M6. `overfitlab` is a dev dependency (PyPI 0.1.1), so `overfitlab trials validate` always runs instead of skipping | The jobs need the commands they call |
| M1-03 | 2026-11-09 | Extras `[edbt]` and `[cla]` exist but are empty until D-02 (M6) and D-03 (M4). `[tardis]` is `tardis-dev>=2.0`; `[bol]` is `overfitlab>=0.1.1` | `edbt-sim` is not on PyPI and a git source would need a Rust build at lock time |
| M1-04 | 2026-11-09 | Tick and lot sizes live in `configs/instruments.yaml`, fees in `configs/fees.yaml` by venue, tier and effective date. The shared capture has no `instruments` dataset; the values are those crypto-lob-alpha recorded from Binance `exchangeInfo` on 2026-11-01. Retail is VIP 0 at maker 2.0 bps and taker 5.0 bps from 2026-11-01, as read by crypto-lob-alpha from the fee page that day; owner confirmation is pending (P-13, before M5). `mid`, `zero` and `rebate` are D-10's hypothetical scenarios and are labelled `hypothetical: true` | DQ-08 needs effective-dated fees; days before 2026-11-01 have no fee row and fail |
| M1-05 | 2026-11-09 | Trial log header is exactly the spec's: the ten contract columns plus `x_config_digest`, `x_selected`, `x_periods_per_year` (365 crypto, 252 for `_lobster` families). With a single daily observation the Sharpe ratio is undefined and `0` is written, because the contract requires a finite value on any row with a series. Strategy families are checked against the spec's list | Contract v1.0 and spec section 7 |
| M1-06 | 2026-11-09 | `mms data check` reads only Parquet footers by default (schema, row counts, `ts_recv` statistics); `--deep` also reads the rows and runs the DQ-13 content checks. Gap-log rows with reason `clock` are reported separately and not counted as downtime, matching crypto-lob-alpha's `NOT_DOWNTIME` | Light reads on the HDD the live recorder writes to |
| M1-07 | 2026-11-09 | Synthetic world: the binomial step σ√dt is rounded to whole ticks (A–S 2008 world: 1,414 ticks of 1e-4 for 0.141421…, a 1.5e-4 relative error in σ); each market order is one trade at `mid + side·floor(D/tick)·tick`, so a quote on the grid at j ticks fills exactly when D ≥ j·tick; the book is a symmetric display around the mid; path i starts i UTC days after `start_utc` | Keeps prices on an integer grid (section 7(b)) while preserving the A·e^(−kδ) fill intensity |
| M1-08 | 2026-11-09 | Replay sorts each Parquet row group and checks the order across row-group and file boundaries instead of sorting a whole day; trades with equal `ts_recv` sort by numeric trade id when every id in the chunk is an integer, else keep file order; events before the first snapshot batch are dropped and counted | Files are hourly by `ts_recv`, so per-chunk sorting gives the global order with bounded memory (NFR-12) |
| M1-09 | 2026-11-09 | `mms data synth` refuses a default output root under `/mnt/*` or on C:. On native Windows `~` expands to C:, so data-writing commands take an explicit `--out` there or run in WSL, as section 7(d) intends for real-data work | Data never goes on C: |
| M1-10 | 2026-11-09 | M1's AC-W-11 was also run (footers only) on one real day, 2026-11-08, read over `\\wsl.localhost`: BTCUSDT 64.2M book_l2 rows and 726k trades, ETHUSDT 58.7M and 899k, SOLUSDT 16.8M and 247k, 24 hourly files each, 13 non-clock gaps totalling about 179 s, fees covered. P-05 (≥ 10 complete days for 3 symbols) is not met yet: the capture began 2026-11-01 22:11 UTC | Early warning for M3: at about 64M book rows a day for BTCUSDT, NFR-01 and NFR-02 are the main risk (R-06) |
| M2-01 | 2026-11-09 | AC-W-02 uses the paper's three tables: γ = 0.1, 0.01 and **1** (Tables 1–3, printed pp. 222–223). The spec lists γ = 0.5, which the paper does not report; the paper is followed, as section 16 requires | Published values exist only for γ = 1 |
| M2-02 | 2026-11-09 | The A–S replication world follows the paper's section 3.3 procedure: `arrivals: bernoulli_step` puts at most one market order per side in each `dt` step, with probability `A·dt` and depth `Exp(k)`, so a quote at `δ ≥ 0` fills with probability `λ(δ)·dt` as in the paper. Its display book sits 10 price units from the mid so it never meets the quotes. The Poisson world (FR-05) is used for the intensity check (AC-W-01) | With Poisson arrivals and quotes refreshed every `dt`, a filled quote is missing until the next step, which lowers fills by about 12% at `λ·dt ≈ 0.27` |
| M2-03 | 2026-11-09 | AC-W-02's "3 standard errors" uses our own sample's SE: `sd/√n` for means and `sd/√(2n)` for standard deviations, n = 1,000 paths. Seeds come from `configs/synthetic/as2008.yaml` (2008, plus 10 × table + strategy) and were set before any comparison. The published numbers carry their own sampling error, which this reading ignores | The spec's literal reading; see the AC-W-02 finding below |
| M2-04 | 2026-11-09 | A model quote that would cross the visible book is pulled to one tick inside it (bid below the best ask, ask above the best bid). The exchange may still reject it against its own, later book | Avoids self-inflicted post-only rejects; DQ-11's rounding away from the mid is kept |
| M2-05 | 2026-11-09 | Quotes fill only against public trades. A trade priced through our price fills the whole remaining quantity; a trade at our price goes through the queue model. Book changes alone never fill | The standard L2 convention (as in hftbacktest); documented in `docs/simulator-contract.md` |
| M2-06 | 2026-11-09 | `requote_threshold_ticks`: a working order is kept when `|new − old| < threshold`, so 0 and 1 both keep only exact matches and 2 tolerates one tick | The spec gives no comparison operator |
| M2-07 | 2026-11-09 | The inventory limit counts orders whose cancel is still in flight, since they can still fill. Near the limit a replacement quote can wait for the cancel's ack | FR-11's worst case |
| M2-08 | 2026-11-09 | One symbol per run; sweeps iterate over symbols. The trial log is `<results_dir>/trial_log.csv`, because the contract requires `runs/<run_id>/returns.parquet` relative to the log; `explog.path` in configs is informational | Keeps every trial row valid when `MMS_RESULTS_DIR` moves |
| M2-09 | 2026-11-09 | Synthetic runs take known `A`, `k`, `σ` from `strategy.params`; calibration from data arrives in M3 (FR-12) | M2 needs no calibration |
| M2-10 | 2026-11-09 | The decision timer starts at the first market event of a day (synthetic paths: at the path start) | Avoids idle timer steps from midnight |
| M2-11 | 2026-11-09 | AC-W-04's direction checks run in `configs/synthetic/queue_latency.yaml`, a BTC-like tick world whose displayed levels sit where a GLFT quote joins. Added latency does not lower fill rates in every world: stale quotes can sit closer to a moved mid and fill more (adversely). Here the lost requote time after fills dominates | Stated so the check is not over-read |
| M2-12 | 2026-11-09 | `--fast` keeps 20,000 paths for the GLFT check | CARA utility is heavy-tailed at γ = 0.5; 4,000 paths gave an unreliable SE. The check takes seconds |
| M2-13 | 2026-11-09 | AC-W-14 reports four parameter sets, including the GLFT paper's Figure 1 setting. Three are within 5% for `|q| ≤ Q/2`; γ = 0.5, Q = 5 is not (5.1%), because the continuous approximation is weak when `α/η` is large. The criterion (≥ 3 sets) is met and the fourth is shown | Not hidden |
| M2-14 | 2026-11-09 | The fallback's reconciliation check also fails on a non-finite residual | A NaN fee passed `abs(residual) > 1e-9` silently; found by a test |
