# Reference hardware

Recorded 2026-11-09 for the benchmarks in `reports/tables/benchmark.csv` (NFR-01, NFR-02, NFR-12).

| Item | Value |
|---|---|
| CPU | Intel Core i9-14900K, 24 cores, 32 logical processors |
| RAM | 128 GB physical; WSL2 sees 62 GB |
| OS | Windows 11 Pro; WSL2 Ubuntu 22.04.5 LTS, kernel 6.18.54.1-microsoft-standard-WSL2 |
| Python | CPython 3.12.10 (Windows, via uv) |
| Data disk | D:, internal 2 TB SATA HDD, holding the WSL ext4 image (`D:\WSL\Ubuntu\ext4.vhdx`) |
| GPU | none used |

## Shared machine

The PC also runs crypto-lob-alpha's live recorder (until about 2026-11-21 for its AC-W-01 window) and event-driven-backtester's Deribit testnet run (until 2026-11-28 20:10 UTC). Heavy runs use at most 8 threads at `nice 10`. Published benchmark numbers wait until neither run is active.

## Data volume (measured)

Footer counts of crypto-lob-alpha's capture for 2026-11-08 (`mms data check`):

| Symbol | book_l2 rows | trades rows |
|---|---|---|
| BTCUSDT | 64,205,039 | 726,369 |
| ETHUSDT | 58,687,569 | 898,975 |
| SOLUSDT | 16,786,270 | 247,161 |

On-disk size per symbol-day is about 170 MB of zstd Parquet for BTCUSDT book_l2 (24 hourly files of about 7 MB) on 2026-11-08; busier days reach 146M book rows (`mms data check` over 2026-11-02 to 11-19, 48 symbol-days, 0 problems).

The decoded event cache (`$DATA_ROOT/interim/events`, docs/decisions.md M3-10) takes about 1.3 GB per BTCUSDT day, 1.1 GB per ETHUSDT day and 0.26 GB per SOLUSDT day: 13 GB for the 15 tuning symbol-days.

Since 2026-11-17 the repository and the WSL distro live on G: (`G:\WSL\Ubuntu`); D: was failing.
