# Market-making simulator and agent

> Optimal quoting under inventory risk, tested on replayed order books, with P&L broken down to its sources.

**Status:** in progress (M3 of 7: real data, calibration and the first P&L decomposition)

| | |
|---|---|
| Track | Hybrid (Researcher + Developer) |
| Estimated effort | 5–7 weeks |
| Stack | Python, plus your [event-driven-backtester](https://github.com/avmhdt/event-driven-backtester) engine; PyTorch for RL |
| Data cost | Free → Paid |
| Strongest for | Market-making QR/QT and quant dev |

## Requirements

- [x] Avellaneda–Stoikov (2008) and Guéant–Lehalle–Fernandez-Tapia (2013) quoting with inventory limits. (FR-09, FR-10, FR-11; AC-W-02, AC-W-03, AC-W-05, AC-W-14)
- [x] Parameters estimated from data: order-arrival intensity by distance from mid, volatility, fee tiers. (FR-08, FR-12; AC-W-10)
- [ ] Simulation with queue position and latency (reuse [event-driven-backtester](https://github.com/avmhdt/event-driven-backtester)).
- [ ] P&L decomposition: spread capture, inventory P&L, adverse selection via markouts at several horizons, and fees and rebates.
- [ ] Quotes skewed by your short-horizon signal from [crypto-lob-alpha](https://github.com/avmhdt/crypto-lob-alpha).
- [ ] Optionally, an RL agent (PPO or SAC) compared with the analytic strategy.

## Data

Raw and licensed data live in `data/`, which is git-ignored. Only download scripts and small synthetic fixtures are committed.

| Dataset | Cost | What it gives you |
|---|---|---|
| [Your L2 recordings (Binance, Bybit, OKX)](https://developers.binance.com) | Free | Perpetual futures work well: deep books and clear fee tiers. |
| [Tardis.dev](https://tardis.dev) | Paid | Long multi-venue history for robustness tests. |
| [LOBSTER samples](https://lobsterdata.com) | Free samples | An equity contrast. |
| Synthetic Poisson order flow | Free | Validates the simulator against closed-form results. |

## Success criteria

### Works (correct and reproducible)

- [x] In the synthetic Poisson world, simulated results match the analytic model. (AC-W-01 to AC-W-04; AC-W-02 with the documented exception M2-20)
- [x] Inventory stays within limits, and the P&L components sum exactly to total P&L. (AC-W-05, AC-W-06, on synthetic and recorded runs)

### Impresses hirers

- [ ] Markout curves that show adverse selection, and how much the signal skew reduces it.
- [ ] Results across several days, assets and fee tiers, with confidence intervals.
- [ ] An honest RL vs analytic comparison that includes training cost and robustness.
- [ ] Latency sensitivity: P&L as a function of added round-trip latency.

## Questions this project must answer

- Where does a market maker’s P&L actually come from?
- What does your strategy do during a large one-sided move?
- How would you quote across correlated instruments?
