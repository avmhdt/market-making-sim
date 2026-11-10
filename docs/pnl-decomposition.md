# P&L decomposition

Code: `src/mmsim/analytics/pnl.py` (FR-17, NFR-04). Tests: `tests/property/test_pnl_identity.py` (AC-W-06).

## Definitions

For fills `i` with signed quantity `q_i` (+ when we buy), price `p_i` and pre-fill mid `m_i` (the exchange-time mid just before the fill's triggering event, DQ-09); the exchange-time mid `m(t)` (as-of); run end `T`; start position `Q0`; headline horizon `h*` (D-05: 5 s); and `t_i⁺ = min(t_i + h*, T)`:

| Component | Formula | Reading |
|---|---|---|
| Spread capture | `Σ q_i (m_i − p_i)` | Edge against the mid at the moment of the fill; non-negative for passive fills inside the spread |
| Adverse selection | `Σ q_i (m(t_i⁺) − m_i)` | How far the mid moved against the fill within `h*` |
| Inventory residual | `Σ q_i (m_T − m(t_i⁺)) + Q0 (m_T − m_0)` | What holding the position after `h*` earned or lost |
| Fees | `− Σ fee_i` | Rebates add |
| Funding | `− Σ_f Q(t_f) · mark_f · rate_f` | Longs pay shorts when the rate is positive |
| Total | `cash_T + Q_T m_T − (cash_0 + Q0 m_0)` | Mark-to-mid, from the cash ledger |

## Identity

`spread + adverse selection + inventory residual + fees + funding = total`

The first three telescope: for each fill they add up to `q_i (m_T − p_i)`, and the start position adds `Q0 (m_T − m_0)`. The cash ledger gives the same sum independently: `cash_T = −Σ q_i p_i`.

## Exactness

Price terms are integers in half-tick × lot units: `mid_x2` is bid + ask in ticks and `2 p_i` is the fill price, so every product is an exact int64. An overflow guard requires every term below 2⁶². The total's price part comes from the integer cash ledger, and the integer identity is checked with `==`. One multiplication by `tick_size / 2 × lot_size` converts to quote currency. Fees and funding are floats summed with `math.fsum`. The overall residual must be at most 1e-9 quote currency, or the run fails (`ReconciliationError`); a non-finite residual also fails.

## Markouts (FR-18)

`mo_mid_bps(i, h) = sign(q_i) (m(t_i + h) − m_i) / m_i × 1e4` and `mo_fill_bps(i, h) = sign(q_i) (m(t_i + h) − p_i) / p_i × 1e4`, at 100 ms, 500 ms, 1 s, 5 s, 10 s, 30 s and 60 s. A markout is invalid when `t_i + h` falls after the data end or inside a gap. Volume-weighted means per horizon are in `metrics.json`; day-block bootstrap CIs arrive with the real-data curves in M3.

## Outputs

- `pnl_attribution.parquet`: per fill `spread_x2`, `as_x2_<h>` for every horizon (headline first), `inv_resid_x2` (headline horizon) and `fee_quote`.
- `metrics.json`: the five components, the total and `reconciliation_residual` (the largest absolute residual over the run's sessions).
