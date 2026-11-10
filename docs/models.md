# Models

Formulas as implemented, with their sources. Units: prices in quote currency, time in seconds, `σ` in price per √s, `A` in fills per second, `k` in 1/price, `γ` in 1/(price × unit). In code, inventory `q` is the position in quote units (lots ÷ quote lots).

## Avellaneda and Stoikov (2008)

Avellaneda, M. and Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance* 8(3), 217–224. Section 3.2. Code: `src/mmsim/models/avellaneda_stoikov.py`.

| Quantity | Formula | Paper |
|---|---|---|
| Reservation price | `r(s, t) = s − q γ σ² (T − t)` | eq. (29) |
| Total spread | `δa + δb = γ σ² (T − t) + (2/γ) ln(1 + γ/k)` | eq. (30) |
| Quotes | `r ∓ (δa + δb)/2` | section 3.2 |

The spread is centred on the reservation price, not the mid. `τ = T − t` for a finite horizon (validation, finite sessions) or `τ = horizon_s` held constant on 24/7 markets (`strategy.tau_mode: rolling`).

## Guéant, Lehalle and Fernandez-Tapia (2013)

Guéant, O., Lehalle, C.-A. and Fernandez-Tapia, J. (2013). Dealing with the inventory risk: a solution to the market making problem. *Mathematics and Financial Economics* 7(4), 477–507; arXiv:1105.3115. The arXiv text numbers propositions and theorems, not equations, so those are cited. Code: `src/mmsim/models/glft.py`.

| Quantity | Formula | Paper |
|---|---|---|
| Constants | `α = (k/2) γ σ²`, `η = A (1 + γ/k)^(−(1 + k/γ))` | Proposition 1 |
| Exact solution | `v(t) = exp(−M (T − t)) · 1`, `M` tridiagonal on `q ∈ [−Q, Q]`, diagonal `α q²`, off-diagonals `−η` | Proposition 2 |
| Exact quotes | `δb(t, q) = (1/k) ln(v_q / v_{q+1}) + c1` for `q ≠ Q`; `δa(t, q) = (1/k) ln(v_q / v_{q−1}) + c1` for `q ≠ −Q` | Theorem 1 |
| Value function | `u(t, x, q, s) = −exp(−γ (x + q s)) v_q(t)^(−γ/k)` | Theorem 1 |
| Asymptotic quotes | `δb ≈ c1 + (q + ½) c2`, `δa ≈ c1 − (q − ½) c2` | section 4, after Proposition 3 |
| | `c1 = (1/γ) ln(1 + γ/k)`, `c2 = √(σ² γ / (2 k A) · (1 + γ/k)^(1 + k/γ))` | |
| Drift extension (M4) | diagonal `α q² − β q` with `β = k μ`; asymptotically adds `∓ μ/(γ σ²) · c2` | Proposition 4, section 5.1 |

The spec's FR-10 formulas agree with the paper. The exact solution is computed from the eigendecomposition of the symmetric `M`, in logs and with the smallest eigenvalue factored out, so long horizons neither overflow nor underflow; it matches `scipy.linalg.expm` to 1e-10 (`tests/unit/test_glft.py`). The certainty equivalent reported for AC-W-03 is `−(1/γ) ln(−u) = x + q s + (1/k) ln v_q(0)`.

## Fee-aware quoting (FR-13)

A maker fee `c` per unit (quote currency; negative is a rebate) earned at distance `δ` nets `δ − c`. With `δ' = δ − c`, the fill intensity is

`A e^(−k δ) = A e^(−k c) · e^(−k δ') = A' e^(−k δ')`, with `A' = A e^(−k c)`.

The problem in `δ'` is the original one with `A'` in place of `A`, so the policy solves it with `A'` and quotes `δ = δ' + c`. This is exact under the exponential intensity. A rebate (`c < 0`) raises `A'` and narrows the quotes. In code `c = maker_bps / 1e4 × mid`.

## Units on real data

`σ` comes from 1 s sampled mid changes and `A`, `k` from the intensity fit of FR-12 (M3). A quote unit is `quote_lots` lots (D-08: the smallest lot multiple with notional ≥ 1,000 USD), and `Q = q_max_units` (D-08: 10).

## Validation

`mms validate synthetic` checks these formulas against known results: the A–S 2008 tables, the GLFT certainty equivalent, and the asymptotic-versus-exact gap (AC-W-14). Results are in `reports/tables/synthetic_validation.csv`.
