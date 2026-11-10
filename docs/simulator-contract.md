# Simulator contract

The types live in `src/mmsim/sim/types.py` and the Protocols in `src/mmsim/sim/protocol.py`; both follow spec section 10 literally. The fallback simulator (`src/mmsim/sim/fallback.py`, M2) is the reference implementation; `tests/contract/test_simulator_contract.py` runs the contract against every available implementation (AC-W-08). The edbt mapping arrives with M6.

## Who imports what

- Strategies, the runner, analytics and RL code import only `mmsim.sim.types` and `mmsim.sim.protocol`.
- Only `mmsim/sim/edbt_adapter.py` imports `edbt`.
- A missing feature is declared through `capabilities()`, never by changing the Protocol. A full implementation declares `{"l2", "queue_position", "latency", "fees", "timers"}`. A run that needs a capability its backend lacks fails at start-up.

## Units and clocks

- Prices are integer ticks and quantities integer lots. Converting from the canonical float columns rounds to the grid and fails if the residual is 1e-6 tick or lot or more.
- Timestamps are int64 ns UTC.
- `ts_exchange` is the matching clock: each stream's `ts_exchange`, made non-decreasing within the stream.
- `ts_local` is the strategy's clock. An exchange event at `t` is visible at `t + feed_delay` (DQ-02). A decision at local time `t` reaches the exchange at `t + order_delay`; an exchange-side order event at `t` is seen at `t + ack_delay`.
- `BookView.mid_x2` is bid + ask in ticks, so mids stay integers.

## Event order (DQ-01, DQ-03)

The replay (`mmsim.data.replay`) feeds every backend the same stream:

1. Each stream in a stable sort by `ts_recv`, then `batch_id` (book_l2) or the numeric trade id (trades), then file row order.
2. Matching time = running maximum of `ts_exchange` within the stream.
3. Streams merged by `(matching time, priority, ts_recv, row index)`, trades before book_l2 at equal matching time.
4. A book_l2 batch (one venue message) is applied atomically and never split; a snapshot batch replaces the book.
5. Replay starts at the first snapshot batch; earlier events are dropped and counted.

On a sequence gap the simulator cancels every resting quote, stops quoting until the next snapshot, and excludes the interval from markouts (DQ-03, M3).

## One timeline

The fallback keeps a single timeline. Market events happen at their matching time. The strategy sees them at `max(matching time, ts_exchange + feed delay)`. Its orders and cancels reach the exchange `order_delay` after the decision, and exchange-side order events (acks, fills) reach it `ack_delay` later. At equal times the order is: market event, order or cancel arrival, strategy-visible delivery, timer. Everything visible at the same time comes back as one `StepResult`, so a book change and a timer at the same instant give one decision on the new book.

The runner decides on every timer tick (D-07, 100 ms) and on every own fill; it never reacts to bare acks, so zero-latency acks cannot loop. `step()` returns `None` once the market data is exhausted and only timers remain.

## Orders

- Quotes are post-only (`TimeInForce.GTX`). A post-only order that would cross on arrival is rejected, never converted to a taker order (DQ-11).
- `submit_limit_ticks` returns the client order id at once; the order reaches the exchange after the order latency and its acceptance is seen after the ack latency.
- `open_orders` and `position_lots` reflect only what the strategy has seen.

## Matching

Resting quotes fill only against public trades, never against book changes alone:

- A trade priced through our price (a sell below our bid, a buy above our ask) fills the whole remaining quantity, because the sweep emptied our level.
- A trade at our price goes through the queue model: it consumes the queue ahead first, and what is left fills us.
- Trades with side 0 (unknown aggressor) never fill quotes and are counted.
- A crossed or locked reconstructed book is counted and not shown to the strategy (DQ-04); excluding a day above 0.01% crossed events is an M3 data rule.

## Fills

Every `Fill` carries the queue ahead at entry, the order latency and the fee in quote currency (positive cost, negative rebate). `mid_x2_before` is the exchange-time mid just before the event that caused the fill (DQ-09); the P&L decomposition and markouts use it.

## Queue models (FR-02, DQ-12)

| Model | Ahead at entry | Trades at our level | Cancellations at our level |
|---|---|---|---|
| `front` | 0 | fill us first | no effect |
| `conservative` | displayed size | reduce ahead first, then fill us | no effect |
| `prob` (power n, default 2) | displayed size | reduce ahead first, then fill us | reduce ahead with probability f(ahead)/(f(ahead)+f(behind)), f(x) = xⁿ |

L2 depth updates often arrive after the trades that caused them. Volume already traded at a level is netted against the next size decrease at that level before the rest counts as cancellation (DQ-12).

## Parity (AC-H-10)

Fallback and edbt must give identical fill lists in the synthetic world (front queue, zero latency) and on the fixture day (conservative queue, constant latency), and fill counts and P&L within 2% averaged over 20 seeds with the `prob` queue.
