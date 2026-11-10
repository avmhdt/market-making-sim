"""Synthetic Poisson order flow in the canonical schemas (FR-05).

The world of Avellaneda and Stoikov (2008), section 4: a mid process plus buy and sell
market orders, each side a Poisson process of rate ``A`` whose depth beyond the mid is
``D ~ Exp(k)``. A resting quote at distance ``delta`` is reached by an order with
``D >= delta``, so it fills at intensity ``A * exp(-k * delta)``. Each market order is
emitted as one canonical trade at ``mid + side * floor(D / tick) * tick``, so a quote on
the tick grid at ``j`` ticks fills exactly when ``D >= j * tick``.

The book is a symmetric display around the mid (``levels`` levels each side from
``half_spread_ticks``), written as an initial snapshot followed by incremental diffs.
Path ``i`` starts ``i`` UTC days after ``start_utc``. All randomness flows from
``numpy.random.SeedSequence([seed, i])`` (NFR-03).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from mmsim.config import SynthConfig
from mmsim.io.parquet import write_canonical
from mmsim.sim.types import InstrumentSpec

__all__ = [
    "NS_PER_DAY",
    "SynthWorld",
    "generate",
    "generate_path",
    "synth_instrument",
    "write_world",
]

NS_PER_DAY = 86_400_000_000_000
BATCH_ID_STRIDE = 1_000_000_000

Int64Array = npt.NDArray[np.int64]


@dataclass(frozen=True)
class SynthWorld:
    """One generated path: canonical frames plus the true mid path for validation."""

    path_index: int
    start_ns: int
    end_ns: int
    book_l2: pl.DataFrame
    trades: pl.DataFrame
    mid_times_ns: Int64Array
    """Exchange times at which the mid takes ``mid_ticks``; the first is ``start_ns``."""
    mid_ticks: Int64Array
    instrument: InstrumentSpec


def synth_instrument(cfg: SynthConfig) -> InstrumentSpec:
    """The instrument of a synthetic world: a fee-free perpetual."""
    return InstrumentSpec(
        venue=cfg.venue,
        symbol=cfg.symbol,
        tick_size=cfg.tick_size,
        lot_size=cfg.lot_size,
        contract_type="perp",
        maker_bps=0.0,
        taker_bps=0.0,
        fee_tier="zero",
    )


def _to_ns(start_ns: int, t: npt.NDArray[np.float64], unit_ns: int) -> Int64Array:
    return start_ns + np.rint(t * unit_ns).astype(np.int64)


def _mid_path(
    cfg: SynthConfig, rng: np.random.Generator, start_ns: int
) -> tuple[Int64Array, Int64Array]:
    s0 = round(cfg.mid.s0 / cfg.tick_size)
    if cfg.mid.kind == "binomial_grid":
        n = round(cfg.horizon / cfg.dt)
        if not math.isclose(n * cfg.dt, cfg.horizon, rel_tol=1e-9):
            raise ValueError("horizon must be a whole number of dt steps")
        step = round(cfg.mid.sigma * math.sqrt(cfg.dt) / cfg.tick_size)
        if step < 1:
            raise ValueError("sigma*sqrt(dt) is below one tick; use a finer tick_size")
        moves = rng.choice(np.array([-1, 1], dtype=np.int64), size=n) * step
        times = _to_ns(start_ns, np.arange(1, n + 1, dtype=np.float64) * cfg.dt, cfg.time_unit_ns)
    else:
        rate = (cfg.mid.sigma / cfg.tick_size) ** 2
        count = int(rng.poisson(rate * cfg.horizon))
        t = np.sort(rng.uniform(0.0, cfg.horizon, size=count))
        times = np.maximum(_to_ns(start_ns, t, cfg.time_unit_ns), start_ns + 1)
        moves = rng.choice(np.array([-1, 1], dtype=np.int64), size=count)
    mid = np.concatenate([[s0], s0 + np.cumsum(moves)]).astype(np.int64)
    return np.concatenate([[start_ns], times]).astype(np.int64), mid


def _levels(mid: int, cfg: SynthConfig) -> tuple[dict[int, int], dict[int, int]]:
    h, lots = cfg.book.half_spread_ticks, cfg.book.level_lots
    bids = {mid - h - i: lots for i in range(cfg.book.levels)}
    asks = {mid + h + i: lots for i in range(cfg.book.levels)}
    return bids, asks


def _recv(ts: Int64Array, cfg: SynthConfig, rng: np.random.Generator) -> Int64Array:
    jitter = (
        rng.integers(0, cfg.feed_jitter_ns + 1, size=ts.size, dtype=np.int64)
        if cfg.feed_jitter_ns
        else np.zeros(ts.size, dtype=np.int64)
    )
    recv = ts + cfg.feed_latency_ns + jitter
    return np.maximum.accumulate(recv) if recv.size else recv


def generate_path(cfg: SynthConfig, path_index: int = 0) -> SynthWorld:
    """Generate path ``path_index`` of the world described by ``cfg``."""
    if cfg.horizon * cfg.time_unit_ns >= NS_PER_DAY:
        raise ValueError("a synthetic path must fit inside one UTC day")
    mid_rng, buy_rng, sell_rng, feed_rng = (
        np.random.Generator(np.random.PCG64(child))
        for child in np.random.SeedSequence([cfg.seed, path_index]).spawn(4)
    )
    start_ns = int(cfg.start_utc.timestamp()) * 1_000_000_000 + path_index * NS_PER_DAY
    end_ns = start_ns + round(cfg.horizon * cfg.time_unit_ns)
    mid_times, mid_ticks = _mid_path(cfg, mid_rng, start_ns)

    # Market orders: Poisson or per-step Bernoulli arrivals, depth D ~ Exp(k).
    trade_t: list[Int64Array] = []
    trade_px: list[Int64Array] = []
    trade_side: list[npt.NDArray[np.int8]] = []
    for side, rng in ((1, buy_rng), (-1, sell_rng)):
        mo = cfg.market_orders
        if mo.arrivals == "poisson":
            count = int(rng.poisson(mo.A * cfg.horizon))
            t = np.sort(rng.uniform(0.0, cfg.horizon, size=count)).astype(np.float64)
            shift = 0.0
        else:
            # One order per side per step, depth Exp(k) - c with c = ln(1 / (A dt)) / k:
            # P(D >= delta) = min(1, A exp(-k delta) dt) for every delta, the paper's rule.
            count = round(cfg.horizon / cfg.dt)
            t = (np.arange(count, dtype=np.float64) + 0.5) * cfg.dt
            shift = math.log(1.0 / (mo.A * cfg.dt)) / mo.k
        depth = rng.exponential(1.0 / mo.k, size=count) - shift
        ts = np.maximum(_to_ns(start_ns, t, cfg.time_unit_ns), start_ns + 1)
        # Trades sort before book changes at equal time, so they see the earlier mid.
        idx = np.searchsorted(mid_times, ts, side="left") - 1
        ticks = np.floor(depth / cfg.tick_size + 1e-9).astype(np.int64)
        trade_t.append(ts)
        trade_px.append(mid_ticks[idx] + side * ticks)
        trade_side.append(np.full(count, side, dtype=np.int8))
    ts_all = np.concatenate(trade_t)
    order = np.argsort(ts_all, kind="stable")
    t_ex = ts_all[order]
    px = np.concatenate(trade_px)[order]
    sd = np.concatenate(trade_side)[order]
    n_trades = t_ex.size
    trades = pl.DataFrame(
        {
            "ts_exchange": t_ex,
            "ts_recv": _recv(t_ex, cfg, feed_rng),
            "venue": [cfg.venue] * n_trades,
            "symbol": [cfg.symbol] * n_trades,
            "trade_id": [str(path_index * BATCH_ID_STRIDE + i + 1) for i in range(n_trades)],
            "price": px.astype(np.float64) * cfg.tick_size,
            "qty": np.full(n_trades, cfg.market_orders.qty_lots * cfg.lot_size),
            "side": sd,
        },
        schema={
            "ts_exchange": pl.Int64,
            "ts_recv": pl.Int64,
            "venue": pl.String,
            "symbol": pl.String,
            "trade_id": pl.String,
            "price": pl.Float64,
            "qty": pl.Float64,
            "side": pl.Int8,
        },
    )

    # Book: snapshot at the start, then one diff batch per mid change.
    rows_t: list[int] = []
    rows_side: list[int] = []
    rows_px: list[int] = []
    rows_lots: list[int] = []
    rows_snap: list[bool] = []
    rows_batch: list[int] = []
    bids, asks = _levels(int(mid_ticks[0]), cfg)
    batch0 = path_index * BATCH_ID_STRIDE + 1
    for side, levels in ((1, bids), (-1, asks)):
        for price, lots in sorted(levels.items()):
            rows_t.append(int(mid_times[0]))
            rows_side.append(side)
            rows_px.append(price)
            rows_lots.append(lots)
            rows_snap.append(True)
            rows_batch.append(batch0)
    for k in range(1, mid_ticks.size):
        new_bids, new_asks = _levels(int(mid_ticks[k]), cfg)
        for side, old, new in ((1, bids, new_bids), (-1, asks, new_asks)):
            for price in sorted(set(old) - set(new)):
                rows_t.append(int(mid_times[k]))
                rows_side.append(side)
                rows_px.append(price)
                rows_lots.append(0)
                rows_snap.append(False)
                rows_batch.append(batch0 + k)
            for price in sorted(set(new) - set(old)):
                rows_t.append(int(mid_times[k]))
                rows_side.append(side)
                rows_px.append(price)
                rows_lots.append(new[price])
                rows_snap.append(False)
                rows_batch.append(batch0 + k)
        bids, asks = new_bids, new_asks
    t_book = np.asarray(rows_t, dtype=np.int64)
    batch_arr = np.asarray(rows_batch, dtype=np.int64)
    # One receive time per batch, so a batch never straddles two receive times.
    first_rows = np.flatnonzero(np.r_[True, batch_arr[1:] != batch_arr[:-1]])
    recv_batch = _recv(t_book[first_rows], cfg, feed_rng)
    counts = np.diff(np.r_[first_rows, batch_arr.size])
    n_rows = t_book.size
    book = pl.DataFrame(
        {
            "ts_exchange": t_book,
            "ts_recv": np.repeat(recv_batch, counts),
            "venue": [cfg.venue] * n_rows,
            "symbol": [cfg.symbol] * n_rows,
            "seq": batch_arr - path_index * BATCH_ID_STRIDE,
            "side": np.asarray(rows_side, dtype=np.int8),
            "price": np.asarray(rows_px, dtype=np.float64) * cfg.tick_size,
            "qty": np.asarray(rows_lots, dtype=np.float64) * cfg.lot_size,
            "is_snapshot": rows_snap,
            "batch_id": batch_arr,
        },
        schema={
            "ts_exchange": pl.Int64,
            "ts_recv": pl.Int64,
            "venue": pl.String,
            "symbol": pl.String,
            "seq": pl.Int64,
            "side": pl.Int8,
            "price": pl.Float64,
            "qty": pl.Float64,
            "is_snapshot": pl.Boolean,
            "batch_id": pl.Int64,
        },
    )
    return SynthWorld(
        path_index=path_index,
        start_ns=start_ns,
        end_ns=end_ns,
        book_l2=book,
        trades=trades,
        mid_times_ns=mid_times,
        mid_ticks=mid_ticks,
        instrument=synth_instrument(cfg),
    )


def generate(cfg: SynthConfig) -> Iterator[SynthWorld]:
    """All ``cfg.paths`` paths, in order."""
    for index in range(cfg.paths):
        yield generate_path(cfg, index)


def write_world(world: SynthWorld, root: Path) -> list[Path]:
    """Write a path's book_l2 and trades into the canonical layout under ``root``."""
    return write_canonical(world.book_l2, root, "book_l2") + write_canonical(
        world.trades, root, "trades"
    )
