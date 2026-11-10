"""The event loop, run directory and experiment-log row (spec section 5, FR-25).

``run_session`` drives one simulator over one session (a UTC day or a synthetic path): a
periodic timer (D-07) plus every own fill triggers a decision; the quote manager applies it
under the inventory limit. ``execute_run`` builds the sessions of a config, runs them,
decomposes P&L (the reconciliation is an assertion), writes every output of section 7(b)
and appends exactly one trial row, also when the run fails.
"""

from __future__ import annotations

import json
import math
import platform
import statistics
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import polars as pl

from mmsim.analytics.markouts import markout_summary, markout_table
from mmsim.analytics.pnl import NS_PER_MS, PnLBreakdown, attribution_table, decompose_pnl, mid_asof
from mmsim.config import MmsConfig, load_synth_config, resolve_dict
from mmsim.data.instruments import FeeCoverageError, load_fee_rows, resolve_instrument
from mmsim.data.loaders import day_files, iter_row_groups, list_days, read_files
from mmsim.data.replay import MarketEvent, replay
from mmsim.data.synth import generate_path
from mmsim.explog import append_failed, append_trial, periods_per_year
from mmsim.io.parquet import write_parquet
from mmsim.io.runlog import RunDir
from mmsim.models.calibration import CalibratedParams
from mmsim.settings import Settings
from mmsim.sim.latency import median_feed_latency_ns
from mmsim.sim.protocol import MarketSimulator
from mmsim.sim.registry import make_simulator
from mmsim.sim.types import BookView, Fill, InstrumentSpec, OrderUpdate, StepResult
from mmsim.strategy.analytic import AnalyticPolicy
from mmsim.strategy.base import QuotePolicy, StrategyState
from mmsim.strategy.quote_manager import sync_quotes

__all__ = [
    "FixedParamsProvider",
    "ParamsProvider",
    "RunOutcome",
    "Session",
    "SessionLog",
    "build_sessions",
    "execute_run",
    "fail_run",
    "finish_run",
    "resolved_config",
    "run_session",
    "session_frames",
]

NS_PER_S = 1_000_000_000
NS_PER_HOUR = 3600 * NS_PER_S
NS_PER_DAY = 24 * NS_PER_HOUR
QUOTE_NOTIONAL_USD = 1_000.0  # D-08


class ParamsProvider(Protocol):
    """Model parameters as known at a local time (no look-ahead)."""

    def update(self, step: StepResult) -> None:
        """See one step's strategy-visible events."""
        ...

    def params(self, now_ns: int) -> CalibratedParams | None:
        """Current parameters, or ``None`` during warm-up."""
        ...


@dataclass(frozen=True)
class FixedParamsProvider:
    """Known parameters (synthetic worlds)."""

    A: float
    k: float
    sigma: float

    def update(self, step: StepResult) -> None:
        """Nothing to learn."""

    def params(self, now_ns: int) -> CalibratedParams:
        """The fixed values."""
        return CalibratedParams(now_ns, self.A, self.k, self.sigma, 1.0, 0)


@dataclass
class Session:
    """One replay unit."""

    day: date
    start_ns: int
    end_ns: int
    instrument: InstrumentSpec
    events: Callable[[], Iterator[MarketEvent]]
    median_feed_ns: int
    seed: int
    timer_start_ns: int | None = None
    """First decision time; ``None`` starts at the first market event."""


@dataclass
class SessionLog:
    """Everything one session produced."""

    session: Session
    orders: list[dict[str, Any]] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    quotes: list[dict[str, Any]] = field(default_factory=list)
    mids: list[tuple[int, int, int]] = field(default_factory=list)
    positions_after: list[int] = field(default_factory=list)
    max_abs_inventory_lots: int = 0
    time_at_limit_ns: int = 0
    quoting_ns: int = 0
    decisions: int = 0
    q_max_lots: int = 0
    quote_lots: int = 0
    sim_stats: dict[str, int] = field(default_factory=dict)


def _order_row(
    ts_local: int, ts_exchange: int | None, order_id: int, event: str, side: int,
    price: int, qty: int, reason: str | None,
) -> dict[str, Any]:  # fmt: skip
    return {
        "ts_local": ts_local,
        "ts_exchange": ts_exchange,
        "order_id": order_id,
        "event": event,
        "side": side,
        "price_ticks": price,
        "qty_lots": qty,
        "reason": reason,
    }


_UPDATE_EVENT = {"accepted": "accept", "rejected": "reject", "cancelled": "cancelled"}


def run_session(
    sim: MarketSimulator,
    session: Session,
    policy_factory: Callable[[int], QuotePolicy],
    params: ParamsProvider,
    *,
    q_max_units: int,
    quote_lots: int | None,
    decision_interval_ns: int,
    requote_threshold_ticks: int,
    mids: list[tuple[int, int, int]],
    tau_finite: bool = False,
) -> SessionLog:
    """Drive ``sim`` through one session.

    ``quote_lots=None`` applies D-08 at the first two-sided book: the smallest lot multiple
    with notional of at least 1,000 USD. ``mids`` is the list the simulator's mid observer
    appends to; it becomes the session's exchange-time mid series.
    """
    log = SessionLog(session=session, mids=mids)
    inst = sim.instrument
    sim.reset(seed=session.seed)
    first = session.timer_start_ns if session.timer_start_ns is not None else sim.now()
    sim.set_timer(at_ns=first, every_ns=decision_interval_ns)
    sides = {"BUY": 1, "SELL": -1}
    book: BookView | None = None
    policy: QuotePolicy | None = None
    position = 0
    last_t = first
    quoting = False
    order_info: dict[int, tuple[int, int]] = {}
    while (step := sim.step()) is not None:
        if step.ts_local > session.end_ns:
            break
        span = step.ts_local - last_t
        if log.q_max_lots and abs(position) >= log.q_max_lots:
            log.time_at_limit_ns += span
        if quoting:
            log.quoting_ns += span
        last_t = step.ts_local
        params.update(step)
        for update in step.order_updates:
            _log_update(log, update, order_info)
        for fill in step.fills:
            position += int(fill.side) * fill.qty_lots
            log.fills.append(fill)
            log.positions_after.append(position)
            log.max_abs_inventory_lots = max(log.max_abs_inventory_lots, abs(position))
            side_v = int(fill.side)
            log.orders.append(
                _order_row(fill.ts_local, fill.ts_exchange, fill.order_id, "fill", side_v,
                           fill.price_ticks, fill.qty_lots, None)
            )  # fmt: skip
        if step.book is not None:
            book = step.book
        if not (step.timer or step.fills) or book is None:
            continue
        if not book.bid_ticks or not book.ask_ticks:
            continue
        current = params.params(step.ts_local)
        if current is None:
            sim.cancel_all(inst.symbol)
            quoting = False
            continue
        if policy is None:
            lots = quote_lots
            if lots is None:
                mid = book.mid_x2 * inst.tick_size / 2.0
                lots = max(1, math.ceil(QUOTE_NOTIONAL_USD / (mid * inst.lot_size) - 1e-9))
            log.quote_lots = lots
            log.q_max_lots = q_max_units * lots
            policy = policy_factory(lots)
        remaining = (session.end_ns - step.ts_local) / NS_PER_S if tau_finite else None
        state = StrategyState(
            ts_local=step.ts_local,
            book=book,
            inventory_lots=sim.position_lots(inst.symbol),
            q_max_lots=log.q_max_lots,
            params=current,
            signal_ticks=None,
            t_remaining_s=remaining,
        )
        decision = policy.decide(state)
        log.decisions += 1
        actions = sync_quotes(
            sim,
            decision,
            q_max_lots=log.q_max_lots,
            requote_threshold_ticks=requote_threshold_ticks,
        )
        quoting = actions.bid_lots > 0 or actions.ask_lots > 0
        for oid in actions.cancelled:
            side, price = order_info.get(oid, (0, 0))
            log.orders.append(
                _order_row(step.ts_local, None, oid, "cancel_req", side, price, 0, None)
            )
        working = {o.order_id: o for o in sim.open_orders(inst.symbol)}
        for oid in actions.submitted:
            o = working[oid]
            order_info[oid] = (sides[o.side.name], o.price_ticks)
            log.orders.append(
                _order_row(step.ts_local, None, oid, "submit", sides[o.side.name], o.price_ticks,
                           o.remaining_lots, None)
            )  # fmt: skip
        log.quotes.append(
            {
                "ts_local": step.ts_local,
                "bid_ticks": decision.bid_ticks,
                "ask_ticks": decision.ask_ticks,
                "reservation_x2": (
                    decision.bid_ticks + decision.ask_ticks
                    if decision.bid_ticks is not None and decision.ask_ticks is not None
                    else None
                ),
                "half_spread_ticks": (
                    (decision.ask_ticks - decision.bid_ticks) / 2.0
                    if decision.bid_ticks is not None and decision.ask_ticks is not None
                    else None
                ),
                "signal": None,
                "A": current.A,
                "k": current.k,
                "sigma": current.sigma,
                "inventory_lots": state.inventory_lots,
            }
        )
    stats = getattr(sim, "stats", None)
    log.sim_stats = asdict(stats) if stats is not None else {}
    return log


def _log_update(log: SessionLog, update: OrderUpdate, info: dict[int, tuple[int, int]]) -> None:
    event = _UPDATE_EVENT.get(update.status)
    if event is None:
        return  # fills are logged from the Fill itself
    side, price = info.get(update.order_id, (0, 0))
    log.orders.append(
        _order_row(update.ts_local, None, update.order_id, event, side, price, 0, update.reason)
    )


# ---- frames ---------------------------------------------------------------------------

FILL_SCHEMA = {
    "ts_exchange": pl.Int64,
    "ts_local": pl.Int64,
    "order_id": pl.Int64,
    "side": pl.Int8,
    "price_ticks": pl.Int64,
    "qty_lots": pl.Int64,
    "liquidity": pl.String,
    "fee_quote": pl.Float64,
    "queue_ahead_lots_at_entry": pl.Int64,
    "order_latency_ns": pl.Int64,
    "mid_x2_before": pl.Int64,
    "position_after_lots": pl.Int64,
}
ORDER_SCHEMA = {
    "ts_local": pl.Int64,
    "ts_exchange": pl.Int64,
    "order_id": pl.Int64,
    "event": pl.String,
    "side": pl.Int8,
    "price_ticks": pl.Int64,
    "qty_lots": pl.Int64,
    "reason": pl.String,
}
QUOTE_SCHEMA = {
    "ts_local": pl.Int64,
    "bid_ticks": pl.Int64,
    "ask_ticks": pl.Int64,
    "reservation_x2": pl.Int64,
    "half_spread_ticks": pl.Float64,
    "signal": pl.Float64,
    "A": pl.Float64,
    "k": pl.Float64,
    "sigma": pl.Float64,
    "inventory_lots": pl.Int64,
}
MID_SCHEMA = {"ts_exchange": pl.Int64, "mid_x2": pl.Int64, "spread_ticks": pl.Int64}


def session_frames(log: SessionLog) -> dict[str, pl.DataFrame]:
    """The session's fills, orders, quotes and mids as typed frames."""
    fills = pl.DataFrame(
        [
            {
                "ts_exchange": f.ts_exchange,
                "ts_local": f.ts_local,
                "order_id": f.order_id,
                "side": int(f.side),
                "price_ticks": f.price_ticks,
                "qty_lots": f.qty_lots,
                "liquidity": f.liquidity,
                "fee_quote": f.fee_quote,
                "queue_ahead_lots_at_entry": f.queue_ahead_lots_at_entry,
                "order_latency_ns": f.order_latency_ns,
                "mid_x2_before": f.mid_x2_before,
                "position_after_lots": pos,
            }
            for f, pos in zip(log.fills, log.positions_after, strict=True)
        ],
        schema=FILL_SCHEMA,
    )
    # The decomposition is by exchange time; strategy-seen order can differ under latency.
    fills = fills.sort("ts_exchange", maintain_order=True)
    return {
        "fills": fills,
        "orders": pl.DataFrame(log.orders, schema=ORDER_SCHEMA),
        "quotes": pl.DataFrame(log.quotes, schema=QUOTE_SCHEMA),
        "mids": pl.DataFrame(log.mids, schema=MID_SCHEMA, orient="row"),
    }


# ---- run lifecycle --------------------------------------------------------------------


def finish_run(
    run: RunDir,
    *,
    explog_path: Path,
    strategy_family: str,
    params: dict[str, object],
    dates: list[date],
    pnl_quote: list[float],
    capital: float,
    metrics: dict[str, Any],
    selected: bool = False,
) -> dict[str, str]:
    """Write ``returns.parquet`` and ``metrics.json``, then append the trial row."""
    if not dates:
        raise ValueError("a finished run needs at least one daily return")
    run.write_returns(dates, pnl_quote, capital)
    run.write_metrics(metrics)
    return append_trial(
        explog_path,
        run_id=run.run_id,
        strategy_family=strategy_family,
        params=params,
        start=dates[0],
        end=dates[-1],
        returns_path=run.returns_path,
        selected=selected,
    )


def fail_run(
    run: RunDir,
    *,
    explog_path: Path,
    strategy_family: str,
    params: dict[str, object],
    error: str,
    start: date | None = None,
    end: date | None = None,
) -> dict[str, str]:
    """Record a failed run: ``metrics.json`` with the error, and an ``n_obs = 0`` row."""
    run.write_metrics({"status": "failed", "error": error})
    return append_failed(
        explog_path,
        run_id=run.run_id,
        strategy_family=strategy_family,
        params=params,
        start=start,
        end=end,
    )


@dataclass
class RunOutcome:
    """A finished run."""

    run: RunDir
    row: dict[str, str]
    metrics: dict[str, Any]
    breakdowns: list[PnLBreakdown]
    logs: list[SessionLog]


def _appender(out: list[tuple[int, int, int]]) -> Callable[[int, int, int], None]:
    def add(t: int, mid_x2: int, spread: int) -> None:
        out.append((t, mid_x2, spread))

    return add


def _session_seed(seed: int, index: int) -> int:
    return int(np.random.SeedSequence([seed, index]).generate_state(1)[0])


def _fees_for(fees_path: Path, venue: str, tier: str, day: date) -> tuple[float, float]:
    rows = [r for r in load_fee_rows(fees_path) if r.venue == venue and r.tier == tier]
    rows = [r for r in rows if r.covers(day)]
    if len(rows) != 1:
        raise FeeCoverageError(f"{venue} tier {tier!r} on {day}: {len(rows)} fee rows")
    return rows[0].maker_bps, rows[0].taker_bps


def build_sessions(cfg: MmsConfig, settings: Settings, repo_root: Path) -> list[Session]:
    """Sessions of a config: synthetic paths in memory, or one per UTC day of partitions."""
    data = cfg.data
    if len(data.symbols) != 1:
        raise ValueError("a run covers one symbol; sweep over symbols instead")
    symbol = data.symbols[0]
    sessions: list[Session] = []
    fees_path = repo_root / cfg.fees.schedule
    if data.source == "synthetic" and data.synthetic_config is not None:
        synth = load_synth_config(repo_root / data.synthetic_config)
        n = data.paths or synth.paths
        tick, lot = synth.tick_size, synth.lot_size
        for i in range(n):
            world = generate_path(synth, i)
            day = datetime.fromtimestamp(world.start_ns / NS_PER_S, tz=UTC).date()
            maker, taker = _fees_for(fees_path, synth.venue, cfg.fees.tier, day)
            inst = InstrumentSpec(
                synth.venue, synth.symbol, tick, lot, "perp", maker, taker, cfg.fees.tier
            )

            def synth_events(w: Any = world) -> Iterator[MarketEvent]:
                return replay(
                    w.book_l2.to_arrow().to_batches(), w.trades.to_arrow().to_batches(), tick, lot
                )

            trades = world.trades
            sessions.append(
                Session(
                    day=day,
                    start_ns=world.start_ns,
                    end_ns=world.end_ns,
                    instrument=inst,
                    events=synth_events,
                    median_feed_ns=median_feed_latency_ns(trades["ts_exchange"], trades["ts_recv"]),
                    seed=_session_seed(cfg.run.seed, i),
                    timer_start_ns=world.start_ns,
                )
            )
        return sessions
    root = settings.mms_cla_data_root if data.source == "cla" else settings.processed_root
    for i, day in enumerate(list_days(root, "book_l2", data.venue, symbol, data.start, data.end)):
        inst = resolve_instrument(
            repo_root / data.instruments, fees_path, data.venue, symbol, day, cfg.fees.tier
        )
        book_files = day_files(root, "book_l2", data.venue, symbol, day)
        trade_files = day_files(root, "trades", data.venue, symbol, day)
        ts = read_files(trade_files, ["ts_exchange", "ts_recv"])

        def day_events(
            b: list[Path] = book_files, t: list[Path] = trade_files, s: InstrumentSpec = inst
        ) -> Iterator[MarketEvent]:
            cols_b = ["ts_exchange", "ts_recv", "seq", "side", "price", "qty", "is_snapshot",
                      "batch_id"]  # fmt: skip
            cols_t = ["ts_exchange", "ts_recv", "trade_id", "price", "qty", "side"]
            return replay(
                iter_row_groups(b, cols_b), iter_row_groups(t, cols_t), s.tick_size, s.lot_size
            )

        start = (day - date(1970, 1, 1)).days * NS_PER_DAY
        sessions.append(
            Session(
                day=day,
                start_ns=start,
                end_ns=start + NS_PER_DAY,
                instrument=inst,
                events=day_events,
                median_feed_ns=(
                    median_feed_latency_ns(ts["ts_exchange"], ts["ts_recv"]) if ts.height else 0
                ),
                seed=_session_seed(cfg.run.seed, i),
            )
        )
    if not sessions:
        raise ValueError(f"no data for {data.venue}/{symbol} in {data.start}..{data.end}")
    return sessions


def _hourly(
    fr: dict[str, pl.DataFrame], start_ns: int, end_ns: int, unit: float
) -> list[tuple[int, float]]:
    fills, mids = fr["fills"], fr["mids"]
    if mids.is_empty():
        return []
    ends = list(range((start_ns // NS_PER_HOUR + 1) * NS_PER_HOUR, end_ns, NS_PER_HOUR))
    ends.append(end_ns)
    t = fills["ts_exchange"].to_numpy().astype(np.int64)
    q = fills["side"].to_numpy().astype(np.int64) * fills["qty_lots"].to_numpy().astype(np.int64)
    p2 = 2 * fills["price_ticks"].to_numpy().astype(np.int64)
    fee = fills["fee_quote"].to_numpy()
    m = mid_asof(
        mids["ts_exchange"].to_numpy().astype(np.int64),
        mids["mid_x2"].to_numpy().astype(np.int64),
        np.asarray(ends, dtype=np.int64),
    )
    out: list[tuple[int, float]] = []
    prev = 0.0
    for h, mid in zip(ends, m.tolist(), strict=True):
        sel = t <= h
        cash = -int(np.sum(q[sel] * p2[sel]))
        pos = int(q[sel].sum())
        value = (cash + pos * mid) * unit - math.fsum(fee[sel].tolist())
        out.append((h, value - prev))
        prev = value
    return out


def _sharpe(values: Sequence[float], periods: float) -> float | None:
    if len(values) < 2:
        return None
    sd = statistics.stdev(values)
    return statistics.fmean(values) / sd * math.sqrt(periods) if sd > 0 else None


def execute_run(
    cfg: MmsConfig,
    *,
    settings: Settings,
    repo_root: Path = Path(),
    resolved: dict[str, Any] | None = None,
    family: str | None = None,
    results_dir: Path | None = None,
    explog_path: Path | None = None,
    policy_factory: Callable[[InstrumentSpec, int], QuotePolicy] | None = None,
    extra_params: dict[str, object] | None = None,
) -> RunOutcome:
    """Run every session of ``cfg``, write the outputs and append the trial row.

    ``policy_factory(instrument, quote_lots)`` replaces the config's analytic policy (used
    by the validation suite's symmetric benchmark); ``extra_params`` join ``params_json``.
    """
    family = family or cfg.strategy.family
    results = results_dir or Path(cfg.run.results_dir)
    # The contract puts runs/<run_id>/ next to the log, so the log lives in the results dir.
    log_path = explog_path or (results / "trial_log.csv")
    run = RunDir.create(results)
    params: dict[str, object] = {
        "strategy": cfg.strategy.model_dump(mode="json"),
        "sim": cfg.sim.model_dump(mode="json"),
        "data": cfg.data.model_dump(mode="json"),
        "fees_tier": cfg.fees.tier,
        "seed": cfg.run.seed,
        "backend": cfg.run.sim,
        **(extra_params or {}),
    }
    if resolved is not None:
        run.write_config(resolved, settings.redacted())
    started = time.perf_counter()
    try:
        outcome = _execute(cfg, settings, repo_root, run, family, params, log_path, policy_factory)
    except Exception as exc:
        fail_run(run, explog_path=log_path, strategy_family=family, params=params,
                 error=f"{type(exc).__name__}: {exc}")  # fmt: skip
        raise
    outcome.metrics["runtime_s"] = time.perf_counter() - started
    run.write_metrics(outcome.metrics)
    return outcome


def _execute(
    cfg: MmsConfig,
    settings: Settings,
    repo_root: Path,
    run: RunDir,
    family: str,
    params: dict[str, object],
    log_path: Path,
    policy_factory: Callable[[InstrumentSpec, int], QuotePolicy] | None,
) -> RunOutcome:
    st = cfg.strategy
    if st.family == "rl":
        raise NotImplementedError("RL policies arrive in M6")
    if st.params is None:
        raise NotImplementedError("calibrated parameters arrive in M3; set strategy.params")
    provider = FixedParamsProvider(st.params.A, st.params.k, st.params.sigma)
    sessions = build_sessions(cfg, settings, repo_root)
    horizons = list(cfg.analysis.markout_horizons_ms)
    headline = cfg.analysis.headline_horizon_ms
    ordered = [headline, *[h for h in horizons if h != headline]]
    logs: list[SessionLog] = []
    breakdowns: list[PnLBreakdown] = []
    names = ("fills", "orders", "quotes", "mids", "attr", "mo")
    frames: dict[str, list[pl.DataFrame]] = {k: [] for k in names}
    daily: dict[date, float] = {}
    hourly: list[tuple[int, float]] = []
    events = 0
    wall = time.perf_counter()
    for session in sessions:
        mids: list[tuple[int, int, int]] = []
        sim = make_simulator(
            cfg.sim, session.events, session.instrument, cfg.run.sim,
            median_feed_ns=session.median_feed_ns, book_depth=cfg.data.book_depth,
            start_ns=session.timer_start_ns, on_mid=_appender(mids),
            seed=session.seed,
        )  # fmt: skip
        inst = session.instrument

        def factory(lots: int, inst: InstrumentSpec = inst) -> QuotePolicy:
            if policy_factory is not None:
                return policy_factory(inst, lots)
            return AnalyticPolicy(
                family=st.family, gamma=st.gamma, tick_size=inst.tick_size, quote_lots=lots,
                q_max_units=st.q_max_units, horizon_s=st.horizon_s, tau_mode=st.tau_mode,
                maker_bps=inst.maker_bps,
            )  # fmt: skip

        log = run_session(
            sim, session, factory, provider,
            q_max_units=st.q_max_units,
            quote_lots=None if st.quote_lots == "auto" else st.quote_lots,
            decision_interval_ns=st.decision_interval_ms * NS_PER_MS,
            requote_threshold_ticks=st.requote_threshold_ticks,
            mids=mids, tau_finite=st.tau_mode == "finite",
        )  # fmt: skip
        logs.append(log)
        events += log.sim_stats.get("market_events", 0)
        fr = session_frames(log)
        if fr["mids"].is_empty():
            continue
        bd = decompose_pnl(fr["fills"], fr["mids"], horizon_ms=headline, instrument=inst,
                           end_ns=session.end_ns)  # fmt: skip
        breakdowns.append(bd)
        daily[session.day] = daily.get(session.day, 0.0) + bd.total
        unit = inst.tick_size / 2.0 * inst.lot_size
        hourly.extend(_hourly(fr, session.start_ns, session.end_ns, unit))
        for key in ("fills", "orders", "quotes", "mids"):
            frames[key].append(fr[key])
        frames["attr"].append(
            attribution_table(fr["fills"], fr["mids"], horizons_ms=ordered, end_ns=session.end_ns)
        )
        frames["mo"].append(markout_table(fr["fills"], fr["mids"], horizons, end_ns=session.end_ns))
    if not breakdowns:
        raise ValueError("no session produced a mid series")
    wall = time.perf_counter() - wall
    outputs = {"attr": "pnl_attribution", "mo": "markouts"}
    for key in names:
        name = outputs.get(key, key)
        write_parquet(pl.concat(frames[key]), run.path / f"{name}.parquet")
    hourly_df = pl.DataFrame(
        {"ts": [h for h, _ in hourly], "pnl_quote": [v for _, v in hourly]},
        schema={"ts": pl.Int64, "pnl_quote": pl.Float64},
    ).with_columns((pl.col("pnl_quote") / cfg.analysis.capital_usd).alias("ret"))
    write_parquet(hourly_df, run.path / "returns_hourly.parquet")
    dates = sorted(daily)
    pnl = [daily[d] for d in dates]
    q = periods_per_year(family)
    rets = [x / cfg.analysis.capital_usd for x in pnl]
    hourly_rets = hourly_df["ret"].to_list()
    all_fills = pl.concat(frames["fills"])
    mo = markout_summary(pl.concat(frames["mo"]))
    metrics: dict[str, Any] = {
        "status": "ok",
        "sessions": len(sessions),
        "symbol": cfg.data.symbols[0],
        "horizon_ms": headline,
        "total_pnl": math.fsum(b.total for b in breakdowns),
        "spread_capture": math.fsum(b.spread for b in breakdowns),
        "adverse_selection": math.fsum(b.adverse_selection for b in breakdowns),
        "inventory_residual": math.fsum(b.inventory_residual for b in breakdowns),
        "fees": math.fsum(b.fees for b in breakdowns),
        "funding": math.fsum(b.funding for b in breakdowns),
        "reconciliation_residual": max(abs(b.residual) for b in breakdowns),
        "fills": all_fills.height,
        "volume_lots": int(all_fills["qty_lots"].sum()) if all_fills.height else 0,
        "max_abs_inventory_lots": max(lg.max_abs_inventory_lots for lg in logs),
        "q_max_lots": max(lg.q_max_lots for lg in logs),
        "time_at_limit_s": sum(lg.time_at_limit_ns for lg in logs) / NS_PER_S,
        "quote_uptime_s": sum(lg.quoting_ns for lg in logs) / NS_PER_S,
        "decisions": sum(lg.decisions for lg in logs),
        "gaps_skipped": 0,
        "side0_trades": sum(lg.sim_stats.get("side0_trades", 0) for lg in logs),
        "crossed_book_events": sum(lg.sim_stats.get("crossed_book_events", 0) for lg in logs),
        "orders_submitted": sum(lg.sim_stats.get("orders_submitted", 0) for lg in logs),
        "orders_rejected": sum(lg.sim_stats.get("orders_rejected", 0) for lg in logs),
        "sharpe_daily_annual": _sharpe(rets, q),
        "sharpe_hourly_annual": _sharpe(hourly_rets, 8760 if q == 365 else 252 * 6.5),
        "markouts_mid_bps": {
            str(h): v
            for h, v in zip(mo["horizon_ms"].to_list(), mo["mo_mid_bps"].to_list(), strict=True)
        },
        "market_events": events,
        "events_per_s": events / wall if wall > 0 else None,
        "package_versions": {
            "python": platform.python_version(),
            "polars": pl.__version__,
            "numpy": np.__version__,
        },
    }
    row = finish_run(run, explog_path=log_path, strategy_family=family, params=params,
                     dates=dates, pnl_quote=pnl, capital=cfg.analysis.capital_usd,
                     metrics=metrics)  # fmt: skip
    return RunOutcome(run=run, row=row, metrics=metrics, breakdowns=breakdowns, logs=logs)


def resolved_config(path: Path, overrides: list[str], settings: Settings) -> dict[str, Any]:
    """The merged config mapping, for ``config.resolved.yaml``."""
    out: dict[str, Any] = json.loads(
        json.dumps(resolve_dict(path, overrides, settings), default=str)
    )
    return out
