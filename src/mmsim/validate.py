"""``mms validate synthetic``: the simulator against known results (FR-19).

Checks, each a row of ``reports/tables/synthetic_validation.csv``:

1. AC-W-01: fill intensity of resting quotes in the Poisson world against ``A exp(-k delta)``
   on a 10-point grid (99% Poisson CIs), and the fitted ``k`` within 3%.
2. AC-W-02: Avellaneda-Stoikov (2008) Tables 1-3 with the NumPy reference and with the
   event-driven fallback (front queue, zero latency), within 3 standard errors.
3. AC-W-03: GLFT exact strategy, Monte Carlo CARA utility against Theorem 1's value
   function, within 3 SE + 1%.
4. AC-W-04: fallback against the reference (mean profit within 3 SE of the difference), and
   lower fill rates with a conservative queue or added latency.
5. AC-W-14: GLFT asymptotic against exact quotes at a converged horizon, within 5% for
   ``|q| <= Q/2``; the gap near the bounds is reported.

Calibration recovery (FR-19 item 4, AC-W-10) joins in M3. Every Monte Carlo and
event-driven run appends one trial-log row (family ``synthetic_validation``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import polars as pl
import yaml
from scipy import stats

from mmsim.config import MmsConfig, SimConfig, load_synth_config
from mmsim.data.replay import MarketEvent, replay
from mmsim.data.synth import generate_path
from mmsim.io.runlog import LF, RunDir
from mmsim.models.glft import (
    glft_asymptotic,
    glft_certainty_equivalent,
    glft_exact,
)
from mmsim.models.reference_mc import as2008_avg_spread, as2008_mc, glft_mc
from mmsim.runner import RunOutcome, execute_run, finish_run
from mmsim.settings import Settings
from mmsim.sim.registry import make_simulator
from mmsim.sim.types import InstrumentSpec, Side
from mmsim.strategy.analytic import SymmetricPolicy
from mmsim.strategy.base import QuotePolicy

__all__ = [
    "Check",
    "ValidationPaths",
    "load_exceptions",
    "run_validation",
    "unexplained_failures",
    "write_report",
]

FAMILY = "synthetic_validation"
NS = 1_000_000_000


@dataclass(frozen=True)
class Check:
    """One validated quantity."""

    check: str
    case: str
    metric: str
    value: float
    target: float
    se: float | None
    z: float | None
    tolerance: str
    passed: bool
    note: str = ""


ExceptionKey = tuple[str, str, str]


def load_exceptions(repo_root: Path) -> dict[ExceptionKey, str]:
    """Owner-accepted misses: ``(check, case, metric) -> decision id``."""
    path = repo_root / "configs" / "synthetic" / "documented_exceptions.yaml"
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {
        (str(e["check"]), str(e["case"]), str(e["metric"])): str(e["decision"])
        for e in raw.get("exceptions", [])
    }


def mark_exceptions(checks: list[Check], exceptions: dict[ExceptionKey, str]) -> list[Check]:
    """Failed rows that are documented exceptions get a note naming the decision."""
    out: list[Check] = []
    for c in checks:
        decision = exceptions.get((c.check, c.case, c.metric))
        if not c.passed and decision is not None:
            c = Check(**{**asdict(c), "note": f"documented exception ({decision})"})
        out.append(c)
    return out


def unexplained_failures(checks: list[Check], exceptions: dict[ExceptionKey, str]) -> list[Check]:
    """Failed rows that are not documented exceptions; any of these fails the run."""
    return [c for c in checks if not c.passed and (c.check, c.case, c.metric) not in exceptions]


@dataclass(frozen=True)
class ValidationPaths:
    """Where the suite reads configs and writes outputs."""

    repo_root: Path
    results_dir: Path
    explog_path: Path
    reports_dir: Path


def _log_series(
    paths: ValidationPaths, params: dict[str, object], values: npt.ArrayLike, start: date
) -> None:
    """A Monte Carlo run as a trial: path ``i`` is day ``start + i``."""
    vals = [float(x) for x in np.asarray(values, dtype=np.float64)]
    run = RunDir.create(paths.results_dir)
    dates = [start + timedelta(days=i) for i in range(len(vals))]
    finish_run(
        run,
        explog_path=paths.explog_path,
        strategy_family=FAMILY,
        params=params,
        dates=dates,
        pnl_quote=vals,
        capital=10_000.0,
        metrics={"status": "ok", "kind": "reference_mc", **params},
    )


def _synth_run_cfg(
    synth_path: str,
    *,
    paths: int,
    family: str,
    gamma: float,
    params: dict[str, float],
    quote_lots: int,
    q_max_units: int,
    decision_ms: int,
    horizon_s: float,
    tau_mode: str,
    sim: dict[str, Any] | None = None,
    seed: int = 0,
) -> MmsConfig:
    zero = {"kind": "constant", "ns": 0}
    sim_cfg = sim or {
        "queue_model": {"kind": "front"},
        "latency": {"feed": zero, "order": zero, "ack": zero},
    }
    return MmsConfig.model_validate(
        {
            "run": {"name": "validate", "seed": seed, "results_dir": "unused"},
            "data": {
                "source": "synthetic",
                "venue": "synthetic",
                "symbols": ["SYN"],
                "synthetic_config": synth_path,
                "paths": paths,
            },
            "fees": {"tier": "zero"},
            "sim": sim_cfg,
            "strategy": {
                "family": family,
                "gamma": gamma,
                "horizon_s": horizon_s,
                "tau_mode": tau_mode,
                "quote_lots": quote_lots,
                "q_max_units": q_max_units,
                "decision_interval_ms": decision_ms,
                "requote_threshold_ticks": 0,
                "params": params,
            },
        }
    )


def _profit_and_q(outcome: RunOutcome) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    profit = np.array([b.total for b in outcome.breakdowns])
    final_q = np.array(
        [sum(int(f.side) * f.qty_lots for f in log.fills) for log in outcome.logs], dtype=float
    )
    return profit, final_q


# ---- AC-W-01 ------------------------------------------------------------------------------


def check_intensity(paths: ValidationPaths, fast: bool) -> tuple[list[Check], dict[str, Any]]:
    """Resting quotes on a 10-point depth grid, front queue, zero latency."""
    synth = load_synth_config(paths.repo_root / "configs" / "synthetic" / "intensity.yaml")
    n_paths = 1 if fast else synth.paths
    grid_ticks = [j * 1000 for j in range(1, 11)]  # delta = 0.1 ... 1.0
    deltas = np.array(grid_ticks, dtype=float) * synth.tick_size
    counts = np.zeros(len(grid_ticks))
    orders = 0
    pnl: list[float] = []
    zero = {"kind": "constant", "ns": 0}
    sim_cfg = SimConfig.model_validate(
        {"queue_model": {"kind": "front"}, "latency": {"feed": zero, "order": zero, "ack": zero}}
    )
    inst = InstrumentSpec(synth.venue, synth.symbol, synth.tick_size, synth.lot_size, "perp",
                          0.0, 0.0, "zero")  # fmt: skip
    for i in range(n_paths):
        world = generate_path(synth, i)
        orders += world.trades.height
        mids: list[int] = []

        def events(w: Any = world) -> Iterator[MarketEvent]:
            return replay(w.book_l2.to_arrow().to_batches(), w.trades.to_arrow().to_batches(),
                          synth.tick_size, synth.lot_size)  # fmt: skip

        def on_mid(t: int, m: int, s: int, out: list[int] = mids) -> None:
            out.append(m)

        sim = make_simulator(sim_cfg, events, inst, start_ns=world.start_ns, on_mid=on_mid)
        sim.reset(seed=i)
        working: dict[int, tuple[int, int]] = {}
        cash_x2 = 0
        pos = 0
        while (step := sim.step()) is not None:
            for fill in step.fills:
                key = working.pop(fill.order_id, None)
                if key is not None:
                    counts[key[1]] += 1
                pos += int(fill.side) * fill.qty_lots
                cash_x2 -= int(fill.side) * fill.qty_lots * 2 * fill.price_ticks
            if step.book is not None and step.book.bid_ticks and step.book.ask_ticks:
                for oid in list(working):
                    sim.cancel(oid)
                    del working[oid]
            book = sim.book(inst.symbol)
            if not (book.bid_ticks and book.ask_ticks):
                continue
            mid = book.mid_x2 // 2
            have = set(working.values())
            for side in (1, -1):
                for j, d in enumerate(grid_ticks):
                    if (side, j) in have:
                        continue
                    oid = sim.submit_limit_ticks(inst.symbol, Side(side), mid - side * d, 1)
                    working[oid] = (side, j)
        pnl.append((cash_x2 + pos * mids[-1]) * synth.tick_size / 2 * synth.lot_size)
    _log_series(paths, {"check": "intensity", "paths": n_paths, "seed": synth.seed}, pnl,
                date(2026, 2, 1))  # fmt: skip
    exposure = 2.0 * n_paths * synth.horizon
    rate = counts / exposure
    expect = synth.market_orders.A * np.exp(-synth.market_orders.k * deltas)
    lo = stats.chi2.ppf(0.005, 2 * counts) / 2 / exposure
    hi = stats.chi2.ppf(0.995, 2 * counts + 2) / 2 / exposure
    checks = [
        Check("AC-W-01", f"delta={d:.1f}", "fill_rate_per_s", float(r), float(e), None, None,
              f"99% Poisson CI [{a:.3f}, {b:.3f}]", bool(a <= e <= b))
        for d, r, e, a, b in zip(deltas, rate, expect, lo, hi, strict=True)
    ]  # fmt: skip
    w = counts
    x, y = deltas, np.log(rate)
    xm = np.sum(w * x) / np.sum(w)
    ym = np.sum(w * y) / np.sum(w)
    k_hat = -float(np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2))
    a_hat = float(np.exp(ym + k_hat * xm))
    k_true = synth.market_orders.k
    note = f"{orders:,} market orders" + ("" if orders >= 100_000 else " (< 1e5: --fast run)")
    checks.append(
        Check("AC-W-01", "fit", "k", k_hat, k_true, None, None, "within 3%",
              abs(k_hat / k_true - 1) <= 0.03, note)
    )  # fmt: skip
    checks.append(
        Check("AC-W-01", "fit", "A", a_hat, synth.market_orders.A, None, None, "reported", True)
    )
    figure = {"delta": deltas.tolist(), "rate": rate.tolist(), "lo": lo.tolist(),
              "hi": hi.tolist(), "expect": expect.tolist()}  # fmt: skip
    return checks, figure


# ---- AC-W-02 and AC-W-04 ---------------------------------------------------------------


def _stat_checks(
    check: str, case: str, profit: npt.NDArray[np.float64], final_q: npt.NDArray[np.float64],
    published: dict[str, float],
) -> list[Check]:  # fmt: skip
    n = profit.size
    out: list[Check] = []
    for metric, sample, key in (
        ("mean_profit", profit, "profit"),
        ("std_profit", profit, "std_profit"),
        ("mean_final_q", final_q, "final_q"),
        ("std_final_q", final_q, "std_final_q"),
    ):
        sd = float(np.std(sample, ddof=1))
        if metric.startswith("mean"):
            value, se = float(np.mean(sample)), sd / math.sqrt(n)
        else:
            value, se = sd, sd / math.sqrt(2 * n)
        target = published[key]
        z = (value - target) / se if se > 0 else 0.0
        out.append(Check(check, case, metric, value, target, se, z, "|z| <= 3", abs(z) <= 3))
    return out


POPULATION_PATHS = 50_000


def _population_rows(
    paths: ValidationPaths,
    case: str, gamma: float, strategy: Literal["inventory", "symmetric"], seed: int,
    sym_spread: float, pub: dict[str, float], n_pub: int,
) -> list[Check]:  # fmt: skip
    """Reported only: a 50,000-path reference estimate, and how far the published value is
    from it in units of the published sample's own standard error (n = 1,000)."""
    big = as2008_mc(gamma, strategy=strategy, n_paths=POPULATION_PATHS, seed=seed + 100_000,
                    symmetric_spread=sym_spread)  # fmt: skip
    _log_series(paths, {"check": "as2008_population", "backend": "reference", "gamma": gamma,
                        "strategy": strategy, "paths": POPULATION_PATHS, "seed": seed + 100_000},
                big.profit, date(2030, 1, 1))  # fmt: skip
    rows: list[Check] = []
    for metric, sample, key in (
        ("mean_profit", big.profit, "profit"),
        ("std_profit", big.profit, "std_profit"),
        ("mean_final_q", big.final_q, "final_q"),
        ("std_final_q", big.final_q, "std_final_q"),
    ):
        sd = float(np.std(sample, ddof=1))
        value = float(np.mean(sample)) if metric.startswith("mean") else sd
        se_pub = sd / math.sqrt(n_pub if metric.startswith("mean") else 2 * n_pub)
        z_pub = (pub[key] - value) / se_pub
        rows.append(Check("AC-W-02 diagnostic", f"{case} reference 50k", metric, value, pub[key],
                          se_pub, z_pub, "reported, post hoc: z of the published value", True,
                          "post hoc diagnostic, not a pass/fail row"))  # fmt: skip
    return rows


def check_as2008(
    paths: ValidationPaths, fast: bool, settings: Settings
) -> tuple[list[Check], dict[str, Any]]:
    """Tables 1-3 of A-S 2008 with the reference MC and the fallback, plus AC-W-04 part 1."""
    table = json.loads(
        (paths.repo_root / "tests" / "fixtures" / "as2008_table.json").read_text(encoding="utf-8")
    )
    synth_rel = "configs/synthetic/as2008.yaml"
    synth = load_synth_config(paths.repo_root / synth_rel)
    setup = table["setup"]
    n_ref = int(setup["paths"])
    n_sim = 100 if fast else synth.paths
    checks: list[Check] = []
    figure: dict[str, Any] = {}
    for t_index, entry in enumerate(table["tables"]):
        gamma = float(entry["gamma"])
        sym_spread = as2008_avg_spread(gamma, setup["sigma"], setup["k"], setup["T"], setup["dt"])
        cases: list[tuple[int, Literal["inventory", "symmetric"]]] = [
            (0, "inventory"),
            (1, "symmetric"),
        ]
        for s_index, strategy in cases:
            case = f"gamma={gamma:g} {strategy}"
            seed = synth.seed + 10 * t_index + s_index
            ref = as2008_mc(gamma, strategy=strategy, n_paths=n_ref, seed=seed,
                            symmetric_spread=sym_spread)  # fmt: skip
            _log_series(paths, {"check": "as2008", "backend": "reference", "gamma": gamma,
                                "strategy": strategy, "paths": n_ref, "seed": seed},
                        ref.profit, date(2030, 1, 1))  # fmt: skip
            pub = entry[strategy]
            checks += _stat_checks("AC-W-02", f"{case} reference", ref.profit, ref.final_q, pub)
            checks.append(Check("AC-W-02", f"{case} reference", "avg_spread", ref.avg_spread,
                                pub["spread"], None, None, "reported", True))  # fmt: skip
            if not fast:
                checks += _population_rows(
                    paths, case, gamma, strategy, seed, sym_spread, pub, n_ref
                )
            cfg = _synth_run_cfg(
                synth_rel, paths=n_sim, family="as", gamma=gamma,
                params={"A": setup["A"], "k": setup["k"], "sigma": setup["sigma"]},
                quote_lots=1, q_max_units=10**6, decision_ms=round(setup["dt"] * 1000),
                horizon_s=setup["T"], tau_mode="finite", seed=seed,
            )  # fmt: skip
            factory: Callable[[InstrumentSpec, int], QuotePolicy] | None = None
            if strategy == "symmetric":

                def factory(
                    inst: InstrumentSpec, lots: int, h: float = sym_spread / 2
                ) -> QuotePolicy:
                    return SymmetricPolicy(h, inst.tick_size, lots)

            outcome = execute_run(
                cfg, settings=settings, repo_root=paths.repo_root, family=FAMILY,
                results_dir=paths.results_dir, explog_path=paths.explog_path,
                policy_factory=factory,
                extra_params={"check": "as2008", "backend": "fallback", "strategy": strategy},
            )  # fmt: skip
            profit, final_q = _profit_and_q(outcome)
            checks += _stat_checks("AC-W-02", f"{case} fallback", profit, final_q, pub)
            figure[case] = {
                "reference": ref.profit.tolist(),
                "fallback": profit.tolist(),
                "published": pub["profit"],
            }
            if gamma == 0.1 and strategy == "inventory":
                diff = float(profit.mean() - ref.profit.mean())
                se = math.sqrt(profit.var(ddof=1) / profit.size
                               + ref.profit.var(ddof=1) / ref.profit.size)  # fmt: skip
                checks.append(Check("AC-W-04", case, "fallback_minus_reference_profit", diff, 0.0,
                                    se, diff / se, "|z| <= 3", abs(diff / se) <= 3))  # fmt: skip
    return checks, figure


def check_queue_latency(paths: ValidationPaths, settings: Settings) -> list[Check]:
    """AC-W-04 part 2: conservative queue and +50 ms latency lower the fill rate."""
    synth_rel = "configs/synthetic/queue_latency.yaml"
    synth = load_synth_config(paths.repo_root / synth_rel)
    mo = synth.market_orders
    zero = {"kind": "constant", "ns": 0}

    def run(queue: str, added_rtt_ns: int) -> int:
        cfg = _synth_run_cfg(
            synth_rel, paths=synth.paths, family="glft", gamma=0.01,
            params={"A": mo.A, "k": mo.k, "sigma": synth.mid.sigma},
            quote_lots=mo.qty_lots, q_max_units=10, decision_ms=100,
            horizon_s=synth.horizon, tau_mode="rolling", seed=synth.seed,
            sim={"queue_model": {"kind": queue},
                 "latency": {"feed": zero, "order": zero, "ack": zero,
                             "added_rtt_ns": added_rtt_ns}},
        )  # fmt: skip
        outcome = execute_run(
            cfg, settings=settings, repo_root=paths.repo_root, family=FAMILY,
            results_dir=paths.results_dir, explog_path=paths.explog_path,
            extra_params={"check": "queue_latency"},
        )  # fmt: skip
        fills: int = outcome.metrics["fills"]
        return fills

    base = run("front", 0)
    cons = run("conservative", 0)
    lat = run("front", 50_000_000)
    return [
        Check("AC-W-04", "conservative vs front", "fills", cons, base, None, None,
              "strictly lower", cons < base),
        Check("AC-W-04", "+50 ms round trip vs 0", "fills", lat, base, None, None,
              "strictly lower", lat < base),
    ]  # fmt: skip


# ---- AC-W-03 and AC-W-14 ---------------------------------------------------------------


def check_glft_ce(paths: ValidationPaths, fast: bool) -> tuple[list[Check], list[dict[str, float]]]:
    """Monte Carlo expected CARA utility against the closed form."""
    raw = yaml.safe_load(
        (paths.repo_root / "configs" / "synthetic" / "glft_ce.yaml").read_text(encoding="utf-8")
    )
    n = int(raw["paths"])
    checks: list[Check] = []
    figure: list[dict[str, float]] = []
    for i, s in enumerate(raw["sets"]):
        g, sig, A, k, Q = s["gamma"], s["sigma"], s["A"], s["k"], int(s["q_max"])
        sol = glft_exact(g, sig, A, k, Q, raw["horizon"])
        ce = glft_certainty_equivalent(sol, 0, 0.0, raw["s0"])
        u_cf = -math.exp(-g * ce)
        seed = int(raw["seed"]) + i
        w = glft_mc(g, sig, A, k, Q, raw["horizon"], raw["dt"], n_paths=n, seed=seed, s0=raw["s0"])
        _log_series(paths, {"check": "glft_ce", "set": s, "paths": n, "seed": seed}, w,
                    date(2040, 1, 1))  # fmt: skip
        u = -np.exp(-g * w)
        mean = float(u.mean())
        se = float(u.std(ddof=1) / math.sqrt(n))
        tol = 3 * se + 0.01 * abs(u_cf)
        case = f"gamma={g:g} sigma={sig:g} A={A:g} k={k:g} Q={Q}"
        ok = abs(mean - u_cf) <= tol
        z = (mean - u_cf) / se
        checks.append(Check("AC-W-03", case, "expected_utility", mean, u_cf, se, z,
                            "<= 3 SE + 1%", ok))  # fmt: skip
        ce_mc = -math.log(-mean) / g
        checks.append(Check("AC-W-03", case, "certainty_equivalent", ce_mc, ce, None, None,
                            "reported", True))  # fmt: skip
        figure.append({"case": i, "ce": ce, "ce_mc": ce_mc})
    return checks, figure


ASYMPTOTIC_SETS = [
    {"gamma": 0.01, "sigma": 0.3, "A": 0.9, "k": 0.3, "q_max": 30},
    {"gamma": 0.1, "sigma": 2.0, "A": 140.0, "k": 1.5, "q_max": 10},
    {"gamma": 0.05, "sigma": 0.3, "A": 0.9, "k": 0.3, "q_max": 20},
    {"gamma": 0.5, "sigma": 2.0, "A": 140.0, "k": 1.5, "q_max": 5},
]
"""The first is the GLFT paper's Figure 1 setting (T = 600 s there)."""


def asymptotic_gap(s: dict[str, float]) -> tuple[float, float, float]:
    """``(converged horizon, max rel. gap for |q| <= Q/2, max rel. gap beyond)``."""
    g, sig, A, k, Q = s["gamma"], s["sigma"], s["A"], s["k"], int(s["q_max"])

    def quotes(T: float) -> npt.NDArray[np.float64]:
        sol = glft_exact(g, sig, A, k, Q, T, n_t=2)
        vals = [sol.deltas(0.0, q) for q in range(-Q + 1, Q)]
        return np.array([[b, a] for b, a in vals], dtype=float)

    T = 1.0
    prev = quotes(T)
    while True:
        cur = quotes(2 * T)
        T *= 2
        if np.max(np.abs(cur / prev - 1)) < 1e-3 or T > 1e7:
            break
        prev = cur
    inner, outer = 0.0, 0.0
    for idx, q in enumerate(range(-Q + 1, Q)):
        b, a = glft_asymptotic(q, g, sig, A, k)
        gap = max(abs(cur[idx, 0] / b - 1), abs(cur[idx, 1] / a - 1))
        if abs(q) <= Q / 2:
            inner = max(inner, gap)
        else:
            outer = max(outer, gap)
    return T, inner, outer


def check_asymptotic() -> list[Check]:
    """AC-W-14: at least three sets within 5%."""
    checks: list[Check] = []
    passes = 0
    for s in ASYMPTOTIC_SETS:
        T, inner, outer = asymptotic_gap(s)
        ok = inner <= 0.05
        passes += ok
        case = "gamma={gamma:g} sigma={sigma:g} A={A:g} k={k:g} Q={q_max}".format(**s)
        checks.append(Check("AC-W-14", case, "max_rel_gap_inner", inner, 0.05, None, None,
                            "<= 5% for |q| <= Q/2", ok, f"converged at T={T:g}"))  # fmt: skip
        checks.append(Check("AC-W-14", case, "max_rel_gap_near_bounds", outer, 0.0, None, None,
                            "reported", True))  # fmt: skip
    checks.append(Check("AC-W-14", "all sets", "sets_within_5pct", passes, 3, None, None,
                        ">= 3 sets", passes >= 3))  # fmt: skip
    return checks


# ---- orchestration ---------------------------------------------------------------------


def run_validation(
    fast: bool, *, settings: Settings, paths: ValidationPaths
) -> tuple[list[Check], dict[str, Any]]:
    """Run every check; returns the rows and the data behind the figure."""
    checks: list[Check] = []
    intensity, fig_int = check_intensity(paths, fast)
    checks += intensity
    as_checks, fig_as = check_as2008(paths, fast, settings)
    checks += as_checks
    checks += check_queue_latency(paths, settings)
    ce_checks, fig_ce = check_glft_ce(paths, fast)
    checks += ce_checks
    checks += check_asymptotic()
    checks = mark_exceptions(checks, load_exceptions(paths.repo_root))
    return checks, {"intensity": fig_int, "as2008": fig_as, "glft_ce": fig_ce}


def _verdict(c: Check) -> str:
    if c.passed:
        return "yes"
    return f"NO, {c.note}" if c.note.startswith("documented exception") else "NO"


def write_report(checks: list[Check], figure: dict[str, Any], reports_dir: Path) -> None:
    """``tables/synthetic_validation.{csv,md}`` and ``figures/synthetic_validation.png``."""
    tables = reports_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame([asdict(c) for c in checks])
    frame.write_csv(tables / "synthetic_validation.csv", line_terminator="\n")
    lines = [
        "| check | case | metric | value | target | SE | z | tolerance | pass |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in checks:
        se = "" if c.se is None else f"{c.se:.4g}"
        z = "" if c.z is None else f"{c.z:.2f}"
        lines.append(
            f"| {c.check} | {c.case} | {c.metric} | {c.value:.6g} | {c.target:.6g} | {se} | {z} "
            f"| {c.tolerance} | {_verdict(c)} |"
        )
    (tables / "synthetic_validation.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline=LF
    )
    _figure(figure, reports_dir / "figures" / "synthetic_validation.png")


def _figure(data: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(1600 / 150, 4.2), dpi=150)
    it = data["intensity"]
    ax = axes[0]
    ax.semilogy(it["delta"], it["expect"], color="#555555", label="A exp(-k delta)")
    err = [np.subtract(it["rate"], it["lo"]), np.subtract(it["hi"], it["rate"])]
    ax.errorbar(
        it["delta"], it["rate"], yerr=err, fmt="o", color="#1f77b4", label="fallback, 99% CI"
    )
    ax.set_xlabel("distance from mid delta")
    ax.set_ylabel("fills per second")
    ax.set_title("Fill intensity (AC-W-01)")
    ax.legend()
    ax = axes[1]
    cases = list(data["as2008"])
    ref = [np.mean(data["as2008"][c]["reference"]) for c in cases]
    sim = [np.mean(data["as2008"][c]["fallback"]) for c in cases]
    pub = [data["as2008"][c]["published"] for c in cases]
    x = np.arange(len(cases))
    ax.bar(x - 0.2, ref, 0.4, label="NumPy reference", color="#1f77b4")
    ax.bar(x + 0.2, sim, 0.4, label="event-driven fallback", color="#ff7f0e")
    ax.scatter(x, pub, color="black", marker="_", s=300, linewidths=2, label="published", zorder=3)
    labels = [
        c.replace("gamma=", "γ=").replace(" inventory", "\ninv.").replace(" symmetric", "\nsym.")
        for c in cases
    ]
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("mean profit (1,000 paths)")
    ax.set_ylim(0, max(max(ref), max(sim)) * 1.35)
    ax.set_title("A-S 2008 Tables 1-3 (AC-W-02)")
    ax.legend(loc="upper center", ncol=3, fontsize=7)
    ax = axes[2]
    ce = data["glft_ce"]
    ax.scatter([c["ce"] for c in ce], [c["ce_mc"] for c in ce], color="#2ca02c")
    lim = [min(c["ce"] for c in ce) - 2, max(c["ce"] for c in ce) + 2]
    ax.plot(lim, lim, color="#555555", linewidth=1)
    ax.set_xlabel("closed-form certainty equivalent")
    ax.set_ylabel("Monte Carlo certainty equivalent")
    ax.set_title("GLFT exact strategy (AC-W-03)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
