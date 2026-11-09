"""The ``mms`` command line (FR-29, spec section 10).

Every command in the spec exists from M1 so that ``--help`` works everywhere; commands
whose milestone has not arrived exit with code 2 and say which milestone brings them.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

from mmsim.config import MmsConfig, load_config, load_synth_config
from mmsim.data.check import run_check
from mmsim.data.synth import generate, write_world
from mmsim.explog import read_log
from mmsim.settings import DataLocationError, Settings, check_data_location, get_settings

__all__ = ["app"]

app = typer.Typer(help="Market-making simulator (market-making-sim).", no_args_is_help=True)
data_app = typer.Typer(help="Data access, generation and ingest.", no_args_is_help=True)
validate_app = typer.Typer(help="Validation against known results.", no_args_is_help=True)
rl_app = typer.Typer(help="Reinforcement-learning agents.", no_args_is_help=True)
explog_app = typer.Typer(help="Experiment log (trial-log contract v1.0).", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(validate_app, name="validate")
app.add_typer(rl_app, name="rl")
app.add_typer(explog_app, name="explog")

DEFAULT_CONFIG = Path("configs/base.yaml")


def _not_yet(milestone: str) -> None:
    typer.echo(f"not implemented yet: arrives in {milestone} (see SPEC section 13)", err=True)
    raise typer.Exit(code=2)


def _day(value: str | None) -> datetime | None:
    return datetime.strptime(value, "%Y-%m-%d") if value else None


def _source_root(cfg: MmsConfig, settings: Settings) -> Path:
    if cfg.data.source == "cla":
        return settings.mms_cla_data_root
    return settings.processed_root


@data_app.command("check")
def data_check(
    config: Annotated[Path, typer.Option(help="Run config naming venue, symbols and window.")] = (
        DEFAULT_CONFIG
    ),
    symbol: Annotated[list[str] | None, typer.Option(help="Symbol(s); default: config's.")] = None,
    start: Annotated[str | None, typer.Option(help="First UTC day, YYYY-MM-DD.")] = None,
    end: Annotated[str | None, typer.Option(help="Last UTC day, YYYY-MM-DD.")] = None,
    deep: Annotated[
        bool, typer.Option(help="Also read rows and run DQ-13 content checks.")
    ] = False,
) -> None:
    """Smoke-test data access: paths, schemas, coverage, gaps, instruments and fees."""
    settings = get_settings()
    cfg = load_config(config, settings=settings)
    s, e = _day(start), _day(end)
    report = run_check(
        _source_root(cfg, settings),
        cfg.data,
        cfg.fees,
        symbols=symbol or None,
        start=s.date() if s else None,
        end=e.date() if e else None,
        deep=deep,
    )
    for line in report.lines():
        typer.echo(line)
    raise typer.Exit(code=0 if report.ok else 1)


@data_app.command("synth")
def data_synth(
    config: Annotated[Path, typer.Option(help="Synthetic-world config.")],
    out: Annotated[Path | None, typer.Option(help="Canonical root; default processed/.")] = None,
    seed: Annotated[int | None, typer.Option(help="Override the config's seed.")] = None,
) -> None:
    """Generate a synthetic Poisson world in the canonical schemas (FR-05)."""
    overrides = [f"seed={seed}"] if seed is not None else []
    cfg = load_synth_config(config, overrides)
    try:
        root = out or check_data_location(get_settings().processed_root)
    except DataLocationError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    files = 0
    for world in generate(cfg):
        files += len(write_world(world, root))
    typer.echo(f"wrote {cfg.paths} path(s), {files} files under {root}")


@data_app.command("funding")
def data_funding(
    symbol: Annotated[list[str], typer.Option(help="Symbol(s).")],
    start: Annotated[str, typer.Option(help="First UTC day.")],
    end: Annotated[str, typer.Option(help="Last UTC day.")],
) -> None:
    """Download Binance USD-M funding history (public REST)."""
    _not_yet("M3")


@data_app.command("ingest-lobster")
def data_ingest_lobster(
    src: Annotated[Path, typer.Option(help="Folder with LOBSTER sample files.")],
    ticker: Annotated[str, typer.Option(help="Ticker, e.g. AAPL.")],
    date_: Annotated[str, typer.Option("--date", help="Sample date.")],
    levels: Annotated[int, typer.Option(help="Book levels.")] = 10,
) -> None:
    """Convert LOBSTER samples to canonical book_l2 and trades."""
    _not_yet("M5")


@data_app.command("ingest-tardis")
def data_ingest_tardis(
    exchange: Annotated[str, typer.Option(help="Tardis exchange id.")],
    symbol: Annotated[str, typer.Option(help="Symbol.")],
    start: Annotated[str, typer.Option(help="First UTC day.")],
    end: Annotated[str, typer.Option(help="Last UTC day.")],
    free_sample_only: Annotated[bool, typer.Option(help="Never call the paid API.")] = True,
) -> None:
    """Download and convert Tardis.dev data (D-01 gated)."""
    _not_yet("M5")


@app.command("calibrate")
def calibrate(
    config: Annotated[Path, typer.Option(help="Run config.")],
    symbol: Annotated[str, typer.Option(help="Symbol.")],
    date_: Annotated[str, typer.Option("--date", help="UTC day.")],
) -> None:
    """Fit A, k and sigma for one symbol-day (FR-12)."""
    _not_yet("M3")


@app.command("run")
def run(
    config: Annotated[Path, typer.Option(help="Run config.")],
    set_: Annotated[list[str] | None, typer.Option("--set", help="key=value override.")] = None,
    sim: Annotated[str | None, typer.Option(help="fallback | edbt")] = None,
    seed: Annotated[int | None, typer.Option(help="Override run.seed.")] = None,
) -> None:
    """Run one strategy configuration through the simulator."""
    _not_yet("M2")


@app.command("sweep")
def sweep(
    config: Annotated[Path, typer.Option(help="Sweep config.")],
    jobs: Annotated[int | None, typer.Option(help="Worker processes.")] = None,
    resume: Annotated[bool, typer.Option(help="Skip finished cells.")] = False,
) -> None:
    """Run a grid of configurations in parallel."""
    _not_yet("M5")


@validate_app.command("synthetic")
def validate_synthetic(
    fast: Annotated[bool, typer.Option(help="CI subset: fewer paths, coarser grid.")] = False,
) -> None:
    """Synthetic validation against closed-form results (FR-19)."""
    _not_yet("M2")


@validate_app.command("signal-parity")
def validate_signal_parity(
    model_dir: Annotated[Path, typer.Option(help="Signal artifact version folder.")],
) -> None:
    """Online vs offline signal parity (AC-W-09)."""
    _not_yet("M4")


@validate_app.command("sim-parity")
def validate_sim_parity(
    config: Annotated[Path, typer.Option(help="Parity config.")],
) -> None:
    """Fallback vs edbt parity (AC-H-10)."""
    _not_yet("M6")


@rl_app.command("train")
def rl_train(
    algo: Annotated[str, typer.Option(help="ppo | sac")],
    config: Annotated[Path, typer.Option(help="RL config.")],
    seed: Annotated[int, typer.Option(help="Seed.")],
    steps: Annotated[int | None, typer.Option(help="Step budget.")] = None,
) -> None:
    """Train a PPO or SAC agent (FR-23)."""
    _not_yet("M6")


@rl_app.command("eval")
def rl_eval(
    run_id: Annotated[str, typer.Option(help="Training run id.")],
    split: Annotated[str, typer.Option(help="val | test")],
    stress: Annotated[str | None, typer.Option(help="Stress scenario.")] = None,
) -> None:
    """Evaluate a trained agent (FR-24)."""
    _not_yet("M6")


@app.command("report")
def report(
    name: Annotated[str, typer.Argument(help="all | synthetic | pnl | markouts | ...")],
) -> None:
    """Regenerate figures and tables (FR-26)."""
    _not_yet("M2 (synthetic), M3 (pnl, markouts) and later")


@explog_app.command("summary")
def explog_summary(
    family: Annotated[str | None, typer.Option(help="Only this strategy_family.")] = None,
    path: Annotated[Path, typer.Option(help="Trial log.")] = Path("results/trial_log.csv"),
    check_limits: Annotated[
        bool, typer.Option(help="Check max |inventory| <= q_max in every run (AC-W-05).")
    ] = False,
) -> None:
    """Trial counts and Sharpe distribution per family."""
    rows = [r for r in read_log(path) if family is None or r["strategy_family"] == family]
    counts = Counter(r["strategy_family"] for r in rows)
    failed = Counter(r["strategy_family"] for r in rows if r["n_obs"] == "0")
    sharpes: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["sharpe_annual"]:
            sharpes[r["strategy_family"]].append(float(r["sharpe_annual"]))
    typer.echo(f"{'family':<24} {'trials':>6} {'failed':>6} {'median_sr':>10} {'max_sr':>8}")
    for fam in sorted(counts):
        values = sharpes.get(fam, [])
        med = f"{statistics.median(values):.3f}" if values else "-"
        top = f"{max(values):.3f}" if values else "-"
        typer.echo(f"{fam:<24} {counts[fam]:>6} {failed[fam]:>6} {med:>10} {top:>8}")
    typer.echo(f"total trials: {len(rows)}")
    if check_limits:
        breaches = 0
        for r in rows:
            metrics_path = path.parent / "runs" / r["run_id"] / "metrics.json"
            if not metrics_path.exists():
                continue
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            if "max_abs_inventory_lots" in m and m["max_abs_inventory_lots"] > m["q_max_lots"]:
                breaches += 1
                typer.echo(f"LIMIT BREACH: {r['run_id']}", err=True)
        if breaches:
            raise typer.Exit(code=1)
        typer.echo("inventory limits: no breach")
