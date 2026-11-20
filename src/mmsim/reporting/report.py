"""``mms report``: tables and figures from run outputs (FR-26).

Every report names its runs through committed configs (``configs/reports.yaml``). For each
config and symbol the latest successful trial with exactly the parameters that config
produces is used; when there is none, the run is executed first. Numbers in tables come
only from run outputs, with 95% day-block bootstrap CIs next to point estimates.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from mmsim.analytics.stats import day_block_bootstrap_ci, weighted_mean
from mmsim.config import load_config, resolve_dict
from mmsim.explog import canonical_json, read_log
from mmsim.runner import execute_run, run_params
from mmsim.settings import Settings

__all__ = ["RunRef", "ensure_runs", "markouts_report", "pnl_report", "write_table"]

COMPONENTS = ("spread", "adverse_selection", "inventory_residual", "fees", "funding")


@dataclass(frozen=True)
class RunRef:
    """A finished run behind a report row."""

    policy: str
    symbol: str
    run_id: str
    path: Path


def write_table(frame: pl.DataFrame, reports_dir: Path, name: str, digits: int = 6) -> Path:
    """``tables/<name>.csv`` and a Markdown twin (display rounding only)."""
    tables = reports_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    csv = tables / f"{name}.csv"
    frame.write_csv(csv, line_terminator="\n")
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "---|" * len(frame.columns)
    rows = []
    for row in frame.iter_rows():
        cells = [
            f"{v:.{digits}g}" if isinstance(v, float) else ("" if v is None else str(v))
            for v in row
        ]
        rows.append("| " + " | ".join(cells) + " |")
    (tables / f"{name}.md").write_text("\n".join([header, rule, *rows]) + "\n", encoding="utf-8",
                                       newline="\n")  # fmt: skip
    return csv


def ensure_runs(
    report: str, *, settings: Settings, repo_root: Path = Path(), run_missing: bool = True
) -> list[RunRef]:
    """The runs behind ``report`` in ``configs/reports.yaml``, executing missing ones."""
    spec = yaml.safe_load((repo_root / "configs" / "reports.yaml").read_text(encoding="utf-8"))
    entry = spec[report]
    results = settings.mms_results_dir
    refs: list[RunRef] = []
    for config in entry["configs"]:
        for symbol in entry["symbols"]:
            overrides = [f"data.symbols=[{symbol}]"]
            path = repo_root / config
            cfg = load_config(path, overrides, settings=settings)
            want = canonical_json(run_params(cfg))
            rows = [
                r
                for r in read_log(results / "trial_log.csv")
                if r["strategy_family"] == cfg.strategy.family
                and r["params_json"] == want
                and r["n_obs"] != "0"
                and (results / "runs" / r["run_id"] / "metrics.json").exists()
            ]
            if rows:
                run_id = rows[-1]["run_id"]
            elif run_missing:
                resolved = json.loads(
                    json.dumps(resolve_dict(path, overrides, settings), default=str)
                )
                outcome = execute_run(cfg, settings=settings, repo_root=repo_root,
                                      resolved=resolved, results_dir=results)  # fmt: skip
                run_id = outcome.run.run_id
            else:
                raise FileNotFoundError(f"no run for {config} {symbol}")
            refs.append(RunRef(cfg.strategy.family, symbol, run_id, results / "runs" / run_id))
    return refs


def _ci(values: Sequence[float], seed: int) -> tuple[float, float]:
    return day_block_bootstrap_ci([np.array([v]) for v in values], lambda x: float(np.mean(x)),
                                  seed=seed)  # fmt: skip


def pnl_report(refs: Sequence[RunRef], reports_dir: Path) -> pl.DataFrame:
    """Mean daily components with CIs, shares of gross spread capture, excluded days."""
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for ref in refs:
        sess = pl.read_parquet(ref.path / "sessions.parquet")
        for r in sess.filter(~pl.col("valid")).iter_rows(named=True):
            excluded.append({"policy": ref.policy, "symbol": ref.symbol, "day": str(r["day"]),
                             "reason": r["reason"]})  # fmt: skip
        valid = sess.filter(pl.col("valid") & pl.col("total").is_not_null())
        row: dict[str, Any] = {"policy": ref.policy, "symbol": ref.symbol, "days": valid.height}
        for i, comp in enumerate((*COMPONENTS, "total")):
            vals = valid[comp].to_list()
            lo, hi = _ci(vals, seed=i)
            row[f"{comp}_per_day"] = float(np.mean(vals)) if vals else float("nan")
            row[f"{comp}_lo"] = lo
            row[f"{comp}_hi"] = hi
        gross = row["spread_per_day"]
        for comp in ("adverse_selection", "inventory_residual", "fees"):
            row[f"{comp}_share_of_spread"] = (
                row[f"{comp}_per_day"] / gross if gross else float("nan")
            )
        fills = valid["fills"].to_numpy().astype(np.float64)
        row["fills_per_day"] = float(fills.mean()) if fills.size else float("nan")
        row["run_id"] = ref.run_id
        rows.append(row)
    table = pl.DataFrame(rows)
    write_table(table, reports_dir, "pnl_decomposition")
    write_table(
        pl.DataFrame(excluded, schema={"policy": pl.String, "symbol": pl.String, "day": pl.String,
                                       "reason": pl.String}),
        reports_dir, "excluded_days",
    )  # fmt: skip
    _pnl_figure(table, reports_dir / "figures" / "pnl_decomposition.png")
    return table


def markouts_report(refs: Sequence[RunRef], reports_dir: Path) -> pl.DataFrame:
    """Volume-weighted markouts by horizon and side, with 95% day-block CIs."""
    rows: list[dict[str, Any]] = []
    for ref in refs:
        sess = pl.read_parquet(ref.path / "sessions.parquet")
        valid_days = set(sess.filter(pl.col("valid"))["day"].to_list())
        mo = pl.read_parquet(ref.path / "markouts.parquet").filter(pl.col("valid"))
        mo = mo.with_columns(
            pl.from_epoch(pl.col("ts_exchange"), time_unit="ns").dt.date().alias("day")
        ).filter(pl.col("day").is_in(list(valid_days)))
        for side_name, side_filter in (("all", None), ("buy", 1), ("sell", -1)):
            part = mo if side_filter is None else mo.filter(pl.col("side") == side_filter)
            for h in sorted(part["horizon_ms"].unique().to_list()):
                ph = part.filter(pl.col("horizon_ms") == h)
                row: dict[str, Any] = {
                    "policy": ref.policy, "symbol": ref.symbol, "side": side_name,
                    "horizon_ms": h, "fills": ph.height,
                }  # fmt: skip
                for col in ("mo_mid_bps", "mo_fill_bps"):
                    by_day = [
                        g.select(col, "qty_lots").to_numpy().astype(np.float64)
                        for _, g in ph.group_by("day", maintain_order=True)
                    ]
                    point = weighted_mean(np.concatenate(by_day)) if by_day else float("nan")
                    lo, hi = day_block_bootstrap_ci(by_day, weighted_mean, n=2000, seed=h)
                    row[col] = point
                    row[f"{col}_lo"] = lo
                    row[f"{col}_hi"] = hi
                rows.append(row)
    table = pl.DataFrame(rows)
    write_table(table, reports_dir, "markouts")
    for symbol in sorted(table["symbol"].unique().to_list()):
        _markout_figure(table.filter(pl.col("symbol") == symbol), symbol,
                        reports_dir / "figures" / f"markouts_{symbol}.png")  # fmt: skip
    return table


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


COLOURS = {"spread": "#2ca02c", "adverse_selection": "#d62728", "inventory_residual": "#9467bd",
           "fees": "#7f7f7f", "funding": "#e377c2"}  # fmt: skip


def _pnl_figure(table: pl.DataFrame, path: Path) -> None:
    plt = _plt()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1600 / 150, 5), dpi=150)
    labels = [f"{r['symbol']}\n{r['policy']}" for r in table.iter_rows(named=True)]
    x = np.arange(len(labels))
    pos = np.zeros(len(labels))
    neg = np.zeros(len(labels))
    for comp in COMPONENTS:
        vals = table[f"{comp}_per_day"].to_numpy()
        base = np.where(vals >= 0, pos, neg)
        ax.bar(x, vals, 0.6, bottom=base, color=COLOURS[comp], label=comp.replace("_", " "))
        pos += np.where(vals >= 0, vals, 0)
        neg += np.where(vals < 0, vals, 0)
    tot = table["total_per_day"].to_numpy()
    err = [tot - table["total_lo"].to_numpy(), table["total_hi"].to_numpy() - tot]
    ax.errorbar(x, tot, yerr=np.nan_to_num(err), fmt="D", color="black", label="total, 95% CI")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("USD per day (tuning days)")
    ax.set_title("Where the P&L comes from: daily components per symbol and policy")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _markout_figure(table: pl.DataFrame, symbol: str, path: Path) -> None:
    plt = _plt()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(1600 / 150, 4.5), dpi=150, sharex=True)
    for ax, col, title in ((axes[0], "mo_mid_bps", "from the pre-fill mid"),
                           (axes[1], "mo_fill_bps", "from the fill price")):  # fmt: skip
        for policy in sorted(table["policy"].unique().to_list()):
            t = table.filter((pl.col("policy") == policy) & (pl.col("side") == "all")).sort(
                "horizon_ms"
            )
            h = t["horizon_ms"].to_numpy() / 1000
            ax.plot(h, t[col].to_numpy(), marker="o", label=policy)
            ax.fill_between(h, t[f"{col}_lo"].to_numpy(), t[f"{col}_hi"].to_numpy(), alpha=0.2)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("horizon (s)")
        ax.set_ylabel("markout (bps, volume-weighted)")
        ax.set_title(f"{symbol}: {title}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
