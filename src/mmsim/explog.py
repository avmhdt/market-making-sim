"""Experiment log in backtest-overfitting-lab's trial-log contract v1.0 (FR-25, AC-W-12).

``results/trial_log.csv`` is append-only and committed; it is the honest trial count.
Every run, sweep cell, validation run and RL evaluation appends exactly one row, failed
runs included (``n_obs = 0``). Appends are serialised with a lock file so parallel sweep
workers never interleave rows. Adapted from the contract's reference writer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow.parquet as pq

__all__ = [
    "COLUMNS",
    "append_failed",
    "append_trial",
    "canonical_json",
    "config_digest",
    "git_sha",
    "new_run_id",
    "periods_per_year",
    "read_log",
]

COLUMNS = (
    "run_id",
    "timestamp",
    "git_sha",
    "strategy_family",
    "params_json",
    "start",
    "end",
    "n_obs",
    "sharpe_annual",
    "returns_path",
    "x_config_digest",
    "x_selected",
    "x_periods_per_year",
)
"""The header, exactly as spec section 10 fixes it."""

FAMILIES = frozenset(
    {
        "as",
        "glft",
        "glft_exact",
        "glft_signal",
        "rl_ppo",
        "rl_sac",
        "synthetic_validation",
        "ablation",
    }
)


def new_run_id() -> str:
    """A UUID v4 in lower-case canonical form."""
    return str(uuid.uuid4())


def canonical_json(params: dict[str, object]) -> str:
    """The contract's canonical JSON of a params mapping."""
    return json.dumps(
        params, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def periods_per_year(strategy_family: str) -> int:
    """q for Sharpe annualisation: 252 for LOBSTER equity families, 365 for 24/7 crypto."""
    return 252 if strategy_family.endswith("_lobster") else 365


def _check_family(strategy_family: str) -> None:
    base = strategy_family.removesuffix("_lobster")
    if base not in FAMILIES:
        raise ValueError(f"unknown strategy_family {strategy_family!r}")


def git_sha(log_path: Path) -> str:
    """HEAD's full hash, with ``-dirty`` if tracked files other than the log changed."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, cwd=log_path.parent
        ).stdout

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        sha = git("rev-parse", "HEAD").strip()
        root = Path(git("rev-parse", "--show-toplevel").strip())
        rel = log_path.resolve().relative_to(root.resolve()).as_posix()
        status = git("status", "--porcelain", "--untracked-files=no", "--", ".", f":(exclude){rel}")
    except (OSError, ValueError, subprocess.CalledProcessError):
        return "0" * 40 + "-dirty"
    return f"{sha}-dirty" if status.strip() else sha


def config_digest(params_json: str, start: date | None, end: date | None, sha: str) -> str:
    """8-hex digest of the params, the data window and the code version (determinism key)."""
    window = f"{start.isoformat() if start else ''}..{end.isoformat() if end else ''}"
    return hashlib.sha256(f"{params_json}|{window}|{sha}".encode()).hexdigest()[:8]


@contextmanager
def _locked(path: Path, timeout_s: float = 60.0) -> Iterator[None]:
    lock = path.with_name(path.name + ".lock")
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"could not lock {path} within {timeout_s} s") from None
            time.sleep(0.01)
    try:
        yield
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)


def _append_row(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(path):
        is_new = not path.exists() or path.stat().st_size == 0
        if not is_new:
            with path.open(encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle), [])
            if tuple(header) != COLUMNS:
                raise ValueError(f"{path}: header {header} is not the contract header")
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, restval="", lineterminator="\n")
            if is_new:
                writer.writeheader()
            writer.writerow(row)


def _base_row(
    path: Path, strategy_family: str, params: dict[str, object], run_id: str
) -> dict[str, str]:
    _check_family(strategy_family)
    uuid.UUID(run_id, version=4)
    return {
        "run_id": run_id,
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha(path),
        "strategy_family": strategy_family,
        "params_json": canonical_json(params),
        "x_periods_per_year": str(periods_per_year(strategy_family)),
    }


def append_trial(
    path: Path,
    *,
    run_id: str,
    strategy_family: str,
    params: dict[str, object],
    start: date,
    end: date,
    returns_path: Path,
    selected: bool = False,
) -> dict[str, str]:
    """Append one evaluated trial; ``returns_path`` is the run's ``returns.parquet``.

    ``n_obs`` and ``sharpe_annual`` are computed from the Parquet series, which must hold
    ``date`` and ``ret`` and match ``start``/``end``. With a single observation the Sharpe
    ratio is undefined; the contract requires a finite value on any row with a series, so
    0 is written (docs/decisions.md, M1-05).
    """
    table = pq.read_table(returns_path, columns=["date", "ret"])
    dates = table.column("date").to_pylist()
    rets = [float(x) for x in table.column("ret").to_pylist()]
    if not dates:
        raise ValueError(f"{returns_path} is empty; log a failed run instead")
    if dates[0] != start or dates[-1] != end:
        raise ValueError(f"window {start}..{end} does not match {returns_path}")
    q = periods_per_year(strategy_family)
    sd = statistics.stdev(rets) if len(rets) > 1 else 0.0
    sharpe = statistics.fmean(rets) / sd * math.sqrt(q) if sd > 0 else 0.0
    row = _base_row(path, strategy_family, params, run_id)
    rel = returns_path.resolve().relative_to(path.parent.resolve()).as_posix()
    if rel != f"runs/{run_id}/returns.parquet":
        raise ValueError(f"returns_path must be runs/{run_id}/returns.parquet, got {rel}")
    row |= {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "n_obs": str(len(rets)),
        "sharpe_annual": f"{sharpe:.10g}",
        "returns_path": rel,
        "x_config_digest": config_digest(row["params_json"], start, end, row["git_sha"]),
        "x_selected": "true" if selected else "false",
    }
    _append_row(path, row)
    return row


def append_failed(
    path: Path,
    *,
    run_id: str,
    strategy_family: str,
    params: dict[str, object],
    start: date | None = None,
    end: date | None = None,
) -> dict[str, str]:
    """Append a failed or aborted run: ``n_obs = 0``, no Sharpe, no series. It counts in N."""
    row = _base_row(path, strategy_family, params, run_id)
    row |= {
        "start": start.isoformat() if start else "",
        "end": end.isoformat() if end else "",
        "n_obs": "0",
        "x_config_digest": config_digest(row["params_json"], start, end, row["git_sha"]),
        "x_selected": "false",
    }
    _append_row(path, row)
    return row


def read_log(path: Path) -> list[dict[str, str]]:
    """All rows of the log (empty if it does not exist)."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is not None and tuple(reader.fieldnames) != COLUMNS:
            raise ValueError(f"{path}: header {reader.fieldnames} is not the contract header")
        return list(reader)
