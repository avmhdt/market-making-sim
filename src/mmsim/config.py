"""Typed YAML configs (spec section 10).

A run config is merged over ``configs/base.yaml`` (found by walking up from the config
file to a ``configs`` directory that holds it), then ``--set key=value`` overrides are
applied, then ``${VAR}`` references are filled from the settings and the environment.
Every model forbids unknown keys, so a typo fails loudly.
"""

from __future__ import annotations

import os
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mmsim.settings import Settings

__all__ = [
    "AnalysisConfig",
    "CalibrationConfig",
    "DataConfig",
    "FeesConfig",
    "LatencyConfig",
    "MmsConfig",
    "SimConfig",
    "StrategyConfig",
    "SynthConfig",
    "apply_overrides",
    "deep_merge",
    "find_base_config",
    "load_config",
    "load_synth_config",
    "load_yaml",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunSection(_Model):
    """Run identity and backend."""

    name: str = "run"
    seed: int = 0
    sim: Literal["fallback", "edbt"] = "fallback"
    results_dir: str = "./results"


class DataConfig(_Model):
    """Which data to replay. ``start`` and ``end`` are inclusive UTC days."""

    source: Literal["cla", "synthetic", "lobster", "tardis"] = "cla"
    venue: str = "binance-usdm"
    symbols: list[str] = Field(default_factory=lambda: ["BTCUSDT"])
    start: date | None = None
    end: date | None = None
    book_depth: int = Field(default=10, ge=1)
    respect_gap_log: bool = True
    instruments: str = "configs/instruments.yaml"

    @model_validator(mode="after")
    def _window(self) -> DataConfig:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError(f"data.start {self.start} is after data.end {self.end}")
        return self


class FeesConfig(_Model):
    """Fee schedule and tier (D-10) and whether funding is charged."""

    schedule: str = "configs/fees.yaml"
    tier: str = "retail"
    funding: bool = True


class QueueModelConfig(_Model):
    """Queue model (FR-02): ``front``, ``conservative`` or ``prob`` with ``f(x) = x**power``."""

    kind: Literal["front", "conservative", "prob"] = "prob"
    power: float = Field(default=2.0, gt=0)


class FeedLatencyConfig(_Model):
    """Feed latency: the recorded ``ts_recv - ts_exchange`` or a constant, plus ``extra_ns``."""

    kind: Literal["recorded", "constant"] = "recorded"
    ns: int | None = Field(default=None, ge=0)
    extra_ns: int = Field(default=0, ge=0)


class OrderLatencyConfig(_Model):
    """Order or ack latency (D-09): the day's median feed latency, a constant or empirical."""

    kind: Literal["median_feed", "constant", "empirical"] = "median_feed"
    ns: int | None = Field(default=None, ge=0)


class LatencyConfig(_Model):
    """All latency models; ``added_rtt_ns`` is swept by the latency study (FR-21)."""

    feed: FeedLatencyConfig = Field(default_factory=FeedLatencyConfig)
    order: OrderLatencyConfig = Field(default_factory=OrderLatencyConfig)
    ack: OrderLatencyConfig = Field(default_factory=OrderLatencyConfig)
    added_rtt_ns: int = Field(default=0, ge=0)


class SimConfig(_Model):
    """Simulator settings shared by every backend."""

    queue_model: QueueModelConfig = Field(default_factory=QueueModelConfig)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    post_only: bool = True


class DepthGrid(_Model):
    """Depth grid in ticks for the intensity fit (FR-12)."""

    min: int = Field(default=1, ge=1)
    max: int = Field(default=20, ge=1)


class CalibrationConfig(_Model):
    """Rolling point-in-time calibration (DQ-05, FR-12)."""

    window_s: float = Field(default=600.0, gt=0)
    refit_every_s: float = Field(default=60.0, gt=0)
    depth_grid_ticks: DepthGrid = Field(default_factory=DepthGrid)
    sigma_sample_ms: int = Field(default=1000, gt=0)
    min_trades: int = Field(default=200, ge=1)
    min_r2: float = Field(default=0.8, ge=0, le=1)


class SignalConfig(_Model):
    """Signal skew (FR-16)."""

    enabled: bool = False
    model_dir: str | None = None
    mode: Literal["reservation_shift", "glft_drift", "stub"] = "reservation_shift"
    alpha: float = Field(default=0.5, ge=0, le=1)
    pull_threshold_ticks: float | None = None
    compute_latency_us: int = Field(default=200, ge=0)


class StrategyConfig(_Model):
    """Quoting policy (FR-09 to FR-11)."""

    family: Literal["as", "glft", "glft_exact", "rl"] = "glft"
    gamma: float = Field(default=0.05, gt=0)
    horizon_s: float = Field(default=3600.0, gt=0)
    quote_lots: int | Literal["auto"] = "auto"
    q_max_units: int = Field(default=10, ge=1)
    decision_interval_ms: int = Field(default=100, gt=0)
    requote_threshold_ticks: int = Field(default=1, ge=0)
    signal: SignalConfig = Field(default_factory=SignalConfig)


class BootstrapConfig(_Model):
    """Block bootstrap for confidence intervals."""

    n: int = Field(default=5000, ge=1)
    block: Literal["day"] = "day"
    alpha: float = Field(default=0.05, gt=0, lt=1)


class AnalysisConfig(_Model):
    """Markouts, bootstrap and return accounting (D-05, D-11)."""

    markout_horizons_ms: list[int] = Field(
        default_factory=lambda: [100, 500, 1000, 5000, 10000, 30000, 60000]
    )
    headline_horizon_ms: int = 5000
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    capital_usd: float = Field(default=10_000.0, gt=0)
    return_freq: str = "1h"


class ExplogConfig(_Model):
    """Experiment log location."""

    path: str = "results/trial_log.csv"


class MmsConfig(_Model):
    """A complete run config."""

    run: RunSection = Field(default_factory=RunSection)
    data: DataConfig = Field(default_factory=DataConfig)
    fees: FeesConfig = Field(default_factory=FeesConfig)
    sim: SimConfig = Field(default_factory=SimConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    explog: ExplogConfig = Field(default_factory=ExplogConfig)


class MidProcessConfig(_Model):
    """Synthetic mid process (FR-05).

    ``binomial_grid``: the mid moves by +-sigma*sqrt(dt) every ``dt`` (A-S 2008 style),
    rounded to whole ticks. ``tick_jump``: +-1 tick jumps at rate sigma**2 / tick**2.
    """

    kind: Literal["binomial_grid", "tick_jump"] = "binomial_grid"
    s0: float = 100.0
    sigma: float = Field(default=2.0, gt=0)


class MarketOrderConfig(_Model):
    """Poisson market orders, each side at rate ``A`` with depth ``D ~ Exp(k)`` beyond mid."""

    A: float = Field(default=140.0, gt=0)
    k: float = Field(default=1.5, gt=0)
    qty_lots: int = Field(default=1, ge=1)


class SynthBookConfig(_Model):
    """Symmetric synthetic book around the mid."""

    levels: int = Field(default=5, ge=1)
    half_spread_ticks: int = Field(default=1, ge=1)
    level_lots: int = Field(default=10, ge=1)


class SynthConfig(_Model):
    """A synthetic Poisson world (FR-05).

    Model time is in units of ``time_unit_ns`` nanoseconds; ``horizon`` and ``dt`` and the
    rates (``sigma`` per sqrt unit, ``A`` per unit) use that unit.
    """

    venue: str = "synthetic"
    symbol: str = "SYN"
    start_utc: datetime = datetime(2026, 1, 1, tzinfo=UTC)
    time_unit_ns: int = Field(default=1_000_000_000, gt=0)
    horizon: float = Field(default=1.0, gt=0)
    dt: float = Field(default=0.005, gt=0)
    tick_size: float = Field(default=0.0001, gt=0)
    lot_size: float = Field(default=1.0, gt=0)
    mid: MidProcessConfig = Field(default_factory=MidProcessConfig)
    market_orders: MarketOrderConfig = Field(default_factory=MarketOrderConfig)
    book: SynthBookConfig = Field(default_factory=SynthBookConfig)
    feed_latency_ns: int = Field(default=0, ge=0)
    feed_jitter_ns: int = Field(default=0, ge=0)
    seed: int = 0
    paths: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _aware(self) -> SynthConfig:
        if self.start_utc.tzinfo is None:
            raise ValueError("start_utc must carry a timezone (use ...Z)")
        if self.dt > self.horizon:
            raise ValueError("dt must not exceed horizon")
        return self


_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping; an empty or comment-only file gives ``{}``."""
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return loaded


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``over`` into a copy of ``base``; non-mapping values replace."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``key.sub=value`` overrides; values are parsed as YAML (``[A, B]``, ``1e3``)."""
    out = dict(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not key=value")
        dotted, raw = item.split("=", 1)
        keys = [part for part in dotted.strip().split(".") if part]
        if not keys:
            raise ValueError(f"override {item!r} has an empty key")
        value = yaml.safe_load(raw)
        patch: dict[str, Any] = {keys[-1]: value}
        for key in reversed(keys[:-1]):
            patch = {key: patch}
        out = deep_merge(out, patch)
    return out


def _interpolate(node: Any, env: dict[str, str]) -> Any:
    if isinstance(node, str):

        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in env:
                raise KeyError(f"${{{name}}} is not set")
            return env[name]

        return _VAR.sub(repl, node)
    if isinstance(node, dict):
        return {key: _interpolate(value, env) for key, value in node.items()}
    if isinstance(node, list):
        return [_interpolate(value, env) for value in node]
    return node


def find_base_config(path: Path) -> Path | None:
    """The nearest ``configs/base.yaml`` above ``path``, or ``None``."""
    for parent in path.resolve().parents:
        candidate = parent / "base.yaml"
        if parent.name == "configs" and candidate.is_file():
            return candidate
    return None


def resolve_dict(
    path: Path, overrides: list[str] | None = None, settings: Settings | None = None
) -> dict[str, Any]:
    """The merged, overridden and interpolated config mapping for ``path``."""
    raw = load_yaml(path)
    base_path = find_base_config(path)
    if base_path is not None and base_path.resolve() != path.resolve():
        raw = deep_merge(load_yaml(base_path), raw)
    raw = apply_overrides(raw, overrides or [])
    settings = settings or Settings()
    env = {key: value for key, value in settings.redacted().items() if value is not None}
    env |= {key: value for key, value in os.environ.items() if key.startswith("MMS_")}
    resolved = _interpolate(raw, env)
    assert isinstance(resolved, dict)
    return resolved


def load_config(
    path: Path, overrides: list[str] | None = None, settings: Settings | None = None
) -> MmsConfig:
    """Load and validate a run config."""
    return MmsConfig.model_validate(resolve_dict(path, overrides, settings))


def load_synth_config(path: Path, overrides: list[str] | None = None) -> SynthConfig:
    """Load and validate a synthetic-world config (not merged with base.yaml)."""
    raw = apply_overrides(load_yaml(path), overrides or [])
    return SynthConfig.model_validate(raw)
