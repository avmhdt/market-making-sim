"""Instruments and effective-dated fee tiers (FR-08, DQ-08).

Tick and lot sizes come from ``configs/instruments.yaml``; maker and taker fees per tier
come from ``configs/fees.yaml`` by effective date. A day with no fee row is an error, so a
run can never silently use the wrong fees.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, cast

import yaml

from mmsim.sim.types import InstrumentSpec

__all__ = [
    "FeeCoverageError",
    "FeeRow",
    "InstrumentRow",
    "load_fee_rows",
    "load_instrument_rows",
    "resolve_instrument",
]


class FeeCoverageError(LookupError):
    """No fee row covers the requested venue, tier and day (DQ-08)."""


@dataclass(frozen=True, slots=True)
class InstrumentRow:
    """Static contract facts of one venue symbol."""

    venue: str
    symbol: str
    tick_size: float
    lot_size: float
    contract_type: Literal["perp", "spot", "equity"]


@dataclass(frozen=True, slots=True)
class FeeRow:
    """Maker and taker fees of one tier, valid from ``effective_from`` to ``effective_to``."""

    venue: str
    tier: str
    maker_bps: float
    taker_bps: float
    effective_from: date
    effective_to: date | None
    hypothetical: bool

    def covers(self, day: date) -> bool:
        """True if this row applies on ``day`` (``effective_to`` inclusive)."""
        return self.effective_from <= day and (self.effective_to is None or day <= self.effective_to)


def _as_date(value: object) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def load_instrument_rows(path: Path) -> list[InstrumentRow]:
    """Read ``instruments.yaml``."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    rows: list[InstrumentRow] = []
    for item in raw.get("instruments", []):
        contract = str(item["contract_type"])
        if contract not in ("perp", "spot", "equity"):
            raise ValueError(f"{path}: contract_type {contract!r} not perp|spot|equity")
        rows.append(
            InstrumentRow(
                venue=str(item["venue"]),
                symbol=str(item["symbol"]),
                tick_size=float(item["tick_size"]),
                lot_size=float(item["lot_size"]),
                contract_type=cast(Literal["perp", "spot", "equity"], contract),
            )
        )
    return rows


def load_fee_rows(path: Path) -> list[FeeRow]:
    """Read ``fees.yaml``."""
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    rows: list[FeeRow] = []
    for item in raw.get("schedules", []):
        end = item.get("effective_to")
        rows.append(
            FeeRow(
                venue=str(item["venue"]),
                tier=str(item["tier"]),
                maker_bps=float(item["maker_bps"]),
                taker_bps=float(item["taker_bps"]),
                effective_from=_as_date(item["effective_from"]),
                effective_to=None if end is None else _as_date(end),
                hypothetical=bool(item.get("hypothetical", False)),
            )
        )
    return rows


def resolve_instrument(
    instruments_path: Path,
    fees_path: Path,
    venue: str,
    symbol: str,
    day: date,
    tier: str,
) -> InstrumentSpec:
    """The instrument with the fee tier in force on ``day``.

    Raises:
        KeyError: If the symbol is not in ``instruments.yaml``.
        FeeCoverageError: If no single fee row covers the venue, tier and day.
    """
    matches = [
        row
        for row in load_instrument_rows(instruments_path)
        if row.venue == venue and row.symbol == symbol
    ]
    if len(matches) != 1:
        raise KeyError(f"{venue}/{symbol}: {len(matches)} rows in {instruments_path}")
    inst = matches[0]
    fees = [
        row
        for row in load_fee_rows(fees_path)
        if row.venue == venue and row.tier == tier and row.covers(day)
    ]
    if len(fees) != 1:
        raise FeeCoverageError(
            f"{venue} tier {tier!r} on {day}: {len(fees)} fee rows in {fees_path} (need 1)"
        )
    fee = fees[0]
    return InstrumentSpec(
        venue=inst.venue,
        symbol=inst.symbol,
        tick_size=inst.tick_size,
        lot_size=inst.lot_size,
        contract_type=inst.contract_type,
        maker_bps=fee.maker_bps,
        taker_bps=fee.taker_bps,
        fee_tier=tier,
    )
