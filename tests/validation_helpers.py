"""Helpers for the validation tests."""

from __future__ import annotations

from mmsim.validate import Check


def failures(checks: list[Check]) -> list[str]:
    """Readable descriptions of the failed checks."""
    return [
        f"{c.check} {c.case} {c.metric}: {c.value:.6g} vs {c.target:.6g}"
        for c in checks
        if not c.passed
    ]
