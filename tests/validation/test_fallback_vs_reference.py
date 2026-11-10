"""AC-W-04: fallback against the reference, and queue/latency direction."""

from __future__ import annotations

import pytest

from mmsim.settings import Settings
from mmsim.validate import ValidationPaths, check_as2008, check_queue_latency
from validation_helpers import failures


@pytest.mark.slow
def test_fallback_matches_reference(vpaths: ValidationPaths, vsettings: Settings) -> None:
    checks, _ = check_as2008(vpaths, fast=True, settings=vsettings)
    assert failures([c for c in checks if c.check == "AC-W-04"]) == []


def test_queue_and_latency_lower_fill_rate(vpaths: ValidationPaths, vsettings: Settings) -> None:
    assert failures(check_queue_latency(vpaths, vsettings)) == []
