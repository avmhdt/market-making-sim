"""AC-W-01: fill intensity of resting quotes matches A exp(-k delta)."""

from __future__ import annotations

import pytest

from mmsim.validate import ValidationPaths, check_intensity
from validation_helpers import failures


@pytest.mark.slow
def test_intensity_full(vpaths: ValidationPaths) -> None:
    checks, _ = check_intensity(vpaths, fast=False)
    assert failures(checks) == []
    assert "< 1e5" not in checks[-2].note
