"""AC-W-03: GLFT exact strategy, Monte Carlo CARA utility against the closed form."""

from __future__ import annotations

import pytest

from mmsim.validate import ValidationPaths, check_asymptotic, check_glft_ce
from validation_helpers import failures


@pytest.mark.slow
def test_glft_certainty_equivalent(vpaths: ValidationPaths) -> None:
    checks, _ = check_glft_ce(vpaths, fast=False)
    assert failures(checks) == []


def test_asymptotic_rows() -> None:
    assert failures(check_asymptotic()) == []
