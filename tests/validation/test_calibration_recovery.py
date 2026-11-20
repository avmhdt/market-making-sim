"""AC-W-10: calibration recovers known A, k and sigma."""

from __future__ import annotations

from mmsim.validate import ValidationPaths, check_calibration
from validation_helpers import failures


def test_calibration_recovery(vpaths: ValidationPaths) -> None:
    checks = check_calibration(vpaths)
    assert len(checks) == 6
    assert failures(checks) == []
