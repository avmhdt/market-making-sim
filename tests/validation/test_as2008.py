"""AC-W-02: Avellaneda-Stoikov (2008) Tables 1-3, reference MC and event-driven fallback."""

from __future__ import annotations

import pytest

from mmsim.settings import Settings
from mmsim.validate import ValidationPaths, check_as2008, load_exceptions, unexplained_failures


@pytest.mark.slow
def test_as2008_tables(vpaths: ValidationPaths, vsettings: Settings) -> None:
    checks, _ = check_as2008(vpaths, fast=False, settings=vsettings)
    exc = load_exceptions(vpaths.repo_root)
    assert unexplained_failures([c for c in checks if c.check == "AC-W-02"], exc) == []
