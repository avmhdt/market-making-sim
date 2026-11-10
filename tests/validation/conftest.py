"""Shared paths for the validation checks (outputs go to tmp_path)."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import REPO
from mmsim.settings import Settings
from mmsim.validate import ValidationPaths


@pytest.fixture
def vpaths(tmp_path: Path) -> ValidationPaths:
    return ValidationPaths(
        repo_root=REPO,
        results_dir=tmp_path / "results",
        explog_path=tmp_path / "results" / "trial_log.csv",
        reports_dir=tmp_path / "reports",
    )


@pytest.fixture
def vsettings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]
