"""Environment settings loaded from ``.env`` with pydantic-settings (spec section 3).

The keys are exactly those of ``.env.example``. ``~`` is expanded; empty optional values
become ``None``. Secrets (``TARDIS_API_KEY``) are :class:`pydantic.SecretStr` and never
printed.
"""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["DataLocationError", "Settings", "check_data_location", "get_settings"]


class Settings(BaseSettings):
    """Paths, simulator backend and runtime knobs from the environment and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    data_root: Path = Field(default=Path("~/data/market-making-sim"))
    mms_data_root: Path | None = None
    mms_cla_data_root: Path = Field(default=Path("~/data/crypto-lob-alpha"))
    mms_signal_model_dir: Path = Field(default=Path("../crypto-lob-alpha/artifacts/models"))
    mms_results_dir: Path = Field(default=Path("./results"))
    mms_reports_dir: Path = Field(default=Path("./reports"))
    mms_sim_backend: Literal["fallback", "edbt"] = "fallback"
    mms_edbt_path: Path | None = None
    mms_log_level: Literal["DEBUG", "INFO", "WARNING"] = "INFO"
    mms_n_jobs: int = Field(default=4, ge=1)
    mms_torch_threads: int = Field(default=4, ge=1)
    tardis_api_key: SecretStr | None = None

    @field_validator("mms_data_root", "mms_edbt_path", "tardis_api_key", mode="before")
    @classmethod
    def _empty_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "data_root",
        "mms_data_root",
        "mms_cla_data_root",
        "mms_signal_model_dir",
        "mms_results_dir",
        "mms_reports_dir",
        "mms_edbt_path",
    )
    @classmethod
    def _expand_user(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value is not None else None

    @property
    def project_data_root(self) -> Path:
        """Root of ``raw/``, ``interim/``, ``processed/``: ``MMS_DATA_ROOT`` or ``DATA_ROOT``."""
        return self.mms_data_root if self.mms_data_root is not None else self.data_root

    @property
    def processed_root(self) -> Path:
        """Canonical datasets this project writes (synthetic, LOBSTER, Tardis)."""
        return self.project_data_root / "processed"

    @property
    def raw_root(self) -> Path:
        """Raw downloads (Tardis, LOBSTER, funding)."""
        return self.project_data_root / "raw"

    def redacted(self) -> dict[str, str | None]:
        """All settings as strings with secrets replaced, for ``config.resolved.yaml``."""
        out: dict[str, str | None] = {}
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, SecretStr):
                out[name.upper()] = "**redacted**"
            else:
                out[name.upper()] = None if value is None else str(value)
        return out


class DataLocationError(ValueError):
    """A data path points at a location the spec forbids (section 7(d))."""


def check_data_location(path: Path, *, windows: bool | None = None) -> Path:
    """Refuse data paths under ``/mnt/*`` (slow 9P mounts) or on the Windows ``C:`` drive.

    Data and run outputs live in the WSL ext4 filesystem under ``DATA_ROOT``. On native
    Windows ``~`` expands to ``C:``, so writing commands must be given an explicit target.
    """
    windows = (os.name == "nt") if windows is None else windows
    text = path.as_posix()
    if text.startswith("/mnt/"):
        raise DataLocationError(f"{path}: data must not live under /mnt/* (use WSL ext4)")
    if windows and PureWindowsPath(str(path)).drive.upper() == "C:":
        raise DataLocationError(f"{path}: data must not live on C: (pass --out or run in WSL)")
    return path


def get_settings() -> Settings:
    """Load settings from the environment and ``.env`` in the working directory."""
    return Settings()
