"""Signal model artifact loader (FR-14, DQ-06; contract in docs/signal-artifact.md).

An artifact version folder holds ``model.txt`` (LightGBM text booster), ``feature_spec.json``,
``metadata.json`` and ``parity_sample.csv``. :meth:`SignalModel.load` checks, in order:

1. all four files exist;
2. ``feature_spec.json`` parses into the contract's structure and its ``spec_version`` major
   is 1 (a vendored ``docs/feature_spec.schema.json``, when present, is applied too);
3. the SHA-256 of ``model.txt``, ``feature_spec.json`` and ``parity_sample.csv`` equal the
   values in ``metadata.json``;
4. the installed LightGBM ``major.minor`` equals ``metadata.lightgbm_version``'s;
5. every feature ``kind`` is implemented by the online engine;
6. (DQ-06) the evaluation start is strictly after ``valid_end``, and the symbol is covered.

Any failure raises :class:`ArtifactError` with the reason.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, ValidationError

__all__ = [
    "SPEC_MAJOR",
    "ArtifactError",
    "FeatureDef",
    "FeatureSpec",
    "Metadata",
    "Normalization",
    "SignalModel",
    "sha256_file",
]

SPEC_MAJOR = 1
FILES = ("model.txt", "feature_spec.json", "metadata.json", "parity_sample.csv")

FloatArray = npt.NDArray[np.float64]


class ArtifactError(ValueError):
    """The artifact breaks the contract or is not eligible for this use."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)


class Normalization(_Model):
    """Per-feature normalisation applied before ``predict``."""

    method: Literal["none", "zscore", "zscore_clip"]
    mean: float = 0.0
    std: float = 1.0
    clip: float | None = None


class FeatureDef(_Model):
    """One model input, precise enough to recompute from the canonical stream."""

    name: str
    kind: str
    dtype: str = "float64"
    levels: list[int] | None = None
    window: int | None = None
    window_unit: Literal["ms", "events"] | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    definition: str = ""
    normalization: Normalization


class Horizon(_Model):
    """Prediction horizon."""

    kind: Literal["clock", "event"]
    value: int
    unit: Literal["ms", "events"]


class Target(_Model):
    """What the model predicts (ticks over a horizon)."""

    name: str
    unit: str
    horizon: Horizon
    latency_ms: int = 0


class FeatureSpec(_Model):
    """``feature_spec.json``."""

    spec_version: str
    model_name: str
    model_version: str
    venue: str
    symbols: list[str]
    tick_size: float
    lot_size: float
    book_depth: int
    timestamp: Literal["ts_recv", "ts_exchange"]
    clock: Literal["time", "event"]
    sample_grid_ms: int | None
    target: Target
    features: list[FeatureDef]
    compatibility: str = ""


class Metadata(_Model):
    """``metadata.json``."""

    git_sha: str
    created_utc: str
    lightgbm_version: str
    model_sha256: str
    files_sha256: dict[str, str]
    fold: int | str
    train_start: date
    train_end: date
    valid_start: date
    valid_end: date
    train_days: int | list[Any]
    validation_days: int | list[Any]
    target_unit: str
    metrics: dict[str, Any]
    test_days: list[date] = []
    provisional: bool = False
    """A pre-release for integration only: no published result may use it."""


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _major_minor(version: str) -> tuple[int, int]:
    parts = version.split(".")
    return int(parts[0]), int(parts[1])


def _schema_check(spec_raw: dict[str, Any], schema_path: Path | None) -> None:
    if schema_path is None or not schema_path.exists():
        return
    import jsonschema

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        jsonschema.validate(spec_raw, schema)
    except jsonschema.ValidationError as exc:
        raise ArtifactError(f"feature_spec.json fails the JSON Schema: {exc.message}") from exc


class SignalModel:
    """A loaded, verified artifact: ``predict`` takes raw feature vectors."""

    def __init__(self, path: Path, spec: FeatureSpec, metadata: Metadata, booster: Any) -> None:
        self.path = path
        self.spec = spec
        self.metadata = metadata
        self.booster = booster
        h = spec.target.horizon
        self.horizon_ns = h.value * 1_000_000 if h.kind == "clock" else 0
        norms = [f.normalization for f in spec.features]
        self._mean = np.array([n.mean if n.method != "none" else 0.0 for n in norms])
        self._std = np.array([n.std if n.method != "none" else 1.0 for n in norms])
        self._clip = np.array(
            [n.clip if n.method == "zscore_clip" and n.clip is not None else np.inf for n in norms]
        )

    @property
    def feature_names(self) -> list[str]:
        """Model input order."""
        return [f.name for f in self.spec.features]

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        eval_start: date | None = None,
        symbol: str | None = None,
        schema_path: Path | None = None,
        supported_kinds: frozenset[str] | None = None,
    ) -> SignalModel:
        """Load and verify ``path`` (a ``<model_name>/<model_version>`` folder)."""
        missing = [f for f in FILES if not (path / f).is_file()]
        if missing:
            raise ArtifactError(f"{path}: missing {', '.join(missing)}")
        spec_raw = json.loads((path / "feature_spec.json").read_text(encoding="utf-8"))
        meta_raw = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        try:
            spec = FeatureSpec.model_validate(spec_raw)
            meta = Metadata.model_validate(meta_raw)
        except ValidationError as exc:
            raise ArtifactError(f"{path}: contract fields: {exc}") from exc
        if int(spec.spec_version.split(".")[0]) != SPEC_MAJOR:
            raise ArtifactError(f"spec_version {spec.spec_version}: need major {SPEC_MAJOR}")
        _schema_check(spec_raw, schema_path)
        if (spec.model_name, spec.model_version) != (path.parent.name, path.name):
            raise ArtifactError(
                f"{path}: folder is not <model_name>/<model_version> "
                f"({spec.model_name}/{spec.model_version})"
            )
        if meta.test_days and min(meta.test_days) <= meta.valid_end:
            raise ArtifactError(f"test day {min(meta.test_days)} is not after valid_end")
        if sha256_file(path / "model.txt") != meta.model_sha256:
            raise ArtifactError("model.txt SHA-256 does not match metadata.model_sha256")
        for name in ("feature_spec.json", "parity_sample.csv"):
            want = meta.files_sha256.get(name)
            if want is None or sha256_file(path / name) != want:
                raise ArtifactError(f"{name} SHA-256 does not match metadata.files_sha256")
        import lightgbm

        if _major_minor(lightgbm.__version__) != _major_minor(meta.lightgbm_version):
            raise ArtifactError(
                f"LightGBM {lightgbm.__version__} installed, artifact built with "
                f"{meta.lightgbm_version} (major.minor must match)"
            )
        if supported_kinds is None:
            from mmsim.signal.online import SUPPORTED_KINDS

            supported_kinds = SUPPORTED_KINDS
        unknown = sorted({f.kind for f in spec.features} - supported_kinds)
        if unknown:
            raise ArtifactError(f"feature kinds not implemented online: {', '.join(unknown)}")
        if eval_start is not None and eval_start <= meta.valid_end:
            raise ArtifactError(
                f"DQ-06: evaluation from {eval_start} is not after valid_end {meta.valid_end}"
            )
        if symbol is not None and symbol not in spec.symbols:
            raise ArtifactError(f"symbol {symbol} not in the artifact's {spec.symbols}")
        booster = lightgbm.Booster(model_file=str(path / "model.txt"))
        names = booster.feature_name()
        expected = [f.name for f in spec.features]
        unnamed = all(n == f"Column_{i}" for i, n in enumerate(names))
        if len(names) != len(expected) or (not unnamed and names != expected):
            raise ArtifactError("model.txt features do not match feature_spec.json order")
        return cls(path, spec, meta, booster)

    def normalize(self, x_raw: npt.ArrayLike) -> FloatArray:
        """Apply each feature's normalisation (z-score, clipped where specified)."""
        x = (np.asarray(x_raw, dtype=np.float64) - self._mean) / self._std
        return np.asarray(np.clip(x, -self._clip, self._clip), dtype=np.float64)

    def predict_normalized(self, x_norm: npt.ArrayLike) -> FloatArray:
        """Booster output for already-normalised rows (2-D) or one row (1-D)."""
        x = np.atleast_2d(np.asarray(x_norm, dtype=np.float64))
        return np.asarray(self.booster.predict(x), dtype=np.float64)

    def predict(self, x_raw: npt.ArrayLike) -> float:
        """Prediction (target units, ticks) for one raw feature vector."""
        return float(self.predict_normalized(self.normalize(x_raw))[0])
