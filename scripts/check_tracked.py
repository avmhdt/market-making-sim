"""Licensed-data guard (NFR-07), run by pre-commit on staged files and by CI on tracked ones.

Rejects anything under data/; .parquet, .csv.gz and .zip files outside tests/fixtures/;
.csv files outside tests/fixtures/, results/ and reports/tables/; and any file over 500 kB
except uv.lock.
"""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

MAX_BYTES = 500_000
SIZE_EXEMPT = frozenset({"uv.lock"})


def _starts(path: PurePosixPath, *prefix: str) -> bool:
    return path.parts[: len(prefix)] == prefix


def problems(names: list[str], root: Path = Path()) -> list[str]:
    """One message per rejected path; paths are relative to ``root``."""
    found: list[str] = []
    for name in names:
        path = PurePosixPath(name.replace("\\", "/"))
        lower = path.name.lower()
        in_fixtures = _starts(path, "tests", "fixtures")
        if _starts(path, "data"):
            found.append(f"{name}: nothing under data/ may be committed")
        elif lower.endswith((".parquet", ".csv.gz", ".zip")) and not in_fixtures:
            found.append(f"{name}: {path.suffix} files are allowed only under tests/fixtures/")
        elif lower.endswith(".csv") and not (
            in_fixtures or _starts(path, "results") or _starts(path, "reports", "tables")
        ):
            found.append(f"{name}: CSV files are allowed only in fixtures, results and tables")
        local = root / name
        if name not in SIZE_EXEMPT and local.is_file() and local.stat().st_size > MAX_BYTES:
            found.append(f"{name}: over 500 kB")
    return found


def main(argv: list[str]) -> int:
    """Print each problem; exit 1 if there is any."""
    found = problems(argv)
    for message in found:
        print(message, file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
