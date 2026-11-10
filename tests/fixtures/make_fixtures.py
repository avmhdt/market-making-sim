"""Regenerate the committed synthetic fixtures (CSV only; Parquet is built in tmp_path).

Run from the repo root: ``uv run python tests/fixtures/make_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path

from mmsim.config import load_synth_config
from mmsim.data.synth import generate_path

HERE = Path(__file__).parent


def make_mini_day() -> None:
    """tests/fixtures/mini_day/{book_l2,trades}.csv from mini_day/config.yaml."""
    cfg = load_synth_config(HERE / "mini_day" / "config.yaml")
    world = generate_path(cfg, 0)
    world.book_l2.write_csv(HERE / "mini_day" / "book_l2.csv", line_terminator="\n")
    world.trades.write_csv(HERE / "mini_day" / "trades.csv", line_terminator="\n")


if __name__ == "__main__":
    make_mini_day()
