"""The decoded event cache replays exactly the original stream (R-06)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from mmsim.data.cache import build_cache, cache_is_fresh, iter_cache
from mmsim.data.loaders import day_files, iter_row_groups, list_days
from mmsim.data.replay import BookBatchEvent, ReplayStats, replay


def events(root: Path):  # type: ignore[no-untyped-def]
    (day,) = list_days(root, "book_l2", "synthetic", "SYNUSDT")
    b = day_files(root, "book_l2", "synthetic", "SYNUSDT", day)
    t = day_files(root, "trades", "synthetic", "SYNUSDT", day)
    return b + t, replay(iter_row_groups(b), iter_row_groups(t), 0.1, 0.001, ReplayStats())


def key_of(e: object) -> tuple[object, ...]:
    if isinstance(e, BookBatchEvent):
        return ("b", e.ts_match, e.ts_exchange, e.ts_recv, e.batch_id, e.seq, e.is_snapshot,
                e.side.tolist(), e.price_ticks.tolist(), e.qty_lots.tolist())  # fmt: skip
    return ("t", e.ts_match, e.ts_exchange, e.ts_recv, e.price_ticks, e.qty_lots, e.side)  # type: ignore[attr-defined]


def test_cache_round_trip_and_freshness(mini_day_root: Path, tmp_path: Path) -> None:
    files, stream = events(mini_day_root)
    original = [key_of(e) for e in stream]
    cache = tmp_path / "cache"
    key = {"tick": 0.1}
    assert not cache_is_fresh(cache, files, key)
    _, again = events(mini_day_root)
    meta = build_cache(cache, again, files, key, ReplayStats())
    assert meta["trades"] + meta["book_batches"] == len(original)
    assert [key_of(e) for e in iter_cache(cache)] == original
    assert cache_is_fresh(cache, files, key)
    assert not cache_is_fresh(cache, files, {"tick": 0.01})
    # A source file re-closed with a new size or time invalidates the cache.
    st = files[0].stat()
    os.utime(files[0], ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert not cache_is_fresh(cache, files, key)
    assert np.load(cache / "events.npz")["book_px"].dtype == np.int32
