"""Decoded event cache per symbol-day (spec risk R-06: "cache decoded events per day").

Decoding a recorded Binance day (tens of millions of book rows in hourly zstd Parquet) costs
far more than simulating it. The merged event stream of a day, exactly as
:func:`mmsim.data.replay.replay` yields it, is therefore written once to NumPy arrays under
``$DATA_ROOT/interim/events`` and replayed from there.

The cache records the size and modification time of every source file; if a recorder file
changed after the cache was built (files can be re-closed later), the cache is rebuilt.
Prices and quantities are stored as int32 when they fit, else int64.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from mmsim.data.replay import BookBatchEvent, MarketEvent, ReplayStats, TradeRowEvent

__all__ = ["CACHE_VERSION", "build_cache", "cache_is_fresh", "iter_cache", "source_manifest"]

CACHE_VERSION = 1
I64 = np.int64


def source_manifest(files: Sequence[Path]) -> list[dict[str, Any]]:
    """Name, size and mtime of each source file."""
    out = []
    for f in files:
        st = f.stat()
        out.append({"path": str(f), "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    return out


def _small(a: npt.NDArray[np.int64]) -> npt.NDArray[Any]:
    info = np.iinfo(np.int32)
    if a.size == 0 or (a.min() >= info.min and a.max() <= info.max):
        return a.astype(np.int32)
    return a


def cache_is_fresh(directory: Path, files: Sequence[Path], key: dict[str, Any]) -> bool:
    """True if ``directory`` holds a cache built from exactly these files and key."""
    meta_path = directory / "meta.json"
    if not meta_path.exists() or not (directory / "events.npz").exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return bool(
        meta.get("version") == CACHE_VERSION
        and meta.get("key") == key
        and meta.get("sources") == source_manifest(files)
    )


def build_cache(
    directory: Path,
    events: Iterator[MarketEvent],
    files: Sequence[Path],
    key: dict[str, Any],
    stats: ReplayStats | None = None,
) -> dict[str, Any]:
    """Materialise ``events`` into ``directory`` (written atomically: tmp then rename)."""
    kinds: list[int] = []
    b_ts: list[tuple[int, int, int, int, int, int]] = []
    b_side: list[npt.NDArray[np.int8]] = []
    b_px: list[npt.NDArray[np.int64]] = []
    b_qty: list[npt.NDArray[np.int64]] = []
    t_rows: list[tuple[int, int, int, int, int, int]] = []
    for ev in events:
        if isinstance(ev, BookBatchEvent):
            kinds.append(1)
            b_ts.append((ev.ts_match, ev.ts_exchange, ev.ts_recv, ev.batch_id, ev.seq,
                         int(ev.is_snapshot)))  # fmt: skip
            b_side.append(ev.side)
            b_px.append(ev.price_ticks)
            b_qty.append(ev.qty_lots)
        else:
            kinds.append(0)
            t_rows.append((ev.ts_match, ev.ts_exchange, ev.ts_recv, ev.price_ticks, ev.qty_lots,
                           ev.side))  # fmt: skip
    sizes = np.array([s.size for s in b_side], dtype=I64)
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(I64)
    arrays: dict[str, npt.NDArray[Any]] = {
        "kind": np.array(kinds, dtype=np.int8),
        "book_meta": np.array(b_ts, dtype=I64).reshape(-1, 6),
        "book_offsets": offsets,
        "book_side": np.concatenate(b_side).astype(np.int8) if b_side else np.zeros(0, np.int8),
        "book_px": _small(np.concatenate(b_px).astype(I64)) if b_px else np.zeros(0, np.int32),
        "book_qty": _small(np.concatenate(b_qty).astype(I64)) if b_qty else np.zeros(0, np.int32),
        "trades": np.array(t_rows, dtype=I64).reshape(-1, 6),
    }
    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / "events.tmp.npz"
    np.savez(tmp, **arrays)  # type: ignore[arg-type]  # numpy stub types **kwds as bool
    tmp.replace(directory / "events.npz")
    meta = {
        "version": CACHE_VERSION,
        "key": key,
        "sources": source_manifest(files),
        "book_batches": len(b_side),
        "book_rows": int(offsets[-1]),
        "trades": len(t_rows),
        "replay_stats": None if stats is None else {
            k: getattr(stats, k) for k in ("book_batches", "trades", "side0_trades",
                                           "dropped_before_snapshot", "exchange_time_clamped",
                                           "off_grid_book_rows")
        },
    }  # fmt: skip
    (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def iter_cache(directory: Path) -> Iterator[MarketEvent]:
    """Replay a cached day: the same events, in the same order, as the original replay."""
    with np.load(directory / "events.npz") as z:
        kind = z["kind"]
        meta = z["book_meta"]
        offsets = z["book_offsets"]
        side = z["book_side"]
        px = z["book_px"]  # int32 when it fits: no int64 copies of a whole day (NFR-12)
        qty = z["book_qty"]
        trades = z["trades"]
    meta_l = meta.tolist()
    trades_l = trades.tolist()
    off = offsets.tolist()
    bi = ti = 0
    for k in kind.tolist():
        if k:
            m = meta_l[bi]
            lo, hi = off[bi], off[bi + 1]
            yield BookBatchEvent(
                ts_match=m[0], ts_exchange=m[1], ts_recv=m[2], batch_id=m[3], seq=m[4],
                is_snapshot=bool(m[5]), side=side[lo:hi], price_ticks=px[lo:hi],
                qty_lots=qty[lo:hi],
            )  # fmt: skip
            bi += 1
        else:
            t = trades_l[ti]
            yield TradeRowEvent(ts_match=t[0], ts_exchange=t[1], ts_recv=t[2], trade_id="",
                                price_ticks=t[3], qty_lots=t[4], side=t[5])  # fmt: skip
            ti += 1
