"""An L2 book in integer ticks and lots, with sorted price ladders per side."""

from __future__ import annotations

import bisect

import numpy as np
import numpy.typing as npt

from mmsim.sim.types import BookView

__all__ = ["L2Book"]


class L2Book:
    """Aggregated depth by price. Bid side is +1, ask side is -1."""

    __slots__ = ("_ask_px", "_asks", "_bid_px", "_bids")

    def __init__(self) -> None:
        self._bids: dict[int, int] = {}
        self._asks: dict[int, int] = {}
        self._bid_px: list[int] = []  # ascending
        self._ask_px: list[int] = []  # ascending

    def clear(self) -> None:
        """Remove every level."""
        self._bids.clear()
        self._asks.clear()
        self._bid_px.clear()
        self._ask_px.clear()

    def size(self, side: int, price: int) -> int:
        """Displayed lots at ``price`` on ``side`` (0 if absent)."""
        return (self._bids if side == 1 else self._asks).get(price, 0)

    def set_level(self, side: int, price: int, lots: int) -> int:
        """Set the size at a level (0 removes it); returns the previous size."""
        levels, ladder = (self._bids, self._bid_px) if side == 1 else (self._asks, self._ask_px)
        old = levels.get(price, 0)
        if lots > 0:
            if old == 0:
                bisect.insort(ladder, price)
            levels[price] = lots
        elif old:
            del levels[price]
            index = bisect.bisect_left(ladder, price)
            del ladder[index]
        return old

    def apply(
        self,
        sides: npt.NDArray[np.int8],
        prices: npt.NDArray[np.int64],
        lots: npt.NDArray[np.int64],
        snapshot: bool,
    ) -> list[tuple[int, int, int, int]]:
        """Apply one batch atomically; returns ``(side, price, old, new)`` for changed levels.

        A snapshot replaces the book: levels absent from it are reported as removed.
        """
        changes: list[tuple[int, int, int, int]] = []
        if snapshot:
            before = {(1, p): s for p, s in self._bids.items()}
            before |= {(-1, p): s for p, s in self._asks.items()}
            self.clear()
            for side, price, size in zip(
                sides.tolist(), prices.tolist(), lots.tolist(), strict=True
            ):
                self.set_level(side, price, size)
            after = {(1, p): s for p, s in self._bids.items()}
            after |= {(-1, p): s for p, s in self._asks.items()}
            for key in before.keys() | after.keys():
                old, new = before.get(key, 0), after.get(key, 0)
                if old != new:
                    changes.append((key[0], key[1], old, new))
            return changes
        for side, price, size in zip(sides.tolist(), prices.tolist(), lots.tolist(), strict=True):
            old = self.set_level(side, price, size)
            if old != size:
                changes.append((side, price, old, size))
        return changes

    @property
    def best_bid(self) -> int | None:
        """Highest bid price, or ``None``."""
        return self._bid_px[-1] if self._bid_px else None

    @property
    def best_ask(self) -> int | None:
        """Lowest ask price, or ``None``."""
        return self._ask_px[0] if self._ask_px else None

    def mid_x2(self) -> int | None:
        """``best_bid + best_ask`` or ``None`` if a side is empty."""
        if not self._bid_px or not self._ask_px:
            return None
        return self._bid_px[-1] + self._ask_px[0]

    def crossed(self) -> bool:
        """True if the book is crossed or locked."""
        return bool(self._bid_px and self._ask_px and self._bid_px[-1] >= self._ask_px[0])

    def view(self, ts_local: int, depth: int) -> BookView:
        """Top ``depth`` levels as a :class:`BookView`."""
        bid_px = self._bid_px[-1 : -depth - 1 : -1] if depth else []
        ask_px = self._ask_px[:depth]
        return BookView(
            ts_local=ts_local,
            bid_ticks=tuple(bid_px),
            bid_lots=tuple(self._bids[p] for p in bid_px),
            ask_ticks=tuple(ask_px),
            ask_lots=tuple(self._asks[p] for p in ask_px),
        )
