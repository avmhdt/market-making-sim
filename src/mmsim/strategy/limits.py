"""Hard inventory limits counting resting orders (FR-11).

The worst case is that every resting order on a side fills, including orders whose cancel
is still in flight, so the limits are ``position + resting bids <= Q_max`` and
``position - resting asks >= -Q_max``. The side that would breach is not quoted (or quoted
smaller).
"""

from __future__ import annotations

__all__ = ["ask_room", "bid_room"]


def bid_room(position_lots: int, resting_bid_lots: int, q_max_lots: int) -> int:
    """Lots that may still be bid without breaching ``+Q_max`` in the worst case."""
    return max(q_max_lots - position_lots - resting_bid_lots, 0)


def ask_room(position_lots: int, resting_ask_lots: int, q_max_lots: int) -> int:
    """Lots that may still be offered without breaching ``-Q_max`` in the worst case."""
    return max(q_max_lots + position_lots - resting_ask_lots, 0)
