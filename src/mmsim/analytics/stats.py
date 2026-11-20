"""Day-block bootstrap confidence intervals.

Intraday observations are dependent, so whole UTC days are resampled with replacement
(spec: ``bootstrap: {block: day}``). The statistic is computed on the rows of the sampled
days; the interval is the percentile interval of ``n`` resamples.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

__all__ = ["day_block_bootstrap_ci", "weighted_mean"]

FloatArray = npt.NDArray[np.float64]


def weighted_mean(rows: FloatArray) -> float:
    """Mean of column 0 weighted by column 1 (volume-weighted markouts)."""
    w = rows[:, 1]
    total = float(w.sum())
    return float((rows[:, 0] * w).sum() / total) if total > 0 else float("nan")


def day_block_bootstrap_ci(
    values_by_day: Sequence[npt.ArrayLike],
    stat: Callable[[FloatArray], float],
    n: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile CI of ``stat`` over day-resampled data; ``(nan, nan)`` with < 2 days."""
    days = [np.asarray(v, dtype=np.float64) for v in values_by_day]
    days = [d for d in days if d.size]
    if len(days) < 2:
        return float("nan"), float("nan")
    rng = np.random.Generator(np.random.PCG64(seed))
    picks = rng.integers(0, len(days), size=(n, len(days)))
    stats = np.empty(n)
    for i, row in enumerate(picks):
        stats[i] = stat(np.concatenate([days[j] for j in row]))
    finite = stats[np.isfinite(stats)]
    if finite.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
