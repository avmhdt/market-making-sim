"""Market-making simulator: A-S and GLFT quoting on replayed order books."""

import os

# The shared PC allows at most 8 threads per heavy job, and Polars' default pool (one
# thread per core) stalls at low priority on a busy machine. Callers may override.
for _var in (
    "POLARS_MAX_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_var, "4")

__version__ = "0.1.0"
