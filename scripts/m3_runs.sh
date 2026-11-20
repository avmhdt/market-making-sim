#!/usr/bin/env bash
# M3 first decomposition: GLFT then A-S on the tuning days, one process per symbol.
# Run in WSL from the repo, with the venv on ext4 (docs/decisions.md M3-09):
#   UV_PROJECT_ENVIRONMENT=$HOME/venvs/market-making-sim bash scripts/m3_runs.sh
# Shared-PC limits: 3 processes x 2 Polars threads, nice 10, ionice idle.
set -u
cd "$(dirname "$0")/.."
PY=${UV_PROJECT_ENVIRONMENT:?set UV_PROJECT_ENVIRONMENT}/bin/mms
LOGS=${DATA_ROOT:-$HOME/data/market-making-sim}/logs/m3
mkdir -p "$LOGS"
export POLARS_MAX_THREADS=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
for sym in BTCUSDT ETHUSDT SOLUSDT; do
  (
    for cfg in glft as; do
      echo "$(date -u +%FT%TZ) start $cfg $sym" >> "$LOGS/$sym.log"
      nice -n 10 ionice -c3 "$PY" run --config "configs/strategies/$cfg.yaml" \
        --set "data.symbols=[$sym]" >> "$LOGS/$sym.log" 2>&1
      rc=$?; echo "$(date -u +%FT%TZ) exit $rc $cfg $sym" >> "$LOGS/$sym.log"
    done
  ) &
done
wait
echo "$(date -u +%FT%TZ) all done" >> "$LOGS/done.log"
