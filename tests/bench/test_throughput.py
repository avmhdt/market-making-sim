"""NFR-01 throughput: fallback replay with GLFT on a 1M-event synthetic stream (AC-W-15).

Local only (`slow`); `scripts/benchmark.py` writes reports/tables/benchmark.csv.
"""

from __future__ import annotations

import time

import pytest

from conftest import REPO
from mmsim.config import SimConfig, load_synth_config
from mmsim.data.replay import replay
from mmsim.data.synth import generate_path
from mmsim.runner import FixedParamsProvider, Session, run_session
from mmsim.sim.registry import make_simulator
from mmsim.strategy.analytic import AnalyticPolicy

TARGET_EVENTS_PER_S = 50_000


def benchmark_stream(events: int = 1_000_000) -> dict[str, float]:
    """Run GLFT through the fallback on a synthetic world of about ``events`` market events."""
    cfg = load_synth_config(REPO / "tests" / "fixtures" / "mini_day" / "config.yaml",
                            ["horizon=50000.0", "dt=1.0", "seed=1", "mid.sigma=0.35",
                             "market_orders.A=4.0"])  # fmt: skip
    world = generate_path(cfg, 0)
    inst = world.instrument
    sim_cfg = SimConfig.model_validate({
        "queue_model": {"kind": "prob"},
        "latency": {"order": {"kind": "constant", "ns": 5_000_000},
                    "ack": {"kind": "constant", "ns": 5_000_000}},
    })  # fmt: skip

    def source():  # type: ignore[no-untyped-def]
        return replay(world.book_l2.to_arrow().to_batches(max_chunksize=200_000),
                      world.trades.to_arrow().to_batches(max_chunksize=200_000),
                      inst.tick_size, inst.lot_size)  # fmt: skip

    mids: list[tuple[int, int, int]] = []
    sim = make_simulator(sim_cfg, source, inst, on_mid=lambda t, m, s: mids.append((t, m, s)))
    session = Session(day=None, start_ns=world.start_ns, end_ns=world.end_ns, instrument=inst,  # type: ignore[arg-type]
                      events=source, median_feed_ns=0, seed=1)  # fmt: skip

    def policy(lots: int) -> AnalyticPolicy:
        return AnalyticPolicy("glft", 0.05, inst.tick_size, lots, 10, 3600.0,
                              lot_size=inst.lot_size)  # fmt: skip

    t0 = time.perf_counter()
    log = run_session(sim, session, policy, FixedParamsProvider(2.0, 2.0, 0.2), q_max_units=10,
                      quote_lots=5, decision_interval_ns=100_000_000, requote_threshold_ticks=1,
                      mids=mids)  # fmt: skip
    wall = time.perf_counter() - t0
    n = log.sim_stats["market_events"]
    return {"market_events": n, "wall_s": wall, "events_per_s": n / wall, "fills": len(log.fills)}


@pytest.mark.slow
def test_throughput() -> None:
    result = benchmark_stream()
    assert result["market_events"] > 900_000
    # NFR-01 is "meet or report": the value is recorded, the assertion only guards regressions.
    assert result["events_per_s"] > 5_000
