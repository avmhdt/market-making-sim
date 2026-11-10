from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from conftest import FIXTURES, REPO, read_fixture
from mmsim.config import load_synth_config
from mmsim.data.loaders import iter_row_groups, list_days, read_day
from mmsim.data.replay import BookBatchEvent, replay
from mmsim.data.schemas import validate_book_l2, validate_trades
from mmsim.data.synth import generate, generate_path, write_world

AS2008 = REPO / "configs" / "synthetic" / "as2008.yaml"
MINI = FIXTURES / "mini_day" / "config.yaml"
INTENSITY = REPO / "configs" / "synthetic" / "intensity.yaml"


def test_bernoulli_step_fill_probability_is_min_one_lambda_dt() -> None:
    """A quote at delta fills with probability min(1, A exp(-k delta) dt) per step."""
    cfg = load_synth_config(AS2008, ["paths=30"])
    depths = []
    for world in generate(cfg):
        ticks = np.rint(world.trades["price"].to_numpy() / cfg.tick_size).astype(np.int64)
        ts = world.trades["ts_exchange"].to_numpy()
        mid = world.mid_ticks[np.searchsorted(world.mid_times_ns, ts, side="left") - 1]
        depths.append(world.trades["side"].to_numpy() * (ticks - mid) * cfg.tick_size)
    d = np.concatenate(depths)
    steps = 2 * cfg.paths * round(cfg.horizon / cfg.dt)
    assert d.size == steps  # one order per side per step
    a_dt = cfg.market_orders.A * cfg.dt
    for delta in (-0.5, -0.1, 0.3, 1.0):
        p = min(1.0, a_dt * math.exp(-cfg.market_orders.k * delta))
        freq = np.count_nonzero(d >= delta - 1e-9) / steps
        assert abs(freq - p) < 4 * math.sqrt(max(p * (1 - p), 1e-6) / steps) + 1e-3


def test_paths_are_deterministic_and_distinct() -> None:
    cfg = load_synth_config(AS2008)
    a, b = generate_path(cfg, 0), generate_path(cfg, 0)
    assert a.book_l2.equals(b.book_l2)
    assert a.trades.equals(b.trades)
    c = generate_path(cfg, 1)
    assert not np.array_equal(a.mid_ticks, c.mid_ticks)
    assert c.start_ns - a.start_ns == 86_400 * 10**9


def test_committed_fixture_matches_its_generator() -> None:
    world = generate_path(load_synth_config(MINI), 0)
    book, trades = read_fixture("book_l2"), read_fixture("trades")
    assert book.get_column("batch_id").to_list() == world.book_l2.get_column("batch_id").to_list()
    assert np.allclose(book.get_column("price").to_numpy(), world.book_l2["price"].to_numpy())
    assert trades.get_column("ts_recv").to_list() == world.trades["ts_recv"].to_list()


def test_binomial_mid_steps_and_horizon() -> None:
    cfg = load_synth_config(AS2008)
    world = generate_path(cfg, 0)
    steps = np.diff(world.mid_ticks)
    assert set(np.abs(steps).tolist()) == {1414}
    assert world.mid_ticks.size == 201
    assert world.mid_times_ns[-1] == world.end_ns
    assert validate_book_l2(world.book_l2) == []
    assert validate_trades(world.trades) == []


def test_trades_print_at_or_beyond_the_pre_trade_mid() -> None:
    world = generate_path(load_synth_config(INTENSITY, ["horizon=5.0"]), 3)
    ticks = np.rint(world.trades["price"].to_numpy() / 1e-4).astype(np.int64)
    ts = world.trades["ts_exchange"].to_numpy()
    mid = world.mid_ticks[np.searchsorted(world.mid_times_ns, ts, side="left") - 1]
    side = world.trades["side"].to_numpy()
    assert np.all(side * (ticks - mid) >= 0)
    assert np.all(ts > world.start_ns)
    assert np.all(ts < world.end_ns)


def test_book_diffs_rebuild_the_symmetric_book() -> None:
    cfg = load_synth_config(MINI)
    world = generate_path(cfg, 0)
    book: dict[tuple[int, int], int] = {}
    k = 0
    for event in replay(
        world.book_l2.to_arrow().to_batches(), world.trades.to_arrow().to_batches(), 0.1, 0.001
    ):
        if not isinstance(event, BookBatchEvent):
            continue
        if event.is_snapshot:
            book.clear()
        for side, price, lots in zip(
            event.side.tolist(), event.price_ticks.tolist(), event.qty_lots.tolist(), strict=True
        ):
            if lots:
                book[(side, price)] = lots
            else:
                book.pop((side, price), None)
        mid = int(world.mid_ticks[k])
        bids = sorted(p for s, p in book if s == 1)
        asks = sorted(p for s, p in book if s == -1)
        assert bids == list(range(mid - 5, mid))
        assert asks == list(range(mid + 1, mid + 6))
        k += 1
    assert k == world.mid_ticks.size


def test_tick_jump_rate_and_feed_latency() -> None:
    cfg = load_synth_config(MINI)
    world = generate_path(cfg, 0)
    jumps = world.mid_ticks.size - 1
    expected = (cfg.mid.sigma / cfg.tick_size) ** 2 * cfg.horizon
    assert abs(jumps - expected) < 5 * math.sqrt(expected)
    lat = world.trades["ts_recv"].to_numpy() - world.trades["ts_exchange"].to_numpy()
    assert lat.min() >= cfg.feed_latency_ns
    assert np.all(np.diff(world.trades["ts_recv"].to_numpy()) >= 0)


def test_market_order_intensity_matches_a_exp_minus_k_delta() -> None:
    """Quick sanity check; the full AC-W-01 test arrives with M2."""
    cfg = load_synth_config(AS2008, ["paths=40"])
    depths: list[np.ndarray] = []
    for world in generate(cfg):
        ticks = np.rint(world.trades["price"].to_numpy() / cfg.tick_size).astype(np.int64)
        ts = world.trades["ts_exchange"].to_numpy()
        mid = world.mid_ticks[np.searchsorted(world.mid_times_ns, ts, side="left") - 1]
        depths.append(world.trades["side"].to_numpy() * (ticks - mid) * cfg.tick_size)
    d = np.concatenate(depths)
    total_time = 2 * cfg.paths * cfg.horizon  # both sides
    for delta in (0.0, 0.5, 1.0):
        rate = np.count_nonzero(d >= delta - 1e-12) / total_time
        expect = cfg.market_orders.A * math.exp(-cfg.market_orders.k * delta)
        assert abs(rate - expect) < 4 * math.sqrt(expect / total_time)


def test_write_world_partitions_and_refuses_overwrite(tmp_path: Path) -> None:
    world = generate_path(load_synth_config(MINI), 0)
    files = write_world(world, tmp_path)
    assert len(files) == 2
    (day,) = list_days(tmp_path, "trades", "synthetic", "SYNUSDT")
    assert read_day(tmp_path, "trades", "synthetic", "SYNUSDT", day).height == world.trades.height
    assert sum(b.num_rows for b in iter_row_groups(files[:1])) == world.book_l2.height
    with pytest.raises(FileExistsError):
        write_world(world, tmp_path)


def test_generator_rejects_impossible_configs() -> None:
    with pytest.raises(ValueError, match="whole number"):
        generate_path(load_synth_config(AS2008, ["dt=0.3", "market_orders.arrivals=poisson"]), 0)
    with pytest.raises(ValueError, match="below one tick"):
        generate_path(load_synth_config(AS2008, ["tick_size=1.0"]), 0)
    with pytest.raises(ValueError, match="one UTC day"):
        generate_path(load_synth_config(AS2008, ["time_unit_ns=100000000000000"]), 0)
    with pytest.raises(ValueError, match=r"A \* dt"):
        load_synth_config(AS2008, ["dt=0.3"])
