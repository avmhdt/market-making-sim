"""Funding download and loading against a mocked endpoint (FR-08)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx

from mmsim.data.funding import download_funding, funding_frame, load_funding

NS_MS = 1_000_000


def test_download_paginates_and_writes(tmp_path: Path) -> None:
    calls: list[dict[str, str]] = []
    t0 = 1_793_577_600_000  # 2026-11-02 00:00 UTC in ms

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        calls.append(params)
        start = int(params["startTime"])
        if start > t0 + 2 * 8 * 3600 * 1000:
            return httpx.Response(200, json=[])
        rows = [
            {"symbol": "BTCUSDT", "fundingTime": t, "fundingRate": "0.0001", "markPrice": "70000.1"}
            for t in (t0, t0 + 8 * 3600 * 1000, t0 + 16 * 3600 * 1000)
            if t >= start
        ]
        rows[-1]["markPrice"] = ""
        return httpx.Response(200, json=rows)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    frame = download_funding(
        "BTCUSDT", date(2026, 11, 2), date(2026, 11, 2), raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed", client=client, pause_s=0.0,
    )  # fmt: skip
    assert frame.height == 3
    assert frame["mark_price"].null_count() == 1
    assert calls[0]["symbol"] == "BTCUSDT" and calls[0]["limit"] == "1000"
    assert list((tmp_path / "raw" / "funding").glob("*.json"))
    loaded = load_funding(
        tmp_path / "processed", "BTCUSDT", t0 * NS_MS, (t0 + 9 * 3600 * 1000) * NS_MS
    )
    assert loaded.height == 2
    assert load_funding(tmp_path / "nothing", "BTCUSDT", 0, 1).is_empty()


def test_frame_dedupes_and_sorts() -> None:
    rows = [
        {"fundingTime": 2, "fundingRate": "0.1", "markPrice": "1"},
        {"fundingTime": 1, "fundingRate": "0.2"},
        {"fundingTime": 2, "fundingRate": "0.1", "markPrice": "1"},
    ]
    f = funding_frame(rows, "X")
    assert f["ts"].to_list() == [1 * NS_MS, 2 * NS_MS]
    assert f["mark_price"].to_list() == [None, 1.0]
