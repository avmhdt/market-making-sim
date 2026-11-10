"""Documented exceptions (M2-20): reported as failed, tolerated only when listed."""

from __future__ import annotations

from conftest import REPO
from mmsim.validate import Check, load_exceptions, mark_exceptions, unexplained_failures


def row(case: str, metric: str, passed: bool) -> Check:
    return Check("AC-W-02", case, metric, 1.0, 2.0, 0.1, 5.0, "|z| <= 3", passed)


def test_only_the_listed_misses_are_tolerated() -> None:
    exc = load_exceptions(REPO)
    assert set(exc) == {
        ("AC-W-02", "gamma=0.1 symmetric reference", "std_profit"),
        ("AC-W-02", "gamma=0.01 symmetric reference", "std_profit"),
        ("AC-W-02", "gamma=1 symmetric reference", "std_final_q"),
    }
    listed = row("gamma=0.1 symmetric reference", "std_profit", False)
    other = row("gamma=0.1 symmetric fallback", "std_profit", False)
    fine = row("gamma=1 symmetric reference", "std_final_q", True)
    marked = mark_exceptions([listed, other, fine], exc)
    assert marked[0].note == "documented exception (M2-20)"
    assert not marked[0].passed  # still reported as a miss
    assert marked[1].note == "" and marked[2].note == ""
    assert unexplained_failures(marked, exc) == [marked[1]]
    assert unexplained_failures([marked[0], marked[2]], exc) == []
    assert load_exceptions(REPO / "nowhere") == {}
