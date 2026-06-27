"""Phase C glue — membership ingestion (the pure incremental diff, no DB).

`diff_snapshot` is how periodic constituent snapshots build point-in-time history: open the
newly-added, close the departed, leave the unchanged alone. Idempotent by construction.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app.services.universe.ingestion import diff_snapshot


def _asof(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def test_diff_opens_new_and_closes_departed():
    current_open = {"AAA", "BBB", "CCC"}
    new_members = {"AAA", "BBB", "DDD"}  # CCC left, DDD joined
    asof = _asof(2024, 6, 1)
    to_open, to_close = diff_snapshot(current_open, new_members, universe_key="NIFTY500", asof=asof)
    assert [r.symbol for r in to_open] == ["DDD"]
    assert to_open[0].valid_from == asof and to_open[0].valid_to is None
    assert to_close == ["CCC"]


def test_diff_is_noop_for_unchanged_snapshot():
    members = {"AAA", "BBB"}
    to_open, to_close = diff_snapshot(members, members, universe_key="X", asof=_asof(2024, 1, 1))
    assert to_open == [] and to_close == []


def test_diff_initial_seed_opens_everything():
    to_open, to_close = diff_snapshot(set(), {"AAA", "BBB"}, universe_key="X", asof=_asof(2024, 1, 1))
    assert sorted(r.symbol for r in to_open) == ["AAA", "BBB"]
    assert to_close == []


def test_diff_is_deterministic_and_sorted():
    to_open, to_close = diff_snapshot(
        {"Z"}, {"M", "A", "Q"}, universe_key="X", asof=_asof(2024, 1, 1))
    assert [r.symbol for r in to_open] == ["A", "M", "Q"]
    assert to_close == ["Z"]
