"""Phase D — market calendar (§9): session boundaries + refresh-cadence firing.

Pure, timezone-explicit, no I/O. Covers the NSE session open/close boundaries (09:14 vs 09:15
vs 15:30 vs 15:31 IST), weekends, a known NSE holiday, crypto-always-open, the naive-datetime
guard, and the daily/weekly/intraday cadence + should_fire logic.
"""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from app.services.execution.market_calendar import (
    IST,
    is_market_open,
    next_refresh_at,
    should_fire,
)
from app.strategy.spec import UniverseRefresh

UTC = timezone.utc


def _ist(y, mo, d, h, mi) -> datetime:
    return datetime(y, mo, d, h, mi, tzinfo=IST)


# ── crypto: 24/7 ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("when", [
    datetime(2025, 6, 21, 3, 0, tzinfo=UTC),     # Saturday, middle of the night
    datetime(2025, 6, 22, 23, 59, tzinfo=UTC),   # Sunday late
    datetime(2025, 12, 25, 12, 0, tzinfo=UTC),   # Christmas (an equity holiday)
])
def test_crypto_always_open(when):
    assert is_market_open(when, "crypto_spot") is True
    assert is_market_open(when, "crypto") is True


# ── NSE session boundaries (IST) ──────────────────────────────────────────────
# 2025-06-20 is a Friday (a normal trading day, not in the holiday list).
@pytest.mark.parametrize("hh,mm,expected", [
    (9, 14, False),    # one minute before open
    (9, 15, True),     # exactly open
    (12, 0, True),     # mid-session
    (15, 30, True),    # exactly close (inclusive)
    (15, 31, False),   # one minute after close
])
def test_nse_session_boundaries(hh, mm, expected):
    assert is_market_open(_ist(2025, 6, 20, hh, mm), "equity_cash") is expected


def test_nse_closed_on_weekend():
    # 2025-06-21 Saturday, 2025-06-22 Sunday — closed even mid-session-clock.
    assert is_market_open(_ist(2025, 6, 21, 11, 0), "equity_cash") is False
    assert is_market_open(_ist(2025, 6, 22, 11, 0), "equity_cash") is False


def test_nse_closed_on_known_holiday():
    # 2025-08-15 Independence Day (a Friday) — a known NSE trading holiday.
    assert is_market_open(_ist(2025, 8, 15, 11, 0), "equity_cash") is False
    # Sanity: the preceding Thursday IS open.
    assert is_market_open(_ist(2025, 8, 14, 11, 0), "equity_cash") is True


def test_library_drives_holidays_beyond_static_snapshot():
    # 2026-03-26 is a Thursday NSE holiday the maintained exchange_calendars knows but the small
    # static fallback does NOT list — proving the live source is the library, not a hardcoded set.
    import app.services.execution.market_calendar as mc

    assert datetime(2026, 3, 26).date() not in mc.NSE_TRADING_HOLIDAYS
    assert is_market_open(_ist(2026, 3, 26, 11, 0), "equity_cash") is False
    # The surrounding Wednesday is a normal session.
    assert is_market_open(_ist(2026, 3, 25, 11, 0), "equity_cash") is True


def test_static_fallback_used_when_library_unavailable(monkeypatch):
    # Force the exchange_calendars calendar to look unavailable → the static snapshot carries it.
    import app.services.execution.market_calendar as mc

    monkeypatch.setattr(mc, "_NSE_CALENDAR", None)        # already-attempted, unavailable
    monkeypatch.setattr(mc, "_logged_sources", set())
    # A snapshot holiday is still closed; an ordinary weekday is still open.
    assert is_market_open(_ist(2025, 8, 15, 11, 0), "equity_cash") is False   # in snapshot
    assert is_market_open(_ist(2025, 6, 20, 11, 0), "equity_cash") is True    # Friday, open


def test_holidays_override():
    # 2025-06-20 (Fri) is normally open; an injected holiday set closes it, and clears the
    # built-in ones (2025-08-15 now open under the override).
    custom = {datetime(2025, 6, 20).date()}
    assert is_market_open(_ist(2025, 6, 20, 11, 0), "equity_cash", holidays=custom) is False
    assert is_market_open(_ist(2025, 8, 15, 11, 0), "equity_cash", holidays=custom) is True


def test_equity_open_uses_ist_not_utc():
    # 04:30 UTC == 10:00 IST → session is open even though the UTC clock looks "early".
    assert is_market_open(datetime(2025, 6, 20, 4, 30, tzinfo=UTC), "equity_cash") is True
    # 10:30 UTC == 16:00 IST → after close.
    assert is_market_open(datetime(2025, 6, 20, 10, 30, tzinfo=UTC), "equity_cash") is False


def test_naive_datetime_rejected():
    with pytest.raises(ValueError):
        is_market_open(datetime(2025, 6, 20, 11, 0), "equity_cash")


# ── refresh cadence: next_refresh_at ──────────────────────────────────────────
def test_daily_next_refresh_later_today():
    refresh = UniverseRefresh(cadence="daily", at="10:15")
    nxt = next_refresh_at(_ist(2025, 6, 20, 9, 0), refresh, "equity_cash")
    assert nxt == _ist(2025, 6, 20, 10, 15)


def test_daily_next_refresh_rolls_to_tomorrow():
    refresh = UniverseRefresh(cadence="daily", at="10:15")
    nxt = next_refresh_at(_ist(2025, 6, 20, 11, 0), refresh, "equity_cash")
    assert nxt == _ist(2025, 6, 21, 10, 15)


def test_daily_at_with_tz_suffix_parses():
    refresh = UniverseRefresh(cadence="daily", at="09:20 IST")
    nxt = next_refresh_at(_ist(2025, 6, 20, 9, 0), refresh, "equity_cash")
    assert nxt == _ist(2025, 6, 20, 9, 20)


def test_crypto_daily_default_at_midnight_utc():
    refresh = UniverseRefresh(cadence="daily")  # no `at` → crypto default 00:00 UTC
    nxt = next_refresh_at(datetime(2025, 6, 20, 5, 0, tzinfo=UTC), refresh, "crypto_spot")
    assert nxt == datetime(2025, 6, 21, 0, 0, tzinfo=UTC)


def test_weekly_anchors_to_monday():
    refresh = UniverseRefresh(cadence="weekly", at="10:15")
    # 2025-06-20 is Friday → next Monday is 2025-06-23.
    nxt = next_refresh_at(_ist(2025, 6, 20, 11, 0), refresh, "equity_cash")
    assert nxt == _ist(2025, 6, 23, 10, 15)


def test_intraday_next_hour_boundary():
    refresh = UniverseRefresh(cadence="intraday")
    nxt = next_refresh_at(_ist(2025, 6, 20, 10, 20), refresh, "equity_cash")
    assert nxt == _ist(2025, 6, 20, 11, 0)


# ── should_fire ───────────────────────────────────────────────────────────────
def test_should_fire_first_tick_when_never_fired():
    refresh = UniverseRefresh(cadence="daily", at="10:15")
    assert should_fire(_ist(2025, 6, 20, 10, 16), refresh, "equity_cash", last_fired_at=None) is True


def test_should_not_fire_before_scheduled_time_same_day():
    refresh = UniverseRefresh(cadence="daily", at="10:15")
    last = _ist(2025, 6, 20, 10, 15)         # already fired at today's slot
    assert should_fire(_ist(2025, 6, 20, 14, 0), refresh, "equity_cash", last_fired_at=last) is False


def test_should_fire_once_per_day_after_slot_elapses():
    refresh = UniverseRefresh(cadence="daily", at="10:15")
    last = _ist(2025, 6, 20, 10, 15)         # fired yesterday's slot
    # Next day, after the slot → due again.
    assert should_fire(_ist(2025, 6, 21, 10, 16), refresh, "equity_cash", last_fired_at=last) is True
    # Next day, BEFORE the slot → not yet.
    assert should_fire(_ist(2025, 6, 21, 9, 0), refresh, "equity_cash", last_fired_at=last) is False


def test_should_fire_naive_last_fired_rejected():
    refresh = UniverseRefresh(cadence="daily", at="10:15")
    with pytest.raises(ValueError):
        should_fire(_ist(2025, 6, 20, 11, 0), refresh, "equity_cash",
                    last_fired_at=datetime(2025, 6, 20, 10, 15))
