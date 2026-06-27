"""
app/services/execution/market_calendar.py — pure market-session + refresh-cadence calendar.

A dynamic-universe deployment re-resolves its membership on a cadence, but ONLY while the
asset's market is actually trading. This module answers two timezone-explicit questions with
no I/O and no ``datetime.now()`` (every function takes ``asof``), so it is trivially unit-
testable and behaves identically in tests, the API container, and the scheduler:

  * :func:`is_market_open` — is ``asof`` inside a trading session for ``asset_class``?
        crypto  → 24/7 (UTC).
        equity  → NSE cash: Mon–Fri 09:15–15:30 **IST**, holiday-aware.
  * :func:`next_refresh_at` / :func:`should_fire` — resolve a :class:`UniverseRefresh`
        (``cadence`` + optional ``at`` wall-clock) into firing instants in the asset's own
        timezone, and decide whether a tick is due given the last fire.

Timezone discipline (Invariant: never compare a naive instant):
  * crypto reasons in **UTC**; equity reasons in **Asia/Kolkata**.
  * ``asof`` / ``last_fired_at`` MUST be timezone-aware — a naive datetime raises, so an
    ambiguous comparison can never slip through.

Holiday source (production-robust, no hand-maintained year lists):
  1. PRIMARY — the maintained ``exchange_calendars`` ``XBOM`` calendar (BSE/NSE share identical
     trading days *and* hours in India). It knows every market closure for all shipped years and
     is refreshed yearly via ``pip``; we never hardcode a holiday it covers.
  2. FALLBACK — a small static ``NSE_TRADING_HOLIDAYS`` snapshot, used ONLY when the library is
     unavailable or the date is beyond the calendar's shipped range. A warning is logged so a
     fallback is never silent.
  3. OVERRIDE — callers may pass their own ``holidays`` set (e.g. from an ingestion table); an
     explicit set wins over both of the above.
The source actually used is logged the first time it is consulted.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Iterable
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

UTC = timezone.utc
IST = ZoneInfo("Asia/Kolkata")

# ── NSE cash session (IST) ────────────────────────────────────────────────────
NSE_OPEN = time(9, 15)        # 09:15 IST — first tick of the continuous session
NSE_CLOSE = time(15, 30)      # 15:30 IST — last tick (inclusive); 15:31 is closed

# Intraday refresh has no explicit interval field on UniverseRefresh; use a documented
# default so "intraday" is well-defined. Override by modelling an explicit cadence later.
_INTRADAY_DEFAULT_INTERVAL_MIN = 60

# Weekly refresh has no weekday field on the spec; anchor it to Monday (start of the trading
# week) at the `at` time. Documented so the choice is explicit, not accidental.
_WEEKLY_ANCHOR_WEEKDAY = 0    # Monday (date.weekday(): Mon=0 … Sun=6)

# ── Holiday calendar ──────────────────────────────────────────────────────────
# India NSE/BSE calendar code in `exchange_calendars`. BSE (XBOM) and NSE observe identical
# trading days and session hours, so XBOM is authoritative for NSE holidays too.
_NSE_CALENDAR_CODE = "XBOM"

# FALLBACK ONLY — a small static snapshot of NSE trading holidays, consulted just when the
# `exchange_calendars` library is unavailable or the date is beyond its shipped range. The
# library (above) is the primary, always-current source; this is the safety net, not the truth.
# Weekends are handled separately (every Sat/Sun is closed regardless of this list).
NSE_TRADING_HOLIDAYS: frozenset[date] = frozenset({
    # 2025
    date(2025, 2, 26),   # Mahashivratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Id-ul-Fitr (Ramzan)
    date(2025, 4, 10),   # Mahavir Jayanti
    date(2025, 4, 14),   # Dr. Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 27),   # Ganesh Chaturthi
    date(2025, 10, 2),   # Gandhi Jayanti / Dussehra
    date(2025, 10, 21),  # Diwali-Laxmi Pujan
    date(2025, 10, 22),  # Diwali-Balipratipada
    date(2025, 11, 5),   # Prakash Gurpurb (Guru Nanak Jayanti)
    date(2025, 12, 25),  # Christmas
    # 2026 (fixed national holidays; refresh the moveable ones when NSE publishes them)
    date(2026, 1, 26),   # Republic Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 12, 25),  # Christmas
})

# Lazily-loaded `exchange_calendars` XBOM calendar. ``False`` = not yet attempted; ``None`` =
# attempted and unavailable (library missing / load failed) → use the static fallback.
_NSE_CALENDAR: Any = False
_logged_sources: set[str] = set()


def _log_source_once(source: str, level: str = "info") -> None:
    """Announce which holiday source served a check, once per distinct source (no silent default)."""
    if source not in _logged_sources:
        getattr(logger, level)("📅 market_calendar | equity holiday source=%s", source)
        _logged_sources.add(source)


def _nse_calendar() -> Any:
    """The cached ``exchange_calendars`` XBOM calendar, or ``None`` if the library is unavailable.

    Imported lazily so this module stays import-light and works even where the optional
    dependency isn't installed (the static snapshot then carries the calendar)."""
    global _NSE_CALENDAR
    if _NSE_CALENDAR is False:
        try:
            import exchange_calendars as xcals  # lazy: optional dependency
            _NSE_CALENDAR = xcals.get_calendar(_NSE_CALENDAR_CODE)
        except Exception as exc:  # noqa: BLE001 — any import/load failure → documented fallback
            logger.warning(
                "📅 market_calendar | exchange_calendars unavailable (%s) — falling back to the "
                "static NSE_TRADING_HOLIDAYS snapshot. `pip install exchange_calendars` for the "
                "always-current calendar.", exc,
            )
            _NSE_CALENDAR = None
    return _NSE_CALENDAR


def _is_nse_trading_day(d: date, holidays: Iterable[date] | None) -> bool:
    """Is ``d`` an NSE trading day (weekday, not a holiday)? Resolves the holiday source per the
    module contract: explicit ``holidays`` override → ``exchange_calendars`` → static snapshot."""
    if holidays is not None:                      # explicit override wins over everything
        _log_source_once("caller-supplied holidays set")
        return d.weekday() < 5 and d not in set(holidays)

    cal = _nse_calendar()
    if cal is not None:
        try:
            import pandas as pd  # exchange_calendars already depends on pandas
            is_session = bool(cal.is_session(pd.Timestamp(d)))
            _log_source_once(f"exchange_calendars '{_NSE_CALENDAR_CODE}'")
            return is_session
        except Exception:  # noqa: BLE001 — out-of-range date or lib hiccup → static fallback
            _log_source_once(
                f"static NSE_TRADING_HOLIDAYS (date {d} beyond exchange_calendars range)",
                level="warning",
            )
    else:
        _log_source_once("static NSE_TRADING_HOLIDAYS (exchange_calendars unavailable)", "warning")
    return d.weekday() < 5 and d not in NSE_TRADING_HOLIDAYS


# ── Asset-class + timezone normalization ──────────────────────────────────────
def _normalize_asset_class(asset_class: str) -> str:
    """Map the various asset-class spellings to a canonical ``"crypto"`` | ``"equity"``.

    Accepts the taxonomy ids (``crypto_spot`` / ``equity_cash``), the chat free-text values
    (``crypto`` / ``indian_stocks`` / ``nse`` …) and venue names — so callers can pass whatever
    they hold without a translation table. Unknown ⇒ ``"equity"`` (the session-gated default,
    fail-safe: a refresh is gated by a session rather than running 24/7 by accident)."""
    m = (asset_class or "").strip().lower()
    if m in {"crypto", "crypto_spot", "binance", "binance_spot", "spot"}:
        return "crypto"
    return "equity"


def market_timezone(asset_class: str) -> ZoneInfo | timezone:
    """The timezone the asset's market reasons in (crypto→UTC, equity→Asia/Kolkata)."""
    return UTC if _normalize_asset_class(asset_class) == "crypto" else IST


def _ensure_aware(dt: datetime, label: str) -> datetime:
    """Reject a naive datetime — the whole module's correctness rests on tz-explicit instants."""
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(
            f"{label} must be timezone-aware; got naive {dt!r}. market_calendar never assumes "
            f"a timezone (crypto→UTC, equity→Asia/Kolkata) — stamp the tz at the call site."
        )
    return dt


# ── Market session ────────────────────────────────────────────────────────────
def is_market_open(
    asof: datetime,
    asset_class: str,
    *,
    holidays: Iterable[date] | None = None,
) -> bool:
    """Is ``asof`` inside a trading session for ``asset_class``?

    crypto → always True (24/7, UTC). equity → True iff ``asof`` (converted to IST) is an NSE
    trading day (weekday, not a holiday — resolved via ``exchange_calendars`` with a static
    fallback) and the time is within ``[09:15, 15:30]`` inclusive. ``asof`` must be timezone-
    aware. ``holidays`` overrides the calendar with a caller-supplied set when supplied.
    """
    _ensure_aware(asof, "asof")
    if _normalize_asset_class(asset_class) == "crypto":
        logger.debug("🕐 is_market_open | crypto | 24/7 → open=True (asof=%s)", asof.isoformat())
        return True

    local = asof.astimezone(IST)
    trading_day = _is_nse_trading_day(local.date(), holidays)
    in_session = trading_day and (NSE_OPEN <= local.time() <= NSE_CLOSE)
    logger.debug(
        "🕐 is_market_open | NSE | asof_ist=%s | trading_day=%s | session=%s–%s | open=%s",
        local.isoformat(), trading_day, NSE_OPEN.strftime("%H:%M"), NSE_CLOSE.strftime("%H:%M"), in_session,
    )
    return in_session


# ── Refresh cadence ───────────────────────────────────────────────────────────
def _refresh_fields(refresh: Any) -> tuple[str, str | None]:
    """Extract ``(cadence, at)`` from a UniverseRefresh model, a dict, or None (→ defaults)."""
    if refresh is None:
        return "daily", None
    cadence = getattr(refresh, "cadence", None)
    at = getattr(refresh, "at", None)
    if cadence is None and isinstance(refresh, dict):
        cadence = refresh.get("cadence")
        at = refresh.get("at")
    return (str(cadence or "daily").strip().lower(), at)


def _parse_hhmm(at: str | None, default: time) -> time:
    """Parse a wall-clock ``at`` into a ``time``. Tolerates ``"09:20"``, ``"9:20"``,
    ``"09:20 IST"`` and stray prefixes; falls back to ``default`` when absent/unparseable."""
    if not at or not str(at).strip():
        return default
    digits: list[str] = []
    for token in str(at).replace("@", " ").split():
        if ":" in token and token.split(":", 1)[0].strip().isdigit():
            digits = token.split(":")
            break
    if not digits:
        return default
    try:
        hh = int(digits[0])
        # Take the leading 2 digits of the minute field so a trailing tz glued on
        # (e.g. "20IST") still parses to 20.
        minute_digits = "".join(ch for ch in digits[1] if ch.isdigit())[:2] if len(digits) > 1 else ""
        mm = int(minute_digits) if minute_digits else 0
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return time(hh, mm)
    except (TypeError, ValueError):
        pass
    return default


def _default_fire_time(asset_class: str) -> time:
    """When no ``at`` is given: equity fires at the open (09:15 IST); crypto at 00:00 UTC."""
    return NSE_OPEN if _normalize_asset_class(asset_class) == "equity" else time(0, 0)


def _at_on(d: date, t: time, tz: ZoneInfo | timezone) -> datetime:
    return datetime(d.year, d.month, d.day, t.hour, t.minute, tzinfo=tz)


def next_refresh_at(asof: datetime, refresh: Any, asset_class: str) -> datetime:
    """The next scheduled refresh instant STRICTLY after ``asof``, in the asset's timezone.

    ``daily``    → next occurrence of the ``at`` time-of-day.
    ``weekly``   → next Monday (``_WEEKLY_ANCHOR_WEEKDAY``) at the ``at`` time-of-day.
    ``intraday`` → next ``_INTRADAY_DEFAULT_INTERVAL_MIN``-minute boundary from midnight.

    Returned tz-aware in market-tz (crypto→UTC, equity→IST). Pure: no ``now()``.
    """
    _ensure_aware(asof, "asof")
    cadence, at = _refresh_fields(refresh)
    tz = market_timezone(asset_class)
    local = asof.astimezone(tz)

    if cadence == "intraday":
        return _next_intraday(local, tz)

    fire_t = _parse_hhmm(at, _default_fire_time(asset_class))
    if cadence == "weekly":
        return _next_weekly(local, fire_t, tz)
    # daily (default)
    today_fire = _at_on(local.date(), fire_t, tz)
    return today_fire if local < today_fire else _at_on(local.date() + timedelta(days=1), fire_t, tz)


def previous_refresh_at(asof: datetime, refresh: Any, asset_class: str) -> datetime:
    """The most recent scheduled refresh instant AT OR BEFORE ``asof`` (mirror of
    :func:`next_refresh_at`).

    This is the anchor for *clock-quantized* universe resolution: resolving a dynamic universe
    with ``asof = previous_refresh_at(now, ...)`` yields a membership that is STABLE within a
    refresh window and re-picks automatically when the window rolls — no stored "last refreshed"
    state required (the clock IS the state)."""
    cadence, at = _refresh_fields(refresh)
    tz = market_timezone(asset_class)
    local = asof.astimezone(tz)

    if cadence == "intraday":
        nxt = _next_intraday(local, tz)
        step = timedelta(minutes=_INTRADAY_DEFAULT_INTERVAL_MIN)
        return nxt if nxt == local else nxt - step

    fire_t = _parse_hhmm(at, _default_fire_time(asset_class))
    if cadence == "weekly":
        nxt = _next_weekly(local, fire_t, tz)
        return nxt if nxt == local else nxt - timedelta(days=7)
    today_fire = _at_on(local.date(), fire_t, tz)
    return today_fire if local >= today_fire else _at_on(local.date() - timedelta(days=1), fire_t, tz)


def _next_intraday(local: datetime, tz: ZoneInfo | timezone) -> datetime:
    step = _INTRADAY_DEFAULT_INTERVAL_MIN
    minutes = local.hour * 60 + local.minute
    floor_min = (minutes // step) * step
    floor_dt = _at_on(local.date(), time(floor_min // 60, floor_min % 60), tz)
    if local <= floor_dt:
        return floor_dt
    return floor_dt + timedelta(minutes=step)


def _next_weekly(local: datetime, fire_t: time, tz: ZoneInfo | timezone) -> datetime:
    days_ahead = (_WEEKLY_ANCHOR_WEEKDAY - local.weekday()) % 7
    candidate = _at_on(local.date() + timedelta(days=days_ahead), fire_t, tz)
    if local < candidate:
        return candidate
    return _at_on(local.date() + timedelta(days=days_ahead + 7), fire_t, tz)


def should_fire(
    asof: datetime,
    refresh: Any,
    asset_class: str,
    last_fired_at: datetime | None,
) -> bool:
    """Is a refresh tick due at ``asof`` given when it ``last_fired_at``?

    A tick is due when a scheduled instant has elapsed that the deployment has not yet fired
    for: i.e. the most recent scheduled instant ``≤ asof`` is strictly after ``last_fired_at``
    (or the deployment has never fired). Pure cadence logic — the scheduler additionally gates
    on :func:`is_market_open` so an equity refresh never runs outside the session.
    """
    _ensure_aware(asof, "asof")
    prev = previous_refresh_at(asof, refresh, asset_class)
    if last_fired_at is None:
        logger.debug("📆 should_fire | never fired before → due=True (prev_slot=%s)", prev.isoformat())
        return True
    _ensure_aware(last_fired_at, "last_fired_at")
    due = last_fired_at < prev
    logger.debug(
        "📆 should_fire | prev_slot=%s | last_fired=%s | due=%s",
        prev.isoformat(), last_fired_at.isoformat(), due,
    )
    return due
