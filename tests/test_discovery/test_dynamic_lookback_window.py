"""Phase 9i — the high/low lookback window is tunable.

Pre-9i the preset hardcoded `MAX(HIGH, 252)` / `MIN(LOW, 252)` for the
52-week test. Users who typed "20-day high" or "26-week breakout" got
the 252-bar scan regardless. Phase 9i adds a `lookback_window_bars`
parameter (default 252) that flows through the chat → builder →
orchestrator → scanner pipeline:

  1. Chat parser extracts "N-day/week/month/year high/low" → bar count
  2. Orchestrator substitutes into the preset's condition placeholders
  3. Orchestrator bumps lookback_days so the OHLCV fetch covers the
     window
  4. Scanner uses the same window for metric computation (distance to
     high/low) so tie-break ordering matches the user's intent
  5. Tie-break labels reflect the actual window ("Closest to 20-day
     high" not "Closest to 52-week high")
  6. No-match summary mentions the window phrase
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


if "asyncpg" not in sys.modules:
    sys.modules["asyncpg"] = types.ModuleType("asyncpg")


_CHAT_SERVICE_PATH = (
    Path(__file__).resolve().parents[2]
    / "app" / "services" / "chat" / "chat_service.py"
)


def _load_extract_overrides():
    """Compile _extract_discovery_parameter_overrides and its constants
    out of chat_service.py without importing the full module (asyncpg
    gap)."""
    import re as re_module
    src = _CHAT_SERVICE_PATH.read_text(encoding="utf-8")
    helpers_idx = src.index("_VOLUME_MULTIPLIER_RE = re.compile")
    block_start = helpers_idx
    while block_start > 0:
        prev_nl = src.rfind("\n", 0, block_start - 1)
        line_start = prev_nl + 1 if prev_nl != -1 else 0
        line = src[line_start:block_start - 1].strip()
        if line.startswith("#"):
            block_start = line_start
        else:
            break
    fn_marker = "def _extract_discovery_parameter_overrides(message"
    fn_idx = src.index(fn_marker, helpers_idx)
    after_fn = re_module.search(
        r"\n(def |async def |class )", src[fn_idx + len(fn_marker):]
    )
    end_idx = (
        fn_idx + len(fn_marker) + after_fn.start() + 1
        if after_fn
        else len(src)
    )
    body = src[block_start:end_idx]
    namespace: dict = {"re": re_module}
    exec(body, namespace)
    return namespace["_extract_discovery_parameter_overrides"]


# ── Chat parser: N-day/week/month/year → bars ───────────────────────────────


def test_extract_captures_20_day_high():
    extract = _load_extract_overrides()
    out = extract("find a stock near 20-day high with volume spike 1.5x")
    assert out.get("lookback_window_bars") == 20.0
    assert out.get("volume_multiplier") == 1.5


def test_extract_captures_26_week_breakout():
    extract = _load_extract_overrides()
    out = extract("intraday strategy on 26 week breakout with 2x volume")
    assert out.get("lookback_window_bars") == 26 * 5


def test_extract_captures_6_month_high():
    extract = _load_extract_overrides()
    out = extract("volume spike near 6-month high")
    assert out.get("lookback_window_bars") == 6 * 21


def test_extract_captures_1_year_breakout():
    extract = _load_extract_overrides()
    out = extract("volume spike on verge of 1-year breakout")
    assert out.get("lookback_window_bars") == 252.0


def test_extract_captures_user_original_52_weeks_phrasing():
    """The user's original prompt ("on verge of 52 weeks high or low")
    must produce a window of 52 weeks ≈ 260 bars. The orchestrator
    then bumps lookback_days to cover that window."""
    extract = _load_extract_overrides()
    out = extract(
        "create intraday strategy on NSE stock whose volume spike up today 2x "
        "and is on verge of 52 weeks high or low"
    )
    assert out.get("lookback_window_bars") == 52 * 5


def test_extract_ignores_lookback_without_high_low_context():
    """`5 days from now` should NOT be interpreted as a window."""
    extract = _load_extract_overrides()
    out = extract("call me back in 5 days about RELIANCE")
    assert "lookback_window_bars" not in out


def test_extract_drops_lookback_out_of_bounds():
    """`5000 days` would blow up the OHLCV fetch — drop it."""
    extract = _load_extract_overrides()
    out = extract("looking for stocks near 5000 days high")
    assert "lookback_window_bars" not in out


def test_extract_drops_lookback_too_small():
    """1-day window is degenerate — drop it."""
    extract = _load_extract_overrides()
    out = extract("breakout near 1 day high")
    assert "lookback_window_bars" not in out


# ── Orchestrator: substitutes the window into conditions ────────────────────


from app.services.discovery.orchestrator import (
    _format_window_phrase,
    _preset_discovery_config,
)
from app.services.strategy.builder import StrategyBuilder


def _builder_with_preset(preset_name: str = "volume_breakout_52w") -> StrategyBuilder:
    b = StrategyBuilder()
    b.strategy_preset = preset_name
    b.timeframe = "5m"
    b.sentiment = "bullish"
    b.experience = "intermediate"
    b.objective = "intraday"
    b.goal = "volume breakout"
    return b


def test_preset_default_window_is_252():
    b = _builder_with_preset()
    cfg = _preset_discovery_config(b)
    big_or = next(c for c in cfg.conditions if "MAX(HIGH" in c)
    assert "MAX(HIGH, 252)" in big_or
    assert "MIN(LOW, 252)"  in big_or
    assert b.discovery_parameters_used["lookback_window_bars"] == 252.0


def test_user_can_set_a_20_day_window():
    b = _builder_with_preset()
    b.discovery_parameter_overrides = {"lookback_window_bars": 20.0}
    cfg = _preset_discovery_config(b)
    big_or = next(c for c in cfg.conditions if "MAX(HIGH" in c)
    assert "MAX(HIGH, 20)" in big_or
    assert "MIN(LOW, 20)"  in big_or
    # default 252 must NOT appear anywhere in the OR clause
    assert "MAX(HIGH, 252)" not in big_or
    assert b.discovery_parameters_used["lookback_window_bars"] == 20.0


def test_user_can_set_a_2_year_window():
    b = _builder_with_preset()
    b.discovery_parameter_overrides = {"lookback_window_bars": 504.0}
    cfg = _preset_discovery_config(b)
    big_or = next(c for c in cfg.conditions if "MAX(HIGH" in c)
    assert "MAX(HIGH, 504)" in big_or
    # lookback_days must be bumped to cover the longer window.
    assert cfg.lookback_days >= 504 + 30, (
        f"lookback_days={cfg.lookback_days} must accommodate window=504 + buffer"
    )


def test_preset_lookback_days_not_lowered_when_user_window_is_smaller():
    """If the user picks a SHORT window (20 days), lookback_days must
    stay at the preset's existing default (280). Don't shrink the data
    fetch window — other conditions in the OR might still need long
    history (e.g. EMA warmup)."""
    b = _builder_with_preset()
    b.discovery_parameter_overrides = {"lookback_window_bars": 20.0}
    cfg = _preset_discovery_config(b)
    assert cfg.lookback_days >= 280


# ── Tie-break labels rewrite to match the active window ─────────────────────


def test_tie_break_labels_keep_52_week_phrasing_at_default_window():
    b = _builder_with_preset()
    cfg = _preset_discovery_config(b)
    labels = [o.label for o in cfg.tie_break_options]
    # at the default 252-bar window, the literal preset label survives
    assert any("52-week" in lab for lab in labels), (
        f"expected default '52-week' phrasing in tie-break labels; got {labels}"
    )


def test_tie_break_labels_rewrite_to_user_window_phrase():
    """When the user picks 20-day, the labels must say "20-day" so the
    user isn't asked to choose "Closest to 52-week high" for a 20-day
    setup."""
    b = _builder_with_preset()
    b.discovery_parameter_overrides = {"lookback_window_bars": 20.0}
    cfg = _preset_discovery_config(b)
    labels = [o.label for o in cfg.tie_break_options]
    assert any("20-day" in lab for lab in labels), (
        f"expected '20-day' in tie-break labels for 20-day window; got {labels}"
    )
    # The misleading default phrasing must be gone.
    assert not any("52-week" in lab for lab in labels), (
        f"expected NO '52-week' phrasing when window=20-day; got {labels}"
    )


def test_format_window_phrase_picks_largest_natural_unit():
    """Helper sanity: render the most natural unit for a given bar count.

    Heuristic: prefer year/month/week only when the larger unit reads
    naturally. Avoid the awkward `4-week` for 20 bars or `1-month` for
    21 bars — keep small windows as days."""
    assert _format_window_phrase(252) == "1-year"
    assert _format_window_phrase(504) == "2-year"
    assert _format_window_phrase(126) == "6-month"
    assert _format_window_phrase(63)  == "3-month"
    assert _format_window_phrase(130) == "26-week"   # 26*5; not divisible by 21
    assert _format_window_phrase(25)  == "5-week"    # just-barely above day threshold
    assert _format_window_phrase(20)  == "20-day"    # below 25-bar week cutoff
    assert _format_window_phrase(15)  == "15-day"    # ditto
    assert _format_window_phrase(7)   == "7-day"


# ── Scanner: metric window matches the active lookback ──────────────────────


from app.services.discovery.scanner import scan_universe


@pytest.mark.asyncio
async def test_scanner_metrics_use_user_supplied_window():
    """When the user picks a 20-day window, the scanner's
    distance_to_52w_high_pct metric must measure distance to the
    20-day high, not the 252-day high. The metric KEY is preserved
    for the tie-break methods that reference it."""
    import pandas as pd

    target = "HDFCBANK.NS"

    def _bar(ts, c, v):
        return {
            "timestamp": ts.isoformat().replace("+00:00", "Z"),
            "open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": v,
        }

    def _series_with_recent_high():
        """Stock at 110 today; in the past 20 bars the high reached
        ~112 (so close is within 2% of 20-day high). Going back 252
        bars the high reached 200 (so close is 45% from 252-day high)."""
        base = pd.Timestamp("2026-05-12 09:30", tz="UTC")
        rows = []
        # 0..30 (oldest) — quiet at 50
        for i in range(30):
            rows.append(_bar(base + pd.Timedelta(minutes=i), 50.0, 100_000.0))
        # 30..130 — pump to 200
        for i in range(30, 130):
            rows.append(_bar(base + pd.Timedelta(minutes=i), 200.0, 100_000.0))
        # 130..280 — drift back to 110
        for i in range(130, 280):
            rows.append(_bar(base + pd.Timedelta(minutes=i), 110.0, 100_000.0))
        # 280..298 — last 18 bars at ~109..112 (just below 20-day high)
        for i in range(280, 298):
            rows.append(_bar(base + pd.Timedelta(minutes=i), 110.0, 100_000.0))
        rows[-15]["high"] = 112.0   # 20-day high
        rows[-15]["close"] = 112.0
        # Final bar — close=110 with volume spike
        rows.append(_bar(base + pd.Timedelta(minutes=298), 110.0, 100_000.0 * 3.0))
        return rows

    def _flat_series():
        base = pd.Timestamp("2026-05-12 09:30", tz="UTC")
        return [_bar(base + pd.Timedelta(minutes=i), 100.0, 100_000.0) for i in range(300)]

    async def stub_fetcher(symbol, interval, from_utc, to_utc):
        if symbol == target:
            return _series_with_recent_high()
        return _flat_series()

    # First, scan with the default 252-bar window. The 252-day high
    # was 200, so distance from close=110 ≈ 81.8%.
    b_default = _builder_with_preset()
    cfg_default = _preset_discovery_config(b_default)
    result_default = await scan_universe(
        cfg_default, interval="5m", fetch_ohlcv=stub_fetcher,
    )
    # The stock won't pass the OR clause (110 is far from 200-high),
    # but inspect the metric directly via _compute_candidate_metrics
    # to verify the window plumbing works.
    from app.services.discovery.scanner import _compute_candidate_metrics, _df_from_records
    df = _df_from_records(_series_with_recent_high())
    m_252 = _compute_candidate_metrics(df, lookback_window_bars=252)
    m_20  = _compute_candidate_metrics(df, lookback_window_bars=20)
    assert m_252["distance_to_52w_high_pct"] > 50.0, (
        "with 252-bar window, distance to peak (200) should be ~80% from close=110"
    )
    assert m_20["distance_to_52w_high_pct"] < 5.0, (
        "with 20-bar window, distance to recent high (112) should be ~2% from close=110"
    )


# ── No-match reply surfaces the active window ───────────────────────────────


from app.services.chat.strategy_flow import build_discovery_no_match_reply


def test_no_match_reply_mentions_user_chosen_window_in_proximity():
    msg = build_discovery_no_match_reply(parameters_used={
        "volume_multiplier": 1.5,
        "near_52w_high_factor": 0.95,
        "near_52w_low_factor": 1.05,
        "lookback_window_bars": 100.0,
    })
    assert "1.5" in msg
    # 100 bars = 20 weeks
    assert "20-week" in msg
    # And the default "52-week" phrasing is NOT present
    assert "52-week" not in msg


def test_no_match_reply_keeps_52_week_phrasing_at_default_window():
    msg = build_discovery_no_match_reply(parameters_used={
        "near_52w_high_factor": 0.98,
        "near_52w_low_factor": 1.02,
        "lookback_window_bars": 252.0,
    })
    assert "52-week" in msg
