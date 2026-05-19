"""Phase 9c — verify the broadened volume_breakout_52w preset matches the
user's actual natural-language prompt + that the OR-discovery qualifies
stocks via any of the four supported setups.

The user's exact original prompt:
    "create intraday strategy on NSE stock whose volume spike up today 2x
     and is having low pull back or is on breaking on verge of 52 weeks
     high or low"
"""
from __future__ import annotations

import sys
import types
from datetime import datetime, timezone

import pandas as pd
import pytest

if "asyncpg" not in sys.modules:
    sys.modules["asyncpg"] = types.ModuleType("asyncpg")

from app.kb import kb
from app.services.discovery.scanner import scan_universe
from app.services.discovery.types import DiscoveryConfig


# ── KB-level: keyword detection on the user's natural-language prompt ───────


@pytest.mark.parametrize("user_prompt", [
    # The user's literal original prompt
    "create intraday strategy on NSE stock whose volume spike up today 2x and is "
    "having low pull back or is on breaking on verge of 52 weeks high or low",
    # Common variants
    "I want a high relative volume strategy near 52-week high",
    "intraday strategy with 2x volume on NSE",
    "near 52 week breakout with volume spike",
    "find stocks with volume spike on verge of 52 weeks high or low",
    "low pullback breakout intraday",
])
def test_volume_breakout_52w_preset_matches_natural_language_prompt(user_prompt):
    """The chat layer's keyword detector must pin the volume_breakout_52w
    preset for these natural-language phrasings. Regression on the gap that
    Phase 9c was created to close."""
    preset = kb.detect_preset_in_text(user_prompt)
    assert preset is not None, f"no preset matched for prompt: {user_prompt!r}"
    assert preset.name == "volume_breakout_52w", (
        f"prompt {user_prompt!r} matched preset {preset.name!r} instead"
    )


def test_keyword_match_picks_longest_when_multiple_apply():
    """Defensive: a prompt mentioning both 'volume breakout' and
    'volume spike breakout' should pick the longer match. The detector
    already picks longest match, so this test guards that the new
    keywords don't accidentally outrank the more-specific ones."""
    preset = kb.detect_preset_in_text(
        "give me a volume spike breakout on NSE with 2x volume"
    )
    assert preset is not None
    assert preset.name == "volume_breakout_52w"


def test_volume_breakout_52w_preset_carries_scan_timeframe_1d():
    """Critical for correctness — without scan_timeframe=1d the 52-week
    conditions evaluate on intraday bars (252 5m bars ≈ 3 days, not 52w)."""
    preset = kb.presets["volume_breakout_52w"]
    discovery = preset.discovery or {}
    assert discovery.get("scan_timeframe") == "1d", (
        "volume_breakout_52w must declare scan_timeframe: 1d so 52-week "
        "conditions evaluate on daily bars regardless of strategy timeframe"
    )


def test_volume_breakout_52w_preset_uses_or_semantics_in_discovery():
    """The discovery condition list must have a composite OR clause (not
    just AND-ing volume + one other condition).

    Phase 9i — the high/low window is now a `{lookback_window_bars}`
    placeholder substituted by the orchestrator. This test exercises
    the substituted form (`_real_volume_breakout_discovery()`) so it
    keeps pinning the semantic invariant: an OR-clause that covers
    pullback + both extremes."""
    cfg = _real_volume_breakout_discovery()
    conditions = list(cfg.conditions)
    assert len(conditions) >= 2
    # Mandatory volume condition
    assert any("VOL" in c and "AVG" in c for c in conditions), (
        "volume × 2 condition missing"
    )
    # OR-clause covering pullback AND both high/low extremes
    or_clause = next((c for c in conditions if " OR " in c.upper()), None)
    assert or_clause is not None, "expected an OR-clause in discovery conditions"
    upper = or_clause.upper()
    assert "EMA(20)" in upper, "OR-clause missing pullback (EMA pullback term)"
    assert "MAX(HIGH, 252)" in upper, "OR-clause missing 52-week high term"
    assert "MIN(LOW, 252)" in upper, "OR-clause missing 52-week low term"


# ── Scanner: scan_timeframe override is honored ──────────────────────────────


@pytest.mark.asyncio
async def test_scan_timeframe_override_is_passed_to_fetcher():
    """When the preset declares scan_timeframe, the fetcher must receive
    that interval — not the strategy's main timeframe."""
    seen_intervals: list[str] = []

    async def stub_fetcher(symbol, interval, from_utc, to_utc):
        seen_intervals.append(interval)
        # Return enough bars to satisfy any condition's warmup
        return _flat_series(n=300)

    discovery = DiscoveryConfig(
        enabled=True,
        conditions=["CLOSE > 0"],     # trivial condition that always passes
        scan_timeframe="1d",
    )
    await scan_universe(
        discovery, interval="5m", fetch_ohlcv=stub_fetcher,
        asof_utc=datetime(2026, 5, 12, tzinfo=timezone.utc),
    )
    # Every fetch should have used 1d, NEVER the 5m main timeframe.
    assert seen_intervals, "fetcher was not called"
    assert all(iv == "1d" for iv in seen_intervals), (
        f"scan_timeframe override ignored — saw intervals {set(seen_intervals)}"
    )


@pytest.mark.asyncio
async def test_scan_timeframe_unset_falls_back_to_main_timeframe():
    """Backward compat: existing presets without scan_timeframe still use
    the strategy's main timeframe."""
    seen_intervals: list[str] = []

    async def stub_fetcher(symbol, interval, from_utc, to_utc):
        seen_intervals.append(interval)
        return _flat_series(n=300)

    discovery = DiscoveryConfig(enabled=True, conditions=["CLOSE > 0"])
    await scan_universe(discovery, interval="15m", fetch_ohlcv=stub_fetcher)
    assert all(iv == "15m" for iv in seen_intervals)


# ── OR-condition disjunction qualifies via any path ─────────────────────────


@pytest.mark.asyncio
async def test_or_clause_qualifies_stock_via_pullback_alone():
    """A stock that satisfies ONLY the bullish-pullback term of the OR
    clause (not 52w-high or 52w-low) should still pass the discovery
    filter, as long as the mandatory volume condition also passes."""
    target = "HDFCBANK.NS"

    async def stub_fetcher(symbol, interval, from_utc, to_utc):
        if symbol == target:
            return _pullback_series_bullish(n=300, end_volume_mult=3.0)
        return _flat_series(n=300, end_volume_mult=1.0)

    discovery = _real_volume_breakout_discovery()
    result = await scan_universe(discovery, interval="5m", fetch_ohlcv=stub_fetcher)
    assert target in {c.symbol for c in result.candidates}, (
        f"pullback-only setup should qualify; got candidates "
        f"{[c.symbol for c in result.candidates]}"
    )


@pytest.mark.asyncio
async def test_or_clause_qualifies_stock_via_52w_high_alone():
    target = "RELIANCE.NS"

    async def stub_fetcher(symbol, interval, from_utc, to_utc):
        if symbol == target:
            return _new_high_series(n=300, end_volume_mult=3.0)
        return _flat_series(n=300, end_volume_mult=1.0)

    discovery = _real_volume_breakout_discovery()
    result = await scan_universe(discovery, interval="5m", fetch_ohlcv=stub_fetcher)
    assert target in {c.symbol for c in result.candidates}


@pytest.mark.asyncio
async def test_or_clause_rejects_stock_when_volume_alone_qualifies():
    """Volume × 2 alone is not enough — must also satisfy ONE of the
    setups inside the OR clause. Sanity check on AND semantics across
    the two top-level conditions."""
    target = "HDFCBANK.NS"

    async def stub_fetcher(symbol, interval, from_utc, to_utc):
        # Use a "barbell" series so the current close is far from BOTH
        # 52w extremes (price peaked at ~200 long ago, troughed at ~50,
        # then settled flat in the middle). Volume spikes at the last
        # bar but no other condition fires.
        if symbol == target:
            return _barbell_series(n=300, end_volume_mult=3.0)
        return _flat_series(n=300, end_volume_mult=1.0)

    discovery = _real_volume_breakout_discovery()
    result = await scan_universe(discovery, interval="5m", fetch_ohlcv=stub_fetcher)
    assert target not in {c.symbol for c in result.candidates}, (
        "barbell series with volume-only spike should not qualify — neither "
        "pullback nor 52w-high nor 52w-low conditions fire"
    )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _bar(ts: pd.Timestamp, c: float, v: float) -> dict:
    return {
        "timestamp": ts.isoformat().replace("+00:00", "Z"),
        "open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": v,
    }


def _flat_series(n: int = 300, end_volume_mult: float = 1.0) -> list[dict]:
    """Flat at 100 throughout — far from any 52w extreme, no pullback."""
    base = pd.Timestamp("2026-05-12 09:30", tz="UTC")
    rows = []
    for i in range(n):
        ts = base - pd.Timedelta(days=(n - 1 - i))
        vol = 100_000.0 if i < n - 1 else 100_000.0 * end_volume_mult
        rows.append(_bar(ts, 100.0, vol))
    return rows


def _new_high_series(n: int = 300, end_volume_mult: float = 1.0) -> list[dict]:
    """Saw-tooth peaking near bar 100, decaying back, with a final spike to
    a fresh 52-week high so the 'CLOSE >= MAX(HIGH, 252) * 0.98' term fires."""
    base = pd.Timestamp("2026-05-12 09:30", tz="UTC")
    rows = []
    for i in range(n):
        ts = base - pd.Timedelta(days=(n - 1 - i))
        if i < 100:
            close = 100.0 + i * 0.5
        else:
            close = max(100.0, 150.0 - (i - 100) * 0.3)
        if i == n - 1:
            close = 200.0       # new 52w high
        vol = 100_000.0 if i < n - 1 else 100_000.0 * end_volume_mult
        rows.append(_bar(ts, close, vol))
    return rows


def _pullback_series_bullish(n: int = 300, end_volume_mult: float = 1.0) -> list[dict]:
    """Strong uptrend, then a tiny dip into the 20-EMA on the last bar
    that immediately reclaims it. Triggers the bullish-pullback term of
    the OR clause WITHOUT being near the 52-week high (peak was earlier
    and the EMA region is well below it)."""
    base = pd.Timestamp("2026-05-12 09:30", tz="UTC")
    rows = []
    # Strong rising trend first
    for i in range(n - 1):
        ts = base - pd.Timedelta(days=(n - 1 - i))
        # Rise then fall back to ~110, well below the 200 peak
        if i < 200:
            close = 100.0 + i * 0.5      # peak ~200 around bar 200
        else:
            close = max(110.0, 200.0 - (i - 200) * 0.9)
        rows.append(_bar(ts, close, 100_000.0))
    # Last bar: dip below recent EMA(20) then close back above (one-bar pullback)
    last_ts = base
    last_close = rows[-1]["close"] - 0.2
    last_low = last_close - 2.0           # dip below; if EMA(20) is around here, it touches
    rows.append({
        "timestamp": last_ts.isoformat().replace("+00:00", "Z"),
        "open": last_close - 1.5,
        "high": last_close + 0.5,
        "low":  last_low,
        "close": last_close,
        "volume": 100_000.0 * end_volume_mult,
    })
    return rows


def _barbell_series(n: int = 300, end_volume_mult: float = 1.0) -> list[dict]:
    """Past peak ~250 then past trough ~50 (so 52w high+low are both far
    from current price), and a steady recent uptrend so EMA(20) lags below
    close (bullish pullback's MIN(LOW,3) <= EMA fails) AND close stays
    above EMA (bearish pullback's CLOSE < EMA fails). Net effect: NONE of
    the four OR terms fire — stock qualifies on volume alone."""
    base = pd.Timestamp("2026-05-12 09:30", tz="UTC")
    rows = []
    for i in range(n):
        ts = base - pd.Timedelta(days=(n - 1 - i))
        if i < 100:
            close = 50.0 + i * 2.0             # 50 → 250 by bar 100 (past 52w high)
        elif i < 200:
            close = 250.0 - (i - 100) * 2.0    # 250 → 50 by bar 200 (past 52w low)
        else:
            # Steady recent uptrend from 100 to 130. Mid-range vs the past
            # extremes (not near 250 or 50). EMA(20) lags close by ~1.5
            # units so MIN(LOW, 3) sits CLEARLY above the EMA → bullish
            # pullback term fails; CLOSE > EMA so bearish pullback also fails.
            close = 100.0 + (i - 200) * 0.3
        vol = 100_000.0 if i < n - 1 else 100_000.0 * end_volume_mult
        rows.append(_bar(ts, close, vol))
    return rows


def _real_volume_breakout_discovery() -> DiscoveryConfig:
    """Materialize the real volume_breakout_52w preset's discovery config
    so tests evaluate the same conditions that production will.

    Phase 9h — the preset's conditions now contain `{placeholder}`
    tokens (volume_multiplier, near_52w_high_factor, …) that the
    orchestrator substitutes from the `parameters` block before
    scanning. This helper applies the same substitution so the tests
    feed concrete numeric thresholds to scan_universe."""
    raw = dict(kb.presets["volume_breakout_52w"].discovery)
    params = {str(k): float(v) for k, v in (raw.get("parameters") or {}).items()}
    if params:
        # Coerce integer-valued floats so e.g. lookback_window_bars=252.0
        # formats as `252` (matches what the orchestrator does).
        cleaned = {
            k: (int(v) if isinstance(v, float) and v.is_integer() else v)
            for k, v in params.items()
        }
        raw["conditions"] = [c.format(**cleaned) for c in raw.get("conditions") or []]
    # tie_break_options come back as plain dicts from the preset YAML; the
    # DiscoveryConfig schema accepts dicts via Pydantic coercion.
    return DiscoveryConfig(**raw)
