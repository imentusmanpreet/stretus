"""Phase 9l — corpus of realistic trader prompts.

Acts as both regression coverage AND living documentation of what the
discovery pipeline supports out-of-the-box (regex extractor; the LLM
fallback handles anything regex misses at runtime). 50+ phrasings
across volume, RSI, VWAP, EMA, 52-week proximity, breakout, pullback,
day high/low, and combinations thereof.

Each test asserts a SUBSET — `expected ⊆ extracted_names` — rather
than equality, because the regex is conservative and may capture
extra primitives the user implied. The looser assertion lets us
prove "the user's intent is captured" without locking in over-
specification.

If a prompt fails here, the regex needs broadening (or the LLM
fallback is the only path that catches it). New trader phrasings
should be appended to the parametrize list as they come up — the
file IS the supported-vocabulary documentation.
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


def _load_extractors():
    """Compile the regex extractors from chat_service.py without
    importing the module (asyncpg gap)."""
    import re as re_module
    from typing import Any, Optional
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
    fn_marker = "def _extract_discovery_conditions("
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
    namespace: dict = {"re": re_module, "Optional": Optional, "Any": Any}
    exec(body, namespace)
    return (
        namespace["_extract_discovery_parameter_overrides"],
        namespace["_extract_discovery_conditions"],
    )


def _names(message: str) -> set[str]:
    extract_overrides, extract_conds = _load_extractors()
    overrides = extract_overrides(message)
    return {c["name"] for c in extract_conds(message, overrides)}


# ── Volume-only prompts ────────────────────────────────────────────────────


@pytest.mark.parametrize("prompt", [
    "create intraday strategy on NSE stock whose volume spike up today 1.2x",
    "find stocks with 2x volume today",
    "scan for stocks where volume is 1.5x its average",
    "stocks with 3x relative volume",
    "any NSE stock with volume above 2x its 20-day average",
    "intraday strategy where today's volume is 2.5x normal",
    "find NSE stocks with elevated volume",
    "high volume stocks for intraday",
    "scan for strong volume on NSE",
    "stocks doing 4x volume today",
    "stocks with volume 2 times the 20-day average",
    "find scrips where volume is more than 1.5x average",
])
def test_volume_only_prompts_capture_volume_spike(prompt):
    names = _names(prompt)
    assert "volume_spike" in names, f"missed in: {prompt!r} → {names}"


# ── 52-week / lookback proximity ──────────────────────────────────────────


@pytest.mark.parametrize("prompt", [
    "stocks near 52-week high",
    "scan for NSE stocks approaching 52-week high",
    "find stocks close to 52w high",
    "intraday strategy on stocks near year high",
    "stocks at 1-year high",
    "scan for stocks near record high",
    "find stocks approaching 20-day high",
    "stocks near 100 day high",
    "stocks near 6-month high",
    "intraday strategy near 52-week peak",
])
def test_near_high_prompts_capture_near_52w_high(prompt):
    names = _names(prompt)
    assert "near_52w_high" in names, f"missed in: {prompt!r} → {names}"


@pytest.mark.parametrize("prompt", [
    "stocks near 52-week low",
    "scan for NSE stocks approaching 52-week low",
    "find stocks close to 52w low",
    "intraday strategy on stocks near year low",
    "stocks at 1-year low",
    "find stocks approaching 20-day low",
    "stocks near 100 day low",
    "stocks near 6-month low",
])
def test_near_low_prompts_capture_near_52w_low(prompt):
    names = _names(prompt)
    assert "near_52w_low" in names, f"missed in: {prompt!r} → {names}"


# ── Breakout / breakdown ──────────────────────────────────────────────────


@pytest.mark.parametrize("prompt", [
    "find stocks breaking above 52-week high",
    "scan for stocks breaking 52-week high",
    "stocks above 52w high",
    "find new 52-week highs",
    "stocks breaking out above year high",
    "stocks crossing above 50-day high",
])
def test_breakout_high_prompts_capture_above_52w_high(prompt):
    names = _names(prompt)
    assert "above_52w_high" in names, f"missed in: {prompt!r} → {names}"


@pytest.mark.parametrize("prompt", [
    "find stocks breaking below 52-week low",
    "scan for stocks breaking 52-week low",
    "stocks below 52w low",
    "find new 52-week lows",
    "stocks breaking down below year low",
])
def test_breakdown_prompts_capture_below_52w_low(prompt):
    names = _names(prompt)
    assert "below_52w_low" in names, f"missed in: {prompt!r} → {names}"


# ── RSI ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("prompt,expected_threshold", [
    ("stocks with RSI above 60", 60.0),
    ("scan for RSI > 70", 70.0),
    ("find stocks where RSI is greater than 65", 65.0),
    ("intraday with RSI above 60 and volume spike", 60.0),
    ("stocks where RSI exceeds 80", 80.0),
])
def test_rsi_above_prompts(prompt, expected_threshold):
    extract_overrides, extract_conds = _load_extractors()
    overrides = extract_overrides(prompt)
    conds = extract_conds(prompt, overrides)
    rsi = next((c for c in conds if c["name"] == "rsi_above"), None)
    assert rsi is not None, f"missed RSI above in: {prompt!r}"
    assert rsi["params"]["threshold"] == expected_threshold


@pytest.mark.parametrize("prompt,expected_threshold", [
    ("stocks with RSI below 30", 30.0),
    ("scan for RSI < 40", 40.0),
    ("find stocks where RSI is less than 35", 35.0),
    ("stocks where RSI under 25", 25.0),
])
def test_rsi_below_prompts(prompt, expected_threshold):
    extract_overrides, extract_conds = _load_extractors()
    overrides = extract_overrides(prompt)
    conds = extract_conds(prompt, overrides)
    rsi = next((c for c in conds if c["name"] == "rsi_below"), None)
    assert rsi is not None, f"missed RSI below in: {prompt!r}"
    assert rsi["params"]["threshold"] == expected_threshold


# ── VWAP ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("prompt", [
    "stocks above VWAP",
    "find scripts above the VWAP",
    "scan for stocks closing above VWAP",
    "candle closes above VWAP",
    "intraday with VWAP confirmation",
    "stocks trading above VWAP",
    "price above VWAP",
])
def test_above_vwap_prompts(prompt):
    names = _names(prompt)
    assert "above_vwap" in names, f"missed in: {prompt!r} → {names}"


@pytest.mark.parametrize("prompt", [
    "stocks below VWAP",
    "scan for stocks closing below VWAP",
    "price below VWAP",
])
def test_below_vwap_prompts(prompt):
    names = _names(prompt)
    assert "below_vwap" in names, f"missed in: {prompt!r} → {names}"


# ── EMA ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("prompt,expected_period", [
    ("stocks above 20 EMA", 20),
    ("price above the 50 EMA", 50),
    ("scan for stocks above 200-day EMA", 200),
    ("stocks where price > EMA(20)", 20),
    ("close above 9 ema", 9),
    ("price above 50 day ema", 50),
])
def test_above_ema_prompts(prompt, expected_period):
    extract_overrides, extract_conds = _load_extractors()
    overrides = extract_overrides(prompt)
    conds = extract_conds(prompt, overrides)
    ema = next((c for c in conds if c["name"] == "above_ema"), None)
    assert ema is not None, f"missed EMA above in: {prompt!r}"
    assert ema["params"]["period"] == expected_period


# ── Pullback ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("prompt", [
    "find stocks with low pullback",
    "shallow pullback to EMA",
    "stocks where pullback is shallow",
    "scan for shallow retracement",
    "stocks with pullback less than 1%",
    "find stocks with pullback < 2%",
    "minor pullback after breakout",
    "limited pullback strategy",
])
def test_pullback_long_prompts_capture_shallow_pullback_long(prompt):
    names = _names(prompt)
    assert "shallow_pullback_long" in names, f"missed in: {prompt!r} → {names}"


@pytest.mark.parametrize("prompt", [
    "bearish stocks with weak recovery after pullback",
    "short stocks where pullback is shallow",
    "downside stocks with low pullback",
])
def test_pullback_short_context_picks_short_variant(prompt):
    names = _names(prompt)
    assert "shallow_pullback_short" in names, f"missed in: {prompt!r} → {names}"


# ── Day high / low ────────────────────────────────────────────────────────


@pytest.mark.parametrize("prompt", [
    "stocks near day high",
    "scan for stocks close to today's high",
    "stocks approaching session high",
    "intraday near day high with volume spike",
    "stock is close to day high",
    "stocks near intraday high",
])
def test_near_day_high_prompts(prompt):
    names = _names(prompt)
    assert "near_day_high" in names, f"missed in: {prompt!r} → {names}"


# ── Multi-condition combinations (the heart of the use case) ──────────────


def test_combo_volume_rsi_vwap_dayhigh():
    names = _names(
        "Build a scanner for stocks where volume spike is above 3x, "
        "RSI is above 60, price is above VWAP, and stock is close to day high"
    )
    assert names >= {"volume_spike", "rsi_above", "above_vwap", "near_day_high"}


def test_combo_volume_pullback_52w_high():
    names = _names(
        "today's volume is 2x higher than 20-day average volume, "
        "price is near 52-week high, and pullback is less than 1%"
    )
    assert names >= {"volume_spike", "near_52w_high", "shallow_pullback_long"}


def test_combo_high_volume_ema_pullback_vwap():
    names = _names(
        "Create a pullback strategy for high-volume stocks where price is "
        "above 20 EMA, pullback is shallow, and candle closes above VWAP"
    )
    assert names >= {"volume_spike", "above_ema", "shallow_pullback_long", "above_vwap"}


def test_combo_breakout_volume_vwap():
    names = _names(
        "Find NSE stocks breaking above 52-week high with strong volume "
        "and create a breakout strategy with VWAP confirmation"
    )
    assert names >= {"above_52w_high", "volume_spike", "above_vwap"}


def test_combo_bearish_52w_low_volume_pullback():
    names = _names(
        "Create bearish intraday strategy for stocks near 52-week low with "
        "2x volume spike and weak recovery after pullback"
    )
    assert names >= {"near_52w_low", "volume_spike", "shallow_pullback_short"}


def test_combo_rsi_oversold_breakdown():
    names = _names(
        "scan for stocks with RSI below 30 breaking below 52-week low"
    )
    assert names >= {"rsi_below", "below_52w_low"}


def test_combo_multi_ema_check():
    names = _names(
        "stocks above 20 EMA and above 50 EMA with strong volume"
    )
    # First EMA period is captured (we don't enforce both — single
    # `above_ema` primitive per call is fine; multi-EMA support is a
    # future enhancement and the LLM can still produce two entries).
    assert "above_ema" in names
    assert "volume_spike" in names


# ── Negative cases: no discovery primitives in unrelated prose ────────────


@pytest.mark.parametrize("prompt", [
    "hi how are you",
    "show me the tutorial",
    "what is the difference between EMA and SMA",
    "5m",
    "ok proceed",
    "i am happy with this",
    "RELIANCE.NS",
    "use Infosys",
])
def test_unrelated_prose_yields_no_primitives(prompt):
    """Don't false-positive on bare confirmations / single tickers /
    educational questions."""
    names = _names(prompt)
    assert names == set(), f"false positive on: {prompt!r} → {names}"
