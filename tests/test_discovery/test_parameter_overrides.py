"""Phase 9h — user-typed discovery thresholds must override the preset's
hardcoded defaults.

The bug: the preset's discovery condition was hardcoded to
`VOL > AVG(VOL, 20) * 2.0`. When the user typed "create intraday
strategy on NSE stock whose volume spike up today 1.2x", the scanner
still used the 2× threshold (ignoring the user's relaxation) and
returned `no stocks matched` even though 1.2× is much more permissive.

The fix: the preset declares a `parameters` block with placeholders in
its conditions (`{volume_multiplier}`, `{near_52w_high_factor}`, …);
the chat layer parses user-typed thresholds and stashes them on
`builder.discovery_parameter_overrides`; the orchestrator merges
defaults + overrides and substitutes them before passing conditions
to the scanner.

These tests pin the entire override pipeline:
  1. _extract_discovery_parameter_overrides parses prose correctly
  2. The orchestrator's `_preset_discovery_config` substitutes the
     merged params into the conditions
  3. Unknown override keys are rejected (typo guard)
  4. The no-match message surfaces the parameters_used so the user
     can see whether their override was honored
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


if "asyncpg" not in sys.modules:
    sys.modules["asyncpg"] = types.ModuleType("asyncpg")


# ── parameter-override extraction (chat_service.py) ─────────────────────────


_CHAT_SERVICE_PATH = (
    Path(__file__).resolve().parents[2]
    / "app" / "services" / "chat" / "chat_service.py"
)


def _load_extract_overrides():
    """Compile _extract_discovery_parameter_overrides + its regex
    constants out of chat_service.py without importing the module
    (asyncpg gap)."""
    import re as re_module
    src = _CHAT_SERVICE_PATH.read_text(encoding="utf-8")
    helpers_idx = src.index("_VOLUME_MULTIPLIER_RE = re.compile")
    # walk back over any comment lines preceding the constant
    block_start = helpers_idx
    while block_start > 0:
        prev_nl = src.rfind("\n", 0, block_start - 1)
        line_start = prev_nl + 1 if prev_nl != -1 else 0
        line = src[line_start:block_start - 1].strip()
        if line.startswith("#"):
            block_start = line_start
        else:
            break
    # The function lives right after the constants; find the next def
    # AFTER `_extract_discovery_parameter_overrides`.
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


def test_extract_overrides_returns_empty_for_empty_message():
    extract = _load_extract_overrides()
    assert extract("") == {}
    assert extract("   ") == {}


def test_extract_overrides_captures_volume_multiplier_when_context_mentions_volume():
    extract = _load_extract_overrides()
    out = extract(
        "create intraday strategy on NSE stock whose volume spike up today 1.2x"
    )
    assert out == {"volume_multiplier": 1.2}


def test_extract_overrides_captures_integer_volume_multiplier():
    extract = _load_extract_overrides()
    assert extract("find stocks with 3x relative volume") == {"volume_multiplier": 3.0}


def test_extract_overrides_ignores_bare_Nx_without_volume_context():
    """`5x leverage` should NOT be interpreted as volume_multiplier."""
    extract = _load_extract_overrides()
    assert extract("I want a 5x leveraged position on RELIANCE") == {}


def test_extract_overrides_drops_volume_multiplier_outside_sanity_bounds():
    """200x volume is almost certainly a typo — drop it rather than
    feeding it into the scanner."""
    extract = _load_extract_overrides()
    assert extract("volume spike 200x today") == {}
    assert extract("volume 0.1x") == {}    # below lower bound


def test_extract_overrides_captures_near_52w_high_percentage():
    """The "5%" extracts as the factor. The "52-week" phrasing also
    extracts as the lookback window (52 × 5 trading days = 260 bars)
    via the Phase 9i parser — both should be captured."""
    extract = _load_extract_overrides()
    out = extract("find stocks within 5% of 52-week high")
    assert out["near_52w_high_factor"] == 0.95
    assert out["lookback_window_bars"] == 260.0


def test_extract_overrides_captures_near_52w_low_percentage():
    extract = _load_extract_overrides()
    out = extract("intraday strategy within 3% of 52 week low")
    assert out["near_52w_low_factor"] == 1.03
    assert out["lookback_window_bars"] == 260.0


def test_extract_overrides_can_capture_multiple_parameters_in_one_message():
    extract = _load_extract_overrides()
    out = extract(
        "volume spike 1.5x today AND within 5% of 52-week high on NSE"
    )
    assert out["volume_multiplier"] == 1.5
    assert out["near_52w_high_factor"] == 0.95
    assert out["lookback_window_bars"] == 260.0


def test_extract_overrides_drops_out_of_range_percentages():
    extract = _load_extract_overrides()
    # 50% from 52-week high is not "near" — refuse the factor. The
    # window is still extracted (the 9i parser doesn't depend on the
    # percentage being valid).
    out_hi = extract("within 50% of 52-week high")
    assert "near_52w_high_factor" not in out_hi
    out_lo = extract("within 0% of 52-week low")
    assert "near_52w_low_factor" not in out_lo


# ── orchestrator: parameter merge + substitution ────────────────────────────


from app.services.discovery.orchestrator import (
    _apply_parameters_to_condition,
    _merge_parameters,
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


def test_merge_parameters_returns_defaults_when_no_overrides():
    merged = _merge_parameters({"volume_multiplier": 2.0, "near_52w_high_factor": 0.98}, None)
    assert merged == {"volume_multiplier": 2.0, "near_52w_high_factor": 0.98}


def test_merge_parameters_overrides_win_over_defaults():
    merged = _merge_parameters(
        {"volume_multiplier": 2.0, "near_52w_high_factor": 0.98},
        {"volume_multiplier": 1.2},
    )
    assert merged == {"volume_multiplier": 1.2, "near_52w_high_factor": 0.98}


def test_merge_parameters_drops_unknown_override_keys():
    """Typoed override keys must NOT silently add new params — that
    would let a bad chat-layer change inject untracked thresholds."""
    merged = _merge_parameters(
        {"volume_multiplier": 2.0},
        {"volume_multiplier": 1.5, "unknown_param": 99.0},
    )
    assert merged == {"volume_multiplier": 1.5}
    assert "unknown_param" not in merged


def test_merge_parameters_drops_non_numeric_values():
    merged = _merge_parameters(
        {"volume_multiplier": 2.0},
        {"volume_multiplier": "not-a-number"},
    )
    # The override is dropped but the default survives.
    assert merged == {"volume_multiplier": 2.0}


def test_apply_parameters_substitutes_placeholders():
    out = _apply_parameters_to_condition(
        "VOL > AVG(VOL, 20) * {volume_multiplier}",
        {"volume_multiplier": 1.2},
    )
    assert out == "VOL > AVG(VOL, 20) * 1.2"


def test_apply_parameters_raises_on_unknown_placeholder():
    """A placeholder referenced in a condition but missing from the
    params dict must blow up loudly (not silently leave the token in
    place)."""
    with pytest.raises(ValueError) as exc:
        _apply_parameters_to_condition(
            "VOL > {missing_param}", {"volume_multiplier": 2.0}
        )
    assert "missing_param" in str(exc.value)


def test_preset_discovery_config_substitutes_default_parameters():
    """No user overrides → use the preset's defaults verbatim.

    Phase 9i — integer-valued params (volume_multiplier=2.0,
    lookback_window_bars=252) render as int strings (`* 2`, `MAX(HIGH,
    252)`), not floats. The AST accepts both, but the int form reads
    cleanly in logs."""
    b = _builder_with_preset()
    cfg = _preset_discovery_config(b)
    assert cfg is not None
    # The first condition must now contain a literal numeric multiplier,
    # not a `{placeholder}` token.
    assert any("AVG(VOL, 20) * 2" in c for c in cfg.conditions), (
        "preset default volume_multiplier=2.0 should appear in the "
        "substituted condition; got: " + repr(cfg.conditions)
    )
    # The 52w factors must also be substituted (0.98 / 1.02).
    big_or = next(c for c in cfg.conditions if "MAX(HIGH, 252)" in c)
    assert "0.98" in big_or
    assert "1.02" in big_or
    # And the effective parameters must be recorded on the builder so
    # the no-match message can show them.
    assert b.discovery_parameters_used == {
        "volume_multiplier": 2.0,
        "near_52w_high_factor": 0.98,
        "near_52w_low_factor": 1.02,
        "lookback_window_bars": 252.0,
    }


def test_preset_discovery_config_substitutes_user_overrides():
    """User override on volume_multiplier → that value appears in the
    condition, while the other defaults are retained."""
    b = _builder_with_preset()
    b.discovery_parameter_overrides = {"volume_multiplier": 1.2}
    cfg = _preset_discovery_config(b)
    assert cfg is not None
    assert any("AVG(VOL, 20) * 1.2" in c for c in cfg.conditions)
    # Other defaults still in place.
    big_or = next(c for c in cfg.conditions if "MAX(HIGH, 252)" in c)
    assert "0.98" in big_or
    # parameters_used reflects merge.
    assert b.discovery_parameters_used["volume_multiplier"] == 1.2
    assert b.discovery_parameters_used["near_52w_high_factor"] == 0.98


def test_preset_discovery_config_ignores_unknown_override_keys():
    b = _builder_with_preset()
    b.discovery_parameter_overrides = {"this_key_does_not_exist": 42.0}
    cfg = _preset_discovery_config(b)
    assert cfg is not None
    # Conditions still substitute the preset defaults; unknown key
    # is silently dropped (with a log warning we don't assert on).
    assert any("AVG(VOL, 20) * 2" in c for c in cfg.conditions)


# ── no-match reply surfaces the threshold used ──────────────────────────────


from app.services.chat.strategy_flow import build_discovery_no_match_reply


def test_no_match_reply_omits_thresholds_when_unspecified():
    """Backward compat: a caller that doesn't pass parameters_used
    gets the same message as before."""
    msg = build_discovery_no_match_reply()
    assert "No stocks matched" in msg
    assert "Thresholds used:" not in msg


def test_no_match_reply_includes_volume_threshold():
    msg = build_discovery_no_match_reply(
        parameters_used={"volume_multiplier": 1.2},
    )
    assert "Thresholds used:" in msg
    assert "1.2" in msg
    assert "20-day average" in msg


def test_no_match_reply_includes_52w_thresholds_translated_to_percent():
    msg = build_discovery_no_match_reply(
        parameters_used={
            "volume_multiplier": 1.5,
            "near_52w_high_factor": 0.95,
            "near_52w_low_factor": 1.03,
        },
    )
    assert "1.5" in msg
    # Factor 0.95 → 5% of 52w high
    assert "5%" in msg or "5 %" in msg
    assert "52-week high" in msg
    # Factor 1.03 → 3% of 52w low
    assert "3%" in msg or "3 %" in msg
    assert "52-week low" in msg


# ── End-to-end: scanner sees the user's threshold ───────────────────────────


import asyncio

from app.services.discovery.scanner import scan_universe


@pytest.mark.asyncio
async def test_user_override_makes_a_previously_failing_stock_qualify():
    """The whole point: at 2.0× the stock fails the mandatory volume
    condition; at 1.2× the same stock qualifies. This test simulates
    the exact code path production runs."""
    import pandas as pd

    target = "HDFCBANK.NS"

    # Bullish-pullback series with end_volume_mult=1.5 — fails 2.0× but
    # passes 1.2×. Other universe stocks have volume_mult=1.0 so they
    # never qualify.
    def _bar(ts, c, v):
        return {
            "timestamp": ts.isoformat().replace("+00:00", "Z"),
            "open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": v,
        }

    def _pullback_series_15x_volume(n=300):
        base = pd.Timestamp("2026-05-12 09:30", tz="UTC")
        rows = []
        # Long trend up so the last bar pulls back to EMA(20) but stays
        # above it (bullish pullback condition).
        for i in range(n):
            ts = base + pd.Timedelta(minutes=i)
            # rising trend
            close = 100.0 + i * 0.05
            vol = 100_000.0
            rows.append(_bar(ts, close, vol))
        # Final bar: shallow pullback near EMA(20), volume × 1.5 the avg
        last_avg_vol = 100_000.0
        rows[-1]["close"] = rows[-2]["close"] - 0.1
        rows[-1]["low"]   = rows[-2]["close"] - 0.6
        rows[-1]["volume"] = last_avg_vol * 1.5
        return rows

    def _flat_series(n=300):
        base = pd.Timestamp("2026-05-12 09:30", tz="UTC")
        return [_bar(base + pd.Timedelta(minutes=i), 100.0, 100_000.0) for i in range(n)]

    async def stub_fetcher(symbol, interval, from_utc, to_utc):
        if symbol == target:
            return _pullback_series_15x_volume()
        return _flat_series()

    b = _builder_with_preset()

    # Run 1: default 2.0× threshold → target fails (its 1.5× < 2.0×).
    cfg_default = _preset_discovery_config(b)
    result_default = await scan_universe(
        cfg_default, interval="5m", fetch_ohlcv=stub_fetcher,
    )
    assert target not in {c.symbol for c in result_default.candidates}, (
        "at default 2.0× volume threshold the target should NOT qualify "
        "(it only has 1.5× volume)"
    )

    # Run 2: user override 1.2× → target qualifies.
    b2 = _builder_with_preset()
    b2.discovery_parameter_overrides = {"volume_multiplier": 1.2}
    cfg_override = _preset_discovery_config(b2)
    result_override = await scan_universe(
        cfg_override, interval="5m", fetch_ohlcv=stub_fetcher,
    )
    assert target in {c.symbol for c in result_override.candidates}, (
        "with user override volume_multiplier=1.2 the target's 1.5× "
        "volume should qualify; got candidates: "
        f"{[c.symbol for c in result_override.candidates]}"
    )
    # The effective threshold was recorded.
    assert b2.discovery_parameters_used["volume_multiplier"] == 1.2
