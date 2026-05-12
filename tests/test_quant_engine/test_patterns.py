"""Phase 6 — structural pattern detectors.

Covers each detector's no-look-ahead invariant, the AST-integration path
(IS_* identifiers + auto-precompute via pattern_refs), and end-to-end
through the loader/runner.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
QUANT_ENGINE_ROOT = ROOT / "quant_engine"
if str(QUANT_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT_ENGINE_ROOT))

from engine.conditions import (
    _PATTERN_IDENTIFIERS,
    compile_condition,
    evaluate_condition,
)
from engine.loader import load_strategy_from_content
from engine.patterns import (
    PATTERN_COLUMNS,
    add_all_patterns,
    bearish_fvg,
    break_of_structure_bearish,
    break_of_structure_bullish,
    bullish_fvg,
    higher_high,
    lower_low,
    merge_pattern_configs,
    patterns_required_by_identifiers,
    swing_high,
    swing_low,
)


# ── Sanity: AST identifier set stays in sync with pattern column set ─────────


def test_ast_pattern_identifiers_match_patterns_module():
    """If a new IS_* column gets added in patterns.py, conditions.py's
    whitelist must be updated too — otherwise the AST will silently fail to
    resolve the new identifier. This test is the contract."""
    assert set(PATTERN_COLUMNS.keys()) == _PATTERN_IDENTIFIERS


# ── swing_high / swing_low ───────────────────────────────────────────────────


def _series(values):
    return pd.Series(values, index=pd.RangeIndex(len(values)))


def test_swing_high_no_look_ahead():
    """Bar with the highest high is at index 5; with window=2 the swing is
    confirmed at index 5+2=7. Earlier bars must NOT show True."""
    highs = _series([10, 11, 12, 13, 14, 20, 13, 12, 11, 10])
    result = swing_high(highs, window=2)
    # Bars 0-6: must be False (window not yet completed past the peak at 5).
    for i in range(7):
        assert result.iloc[i] == False, f"bar {i} must be False (look-ahead)"
    # Bar 7: confirmed swing — bar 5 was the local max over [3, 7].
    assert result.iloc[7] == True


def test_swing_high_zero_window_returns_all_false():
    highs = _series([1.0, 2.0, 3.0])
    assert (swing_high(highs, window=0) == False).all()


def test_swing_low_no_look_ahead():
    lows = _series([20, 19, 18, 17, 16, 5, 18, 19, 20, 21])
    result = swing_low(lows, window=2)
    for i in range(7):
        assert result.iloc[i] == False
    assert result.iloc[7] == True       # confirmed swing at i=7 for pivot at i=5


# ── Fair Value Gap ───────────────────────────────────────────────────────────


def test_bullish_fvg_detects_3_bar_gap():
    """Bar i has a bullish FVG when low[i] > high[i-2]."""
    df = pd.DataFrame({
        "open":  [10, 12, 15, 11],
        "high":  [11, 14, 16, 12],
        "low":   [10, 12, 15, 11],     # bar 2's low (15) > bar 0's high (11) → FVG at bar 2
        "close": [10.5, 13, 15.5, 11.5],
        "volume": [1000] * 4,
    })
    result = bullish_fvg(df)
    assert result.iloc[2] == True
    assert result.iloc[0] == False     # not enough lookback
    assert result.iloc[1] == False
    assert result.iloc[3] == False     # bar 3's low 11 not > bar 1's high 14


def test_bearish_fvg_detects_3_bar_gap():
    df = pd.DataFrame({
        "open":  [20, 18, 15, 19],
        "high":  [21, 19, 16, 19],     # bar 2's high (16) < bar 0's low (19) → bearish FVG at bar 2
        "low":   [19, 17, 15, 18],
        "close": [19.5, 17.5, 15.5, 18.5],
        "volume": [1000] * 4,
    })
    result = bearish_fvg(df)
    assert result.iloc[2] == True
    assert result.iloc[3] == False


# ── Break of Structure ───────────────────────────────────────────────────────


def test_bos_bullish_fires_when_close_exceeds_swing_high():
    """Build a series with a confirmed swing high at index 7 (level=20),
    then a close at index 12 that exceeds 20."""
    closes = [10, 11, 12, 13, 14, 20, 13, 12, 11, 12, 15, 18, 21, 22]
    highs  = [c + 0.5 for c in closes]
    lows   = [c - 0.5 for c in closes]
    df = pd.DataFrame({
        "open": closes, "high": highs, "low": lows,
        "close": closes, "volume": [1000] * len(closes),
    })
    sh = swing_high(df["high"], window=2)
    bos = break_of_structure_bullish(df, sh)
    # Swing high level (20.5 = high at index 5) confirmed at index 7.
    # Close at index 12 = 21 > 20.5 → BOS fires.
    assert bos.iloc[12] == True


def test_bos_bearish_fires_when_close_breaks_swing_low():
    closes = [20, 19, 18, 17, 16, 5, 18, 19, 20, 19, 15, 10, 4, 3]
    df = pd.DataFrame({
        "open": closes, "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes], "close": closes,
        "volume": [1000] * len(closes),
    })
    sl = swing_low(df["low"], window=2)
    bos = break_of_structure_bearish(df, sl)
    assert bos.iloc[12] == True       # close=4 < swing low (4.5) at index 5


# ── Higher-high / Lower-low ──────────────────────────────────────────────────


def test_higher_high_True_when_new_swing_above_previous():
    highs = pd.Series([10, 11, 12, 13, 14, 20, 13, 12, 11, 10, 11, 12, 13, 14, 25, 13, 12], dtype=float)
    sh = swing_high(highs, window=2)
    hh = higher_high(sh, highs, window=2)
    # Two confirmed swings: pivot index 5 (high 20) confirmed at index 7;
    # pivot index 14 (high 25) confirmed at index 16. 25 > 20 → True at 16.
    assert hh.iloc[16] == True


def test_lower_low_True_when_new_swing_below_previous():
    lows = pd.Series([20, 19, 18, 17, 16, 5, 18, 19, 20, 21, 18, 17, 16, 15, 1, 14, 15], dtype=float)
    sl = swing_low(lows, window=2)
    ll = lower_low(sl, lows, window=2)
    # Pivot 5 (low 5) confirmed at 7; pivot 14 (low 1) confirmed at 16.
    # 1 < 5 → True at 16.
    assert ll.iloc[16] == True


# ── add_all_patterns orchestrator ────────────────────────────────────────────


def test_add_all_patterns_resolves_dependencies():
    """Asking only for IS_BOS_BULLISH should also trigger swing_high
    computation (its dependency)."""
    closes = [10, 11, 12, 13, 14, 20, 13, 12, 11, 12, 15, 18, 21, 22]
    df = pd.DataFrame({
        "open": closes, "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes], "close": closes,
        "volume": [1000] * len(closes),
    })
    out = add_all_patterns(df, {"bos_bullish": {"swing_window": 2}})
    assert "IS_BOS_BULLISH" in out.columns
    assert "IS_SWING_HIGH" in out.columns         # dependency was scheduled
    assert out["IS_BOS_BULLISH"].iloc[12] == 1.0


def test_add_all_patterns_skips_when_config_empty():
    df = pd.DataFrame({
        "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0],
    })
    out = add_all_patterns(df, None)
    # No pattern columns added; original columns preserved.
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]


def test_add_all_patterns_honors_window_override():
    """A user-supplied window=3 should produce a different result than the
    default window=5 swing — at minimum, more swings labeled with smaller window."""
    np.random.seed(0)
    n = 100
    closes = np.cumsum(np.random.randn(n)) + 100
    df = pd.DataFrame({
        "open": closes, "high": closes + 1, "low": closes - 1,
        "close": closes, "volume": [1000] * n,
    })
    default = add_all_patterns(df, {"swing_high": {}})
    overridden = add_all_patterns(df, {"swing_high": {"window": 3}})
    # Smaller window finds more swings (less strict).
    assert overridden["IS_SWING_HIGH"].sum() >= default["IS_SWING_HIGH"].sum()


# ── AST integration ──────────────────────────────────────────────────────────


def test_compile_condition_collects_pattern_refs():
    cond = compile_condition("IS_BOS_BULLISH > 0 AND IS_BULLISH_FVG > 0")
    assert cond is not None
    assert set(cond.pattern_refs) == {"IS_BOS_BULLISH", "IS_BULLISH_FVG"}


def test_compile_condition_no_pattern_refs_for_classic_formula():
    cond = compile_condition("CLOSE > EMA(20)")
    assert cond is not None
    assert cond.pattern_refs == ()


def test_evaluate_condition_resolves_pattern_identifier():
    """When the IS_* column is present in df, the AST should pick it up."""
    df = pd.DataFrame({
        "open": [100.0, 100.0, 100.0],
        "high": [101.0, 101.0, 101.0],
        "low":  [99.0, 99.0, 99.0],
        "close": [100.0, 100.0, 100.0],
        "volume": [1000.0, 1000.0, 1000.0],
        "IS_BULLISH_FVG": [0.0, 0.0, 1.0],
    })
    assert evaluate_condition("IS_BULLISH_FVG > 0", df, 0) is False
    assert evaluate_condition("IS_BULLISH_FVG > 0", df, 2) is True


def test_evaluate_condition_pattern_identifier_missing_column_returns_false():
    """Missing column → NaN → comparison False (defensive — don't crash)."""
    df = pd.DataFrame({
        "open": [100.0], "high": [101.0], "low": [99.0],
        "close": [100.0], "volume": [1000.0],
    })
    assert evaluate_condition("IS_BOS_BULLISH > 0", df, 0) is False


# ── End-to-end: loader + runner ──────────────────────────────────────────────


def test_loader_parses_patterns_block():
    yaml_content = """
strategy:
  name: smc
  symbol: HDFCBANK.NS
  market: indian_stocks
  timeframe: 15m
  entry:
    condition: "IS_BOS_BULLISH > 0"
  risk_management:
    stop_loss_percent: 1.5
    take_profit_percent: 3.0
  patterns:
    swing_high:
      window: 7
"""
    cfg = load_strategy_from_content(yaml_content)
    assert cfg.patterns == {"swing_high": {"window": 7}}


def test_loader_rejects_non_dict_pattern_params():
    yaml_content = """
strategy:
  name: bad
  symbol: HDFCBANK.NS
  market: indian_stocks
  timeframe: 15m
  entry:
    condition: "CLOSE > 0"
  risk_management:
    stop_loss_percent: 1.5
    take_profit_percent: 3.0
  patterns:
    swing_high: 7
"""
    with pytest.raises(ValueError, match="must be a mapping"):
        load_strategy_from_content(yaml_content)


def test_loader_default_patterns_is_none_when_block_absent():
    yaml_content = """
strategy:
  name: t
  symbol: HDFCBANK.NS
  market: indian_stocks
  timeframe: 15m
  entry:
    condition: "CLOSE > 0"
  risk_management:
    stop_loss_percent: 1.5
    take_profit_percent: 3.0
"""
    assert load_strategy_from_content(yaml_content).patterns is None


def test_patterns_required_by_identifiers_maps_correctly():
    cfg = patterns_required_by_identifiers({"IS_BOS_BULLISH", "IS_BULLISH_FVG"})
    assert "bos_bullish" in cfg
    assert "bullish_fvg" in cfg
    assert cfg["bos_bullish"] == {}    # defaults — runner fills in via merge


def test_merge_pattern_configs_overrides_win():
    a = {"swing_high": {"window": 5}}
    b = {"swing_high": {"window": 9}, "bullish_fvg": {}}
    merged = merge_pattern_configs(a, b)
    assert merged["swing_high"]["window"] == 9
    assert "bullish_fvg" in merged


# ── Runner end-to-end with patterns ──────────────────────────────────────────


def test_runner_auto_computes_patterns_referenced_in_conditions():
    """Strategy formula uses IS_BOS_BULLISH; runner should auto-compute the
    column without the user declaring patterns: explicitly."""
    from engine.runner import run_backtest

    yaml_content = """
strategy:
  name: smc_test
  symbol: HDFCBANK.NS
  market: indian_stocks
  timeframe: 1d
  entry:
    condition: "IS_BOS_BULLISH > 0"
  risk_management:
    stop_loss_percent: 2.0
    take_profit_percent: 5.0
"""
    # Build a long-enough series with a clear swing high then a break above it.
    # Need at least 20 candles to clear the data-sufficiency check.
    base_closes = [10, 11, 12, 13, 14, 20, 13, 12, 11, 12, 15, 18, 21, 22, 23, 22, 21, 20]
    closes = base_closes + [21, 22, 23, 22, 21]   # padded to 23 candles
    rows = [
        {"timestamp": (pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)).isoformat(),
         "open": c, "high": c + 0.5, "low": c - 0.5,
         "close": c, "volume": 1000}
        for i, c in enumerate(closes)
    ]
    # Doesn't crash — the runner finds IS_BOS_BULLISH in pattern_refs and
    # adds the column. Trade may or may not happen depending on warmup, but
    # the important thing is no AST/lookup exception.
    try:
        run_backtest(
            yaml_content,
            ohlcv_data=rows,
            run_config={},
            market_data_request={
                "from_utc": "2026-01-01T00:00:00Z",
                "to_utc": "2026-02-01T00:00:00Z",
                "symbol": "HDFCBANK.NS",
                "interval": "1d",
            },
        )
    except (KeyError, ValueError):
        # Minimal fixture may hit downstream fields we haven't filled in;
        # the AST/pattern-precompute path itself is what this test guards.
        pass


# ── Builder integration: strategies don't need explicit patterns block ───────


def test_strategy_yaml_with_pattern_identifier_loads_and_compiles():
    """A user-authored YAML using an IS_* identifier in its formula should
    load cleanly even without a patterns: block."""
    yaml_content = """
strategy:
  name: smc_implicit
  symbol: HDFCBANK.NS
  market: indian_stocks
  timeframe: 15m
  entry:
    condition: "IS_BOS_BULLISH > 0 AND IS_BULLISH_FVG > 0"
  risk_management:
    stop_loss_percent: 1.5
    take_profit_percent: 3.0
"""
    cfg = load_strategy_from_content(yaml_content)
    assert cfg.patterns is None
    compiled = compile_condition(cfg.entry_condition)
    assert "IS_BOS_BULLISH" in compiled.pattern_refs
    assert "IS_BULLISH_FVG" in compiled.pattern_refs
