"""
Phase 5 — catalog-picker generality corpus (MEASUREMENT, not a one-off fix).

A broad spread of DIFFERENT signals, each written the natural way a user would
(with connective words like "is", "drops", "turns", "the"), across momentum,
trend, volatility, volume, and candlestick families. This is how we keep "does
the picker honour what the user asked for?" a number we can grow — add new
(prompt → expected signal) rows freely.

Known gaps are kept as xfail with a reason, so the corpus is honest about what
still needs work rather than hiding it.
"""
from __future__ import annotations

import pytest

from app.kb import kb
from app.planner.catalog_signal_picker import pick_plan_from_catalog


def _entry_names(prompt: str, sentiment: str) -> set[str]:
    pick = pick_plan_from_catalog(prompt, kb=kb, timeframe="15m", sentiment=sentiment)
    return {s.get("name") for s in pick.signal_plan.get("entry", []) if s.get("name")}


def _exit_names(prompt: str, sentiment: str) -> set[str]:
    pick = pick_plan_from_catalog(prompt, kb=kb, timeframe="15m", sentiment=sentiment)
    return {s.get("name") for s in pick.signal_plan.get("exit", []) if s.get("name")}


# (prompt, expected entry signal, sentiment) — natural phrasing across families.
ENTRY_CORPUS = [
    # momentum
    ("Buy RELIANCE when the RSI is oversold below 30.",          "rsi_oversold",        "bullish"),
    ("Go long when RSI is above 50.",                            "rsi_above_50",        "bullish"),
    ("Long when the stochastic is oversold.",                    "stoch_oversold",      "bullish"),
    ("Buy when CCI is below -100.",                              "cci_oversold",        "bullish"),
    ("Enter when the Williams %R is oversold.",                  "williams_oversold",   "bullish"),
    ("Go long when the MFI drops below 20.",                     "mfi_oversold",        "bullish"),
    ("Enter when CMO is oversold.",                              "cmo_oversold",        "bullish"),
    ("Long when the ultimate oscillator is oversold.",           "ultosc_oversold",     "bullish"),
    ("Go long when ROC turns positive.",                         "roc_positive",        "bullish"),
    ("Enter when momentum is positive.",                         "momentum_positive",   "bullish"),
    ("Buy when the PPO is positive.",                             "ppo_positive",        "bullish"),
    ("Enter when the MACD crosses above its signal line.",       "macd_bullish_cross",  "bullish"),
    ("Sell when the MACD crosses below the signal line.",        "macd_bearish_cross",  "bearish"),
    # trend
    ("Buy when the 9 EMA crosses above the 21 EMA.",             "ema_cross_up",        "bullish"),
    ("Go short when the fast EMA crosses below the slow EMA.",   "ema_cross_down",      "bearish"),
    ("Enter when the price is above the 50 EMA.",                "price_above_ema",     "bullish"),
    ("Buy when ADX is above 25.",                                "adx_strong_trend",    "bullish"),
    ("Buy when ADXR is above 25.",                               "adxr_strong_trend",   "bullish"),
    ("Enter when the supertrend turns bullish.",                 "supertrend_bullish",  "bullish"),
    ("Short when the supertrend flips bearish.",                 "supertrend_bearish",  "bearish"),
    ("Long when the Aroon is bullish.",                          "aroon_bullish",       "bullish"),
    ("Buy when price is above the parabolic SAR.",               "sar_bullish",         "bullish"),
    ("Enter when price is above the KAMA.",                      "price_above_kama",    "bullish"),
    # volatility / channels
    ("Buy when the price moves above the upper Bollinger band.", "price_above_bb_upper","bullish"),
    ("Sell when price falls below the lower Bollinger band.",    "price_below_bb_lower","bearish"),
    ("Buy on a Donchian breakout above the 20-period high.",     "donchian_breakout_up","bullish"),
    # volume
    ("Go long when volume spikes above average.",                "volume_spike",        "bullish"),
    ("Enter when OBV is rising.",                                "obv_rising",          "bullish"),
    # vwap
    ("Buy when price reclaims the VWAP.",                        "vwap_reclaim_bullish","bullish"),
    ("Long when price is 1.5% below VWAP.",                      "vwap_below_pct",      "bullish"),
    ("Short when price is 2% above VWAP.",                       "vwap_above_pct",      "bearish"),
    # candlesticks
    ("Enter long on a bullish engulfing candle.",               "bullish_engulfing",   "bullish"),
    ("Short on a bearish engulfing pattern.",                    "bearish_engulfing",   "bearish"),
    ("Buy on a hammer candle.",                                  "hammer_candle",       "bullish"),
    ("Sell on a shooting star.",                                 "shooting_star_candle","bearish"),
    ("Enter when a morning star forms.",                         "morning_star",        "bullish"),
]


@pytest.mark.parametrize("prompt,expected,sentiment", ENTRY_CORPUS)
def test_entry_signal_picked(prompt, expected, sentiment):
    got = _entry_names(prompt, sentiment)
    assert expected in got, f"{expected!r} not picked for {prompt!r}; got {got}"


# Overbought oscillators are modelled as long-EXIT triggers; test that role.
EXIT_CORPUS = [
    ("Exit the long when the RSI is overbought.",       "rsi_overbought"),
    ("Close longs when the stochastic is overbought.",  "stoch_overbought"),
    ("Take profit when CCI rises above 100.",           "cci_overbought"),
]


@pytest.mark.parametrize("prompt,expected", EXIT_CORPUS)
def test_exit_signal_picked(prompt, expected):
    assert expected in _exit_names(prompt, "bullish"), (
        f"{expected!r} not picked as exit for {prompt!r}"
    )


# ── VWAP percentage-distance cards (the validated "N% from VWAP" gap) ──────────

def _entry_params(prompt: str, sentiment: str, name: str) -> dict:
    pick = pick_plan_from_catalog(prompt, kb=kb, timeframe="5m", sentiment=sentiment)
    for s in pick.signal_plan.get("entry", []):
        if s.get("name") == name:
            return s.get("params") or {}
    return {}


@pytest.mark.parametrize("prompt,name,sentiment,threshold", [
    ("Long when price is 1.5% below VWAP.",        "vwap_below_pct", "bullish", 1.5),
    ("Buy when price is 1.5 percent below VWAP.",  "vwap_below_pct", "bullish", 1.5),
    ("Short when price is 2% above VWAP.",         "vwap_above_pct", "bearish", 2.0),
])
def test_vwap_pct_threshold_extracted(prompt, name, sentiment, threshold):
    # The trader's exact percentage must be captured, not silently defaulted.
    assert _entry_params(prompt, sentiment, name).get("threshold") == threshold


def test_vwap_pct_distinct_from_zscore():
    # Std-dev phrasing keeps selecting the z-score card, not the percent card.
    assert "vwap_below_pct" not in _entry_names("Enter when VWAP z-score is below -2.", "bullish")


# ── Known gaps (honest xfail — each is a distinct, documented issue) ───────────

@pytest.mark.xfail(reason="TRIX phrase doesn't handle 'crosses above zero' "
                          "(word-number 'zero' + 'crosses') — card-phrase gap.")
def test_trix_word_number_gap():
    assert "trix_positive" in _entry_names("Buy when TRIX crosses above zero.", "bullish")


@pytest.mark.xfail(reason="Short-entry on an overbought oscillator needs the "
                          "dual-direction / mirror machinery (overbought cards are "
                          "exit_trigger by design). Separate gap from phrase matching.")
def test_overbought_as_short_entry_dual_direction_gap():
    assert "rsi_overbought" in _entry_names("Short when the RSI is overbought above 70.", "bearish")
