"""
tests/test_planner/corpus.py — Measurement corpus for SDL golden-SDL tests.

Each CorpusEntry is a (prompt, expected-SDL-structure, requirements-list) triple.
Requirements are atomic user clauses; the offline judge verifies that every
requirement appears in provenance (field_sources=user OR unmapped_details).

This file NEVER drives prompt-specific code in production — it only drives
measurement tests. Add entries freely; do NOT add code that branches on
prompt text outside of tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PromptRequirement:
    """One atomic user clause.

    text       — the user's words (verbatim or paraphrased).
    category   — signal | risk | universe | direction | gate | timeframe
    sdl_hint   — dotted SDL path where this should appear, or "unmapped" if expected to be missed.
    """
    text: str
    category: str
    sdl_hint: str  # e.g. "legs.0.entry.trigger", "risk.stop_loss", "unmapped"


@dataclass
class CorpusEntry:
    prompt: str
    description: str  # human label
    requirements: list[PromptRequirement] = field(default_factory=list)
    # Expected golden SDL fields (subset to check — not the full SDL)
    expected: dict[str, Any] = field(default_factory=dict)


# ── Corpus ────────────────────────────────────────────────────────────────────

CORPUS: list[CorpusEntry] = [

    CorpusEntry(
        prompt="Buy ETH on the 15-minute chart when RSI drops below 30. Stop loss 2%, take profit 2:1.",
        description="simple_eth_rsi_mean_reversion",
        requirements=[
            PromptRequirement("Buy (long direction)",     "direction",  "legs.0.direction"),
            PromptRequirement("ETH symbol",               "universe",   "universe.symbol"),
            PromptRequirement("15-minute timeframe",      "timeframe",  "context.timeframe"),
            PromptRequirement("RSI drops below 30",       "signal",     "legs.0.entry.trigger"),
            PromptRequirement("Stop loss 2%",             "risk",       "risk.stop_loss"),
            PromptRequirement("Take profit 2:1",          "risk",       "risk.take_profit"),
        ],
        expected={
            "universe.type":                   "static",
            "universe.asset_class":            "crypto_spot",
            "context.timeframe":               "15m",
            "legs.0.direction":                "long",
            "legs.0.entry.trigger.name_family":"RSI",
            "risk.stop_loss.type":             "percent",
            "risk.stop_loss.value":            2.0,
            "risk.take_profit.type":           "rr",
            "risk.take_profit.ratio":          2.0,
        },
    ),

    CorpusEntry(
        prompt=(
            "Enter above Bollinger Band upper on BTC 1-hour chart. "
            "Stop loss 1.5% below entry."
        ),
        description="btc_bollinger_breakout_explicit_indicator",
        requirements=[
            PromptRequirement("Bollinger Band (explicit, not Keltner)", "signal",   "legs.0.entry.trigger"),
            PromptRequirement("BTC symbol",                              "universe", "universe.symbol"),
            PromptRequirement("1-hour timeframe",                        "timeframe","context.timeframe"),
            PromptRequirement("Stop loss 1.5%",                         "risk",     "risk.stop_loss"),
        ],
        expected={
            "universe.type":                      "static",
            "universe.asset_class":               "crypto_spot",
            "context.timeframe":                  "1h",
            "legs.0.entry.trigger.name_family":   "BB",      # must be BB, NOT Keltner
            "risk.stop_loss.type":                "percent",
            "risk.stop_loss.value":               1.5,
        },
    ),

    CorpusEntry(
        prompt=(
            "On the 15-minute chart, trade the NSE stock with the highest relative volume "
            "that is also above its VWAP. Go long on a breakout above the opening-range high. "
            "ATR stop loss."
        ),
        description="nse_dynamic_orb_highest_rvol",
        requirements=[
            PromptRequirement("NSE stock (equity_cash)",           "universe",  "universe"),
            PromptRequirement("Highest relative volume (rvol rank)", "universe", "universe.rank.by"),
            PromptRequirement("Above VWAP screen condition",       "universe",  "universe.screen"),
            PromptRequirement("15-minute timeframe",               "timeframe", "context.timeframe"),
            PromptRequirement("Long direction",                    "direction", "legs.0.direction"),
            PromptRequirement("Opening-range high breakout",       "signal",    "legs.0.entry.trigger"),
            PromptRequirement("ATR stop loss",                     "risk",      "risk.stop_loss"),
        ],
        expected={
            "universe.type":       "dynamic",
            "universe.asset_class":"equity_cash",
            "universe.rank.by":    "rvol",
            "context.timeframe":   "15m",
            "legs.0.direction":    "long",
        },
    ),

    CorpusEntry(
        prompt=(
            "Long entry on HDFCBANK 15m when EMA(9) crosses above EMA(21) and RSI is above 50. "
            "Stop loss 1% below entry, take profit 2:1. Only trade when ADX > 25."
        ),
        description="hdfcbank_ema_rsi_confluence",
        requirements=[
            PromptRequirement("HDFCBANK symbol",                "universe",  "universe.symbol"),
            PromptRequirement("15m timeframe",                  "timeframe", "context.timeframe"),
            PromptRequirement("EMA(9) crosses above EMA(21)",   "signal",    "legs.0.entry.trigger"),
            PromptRequirement("RSI above 50 (filter)",         "signal",    "legs.0.entry.filters"),
            PromptRequirement("Stop loss 1%",                  "risk",      "risk.stop_loss"),
            PromptRequirement("Take profit 2:1",               "risk",      "risk.take_profit"),
            PromptRequirement("ADX > 25 condition",            "signal",    "legs.0.entry.filters"),
        ],
        expected={
            "universe.type":       "static",
            "universe.asset_class":"equity_cash",
            "context.timeframe":   "15m",
            "legs.0.direction":    "long",
        },
    ),

    CorpusEntry(
        prompt=(
            "Trade TATASTEEL on the daily chart. Long when 50-day SMA is above 200-day SMA. "
            "Short when 50-day SMA is below 200-day SMA. Use 2% stop, 4% target."
        ),
        description="tatasteel_sma_dual_direction",
        requirements=[
            PromptRequirement("TATASTEEL symbol",          "universe",  "universe.symbol"),
            PromptRequirement("Daily chart",               "timeframe", "context.timeframe"),
            PromptRequirement("Long on SMA(50) > SMA(200)", "signal",   "legs.0.entry.trigger"),
            PromptRequirement("Short on SMA(50) < SMA(200)","signal",   "legs.1.entry.trigger"),
            PromptRequirement("2% stop",                   "risk",      "risk.stop_loss"),
            PromptRequirement("4% target",                 "risk",      "risk.take_profit"),
        ],
        expected={
            "universe.type":  "static",
            "context.timeframe": "1d",
        },
    ),

    CorpusEntry(
        prompt=(
            "Buy ETH 15m when RSI < 30. Only trade in trending regime. "
            "Skip trading on earnings dates: 2025-01-31, 2025-07-31. "
            "Stop 2%, target top-20 parallel (I want to trade multiple assets simultaneously)."
        ),
        description="eth_with_unsupported_features",
        requirements=[
            PromptRequirement("ETH symbol",                      "universe",  "universe.symbol"),
            PromptRequirement("RSI < 30",                        "signal",    "legs.0.entry.trigger"),
            PromptRequirement("Trending regime gate",            "gate",      "gates.regime"),
            PromptRequirement("Skip earnings dates",             "gate",      "gates.event"),
            PromptRequirement("Stop 2%",                         "risk",      "risk.stop_loss"),
            PromptRequirement("top-20 parallel (UNSUPPORTED)",   "universe",  "unmapped"),
        ],
        expected={
            "universe.type":       "static",
            "universe.asset_class":"crypto_spot",
        },
    ),

]
