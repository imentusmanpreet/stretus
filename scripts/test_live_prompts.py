"""
scripts/test_live_prompts.py — Replay the exact plans the live chat system
produced and confirm the subtractive constraint compiler now removes the
preset-fabricated filters and exits.

For each of the 4 prompts the user reported, we feed the original message
plus the broken plan (as captured from production) into
apply_semantic_constraints and print what survives.
"""
from __future__ import annotations

import copy

from app.planner.constraint_compiler import apply_semantic_constraints
from app.planner.semantic_extractor import SemanticExtractor


class FakeBuilder:
    def __init__(self, *, timeframe, sentiment="bullish", original_user_prompt=""):
        self.timeframe = timeframe
        self.sentiment = sentiment
        self.original_user_prompt = original_user_prompt
        self.goal = None
        self.semantic_intent = None
        self.stop_loss_spec = None
        self.trailing_stop_spec = None
        self.risk_execution_config = {}
        self.execution_layers = {}
        self.stop_loss = None
        self.take_profit = None
        self.volume_ratio_threshold = None
        self.rsi_entry_band_min = None
        self.rsi_entry_band_max = None


CASES = [
    {
        "name": "RELIANCE / 5m / RSI",
        "tf": "5m",
        "prompt": (
            "Build a simple intraday strategy for RELIANCE on 5-minute timeframe using RSI. "
            "Enter long when RSI crosses above 30 from oversold zone and candle closes bullish. "
            "Exit when RSI reaches above 65 or stop loss is hit."
        ),
        "production_plan": {
            "entry": [
                {"name": "zscore_oversold",          "params": {"window": 20, "threshold": -2}, "signal_type": "TRIGGER", "timeframe": "5m"},
                {"name": "rsi_oversold",             "params": {"window": 14, "threshold": 30}, "signal_type": "FILTER",  "timeframe": "5m"},
                {"name": "bullish_rejection_candle", "params": {"wick_ratio": 0.66},            "signal_type": "FILTER",  "timeframe": "5m"},
            ],
            "exit": [
                {"name": "rsi_overbought", "params": {"window": 14, "threshold": 70}, "signal_type": "TRIGGER", "timeframe": "5m"},
            ],
        },
    },
    {
        "name": "TCS / 15m / SMA 20",
        "tf": "15m",
        "prompt": (
            "Create a beginner-friendly swing trading strategy for TCS using 15-minute candles. "
            "Buy when price crosses above the 20 SMA and sell when price closes below the 20 SMA. "
            "Add stop loss below previous candle low and target 2:1 reward risk ratio."
        ),
        "production_plan": {
            "entry": [
                {"name": "ema_above",            "params": {"window_fast": 9, "window_slow": 21}, "signal_type": "TRIGGER", "timeframe": "15m"},
                {"name": "price_above_ema",      "params": {"window": 20},                        "signal_type": "FILTER",  "timeframe": "15m"},
                {"name": "reference_above_sma",  "params": {"window": 50},                        "signal_type": "FILTER",  "timeframe": "15m"},
                {"name": "supertrend_bullish",   "params": {"window": 10, "multiplier": 3},       "signal_type": "FILTER",  "timeframe": "15m"},
            ],
            "exit": [
                {"name": "price_below_ema", "params": {"window": 20}, "signal_type": "TRIGGER", "timeframe": "15m"},
            ],
        },
    },
    {
        "name": "HDFCBANK / 5m / VWAP",
        "tf": "5m",
        "prompt": (
            "Design a beginner intraday strategy for HDFCBANK using VWAP on 5-minute timeframe. "
            "Buy only when price stays above VWAP with bullish candles and volume confirmation. "
            "Use trailing stop loss after 1% profit."
        ),
        "production_plan": {
            "entry": [
                {"name": "vwap_reclaim_bullish",     "params": {},                                "signal_type": "TRIGGER", "timeframe": "5m"},
                {"name": "bullish_rejection_candle", "params": {"wick_ratio": 0.66},              "signal_type": "FILTER",  "timeframe": "5m"},
                {"name": "rsi_oversold",             "params": {"window": 14, "threshold": 30},   "signal_type": "FILTER",  "timeframe": "5m"},
            ],
            "exit": [
                {"name": "vwap_bearish", "params": {}, "signal_type": "TRIGGER", "timeframe": "5m"},
            ],
        },
    },
    {
        "name": "ITC / 15m / breakout",
        "tf": "15m",
        "prompt": (
            "Create a breakout trading strategy for ITC on 15-minute chart. "
            "Enter buy trade when price breaks previous day high with strong bullish candle and higher than average volume. "
            "Keep stop loss below breakout candle low."
        ),
        "production_plan": {
            "entry": [
                {"name": "n_bar_high_breakout", "params": {"window": 20},                       "signal_type": "TRIGGER", "timeframe": "15m"},
                {"name": "volume_spike",        "params": {"window": 20, "multiplier": 1.5},    "signal_type": "FILTER",  "timeframe": "15m"},
                {"name": "bb_squeeze",          "params": {"window": 20, "window_dev": 2},      "signal_type": "FILTER",  "timeframe": "15m"},
            ],
            "exit": [
                {"name": "ema_cross_down", "params": {"window_fast": 9, "window_slow": 21}, "signal_type": "TRIGGER", "timeframe": "15m"},
            ],
        },
    },
]


def _summarize(plan: dict, key: str) -> str:
    return " + ".join(
        f"{s.get('name')}({s.get('signal_type')})"
        for s in plan.get(key) or []
    ) or "—"


def run():
    ext = SemanticExtractor()
    width = 78
    for case in CASES:
        print("=" * width)
        print(case["name"])
        print("=" * width)
        prompt = case["prompt"]
        print(prompt)
        print()

        semantic = ext.extract(prompt)
        builder = FakeBuilder(
            timeframe=case["tf"], sentiment="bullish",
            original_user_prompt=prompt,
        )
        before = copy.deepcopy(case["production_plan"])
        after = apply_semantic_constraints(
            case["production_plan"], builder,
            semantic_instructions=semantic, source_prompt=prompt,
        )

        print(f"BEFORE (live production):")
        print(f"  entry: {_summarize(before, 'entry')}")
        print(f"  exit : {_summarize(before, 'exit')}")
        print()
        print(f"AFTER (with subtract + Phase 12 consumers):")
        print(f"  entry: {_summarize(after, 'entry')}")
        print(f"  exit : {_summarize(after, 'exit')}")
        applied = after.get("_constraint_compiler_applied") or []
        if applied:
            print(f"  audit: {applied}")
        print()


if __name__ == "__main__":
    run()
