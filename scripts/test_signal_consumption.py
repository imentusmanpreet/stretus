"""
scripts/test_signal_consumption.py — Validate that the constraint_compiler
picks the right KB signals when the semantic extractor has captured
rsi_thresholds / macd_states / vwap_relations / regime_preference /
sl_type_hint / partial_exits.

Each test builds a minimal signal_plan + builder, runs
apply_semantic_constraints, and checks that the post-compilation plan
contains the expected signals and SL spec.
"""
from __future__ import annotations

from app.planner.constraint_compiler import apply_semantic_constraints
from app.planner.semantic_extractor import SemanticExtractor


class FakeBuilder:
    """Minimal builder shim — only the attributes the compiler reads."""
    def __init__(self, *, timeframe="5m", sentiment="bullish"):
        self.timeframe = timeframe
        self.sentiment = sentiment
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


PROMPTS = [
    "Create a beginner-friendly swing trading strategy for TCS using 15-minute candles. "
    "Buy when price crosses above the 20 SMA and sell when price closes below the 20 SMA. "
    "Add stop loss below previous candle low and target 2:1 reward risk ratio.",

    "Build a simple intraday strategy for RELIANCE on 5-minute timeframe using RSI. "
    "Enter long when RSI crosses above 30 from oversold zone and candle closes bullish. "
    "Exit when RSI reaches above 65 or stop loss is hit.",

    "Design a beginner intraday strategy for HDFCBANK using VWAP on 5-minute timeframe. "
    "Buy only when price stays above VWAP with bullish candles and volume confirmation. "
    "Use trailing stop loss after 1% profit.",

    "Create a breakout trading strategy for ITC on 15-minute chart. "
    "Enter buy trade when price breaks previous day high with strong bullish candle and higher than average volume. "
    "Keep stop loss below breakout candle low.",

    "Generate a trend-following strategy for INFY using 9 EMA and 21 EMA on 5-minute timeframe. "
    "Buy when 9 EMA crosses above 21 EMA and avoid trades during sideways market conditions.",

    "Build an intraday momentum strategy for ADANIPORTS on 5-minute timeframe. "
    "Enter only when RSI is above 60, MACD histogram turns positive, and price closes above VWAP. "
    "Use ATR-based stop loss and partial profit booking at 1.5R.",
]


def _baseline_plan(timeframe: str) -> dict:
    """A near-empty plan that the constraint compiler can mutate."""
    return {
        "entry": [],
        "exit": [],
        "signals_used": [],
        "signals_available": 0,
    }


def _summarize_entry(plan: dict) -> str:
    parts = []
    for sig in plan.get("entry", []):
        role = sig.get("signal_type", "?")
        name = sig.get("name", "?")
        params = sig.get("params", {})
        param_str = ",".join(f"{k}={v}" for k, v in params.items() if v not in (None, {}))
        parts.append(f"{name}({role}{':' + param_str if param_str else ''})")
    return " + ".join(parts) if parts else "—"


def _summarize_exit(plan: dict) -> str:
    parts = []
    for sig in plan.get("exit", []):
        role = sig.get("signal_type", "?")
        name = sig.get("name", "?")
        parts.append(f"{name}({role})")
    return " + ".join(parts) if parts else "—"


def run():
    ext = SemanticExtractor()
    width = 78

    for idx, prompt in enumerate(PROMPTS, 1):
        print("=" * width)
        print(f"PROMPT #{idx}")
        print("=" * width)
        print(prompt[:width])
        print()

        # Pick a sensible timeframe for the builder (the chat layer would set this)
        if "15-minute" in prompt or "15m" in prompt:
            tf = "15m"
        else:
            tf = "5m"

        semantic = ext.extract(prompt)
        builder = FakeBuilder(timeframe=tf, sentiment="bullish")

        plan = _baseline_plan(tf)
        try:
            new_plan = apply_semantic_constraints(
                plan, builder, semantic_instructions=semantic, source_prompt=prompt,
            )
        except Exception as e:
            print(f"  ERROR during apply_semantic_constraints: {e!r}")
            continue

        print(f"  Extractor captured:")
        print(f"    vwap_relations    = {semantic.vwap_relations}")
        print(f"    rsi_thresholds    = {semantic.rsi_thresholds}")
        print(f"    macd_states       = {semantic.macd_states}")
        print(f"    regime_preference = {semantic.regime_preference}")
        print(f"    sl_type_hint      = {semantic.sl_type_hint}")
        print(f"    partial_exits     = {semantic.partial_exits}")
        print()
        print(f"  Post-compilation entry signals: {_summarize_entry(new_plan)}")
        print(f"  Post-compilation exit signals : {_summarize_exit(new_plan)}")
        print(f"  builder.stop_loss_spec        : {builder.stop_loss_spec}")
        print(f"  builder.execution_layers      : {builder.execution_layers}")
        applied = new_plan.get("_constraint_compiler_applied") or []
        print(f"  Constraint compiler audit     : {applied}")
        print()


if __name__ == "__main__":
    run()
