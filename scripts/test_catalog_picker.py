"""
scripts/test_catalog_picker.py — Stress-test the catalog signal picker
against a broad set of representative prompts.

Each test case lists the user's prompt and the signals we expect the picker
to produce. The harness reports any mismatch so we can iterate on phrase
patterns without touching Python.

Run:

    PYTHONPATH=. python3 scripts/test_catalog_picker.py
"""
from __future__ import annotations

from dataclasses import dataclass

from app.kb.loader import KB
from app.planner.catalog_signal_picker import (
    auto_fill_missing_families,
    pick_plan_from_catalog,
)
from app.planner.semantic_extractor import SemanticExtractor


@dataclass
class Case:
    name: str
    prompt: str
    timeframe: str
    sentiment: str = "bullish"
    # Expectations — each is a list of signal names that should appear in the
    # respective role. The picker may include additional matches (filters); the
    # check is "expected ⊆ found" unless `strict` is True.
    must_have_trigger: list[str] = None
    must_have_filters: list[str] = None
    must_have_exits: list[str] = None
    must_have_anywhere: list[str] = None     # accepted in any role
    must_not_have: list[str] = None
    # Detected by extractor that we expect to be true
    expects_exit_on_opposite: bool = False
    min_confidence: float = 0.6

    def __post_init__(self):
        self.must_have_trigger = self.must_have_trigger or []
        self.must_have_filters = self.must_have_filters or []
        self.must_have_exits = self.must_have_exits or []
        self.must_have_anywhere = self.must_have_anywhere or []
        self.must_not_have = self.must_not_have or []


CASES: list[Case] = [
    # ── User's 5 original prompts ──────────────────────────────────────────
    Case(
        name="TCS / SMA 20",
        prompt=(
            "Create a beginner-friendly swing trading strategy for TCS using "
            "15-minute candles. Buy when price crosses above the 20 SMA and "
            "sell when price closes below the 20 SMA. Add stop loss below "
            "previous candle low and target 2:1 reward risk ratio."
        ),
        timeframe="15m",
        must_have_trigger=["price_cross_above_sma"],
        must_have_exits=["price_below_sma"],
        must_not_have=["ema_above", "ema_cross_up", "supertrend_bullish", "reference_above_sma"],
    ),
    Case(
        name="RELIANCE / RSI 30→65",
        prompt=(
            "Build a simple intraday strategy for RELIANCE on 5-minute "
            "timeframe using RSI. Enter long when RSI crosses above 30 from "
            "oversold zone and candle closes bullish. Exit when RSI reaches "
            "above 65 or stop loss is hit."
        ),
        timeframe="5m",
        must_have_trigger=["rsi_cross_up"],
        must_have_exits=["rsi_overbought"],
        must_not_have=["zscore_oversold"],
    ),
    Case(
        name="HDFCBANK / VWAP",
        prompt=(
            "Design a beginner intraday strategy for HDFCBANK using VWAP on "
            "5-minute timeframe. Buy only when price stays above VWAP with "
            "bullish candles and volume confirmation. Use trailing stop loss "
            "after 1% profit."
        ),
        timeframe="5m",
        must_have_filters=["vwap_bullish", "volume_spike"],
        must_not_have=["rsi_oversold", "vwap_reclaim_bullish"],
    ),
    Case(
        name="ITC / breakout",
        prompt=(
            "Create a breakout trading strategy for ITC on 15-minute chart. "
            "Enter buy trade when price breaks previous day high with strong "
            "bullish candle and higher than average volume. Keep stop loss "
            "below breakout candle low."
        ),
        timeframe="15m",
        must_have_filters=["volume_spike"],
        must_not_have=["bb_squeeze", "ema_cross_down"],
    ),
    Case(
        name="INFY / EMA 9/21 + avoid sideways",
        prompt=(
            "Generate a trend-following strategy for INFY using 9 EMA and 21 "
            "EMA on 5-minute timeframe. Buy when 9 EMA crosses above 21 EMA "
            "and avoid trades during sideways market conditions."
        ),
        timeframe="5m",
        must_have_trigger=["ema_cross_up"],
        must_have_filters=["adx_strong_trend"],
    ),
    Case(
        name="ADANIPORTS / RSI+MACD+VWAP+ATR",
        prompt=(
            "Build an intraday momentum strategy for ADANIPORTS on 5-minute "
            "timeframe. Enter only when RSI is above 60, MACD histogram turns "
            "positive, and price closes above VWAP. Use ATR-based stop loss "
            "and partial profit booking at 1.5R."
        ),
        timeframe="5m",
        must_have_filters=["rsi_above_60", "macd_positive", "vwap_bullish"],
    ),
    Case(
        name="WIPRO / MACD crossover + EMA 50",
        prompt=(
            "Build a beginner-friendly MACD crossover strategy for WIPRO "
            "using 5-minute candles. Enter long when MACD line crosses above "
            "signal line and price is above 50 EMA. Exit on opposite "
            "crossover or target hit."
        ),
        timeframe="5m",
        must_have_trigger=["macd_bullish_cross"],
        must_have_filters=["price_above_ema"],
        must_have_exits=["macd_bearish_cross"],
        expects_exit_on_opposite=True,
        must_not_have=["ema_above", "price_below_ema"],
    ),

    # ── Beginner-friendly simple prompts ────────────────────────────────────
    Case(
        name="Beginner — buy on golden cross",
        prompt="Buy when 50 SMA crosses above 200 SMA. Simple golden cross strategy on 1h chart.",
        timeframe="1h",
        must_have_trigger=["sma_cross_up"],
    ),
    Case(
        name="Beginner — supertrend buy",
        prompt="Use supertrend bullish as the entry on RELIANCE 15m.",
        timeframe="15m",
        must_have_trigger=["supertrend_bullish"],
    ),
    Case(
        name="Beginner — bollinger upper breakout",
        prompt="Long when price closes above upper bollinger band on 15m chart.",
        timeframe="15m",
        must_have_trigger=["price_above_bb_upper"],
    ),
    Case(
        name="Beginner — bollinger squeeze",
        prompt="Trade after a bollinger squeeze and high volume.",
        timeframe="5m",
        must_have_filters=["bb_squeeze", "volume_spike"],
    ),

    # ── Intermediate — multi-indicator confluence ───────────────────────────
    Case(
        name="Intermediate — RSI 50 + MACD positive + VWAP",
        prompt=(
            "On 5m, enter long when RSI is above 50, MACD is positive, "
            "and price is above VWAP."
        ),
        timeframe="5m",
        must_have_filters=["rsi_above_50", "macd_positive", "vwap_bullish"],
    ),
    Case(
        name="Intermediate — stoch oversold reversal",
        prompt="Enter long when stochastic is below 20 (oversold) on 5m chart.",
        timeframe="5m",
        must_have_trigger=["stoch_oversold"],
    ),
    Case(
        name="Intermediate — MFI overbought exit",
        prompt="Long on RSI > 50 and exit when MFI is above 80.",
        timeframe="15m",
        must_have_filters=["rsi_above_50"],
        must_have_exits=["mfi_overbought"],
    ),
    Case(
        name="Intermediate — CCI oversold long",
        prompt="Enter long when CCI is below -100 oversold and exit when CCI above 100 overbought.",
        timeframe="5m",
        must_have_trigger=["cci_oversold"],
        must_have_exits=["cci_overbought"],
    ),
    Case(
        name="Intermediate — keltner breakout up",
        prompt="Long on a keltner channel breakout up with volume confirmation.",
        timeframe="15m",
        must_have_trigger=["keltner_breakout_up"],
        must_have_filters=["volume_spike"],
    ),

    # ── Advanced — ICT / structure / FVG ───────────────────────────────────
    Case(
        name="Advanced — bullish FVG entry",
        prompt="Enter long on a bullish fair value gap with market structure break to the upside.",
        timeframe="5m",
        # Picker promotes one signal to TRIGGER, demotes the other to FILTER —
        # either ordering is acceptable. We check that BOTH signals end up in
        # the plan via the special pseudo-role checked in `must_have_anywhere`.
        must_have_anywhere=["bullish_fvg_signal", "bullish_market_structure_break"],
    ),
    Case(
        name="Advanced — opening range breakout (ORB)",
        prompt="ORB long trade when price breaks the opening range with above-average volume.",
        timeframe="5m",
        must_have_trigger=["opening_range_breakout"],
        must_have_filters=["volume_spike"],
    ),

    # ── Trend / structure filters ──────────────────────────────────────────
    Case(
        name="Trend filter — uptrend structure",
        prompt="Only trade in an uptrend structure with higher highs and higher lows.",
        timeframe="1h",
        must_have_filters=["uptrend_structure"],
    ),
    Case(
        name="Reference filter — outperforming NIFTY",
        prompt="Enter long only when the stock is outperforming NIFTY benchmark.",
        timeframe="15m",
        must_have_filters=["rs_outperforms_reference"],
    ),

    # ── Exit-on-opposite tests ─────────────────────────────────────────────
    Case(
        name="Exit on opposite — MACD",
        prompt="Long on MACD bullish cross. Exit when it reverses.",
        timeframe="15m",
        must_have_trigger=["macd_bullish_cross"],
        must_have_exits=["macd_bearish_cross"],
        expects_exit_on_opposite=True,
    ),
    Case(
        name="Exit on opposite — supertrend",
        prompt="Long when supertrend turns bullish. Exit on opposite signal.",
        timeframe="5m",
        must_have_trigger=["supertrend_bullish"],
        must_have_exits=["supertrend_bearish"],
        expects_exit_on_opposite=True,
    ),

    # ── Bearish / short prompts ────────────────────────────────────────────
    Case(
        name="Short — MACD bearish cross",
        prompt="Short when MACD bearish cross occurs on 15m.",
        timeframe="15m",
        sentiment="bearish",
        must_have_trigger=["macd_bearish_cross"],
    ),
    Case(
        name="Short — death cross",
        prompt="Short on EMA death cross (50 EMA crosses below 200 EMA).",
        timeframe="1d",
        sentiment="bearish",
        must_have_trigger=["ema_cross_down"],
    ),

    # ── Numeric extraction tests ───────────────────────────────────────────
    Case(
        name="Numeric — ADX threshold 30",
        prompt="Filter trades to only fire when ADX is above 30 (strong trend).",
        timeframe="15m",
        must_have_filters=["adx_strong_trend"],
    ),
    Case(
        name="Numeric — volume 2x average",
        prompt="Long on breakout with volume at least 2x average.",
        timeframe="15m",
        must_have_filters=["volume_spike"],
    ),

    # ── Z-score & statistical ──────────────────────────────────────────────
    Case(
        name="Z-score reversion",
        prompt="Mean reversion long when z-score is below -2 standard deviations.",
        timeframe="5m",
        must_have_trigger=["zscore_oversold"],
    ),

    # ── Gap strategies ─────────────────────────────────────────────────────
    Case(
        name="Gap and go long",
        prompt="Bullish gap-and-go strategy on the open with strong volume.",
        timeframe="5m",
        must_have_trigger=["gap_and_go_bullish"],
        must_have_filters=["volume_spike"],
    ),

    # ── Candle patterns ────────────────────────────────────────────────────
    Case(
        name="Inside bar filter",
        prompt="Add an inside bar filter to the entry.",
        timeframe="15m",
        must_have_filters=["inside_bar"],
    ),
    Case(
        name="Hammer candle entry",
        prompt="Enter long on a hammer candle near support.",
        timeframe="15m",
        must_have_filters=["bullish_rejection_candle"],
    ),

    # ── Bearish indicators ─────────────────────────────────────────────────
    Case(
        name="OBV falling exit",
        prompt="Exit the long when OBV is falling for confirmation.",
        timeframe="15m",
        must_have_filters=["obv_falling"],
    ),

    # ── User's NEW prompt: support-resistance with SMA ─────────────────────
    Case(
        name="ICICIBANK / SMA 200 as dynamic support",
        prompt=(
            "Build a simple support-resistance strategy using 200-day SMA "
            "as dynamic support for ICICIBANK.NS."
        ),
        timeframe="1d",
        must_have_anywhere=["is_above_sma"],
    ),

    # ── Support / resistance phrasings (auto-fill territory) ───────────────
    Case(
        name="50 EMA as support",
        prompt="Strategy uses 50 EMA as dynamic support on RELIANCE.",
        timeframe="1h",
        must_have_anywhere=["price_above_ema"],
        min_confidence=0.0,    # auto-fill produces this, picker phrase miss
    ),
    Case(
        name="VWAP as support intraday",
        prompt="Intraday long setup with VWAP as support on HDFCBANK 5m.",
        timeframe="5m",
        must_have_anywhere=["vwap_bullish"],
        min_confidence=0.0,
    ),
    Case(
        name="20 SMA support level",
        prompt="Take longs when price holds above the 20 SMA level.",
        timeframe="15m",
        must_have_anywhere=["is_above_sma"],
        min_confidence=0.0,
    ),
    Case(
        name="200 SMA trend filter",
        prompt="Only trade longs when stock trades above the 200 SMA.",
        timeframe="1d",
        must_have_anywhere=["is_above_sma"],
        min_confidence=0.0,
    ),

    # ── Auto-fill backstop scenarios (picker can't directly match) ─────────
    Case(
        name="Vague 'use 100 SMA'",
        prompt="A simple strategy using 100 SMA on TCS.",
        timeframe="1d",
        must_have_anywhere=["is_above_sma"],
        min_confidence=0.0,           # auto-fill, not picker confidence
    ),
    Case(
        name="Mentions RSI without details",
        prompt="Build a strategy on INFY 5m that uses RSI.",
        timeframe="5m",
        must_have_anywhere=["rsi_above_50"],   # any RSI signal acceptable
        min_confidence=0.0,
    ),
    Case(
        name="Mentions MACD vaguely",
        prompt="Make a strategy on WIPRO 5m using MACD for momentum.",
        timeframe="5m",
        must_have_anywhere=["macd_positive"],   # any MACD signal acceptable
        min_confidence=0.0,
    ),
    Case(
        name="Mentions ADX without value",
        prompt="Trend strategy on RELIANCE 15m with ADX filter.",
        timeframe="15m",
        must_have_anywhere=["adx_strong_trend"],
        min_confidence=0.0,
    ),

    # ── Natural state phrasings ────────────────────────────────────────────
    Case(
        name="Holds above 50 EMA",
        prompt="Long entries when price holds above the 50 EMA.",
        timeframe="15m",
        must_have_anywhere=["price_above_ema"],
        min_confidence=0.0,
    ),
    Case(
        name="Trading below 200 SMA short",
        prompt="Short when price is trading below 200 SMA on the daily.",
        timeframe="1d",
        sentiment="bearish",
        must_have_anywhere=["price_below_sma"],
        min_confidence=0.0,
    ),

    # ── ATR-based filters ───────────────────────────────────────────────────
    Case(
        name="High volatility filter",
        prompt="Only trade in high volatility regime on NIFTY 5m.",
        timeframe="5m",
        must_have_anywhere=["atr_high_volatility"],
    ),
    Case(
        name="Low volatility (squeeze)",
        prompt="Trade only when ATR is contracting (low volatility).",
        timeframe="15m",
        must_have_anywhere=["atr_low_volatility"],
    ),
]


def _names_in_role(plan: dict, role: str) -> list[str]:
    bucket = "entry" if role in ("trigger", "filter") else "exit"
    needed_type = "TRIGGER" if role in ("trigger", "exit") else "FILTER"
    return [
        (s.get("name") or "")
        for s in plan.get(bucket) or []
        if isinstance(s, dict) and (s.get("signal_type") or "").upper() == needed_type
    ]


def run(case: Case, kb: KB, ext: SemanticExtractor) -> tuple[bool, list[str]]:
    """Run one test case. Returns (passed, list_of_messages)."""
    msgs: list[str] = []
    sem = ext.extract(case.prompt)
    eoo = case.expects_exit_on_opposite or sem.exit_on_opposite

    pick = pick_plan_from_catalog(
        case.prompt, kb=kb, timeframe=case.timeframe,
        sentiment=case.sentiment, exit_on_opposite=eoo,
    )
    plan = pick.signal_plan
    # Run the auto-fill backstop just like chat_service does. This ensures
    # the tests reflect the production path end-to-end.
    auto_fill_missing_families(
        plan, case.prompt, kb=kb, timeframe=case.timeframe, sentiment=case.sentiment,
    )

    triggers = _names_in_role(plan, "trigger")
    filters  = _names_in_role(plan, "filter")
    exits    = _names_in_role(plan, "exit")

    ok = True

    # Required signals
    for n in case.must_have_trigger:
        if n not in triggers:
            ok = False
            msgs.append(f"  ✗ trigger {n!r} missing (got triggers={triggers})")
    for n in case.must_have_filters:
        if n not in filters:
            ok = False
            msgs.append(f"  ✗ filter {n!r} missing (got filters={filters})")
    for n in case.must_have_exits:
        if n not in exits:
            ok = False
            msgs.append(f"  ✗ exit {n!r} missing (got exits={exits})")

    # Forbidden signals
    all_picked = set(triggers + filters + exits)
    for n in case.must_not_have:
        if n in all_picked:
            ok = False
            msgs.append(f"  ✗ should NOT include {n!r}")
    # Role-agnostic checks (signal must appear in entry or exit, any role)
    for n in case.must_have_anywhere:
        if n not in all_picked:
            ok = False
            msgs.append(f"  ✗ {n!r} missing from plan (any role)")

    # Confidence floor
    if pick.confidence < case.min_confidence:
        ok = False
        msgs.append(f"  ✗ confidence {pick.confidence:.2f} < {case.min_confidence}")

    return ok, msgs


def main() -> None:
    kb = KB()
    ext = SemanticExtractor()

    print(f"Loaded {len(kb.signals)} signals; {sum(1 for c in kb.signals.values() if c.match_when)} have match_when blocks.\n")

    pass_count = 0
    fail_count = 0
    for case in CASES:
        ok, msgs = run(case, kb, ext)
        marker = "✓" if ok else "✗"
        print(f"{marker} {case.name}")
        for m in msgs:
            print(m)
        if ok:
            pass_count += 1
        else:
            fail_count += 1

    print()
    print(f"=== {pass_count}/{len(CASES)} cases pass ({fail_count} fail) ===")


if __name__ == "__main__":
    main()
