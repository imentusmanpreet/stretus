"""
scripts/generate_phase1_signal_cards.py
════════════════════════════════════════
Phase 1 — author one catalog SignalCard per new TA-Lib-backed indicator signal.

Run from the repo root:  python scripts/generate_phase1_signal_cards.py

Emits app/kb/signals/<name>.yaml for every spec below (consistent structure,
schema-valid, params across all supported timeframes) and appends the new names
to app/kb/signals/_registry.yaml under a "Phase 1 — additional indicators" section.
Idempotent: re-running overwrites the card files and de-dups registry entries.

This generator is the source of record for HOW the Phase 1 cards were produced;
the YAML it writes is the artifact the loader consumes.
"""
from __future__ import annotations

from pathlib import Path

import yaml

TF = ["1m", "3m", "5m", "10m", "15m", "30m", "1h", "1d"]
SIGNALS_DIR = Path(__file__).resolve().parent.parent / "app" / "kb" / "signals"

# Intent profiles (keys: trend, momentum, breakout, reversal, scalping,
# mean_reversion, confirmation) reused across families.
I_REVERSAL = {"trend": 0.0, "momentum": 0.4, "breakout": 0.0, "reversal": 1.0,
              "scalping": 0.5, "mean_reversion": 1.0, "confirmation": 0.5}
I_MOMENTUM = {"trend": 0.7, "momentum": 1.0, "breakout": 0.4, "reversal": 0.2,
              "scalping": 0.6, "mean_reversion": 0.0, "confirmation": 0.6}
I_TREND = {"trend": 1.0, "momentum": 0.6, "breakout": 0.5, "reversal": 0.0,
           "scalping": 0.3, "mean_reversion": 0.0, "confirmation": 0.8}
I_BREAKOUT = {"trend": 0.7, "momentum": 0.6, "breakout": 1.0, "reversal": 0.0,
              "scalping": 0.4, "mean_reversion": 0.0, "confirmation": 0.4}
I_PATTERN = {"trend": 0.3, "momentum": 0.4, "breakout": 0.3, "reversal": 0.8,
             "scalping": 0.5, "mean_reversion": 0.5, "confirmation": 1.0}
I_VOLUME = {"trend": 0.6, "momentum": 0.6, "breakout": 0.5, "reversal": 0.2,
            "scalping": 0.4, "mean_reversion": 0.0, "confirmation": 1.0}

EXP_DEFAULT = {"beginner": 0.6, "intermediate": 1.0, "expert": 1.0}

INTRADAY_OK = ["5m", "10m", "15m", "30m", "1h", "1d"]


def card(name, category, direction, kind, roles, formula, description, params,
         intent_fit, *, works_best_on=INTRADAY_OK, weak_on=("1m",),
         pairs_well_with=(), contradicts=(), phrases=(), acts_as="entry_trigger",
         mirrors_to=None, extract_params=None, contra_sentiment=None):
    """Build one SignalCard dict. `params` is param→value applied to every TF."""
    data = {
        "name": name,
        "category": category,
        "direction": direction,
        "kind": kind,
        "roles": list(roles),
        "formula": formula,
        "description": description,
        "works_best_on": list(works_best_on),
        "weak_on": list(weak_on),
        "unsupported_on": [],
        "contraindicated_when": (
            [{"sentiment": contra_sentiment}] if contra_sentiment else []
        ),
        "pairs_well_with": list(pairs_well_with),
        "contradicts": list(contradicts),
        "params_by_timeframe": {tf: dict(params) for tf in TF},
        "experience_fit": dict(EXP_DEFAULT),
        "intent_fit": dict(intent_fit),
    }
    if phrases:
        mw = {"phrases": list(phrases), "acts_as": acts_as}
        if mirrors_to:
            mw["mirrors_to"] = mirrors_to
        if extract_params:
            mw["extract_params"] = extract_params
        data["match_when"] = mw
    return data


# Common extract_params blocks.
EX_WIN_THR = {
    "threshold": {"from_group": 1, "default": None, "kind": "float"},
    "window": {"from_group": None, "default": 14, "kind": "int"},
}
EX_WIN = {"window": {"from_group": 1, "default": 14, "kind": "int"}}

CARDS = [
    # ── Momentum oscillators — overbought / oversold (mean-reversion) ──────────
    card("williams_oversold", "momentum", "bullish", "threshold",
         ["entry_trigger", "exit_trigger", "exit_filter"],
         "WILLR({window}) < {threshold}",
         "Williams %R deeply negative — oversold, mean-reversion bullish entry.",
         {"window": 14, "threshold": -80.0}, I_REVERSAL,
         contradicts=["williams_overbought"], contra_sentiment="bearish",
         phrases=[r"williams?\s*%?\s*r\s+oversold", r"%r\s+(?:below|under)\s*(-?\d+)"],
         mirrors_to="williams_overbought", extract_params=EX_WIN_THR),
    card("williams_overbought", "momentum", "bearish", "threshold",
         ["entry_trigger", "exit_trigger", "exit_filter"],
         "WILLR({window}) > {threshold}",
         "Williams %R near zero — overbought, mean-reversion bearish entry / long exit.",
         {"window": 14, "threshold": -20.0}, I_REVERSAL,
         contradicts=["williams_oversold"], contra_sentiment="bullish",
         phrases=[r"williams?\s*%?\s*r\s+overbought", r"%r\s+(?:above|over)\s*(-?\d+)"],
         mirrors_to="williams_oversold", extract_params=EX_WIN_THR),
    card("cmo_oversold", "momentum", "bullish", "threshold",
         ["entry_trigger", "exit_filter"],
         "CMO({window}) < {threshold}",
         "Chande Momentum Oscillator oversold — bullish mean-reversion entry.",
         {"window": 14, "threshold": -50.0}, I_REVERSAL,
         contradicts=["cmo_overbought"], contra_sentiment="bearish",
         phrases=[r"cmo\s+oversold", r"chande\s+momentum\s+oversold"],
         mirrors_to="cmo_overbought", extract_params=EX_WIN_THR),
    card("cmo_overbought", "momentum", "bearish", "threshold",
         ["entry_trigger", "exit_filter"],
         "CMO({window}) > {threshold}",
         "Chande Momentum Oscillator overbought — bearish entry / long exit.",
         {"window": 14, "threshold": 50.0}, I_REVERSAL,
         contradicts=["cmo_oversold"], contra_sentiment="bullish",
         phrases=[r"cmo\s+overbought", r"chande\s+momentum\s+overbought"],
         mirrors_to="cmo_oversold", extract_params=EX_WIN_THR),
    card("ultosc_oversold", "momentum", "bullish", "threshold",
         ["entry_trigger", "exit_filter"],
         "ULTOSC < {threshold}",
         "Ultimate Oscillator oversold — multi-timeframe bullish mean-reversion entry.",
         {"threshold": 30.0}, I_REVERSAL,
         contradicts=["ultosc_overbought"], contra_sentiment="bearish",
         phrases=[r"ultimate\s+oscillator\s+oversold", r"ultosc\s+(?:below|under)\s*(\d+)"],
         mirrors_to="ultosc_overbought",
         extract_params={"threshold": {"from_group": 1, "default": 30, "kind": "float"}}),
    card("ultosc_overbought", "momentum", "bearish", "threshold",
         ["entry_trigger", "exit_filter"],
         "ULTOSC > {threshold}",
         "Ultimate Oscillator overbought — bearish entry / long exit.",
         {"threshold": 70.0}, I_REVERSAL,
         contradicts=["ultosc_oversold"], contra_sentiment="bullish",
         phrases=[r"ultimate\s+oscillator\s+overbought", r"ultosc\s+(?:above|over)\s*(\d+)"],
         mirrors_to="ultosc_oversold",
         extract_params={"threshold": {"from_group": 1, "default": 70, "kind": "float"}}),

    # ── Momentum — directional (above / below zero) ───────────────────────────
    card("roc_positive", "momentum", "bullish", "threshold",
         ["entry_trigger", "entry_filter"],
         "ROC({window}) > {threshold}",
         "Rate of change positive — price higher than n bars ago, bullish momentum.",
         {"window": 10, "threshold": 0.0}, I_MOMENTUM,
         contradicts=["roc_negative"], contra_sentiment="bearish",
         phrases=[r"roc\s+(?:positive|above\s+0|>\s*0)", r"rate\s+of\s+change\s+positive"],
         mirrors_to="roc_negative", extract_params=EX_WIN),
    card("roc_negative", "momentum", "bearish", "threshold",
         ["entry_trigger", "entry_filter"],
         "ROC({window}) < {threshold}",
         "Rate of change negative — price lower than n bars ago, bearish momentum.",
         {"window": 10, "threshold": 0.0}, I_MOMENTUM,
         contradicts=["roc_positive"], contra_sentiment="bullish",
         phrases=[r"roc\s+(?:negative|below\s+0|<\s*0)", r"rate\s+of\s+change\s+negative"],
         mirrors_to="roc_positive", extract_params=EX_WIN),
    card("momentum_positive", "momentum", "bullish", "threshold",
         ["entry_trigger", "entry_filter"],
         "MOM({window}) > {threshold}",
         "Momentum positive — price above its value n bars ago.",
         {"window": 10, "threshold": 0.0}, I_MOMENTUM,
         contra_sentiment="bearish",
         phrases=[r"momentum\s+(?:is\s+)?positive", r"positive\s+momentum"],
         extract_params=EX_WIN),
    card("trix_positive", "momentum", "bullish", "threshold",
         ["entry_trigger", "entry_filter"],
         "TRIX({window}) > {threshold}",
         "TRIX above zero — triple-smoothed momentum turning bullish.",
         {"window": 15, "threshold": 0.0}, I_MOMENTUM,
         contradicts=["trix_negative"], contra_sentiment="bearish",
         phrases=[r"trix\s+(?:positive|above\s+0|>\s*0|bullish)"],
         mirrors_to="trix_negative", extract_params=EX_WIN),
    card("trix_negative", "momentum", "bearish", "threshold",
         ["entry_trigger", "entry_filter"],
         "TRIX({window}) < {threshold}",
         "TRIX below zero — triple-smoothed momentum turning bearish.",
         {"window": 15, "threshold": 0.0}, I_MOMENTUM,
         contradicts=["trix_positive"], contra_sentiment="bullish",
         phrases=[r"trix\s+(?:negative|below\s+0|<\s*0|bearish)"],
         mirrors_to="trix_positive", extract_params=EX_WIN),
    card("ppo_positive", "momentum", "bullish", "threshold",
         ["entry_trigger", "entry_filter"],
         "PPO > {threshold}",
         "Percentage Price Oscillator positive — fast EMA above slow EMA (bullish).",
         {"threshold": 0.0}, I_MOMENTUM,
         contradicts=["ppo_negative"], contra_sentiment="bearish",
         phrases=[r"ppo\s+(?:positive|above\s+0|bullish)"], mirrors_to="ppo_negative"),
    card("ppo_negative", "momentum", "bearish", "threshold",
         ["entry_trigger", "entry_filter"],
         "PPO < {threshold}",
         "Percentage Price Oscillator negative — fast EMA below slow EMA (bearish).",
         {"threshold": 0.0}, I_MOMENTUM,
         contradicts=["ppo_positive"], contra_sentiment="bullish",
         phrases=[r"ppo\s+(?:negative|below\s+0|bearish)"], mirrors_to="ppo_positive"),

    # ── Stochastic ─────────────────────────────────────────────────────────────
    card("stoch_bullish_cross", "momentum", "bullish", "crossover",
         ["entry_trigger"],
         "STOCH_K > STOCH_D",
         "Stochastic %K crosses above %D — bullish momentum trigger.",
         {"fastk": 14}, I_MOMENTUM,
         contradicts=["stoch_bearish_cross"], contra_sentiment="bearish",
         phrases=[r"stoch(?:astic)?\s+bullish\s+cross", r"%k\s+crosses?\s+above\s+%d"],
         mirrors_to="stoch_bearish_cross"),
    card("stoch_bearish_cross", "momentum", "bearish", "crossover",
         ["entry_trigger"],
         "STOCH_K < STOCH_D",
         "Stochastic %K crosses below %D — bearish momentum trigger.",
         {"fastk": 14}, I_MOMENTUM,
         contradicts=["stoch_bullish_cross"], contra_sentiment="bullish",
         phrases=[r"stoch(?:astic)?\s+bearish\s+cross", r"%k\s+crosses?\s+below\s+%d"],
         mirrors_to="stoch_bullish_cross"),

    # ── Trend strength / direction filters ─────────────────────────────────────
    card("adxr_strong_trend", "trend", "neutral", "regime",
         ["entry_filter"],
         "ADXR({window}) > {threshold}",
         "ADXR above threshold — sustained trend strength (either direction). Regime filter.",
         {"window": 14, "threshold": 25.0}, I_TREND,
         pairs_well_with=["ema_above", "supertrend_bullish"],
         phrases=[r"adxr\s+(?:>|>=|above|over)\s*(\d+)", r"adxr\s+strong"],
         acts_as="entry_filter", extract_params=EX_WIN_THR),
    card("aroon_bullish", "trend", "bullish", "regime",
         ["entry_trigger", "entry_filter"],
         "AROONOSC({window}) > {threshold}",
         "Aroon oscillator strongly positive — recent highs dominate, bullish trend.",
         {"window": 14, "threshold": 50.0}, I_TREND,
         contradicts=["aroon_bearish"], contra_sentiment="bearish",
         phrases=[r"aroon\s+(?:bullish|up)", r"aroon\s+oscillator\s+(?:above|>)\s*(\d+)"],
         mirrors_to="aroon_bearish", extract_params=EX_WIN_THR),
    card("aroon_bearish", "trend", "bearish", "regime",
         ["entry_trigger", "entry_filter"],
         "AROONOSC({window}) < {threshold}",
         "Aroon oscillator strongly negative — recent lows dominate, bearish trend.",
         {"window": 14, "threshold": -50.0}, I_TREND,
         contradicts=["aroon_bullish"], contra_sentiment="bullish",
         phrases=[r"aroon\s+(?:bearish|down)"],
         mirrors_to="aroon_bullish", extract_params=EX_WIN_THR),
    card("sar_bullish", "trend", "bullish", "regime",
         ["entry_trigger", "entry_filter", "exit_filter"],
         "CLOSE > SAR",
         "Price above Parabolic SAR — bullish trend; SAR trails below price.",
         {"acceleration": 0.02, "maximum": 0.2}, I_TREND,
         contradicts=["sar_bearish"], contra_sentiment="bearish",
         phrases=[r"(?:price\s+)?above\s+(?:the\s+)?(?:parabolic\s+)?sar", r"sar\s+bullish"],
         mirrors_to="sar_bearish"),
    card("sar_bearish", "trend", "bearish", "regime",
         ["entry_trigger", "entry_filter", "exit_filter"],
         "CLOSE < SAR",
         "Price below Parabolic SAR — bearish trend; SAR trails above price.",
         {"acceleration": 0.02, "maximum": 0.2}, I_TREND,
         contradicts=["sar_bullish"], contra_sentiment="bullish",
         phrases=[r"(?:price\s+)?below\s+(?:the\s+)?(?:parabolic\s+)?sar", r"sar\s+bearish"],
         mirrors_to="sar_bullish"),

    # ── Adaptive / alternative moving averages — price position ────────────────
    card("price_above_kama", "trend", "bullish", "regime",
         ["entry_trigger", "entry_filter"],
         "CLOSE > KAMA({window})",
         "Close above Kaufman Adaptive MA — adaptive bullish trend filter.",
         {"window": 30}, I_TREND, contradicts=["price_below_kama"],
         contra_sentiment="bearish",
         phrases=[r"(?:price|close)\s+above\s+kama", r"above\s+(?:the\s+)?adaptive\s+(?:moving\s+)?average"],
         mirrors_to="price_below_kama", extract_params=EX_WIN),
    card("price_below_kama", "trend", "bearish", "regime",
         ["entry_trigger", "entry_filter"],
         "CLOSE < KAMA({window})",
         "Close below Kaufman Adaptive MA — adaptive bearish trend filter.",
         {"window": 30}, I_TREND, contradicts=["price_above_kama"],
         contra_sentiment="bullish",
         phrases=[r"(?:price|close)\s+below\s+kama"],
         mirrors_to="price_above_kama", extract_params=EX_WIN),
    card("price_above_dema", "trend", "bullish", "regime",
         ["entry_trigger", "entry_filter"],
         "CLOSE > DEMA({window})",
         "Close above Double EMA — low-lag bullish trend filter.",
         {"window": 20}, I_TREND, contra_sentiment="bearish",
         phrases=[r"(?:price|close)\s+above\s+dema", r"double\s+ema"],
         extract_params=EX_WIN),
    card("price_above_tema", "trend", "bullish", "regime",
         ["entry_trigger", "entry_filter"],
         "CLOSE > TEMA({window})",
         "Close above Triple EMA — very-low-lag bullish trend filter.",
         {"window": 20}, I_TREND, contra_sentiment="bearish",
         phrases=[r"(?:price|close)\s+above\s+tema", r"triple\s+ema"],
         extract_params=EX_WIN),
    card("price_above_wma", "trend", "bullish", "regime",
         ["entry_trigger", "entry_filter"],
         "CLOSE > WMA({window})",
         "Close above Weighted MA — recency-weighted bullish trend filter.",
         {"window": 20}, I_TREND, contra_sentiment="bearish",
         phrases=[r"(?:price|close)\s+above\s+wma", r"weighted\s+moving\s+average"],
         extract_params=EX_WIN),

    # ── Donchian channel breakout ──────────────────────────────────────────────
    card("donchian_breakout_up", "trend", "bullish", "regime",
         ["entry_trigger"],
         "CLOSE > DONCHIAN_UPPER({window})",
         "Close breaks above the prior n-bar high (Donchian upper) — bullish breakout.",
         {"window": 20}, I_BREAKOUT, contradicts=["donchian_breakdown"],
         contra_sentiment="bearish",
         phrases=[r"donchian\s+breakout", r"break(?:s|out)?\s+(?:above\s+)?(?:the\s+)?(\d+)[\s-]*(?:bar|day|period)\s+high",
                  r"breaks?\s+(?:the\s+)?(?:n-?bar\s+)?high"],
         mirrors_to="donchian_breakdown", extract_params=EX_WIN),
    card("donchian_breakdown", "trend", "bearish", "regime",
         ["entry_trigger"],
         "CLOSE < DONCHIAN_LOWER({window})",
         "Close breaks below the prior n-bar low (Donchian lower) — bearish breakdown.",
         {"window": 20}, I_BREAKOUT, contradicts=["donchian_breakout_up"],
         contra_sentiment="bullish",
         phrases=[r"donchian\s+breakdown", r"breaks?\s+(?:below\s+)?(?:the\s+)?(\d+)[\s-]*(?:bar|day|period)\s+low"],
         mirrors_to="donchian_breakout_up", extract_params=EX_WIN),

    # ── Volume ──────────────────────────────────────────────────────────────────

    # ── Candlestick patterns (TA-Lib CDL*) — confirmation ───────────────────────
    card("bullish_engulfing", "pattern", "bullish", "pattern",
         ["entry_trigger", "entry_filter"],
         "CDL_ENGULFING > {threshold}",
         "Bullish engulfing candle — strong reversal confirmation to the upside.",
         {"threshold": 0}, I_PATTERN, contradicts=["bearish_engulfing"],
         contra_sentiment="bearish",
         phrases=[r"bullish\s+engulfing", r"engulfing\s+(?:candle\s+)?(?:bull|up)"],
         mirrors_to="bearish_engulfing"),
    card("bearish_engulfing", "pattern", "bearish", "pattern",
         ["entry_trigger", "entry_filter"],
         "CDL_ENGULFING < {threshold}",
         "Bearish engulfing candle — strong reversal confirmation to the downside.",
         {"threshold": 0}, I_PATTERN, contradicts=["bullish_engulfing"],
         contra_sentiment="bullish",
         phrases=[r"bearish\s+engulfing"], mirrors_to="bullish_engulfing"),
    card("hammer_candle", "pattern", "bullish", "pattern",
         ["entry_trigger", "entry_filter"],
         "CDL_HAMMER > {threshold}",
         "Hammer candle — long lower wick rejection, bullish reversal confirmation.",
         {"threshold": 0}, I_PATTERN, contra_sentiment="bearish",
         phrases=[r"hammer\s+candle", r"\bhammer\b"]),
    card("shooting_star_candle", "pattern", "bearish", "pattern",
         ["entry_trigger", "entry_filter"],
         "CDL_SHOOTINGSTAR < {threshold}",
         "Shooting-star candle — long upper wick rejection, bearish reversal confirmation.",
         {"threshold": 0}, I_PATTERN, contra_sentiment="bullish",
         phrases=[r"shooting\s+star"]),
    card("doji_indecision", "pattern", "neutral", "pattern",
         ["entry_filter", "exit_filter"],
         "CDL_DOJI != {threshold}",
         "Doji — open ≈ close, market indecision. Use as a caution / confirmation filter.",
         {"threshold": 0}, I_PATTERN,
         phrases=[r"\bdoji\b", r"indecision\s+candle"], acts_as="entry_filter"),
    card("morning_star", "pattern", "bullish", "pattern",
         ["entry_trigger", "entry_filter"],
         "CDL_MORNINGSTAR > {threshold}",
         "Morning-star three-candle pattern — bullish reversal confirmation.",
         {"threshold": 0}, I_PATTERN, contradicts=["evening_star"],
         contra_sentiment="bearish",
         phrases=[r"morning\s+star"], mirrors_to="evening_star"),
    card("evening_star", "pattern", "bearish", "pattern",
         ["entry_trigger", "entry_filter"],
         "CDL_EVENINGSTAR < {threshold}",
         "Evening-star three-candle pattern — bearish reversal confirmation.",
         {"threshold": 0}, I_PATTERN, contradicts=["morning_star"],
         contra_sentiment="bullish",
         phrases=[r"evening\s+star"], mirrors_to="morning_star"),
    card("three_white_soldiers", "pattern", "bullish", "pattern",
         ["entry_trigger", "entry_filter"],
         "CDL_3WHITESOLDIERS > {threshold}",
         "Three white soldiers — three strong up candles, bullish continuation/reversal.",
         {"threshold": 0}, I_PATTERN, contradicts=["three_black_crows"],
         contra_sentiment="bearish",
         phrases=[r"three\s+white\s+soldiers"], mirrors_to="three_black_crows"),
    card("three_black_crows", "pattern", "bearish", "pattern",
         ["entry_trigger", "entry_filter"],
         "CDL_3BLACKCROWS < {threshold}",
         "Three black crows — three strong down candles, bearish continuation/reversal.",
         {"threshold": 0}, I_PATTERN, contradicts=["three_white_soldiers"],
         contra_sentiment="bullish",
         phrases=[r"three\s+black\s+crows"], mirrors_to="three_white_soldiers"),
]


def main() -> None:
    names = []
    for c in CARDS:
        names.append(c["name"])
        path = SIGNALS_DIR / f"{c['name']}.yaml"
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(c, fh, sort_keys=False, allow_unicode=True, width=100)
    print(f"wrote {len(names)} cards")

    # Append to registry (de-duped), under a clearly-marked Phase 1 section.
    reg_path = SIGNALS_DIR / "_registry.yaml"
    raw = yaml.safe_load(reg_path.read_text(encoding="utf-8")) or {}
    enabled = list(raw.get("enabled") or [])
    existing = set(enabled)
    added = [n for n in names if n not in existing]
    if added:
        lines = reg_path.read_text(encoding="utf-8").rstrip("\n").splitlines()
        lines.append("")
        lines.append("  # ── Phase 1 — additional indicator signals ───────")
        lines.append("  # (Every engine indicator is TA-Lib-backed; these are just the")
        lines.append("  #  Phase 1 signal cards, not a separate engine.)")
        lines.extend(f"  - {n}" for n in added)
        reg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"registry: +{len(added)} new entries ({len(enabled) + len(added)} total)")


if __name__ == "__main__":
    main()
