"""
scripts/migrate_signals_match_when.py — Add `match_when` blocks to every
signal YAML in app/kb/signals/ programmatically.

Run once:

    PYTHONPATH=. python3 scripts/migrate_signals_match_when.py

The script is idempotent — re-running skips files that already have a
`match_when` block, and re-applying after a phrase edit only re-writes
files whose computed block differs from the existing one.

Design notes:

  * The phrase list for each signal is derived from its name/category, NOT
    hand-curated per-signal here. We want one rule per family that is easy
    to extend. After the script runs, individual YAMLs can be hand-edited
    to add custom phrases — the script honours pre-existing `match_when`
    and only fills in when absent.

  * Mirror exits link a bullish signal to its bearish twin (and vice versa)
    so the "exit on opposite crossover" picker rule has a target.

  * `requires_keyword` / `forbids_keyword` are used to disambiguate close
    cousins, e.g. vwap_bullish should NOT match when the prompt contains
    "vwap reclaim" because vwap_reclaim_bullish will win.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
SIGNALS_DIR = REPO / "app" / "kb" / "signals"


# ── Phrase templates by signal name ───────────────────────────────────────────
#
# Each entry: signal_name → match_when dict. We define them all in one place
# so the picker behaviour is auditable from a single source.

SignalSpec = dict[str, Any]
MATCH_SPECS: dict[str, SignalSpec] = {

    # ── ADX ────────────────────────────────────────────────────────────────
    "adx_bullish_di": {
        "phrases": [
            r"adx\s+bullish",
            r"\+di\s+(?:above|crosses?\s+above|>)\s+-di",
            r"\+di\s*>\s*-di",
            r"adx\s+(?:long|buy)\s+(?:setup|signal)",
            r"directional\s+index\s+(?:turns?\s+)?bullish",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "adx_bearish_di",
    },
    "adx_bearish_di": {
        "phrases": [
            r"adx\s+bearish",
            r"-di\s+(?:above|crosses?\s+above|>)\s*\+di",
            r"-di\s*>\s*\+di",
            r"adx\s+(?:short|sell)\s+(?:setup|signal)",
            r"directional\s+index\s+(?:turns?\s+)?bearish",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "adx_bullish_di",
    },
    "adx_strong_trend": {
        "phrases": [
            r"adx\s*(?:>|>=|≥|above|over)\s*(\d+)",
            r"adx\s+(?:is\s+)?(?:strong|strongly\s+trending)",
            r"strong\s+(?:adx\s+)?trend",
            # Tolerate intervening words: "avoid trades during sideways market"
            r"avoid\b[\s\S]{0,40}\b(?:sideways|range[\s\-]+bound|chop(?:py)?|consolidation)",
            r"(?:only|prefer)\b[\s\S]{0,20}\btrend(?:ing|s)?\b",
            r"trend(?:ing)?\s+(?:only|filter|markets?)",
        ],
        "acts_as": "entry_filter",
        "extract_params": {
            "threshold": {"from_group": 1, "default": 25, "kind": "int"},
            "window":    {"from_group": None, "default": 14, "kind": "int"},
        },
    },

    # ── ATR ────────────────────────────────────────────────────────────────
    "atr_high_volatility": {
        "phrases": [
            r"high\s+(?:realised\s+|realized\s+)?volatility",
            r"atr\s+(?:is\s+|expanding|surges?|spikes?)",
            r"volatile\s+(?:market|regime)",
            r"high[\s\-]*atr",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "atr_low_volatility",
    },
    "atr_low_volatility": {
        "phrases": [
            r"low\s+(?:realised\s+|realized\s+)?volatility",
            r"atr\s+(?:is\s+)?(?:contracting|compressed|tight)",
            r"quiet\s+market",
            r"low[\s\-]*atr",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "atr_high_volatility",
    },

    # ── Bollinger Bands ────────────────────────────────────────────────────
    "bb_squeeze": {
        "phrases": [
            r"bollinger\s+(?:band(?:s)?\s+)?squeeze",
            r"bb\s+squeeze",
            r"band\s+contraction",
            r"volatility\s+(?:contraction|squeeze)",
        ],
        "acts_as": "entry_filter",
    },
    "price_above_bb_upper": {
        "phrases": [
            r"(?:price|close)s?\s+(?:closes?\s+|breaks?\s+|is\s+)?above\s+(?:upper\s+)?bollinger",
            r"(?:price|close)s?\s+above\s+bb\s+upper",
            r"bollinger\s+upper\s+(?:band\s+)?(?:break|breach)",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "price_below_bb_lower",
    },
    "price_below_bb_lower": {
        "phrases": [
            r"(?:price|close)s?\s+(?:closes?\s+|breaks?\s+|is\s+)?below\s+(?:lower\s+)?bollinger",
            r"(?:price|close)s?\s+below\s+bb\s+lower",
            r"bollinger\s+lower\s+(?:band\s+)?(?:break|breach)",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "price_above_bb_upper",
    },

    # ── FVG ────────────────────────────────────────────────────────────────
    "bullish_fvg_signal": {
        "phrases": [
            r"bullish\s+(?:fair[\s\-]+value[\s\-]+gap|fvg)",
            r"fvg\s+long",
            r"long\s+(?:on\s+)?(?:bullish\s+)?fvg",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "bearish_fvg_signal",
    },
    "bearish_fvg_signal": {
        "phrases": [
            r"bearish\s+(?:fair[\s\-]+value[\s\-]+gap|fvg)",
            r"fvg\s+short",
            r"short\s+(?:on\s+)?(?:bearish\s+)?fvg",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "bullish_fvg_signal",
    },

    # ── Market structure ───────────────────────────────────────────────────
    "bullish_market_structure_break": {
        "phrases": [
            r"bullish\s+market\s+structure\s+(?:break|shift)",
            r"market\s+structure\s+(?:turns?\s+)?bullish",
            # "market structure break to the upside" / "MSB up"
            r"market\s+structure\s+break\b[\s\S]{0,30}(?:upside|up|long|bullish)",
            r"break\s+of\s+(?:bearish\s+)?structure",
            r"\bms[bs]\s+(?:bullish|long)",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "bearish_market_structure_break",
    },
    "bearish_market_structure_break": {
        "phrases": [
            r"bearish\s+market\s+structure\s+(?:break|shift)",
            r"market\s+structure\s+(?:turns?\s+)?bearish",
            r"market\s+structure\s+break\b[\s\S]{0,30}(?:downside|down|short|bearish)",
            r"break\s+of\s+(?:bullish\s+)?structure",
            r"\bms[bs]\s+(?:bearish|short)",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "bullish_market_structure_break",
    },

    # ── Candle patterns ────────────────────────────────────────────────────
    "bullish_rejection_candle": {
        "phrases": [
            r"bullish\s+rejection\s+candle",
            r"long[\s\-]*wick\s+down",
            r"hammer\s+candle",
            r"bullish\s+pin\s*bar",
            r"rejection\s+candle\s+(?:at|near)\s+(?:support|low)",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "bearish_rejection_candle",
    },
    "bearish_rejection_candle": {
        "phrases": [
            r"bearish\s+rejection\s+candle",
            r"long[\s\-]*wick\s+up",
            r"shooting\s+star",
            r"bearish\s+pin\s*bar",
            r"rejection\s+candle\s+(?:at|near)\s+(?:resistance|high)",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "bullish_rejection_candle",
    },
    "inside_bar": {
        "phrases": [
            r"\binside\s+bar\b",
            r"inside\s+day",
            r"contraction\s+bar",
        ],
        "acts_as": "entry_filter",
    },

    # ── CCI ────────────────────────────────────────────────────────────────
    "cci_overbought": {
        "phrases": [
            r"cci\s+(?:is\s+|crosses?\s+|reaches?\s+)?(?:above|over|>=|>)\s*(\d+)",
            r"cci\s+overbought",
        ],
        "acts_as": "exit_trigger",
        "mirrors_to": "cci_oversold",
        "extract_params": {
            "threshold": {"from_group": 1, "default": 100, "kind": "int"},
        },
    },
    "cci_oversold": {
        "phrases": [
            r"cci\s+(?:is\s+|crosses?\s+|reaches?\s+)?(?:below|under|<=|<)\s*-?(\d+)",
            r"cci\s+oversold",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "cci_overbought",
        "extract_params": {
            "threshold": {"from_group": 1, "default": 100, "kind": "int"},
        },
    },

    # ── Chaikin / OBV / MFI ───────────────────────────────────────────────
    "chaikin_money_flow_positive": {
        "phrases": [
            r"chaikin\s+money\s+flow\s+(?:is\s+)?positive",
            r"cmf\s+(?:is\s+)?(?:positive|>\s*0|above\s+(?:zero|0))",
            r"money\s+flow\s+bullish",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "chaikin_money_flow_negative",
    },
    "chaikin_money_flow_negative": {
        "phrases": [
            r"chaikin\s+money\s+flow\s+(?:is\s+)?negative",
            r"cmf\s+(?:is\s+)?(?:negative|<\s*0|below\s+(?:zero|0))",
            r"money\s+flow\s+bearish",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "chaikin_money_flow_positive",
    },
    "obv_rising": {
        "phrases": [
            r"obv\s+(?:is\s+)?(?:rising|positive|climbing|trending\s+up)",
            r"on[\s\-]+balance\s+volume\s+(?:rising|positive|climbing)",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "obv_falling",
    },
    "obv_falling": {
        "phrases": [
            r"obv\s+(?:is\s+)?(?:falling|negative|declining|trending\s+down)",
            r"on[\s\-]+balance\s+volume\s+(?:falling|negative|declining)",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "obv_rising",
    },
    "mfi_overbought": {
        "phrases": [
            r"mfi\s+(?:is\s+|crosses?\s+|reaches?\s+)?(?:above|over|>=|>)\s*(\d+)",
            r"mfi\s+overbought",
            r"money\s+flow\s+(?:index\s+)?(?:above|>)\s*(\d+)",
        ],
        "acts_as": "exit_trigger",
        "mirrors_to": "mfi_oversold",
        "extract_params": {
            "threshold": {"from_group": 1, "default": 80, "kind": "int"},
        },
    },
    "mfi_oversold": {
        "phrases": [
            r"mfi\s+(?:is\s+|crosses?\s+|reaches?\s+)?(?:below|under|<=|<)\s*(\d+)",
            r"mfi\s+oversold",
            r"money\s+flow\s+(?:index\s+)?(?:below|<)\s*(\d+)",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "mfi_overbought",
        "extract_params": {
            "threshold": {"from_group": 1, "default": 20, "kind": "int"},
        },
    },

    # ── Trend structure ────────────────────────────────────────────────────
    "uptrend_structure": {
        "phrases": [
            r"uptrend(?:ing)?(?:\s+structure)?",
            r"higher\s+highs?\s+and\s+higher\s+lows?",
            r"market\s+(?:in\s+)?uptrend",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "downtrend_structure",
    },
    "downtrend_structure": {
        "phrases": [
            r"downtrend(?:ing)?(?:\s+structure)?",
            r"lower\s+highs?\s+and\s+lower\s+lows?",
            r"market\s+(?:in\s+)?downtrend",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "uptrend_structure",
    },

    # ── EMA ────────────────────────────────────────────────────────────────
    "ema_cross_up": {
        "phrases": [
            r"(\d+)\s*ema\s+(?:crosses?|crossed)\s+(?:above|up\s+through)\s+(\d+)\s*ema",
            r"ema\s+golden\s+cross",
            r"(?:fast|short)\s+ema\s+crosses?\s+(?:above|up)\s+(?:slow|long)\s+ema",
            r"ema\s+bullish\s+cross(?:over)?",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "ema_cross_down",
        "extract_params": {
            "window_fast": {"from_group": 1, "default": None, "kind": "int"},
            "window_slow": {"from_group": 2, "default": None, "kind": "int"},
        },
    },
    "ema_cross_down": {
        "phrases": [
            r"(\d+)\s*ema\s+(?:crosses?|crossed)\s+(?:below|down\s+through)\s+(\d+)\s*ema",
            r"ema\s+death\s+cross",
            r"(?:fast|short)\s+ema\s+crosses?\s+(?:below|down)\s+(?:slow|long)\s+ema",
            r"ema\s+bearish\s+cross(?:over)?",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "ema_cross_up",
        "extract_params": {
            "window_fast": {"from_group": 1, "default": None, "kind": "int"},
            "window_slow": {"from_group": 2, "default": None, "kind": "int"},
        },
    },
    "ema_above": {
        "phrases": [
            # Fast > slow EMA stance, NOT a cross — for trend confirmation
            r"ema\s*\(?\s*(\d+)\s*\)?\s+(?:above|>)\s+ema\s*\(?\s*(\d+)\s*\)?",
            r"fast\s+ema\s+(?:is\s+)?above\s+slow\s+ema",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "ema_below",
        "extract_params": {
            "window_fast": {"from_group": 1, "default": 9, "kind": "int"},
            "window_slow": {"from_group": 2, "default": 21, "kind": "int"},
        },
    },
    "ema_below": {
        "phrases": [
            r"ema\s*\(?\s*(\d+)\s*\)?\s+(?:below|<)\s+ema\s*\(?\s*(\d+)\s*\)?",
            r"fast\s+ema\s+(?:is\s+)?below\s+slow\s+ema",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "ema_above",
        "extract_params": {
            "window_fast": {"from_group": 1, "default": 9, "kind": "int"},
            "window_slow": {"from_group": 2, "default": 21, "kind": "int"},
        },
    },
    "ema_pullback_bullish": {
        "phrases": [
            r"(?:bullish\s+)?ema\s+pullback",
            r"pullback\s+to\s+(?:the\s+)?(?:rising\s+)?ema",
            r"buy\s+on\s+(?:ema\s+)?pullback",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "ema_pullback_bearish",
    },
    "ema_pullback_bearish": {
        "phrases": [
            r"(?:bearish\s+)?ema\s+pullback",
            r"rally\s+to\s+(?:the\s+)?(?:falling\s+)?ema",
            r"short\s+on\s+(?:ema\s+)?(?:pullback|rally)",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "ema_pullback_bullish",
    },
    "ema_sloping_up": {
        "phrases": [
            r"ema\s+(?:slope\s+(?:is\s+)?)?(?:positive|rising|sloping\s+up|upward)",
            r"(?:rising|positive)\s+ema\s+slope",
            r"ema\s+(?:trending\s+)?upward",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "ema_sloping_down",
    },
    "ema_sloping_down": {
        "phrases": [
            r"ema\s+(?:slope\s+(?:is\s+)?)?(?:negative|falling|sloping\s+down|downward)",
            r"(?:falling|negative)\s+ema\s+slope",
            r"ema\s+(?:trending\s+)?downward",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "ema_sloping_up",
    },
    "price_above_ema": {
        "phrases": [
            r"(?:price|close)s?\s+(?:is\s+|stays?\s+|closes?\s+|are\s+)?(?:above|over|>)\s+(?:the\s+)?(\d+)\s*ema",
            r"(?:price|close)s?\s+(?:above|over|>)\s+ema\s*\(?\s*(\d+)\s*\)?",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "price_below_ema",
        "extract_params": {
            "window": {"from_group": 1, "default": 20, "kind": "int"},
        },
    },
    "price_below_ema": {
        "phrases": [
            r"(?:price|close)s?\s+(?:is\s+|stays?\s+|closes?\s+|are\s+)?(?:below|under|<)\s+(?:the\s+)?(\d+)\s*ema",
            r"(?:price|close)s?\s+(?:below|under|<)\s+ema\s*\(?\s*(\d+)\s*\)?",
        ],
        "acts_as": "exit_trigger",
        "mirrors_to": "price_above_ema",
        "extract_params": {
            "window": {"from_group": 1, "default": 20, "kind": "int"},
        },
    },

    # ── SMA ────────────────────────────────────────────────────────────────
    "sma_cross_up": {
        "phrases": [
            r"(\d+)\s*sma\s+(?:crosses?|crossed)\s+(?:above|up\s+through)\s+(\d+)\s*sma",
            r"sma\s+golden\s+cross",
            r"(?:fast|short)\s+sma\s+crosses?\s+(?:above|up)\s+(?:slow|long)\s+sma",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "sma_cross_down",
        "extract_params": {
            "window_fast": {"from_group": 1, "default": None, "kind": "int"},
            "window_slow": {"from_group": 2, "default": None, "kind": "int"},
        },
    },
    "sma_cross_down": {
        "phrases": [
            r"(\d+)\s*sma\s+(?:crosses?|crossed)\s+(?:below|down\s+through)\s+(\d+)\s*sma",
            r"sma\s+death\s+cross",
            r"(?:fast|short)\s+sma\s+crosses?\s+(?:below|down)\s+(?:slow|long)\s+sma",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "sma_cross_up",
        "extract_params": {
            "window_fast": {"from_group": 1, "default": None, "kind": "int"},
            "window_slow": {"from_group": 2, "default": None, "kind": "int"},
        },
    },
    "is_above_sma": {
        "phrases": [
            # Trend filter framings
            r"(?:price|close)s?\s+(?:is\s+|stays?\s+)?above\s+(?:the\s+)?(\d+)[\s\-]*(?:period\s+|day\s+)?sma\s+(?:filter|trend)?",
            r"price\s+above\s+(?:long[\s\-]+term\s+)?sma",
            # Support framings: "200 SMA as support", "200-day SMA as dynamic support",
            # "use 200 SMA as a floor", "SMA(200) acts as support".
            r"(\d+)[\s\-]*(?:day|period)?\s*sma\s+(?:as\s+)?(?:a\s+)?(?:dynamic\s+|key\s+|major\s+)?(?:support|floor|baseline)",
            r"sma\s*\(?\s*(\d+)\s*\)?\s+(?:as\s+|acts?\s+as\s+)?(?:a\s+)?(?:dynamic\s+)?(?:support|floor)",
            # Strategy framings: "support-resistance strategy using 200 SMA"
            r"(?:support[\s\-]*resistance|support)\s+(?:strategy\s+)?(?:using\s+|with\s+|via\s+|on\s+)?(?:the\s+)?(\d+)[\s\-]*(?:day|period)?\s*sma",
            # Bare "SMA as support / floor" without a number (window defaults to 50)
            r"sma\s+(?:as\s+|acts?\s+as\s+)?(?:dynamic\s+)?(?:support|floor)",
        ],
        "acts_as": "entry_filter",
        "extract_params": {
            "window": {"from_group": 1, "default": 50, "kind": "int"},
        },
    },
    "price_below_sma": {
        "phrases": [
            r"(?:price|close)s?\s+(?:is\s+|stays?\s+|closes?\s+|are\s+)?(?:below|under|<)\s+(?:the\s+)?(\d+)\s*sma",
            r"(?:price|close)s?\s+(?:below|under|<)\s+sma\s*\(?\s*(\d+)\s*\)?",
            # Resistance framings: "200 SMA as resistance", "SMA as ceiling"
            r"(\d+)[\s\-]*(?:day|period)?\s*sma\s+(?:as\s+)?(?:a\s+)?(?:dynamic\s+|key\s+|major\s+)?(?:resistance|ceiling|overhead)",
            r"sma\s*\(?\s*(\d+)\s*\)?\s+(?:as\s+|acts?\s+as\s+)?(?:a\s+)?(?:dynamic\s+)?(?:resistance|ceiling)",
        ],
        "acts_as": "exit_trigger",
        "mirrors_to": "price_cross_above_sma",
        "extract_params": {
            "window": {"from_group": 1, "default": 20, "kind": "int"},
        },
    },
    "price_cross_above_sma": {
        "phrases": [
            r"(?:price|close)s?\s+crosses?\s+(?:above|up\s+through)\s+(?:the\s+)?(\d+)\s*sma",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "price_cross_below_sma",
        "extract_params": {
            "window": {"from_group": 1, "default": 20, "kind": "int"},
        },
    },
    "price_cross_below_sma": {
        "phrases": [
            r"(?:price|close)s?\s+crosses?\s+(?:below|down\s+through)\s+(?:the\s+)?(\d+)\s*sma",
        ],
        "acts_as": "exit_trigger",
        "mirrors_to": "price_cross_above_sma",
        "extract_params": {
            "window": {"from_group": 1, "default": 20, "kind": "int"},
        },
    },

    # ── Gap ────────────────────────────────────────────────────────────────
    "gap_up_open": {
        "phrases": [
            r"gap[\s\-]+up\s+(?:open(?:ing)?|day)",
            r"opens?\s+(?:with\s+)?(?:a\s+)?gap[\s\-]+up",
            r"gap[\s\-]+up\s+entry",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "gap_down_open",
    },
    "gap_down_open": {
        "phrases": [
            r"gap[\s\-]+down\s+(?:open(?:ing)?|day)",
            r"opens?\s+(?:with\s+)?(?:a\s+)?gap[\s\-]+down",
            r"gap[\s\-]+down\s+entry",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "gap_up_open",
    },
    "gap_and_go_bullish": {
        "phrases": [
            r"gap[\s\-]+and[\s\-]+go\s+(?:bullish|long|up)",
            r"bullish\s+gap[\s\-]+and[\s\-]+go",
            r"gap\s+up\s+and\s+continuation",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "gap_and_go_bearish",
    },
    "gap_and_go_bearish": {
        "phrases": [
            r"gap[\s\-]+and[\s\-]+go\s+(?:bearish|short|down)",
            r"bearish\s+gap[\s\-]+and[\s\-]+go",
            r"gap\s+down\s+and\s+continuation",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "gap_and_go_bullish",
    },

    # ── Volume ─────────────────────────────────────────────────────────────
    "volume_spike": {
        "phrases": [
            r"volume\s+spike",
            r"volume\s+surge",
            r"(?:high(?:er)?|above[\s\-]+average|strong)\s+volume",
            # "higher than average volume", "more than average volume"
            r"(?:high(?:er)?|more|greater)\s+than\s+(?:the\s+)?(?:avg|average)\s+volume",
            r"volume\s+(?:>|above|over|at\s+least)\s*(\d+(?:\.\d+)?)\s*(?:x|×|times?)",
            r"(\d+(?:\.\d+)?)\s*(?:x|×|times?)\s+(?:the\s+)?(?:avg|average)\s+volume",
            r"volume\s+confirmation",
            r"with\s+volume",
        ],
        "acts_as": "entry_filter",
        "extract_params": {
            "multiplier": {"from_group": 1, "default": 1.5, "kind": "float"},
            "window":     {"from_group": None, "default": 20, "kind": "int"},
        },
    },
    "high_delivery_volume": {
        "phrases": [
            r"high\s+delivery\s+volume",
            r"strong\s+delivery",
            r"delivery\s+percentage\s+(?:high|above)",
        ],
        "acts_as": "entry_filter",
    },

    # ── Keltner ────────────────────────────────────────────────────────────
    "keltner_breakout_up": {
        "phrases": [
            r"keltner\s+(?:channel\s+)?(?:breakout|break)\s+(?:up|above|bullish)",
            r"price\s+breaks?\s+(?:above\s+)?upper\s+keltner",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "keltner_breakout_down",
    },
    "keltner_breakout_down": {
        "phrases": [
            r"keltner\s+(?:channel\s+)?(?:breakout|break)\s+(?:down|below|bearish)",
            r"price\s+breaks?\s+(?:below\s+)?lower\s+keltner",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "keltner_breakout_up",
    },

    # ── MACD ───────────────────────────────────────────────────────────────
    "macd_bullish_cross": {
        "phrases": [
            r"macd\s+(?:line\s+)?(?:crosses?|crossed|cross)\s+above\s+signal",
            r"macd\s+(?:bullish|positive)\s+cross(?:over)?",
            r"macd\s+up\s+cross",
            r"macd\s+golden\s+cross",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "macd_bearish_cross",
    },
    "macd_bearish_cross": {
        "phrases": [
            r"macd\s+(?:line\s+)?(?:crosses?|crossed|cross)\s+below\s+signal",
            r"macd\s+(?:bearish|negative)\s+cross(?:over)?",
            r"macd\s+down\s+cross",
            r"macd\s+death\s+cross",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "macd_bullish_cross",
    },
    "macd_positive": {
        "phrases": [
            r"macd\s+(?:line\s+)?(?:is\s+|>)?\s*positive",
            r"macd\s+(?:histogram\s+)?(?:turns?\s+|is\s+)?positive",
            r"macd\s+above\s+(?:zero|0)",
            r"macd\s+histogram\s+(?:>|above)\s*0",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "macd_negative",
        "forbids_keyword": [r"macd\s+(?:line\s+)?cross(?:es)?\s+above\s+signal"],
    },
    "macd_negative": {
        "phrases": [
            r"macd\s+(?:line\s+)?(?:is\s+|<)?\s*negative",
            r"macd\s+(?:histogram\s+)?(?:turns?\s+|is\s+)?negative",
            r"macd\s+below\s+(?:zero|0)",
            r"macd\s+histogram\s+(?:<|below)\s*0",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "macd_positive",
        "forbids_keyword": [r"macd\s+(?:line\s+)?cross(?:es)?\s+below\s+signal"],
    },

    # ── Breakouts ──────────────────────────────────────────────────────────
    "n_bar_high_breakout": {
        "phrases": [
            r"break(?:s|out)?\s+(?:above\s+|over\s+)?(\d+)[\s\-]*bar\s+high",
            r"new\s+(\d+)[\s\-]*bar\s+high",
            r"highest\s+high\s+(?:in\s+|of\s+(?:the\s+)?last\s+)?(\d+)\s+bars?",
            # Generic breakout phrasings (no specific bar count)
            r"\bbreakout\s+(?:trading\s+)?strategy",
            r"\b(?:price|stock)\s+breaks?\s+out",
            r"breakout\s+above\s+(?:the\s+)?(?:previous|prior|recent)\s+high",
            r"(?:break|breakout)\s+(?:of\s+|over\s+)?(?:previous|prior)\s+(?:day|session)\s+high",
            r"clean\s+(?:intraday\s+)?breakout",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "n_bar_low_breakdown",
        "extract_params": {
            "window": {"from_group": 1, "default": 20, "kind": "int"},
        },
    },
    "n_bar_low_breakdown": {
        "phrases": [
            r"break(?:s|down)?\s+(?:below\s+|under\s+)?(\d+)[\s\-]*bar\s+low",
            r"new\s+(\d+)[\s\-]*bar\s+low",
            r"lowest\s+low\s+(?:in\s+|of\s+(?:the\s+)?last\s+)?(\d+)\s+bars?",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "n_bar_high_breakout",
        "extract_params": {
            "window": {"from_group": 1, "default": 20, "kind": "int"},
        },
    },
    "near_52_week_high": {
        "phrases": [
            r"(?:near|approaching|close\s+to)\s+52[\s\-]*week\s+high",
            r"all[\s\-]*time\s+high\s+(?:area|zone)",
            r"new\s+52[\s\-]*week\s+high",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "near_52_week_low",
    },
    "near_52_week_low": {
        "phrases": [
            r"(?:near|approaching|close\s+to)\s+52[\s\-]*week\s+low",
            r"new\s+52[\s\-]*week\s+low",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "near_52_week_high",
    },
    "opening_range_breakout": {
        "phrases": [
            r"opening\s+range\s+breakout",
            r"\bor\s*[\s\-]*breakout",
            r"breaks?\s+(?:above\s+)?opening\s+range",
            r"\borb\b",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "opening_range_breakdown",
    },
    "opening_range_breakdown": {
        "phrases": [
            r"opening\s+range\s+breakdown",
            r"\bor\s*[\s\-]*breakdown",
            r"breaks?\s+(?:below\s+)?opening\s+range",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "opening_range_breakout",
    },

    # ── RSI ────────────────────────────────────────────────────────────────
    "rsi_above_50": {
        "phrases": [
            r"rsi\s+(?:is\s+|crosses?\s+)?(?:above|over|>)\s*50\b",
            r"rsi\s+>\s*50",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "rsi_below_50",
        "extract_params": {
            "threshold": {"from_group": None, "default": 50, "kind": "int"},
            "window":    {"from_group": None, "default": 14, "kind": "int"},
        },
    },
    "rsi_above_60": {
        "phrases": [
            r"rsi\s+(?:is\s+|crosses?\s+)?(?:above|over|>)\s*60\b",
            r"rsi\s+>\s*60",
            r"strong\s+rsi",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "rsi_below_40",
        "extract_params": {
            "threshold": {"from_group": None, "default": 60, "kind": "int"},
            "window":    {"from_group": None, "default": 14, "kind": "int"},
        },
    },
    "rsi_below_50": {
        "phrases": [
            r"rsi\s+(?:is\s+|crosses?\s+)?(?:below|under|<)\s*50\b",
            r"rsi\s+<\s*50",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "rsi_above_50",
        "extract_params": {
            "threshold": {"from_group": None, "default": 50, "kind": "int"},
            "window":    {"from_group": None, "default": 14, "kind": "int"},
        },
    },
    "rsi_below_40": {
        "phrases": [
            r"rsi\s+(?:is\s+|crosses?\s+)?(?:below|under|<)\s*40\b",
            r"rsi\s+<\s*40",
            r"weak\s+rsi",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "rsi_above_60",
        "extract_params": {
            "threshold": {"from_group": None, "default": 40, "kind": "int"},
            "window":    {"from_group": None, "default": 14, "kind": "int"},
        },
    },
    "rsi_cross_up": {
        "phrases": [
            r"rsi\s+crosses?\s+(?:above|up\s+through)\s+(\d+(?:\.\d+)?)",
            r"rsi\s+(?:exits?|breaks?\s+out\s+of)\s+oversold",
            r"rsi\s+from\s+oversold",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "rsi_cross_down",
        "extract_params": {
            "threshold": {"from_group": 1, "default": 30, "kind": "float"},
            "window":    {"from_group": None, "default": 14, "kind": "int"},
        },
    },
    "rsi_cross_down": {
        "phrases": [
            r"rsi\s+crosses?\s+(?:below|down\s+through)\s+(\d+(?:\.\d+)?)",
            r"rsi\s+(?:exits?|drops\s+out\s+of)\s+overbought",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "rsi_cross_up",
        "extract_params": {
            "threshold": {"from_group": 1, "default": 70, "kind": "float"},
            "window":    {"from_group": None, "default": 14, "kind": "int"},
        },
    },
    "rsi_overbought": {
        "phrases": [
            # "RSI reaches above 65" / "RSI > 70" / "exit when RSI reaches 80"
            r"rsi\s+reaches?\s+(?:above|over|>=|>)?\s*(\d{2})",
            # "exit when RSI is above 65 / 70 / 75 / 80 / 85 / 90"
            r"(?:exit|sell|close)\b[\s\S]{0,40}\brsi\s+(?:is\s+|crosses?\s+)?(?:above|over|>=|>)\s*(6[5-9]|7\d|8\d|9\d)\b",
            r"rsi\s+overbought",
        ],
        "acts_as": "exit_trigger",
        "mirrors_to": "rsi_oversold",
        "extract_params": {
            "threshold": {"from_group": 1, "default": 70, "kind": "float"},
            "window":    {"from_group": None, "default": 14, "kind": "int"},
        },
    },
    "rsi_oversold": {
        "phrases": [
            r"rsi\s+(?:is\s+|reaches?\s+|crosses?\s+)?(?:below|under|<=|<)\s*([12]\d|0\d|3[0-5])\b",
            r"rsi\s+oversold",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "rsi_overbought",
        "extract_params": {
            "threshold": {"from_group": 1, "default": 30, "kind": "float"},
            "window":    {"from_group": None, "default": 14, "kind": "int"},
        },
    },

    # ── Reference / RS ─────────────────────────────────────────────────────
    "reference_above_sma": {
        "phrases": [
            r"reference\s+(?:index\s+)?above\s+sma",
            r"nifty\s+above\s+(?:its\s+)?sma",
            r"banknifty\s+above\s+(?:its\s+)?sma",
            r"benchmark\s+above\s+sma",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "reference_below_sma",
    },
    "reference_below_sma": {
        "phrases": [
            r"reference\s+(?:index\s+)?below\s+sma",
            r"nifty\s+below\s+(?:its\s+)?sma",
            r"banknifty\s+below\s+(?:its\s+)?sma",
            r"benchmark\s+below\s+sma",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "reference_above_sma",
    },
    "rs_outperforms_reference": {
        "phrases": [
            r"outperform(?:s|ing)?\s+(?:nifty|banknifty|reference|benchmark|index)",
            r"relative\s+strength\s+(?:to|vs)\s+(?:nifty|banknifty|reference|benchmark)",
            r"stronger\s+than\s+(?:nifty|banknifty|the\s+market)",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "rs_underperforms_reference",
    },
    "rs_underperforms_reference": {
        "phrases": [
            r"underperform(?:s|ing)?\s+(?:nifty|banknifty|reference|benchmark|index)",
            r"weaker\s+than\s+(?:nifty|banknifty|the\s+market)",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "rs_outperforms_reference",
    },

    # ── Stochastic ─────────────────────────────────────────────────────────
    "stoch_overbought": {
        "phrases": [
            r"stoch(?:astic)?\s+(?:is\s+|crosses?\s+|reaches?\s+)?(?:above|over|>=|>)\s*(\d+)",
            r"stoch(?:astic)?\s+overbought",
        ],
        "acts_as": "exit_trigger",
        "mirrors_to": "stoch_oversold",
        "extract_params": {
            "threshold": {"from_group": 1, "default": 80, "kind": "int"},
        },
    },
    "stoch_oversold": {
        "phrases": [
            r"stoch(?:astic)?\s+(?:is\s+|crosses?\s+|reaches?\s+)?(?:below|under|<=|<)\s*(\d+)",
            r"stoch(?:astic)?\s+oversold",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "stoch_overbought",
        "extract_params": {
            "threshold": {"from_group": 1, "default": 20, "kind": "int"},
        },
    },

    # ── Supertrend ─────────────────────────────────────────────────────────
    "supertrend_bullish": {
        "phrases": [
            r"super[\s\-]*trend\s+(?:turns?\s+|is\s+)?(?:bullish|positive|green|up)",
            r"super[\s\-]*trend\s+long",
            r"super[\s\-]*trend\s+buy",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "supertrend_bearish",
    },
    "supertrend_bearish": {
        "phrases": [
            r"super[\s\-]*trend\s+(?:turns?\s+|is\s+)?(?:bearish|negative|red|down)",
            r"super[\s\-]*trend\s+short",
            r"super[\s\-]*trend\s+sell",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "supertrend_bullish",
    },

    # ── VWAP ───────────────────────────────────────────────────────────────
    "vwap_bullish": {
        "phrases": [
            r"(?:price|close)s?\s+(?:is\s+|stays?\s+|closes?\s+|are\s+)?above\s+vwap",
            r"(?:above|over|>)\s+vwap",
            r"trading\s+(?:above|over)\s+vwap",
        ],
        "acts_as": "entry_filter",
        "mirrors_to": "vwap_bearish",
        # Don't fire when the user specifically said "reclaim"
        "forbids_keyword": [r"vwap\s+reclaim", r"vwap\s+(?:bounce|rejection)"],
    },
    "vwap_bearish": {
        "phrases": [
            r"(?:price|close)s?\s+(?:is\s+|stays?\s+|closes?\s+|are\s+)?below\s+vwap",
            r"(?:below|under|<)\s+vwap",
            r"trading\s+(?:below|under)\s+vwap",
        ],
        "acts_as": "exit_trigger",
        "mirrors_to": "vwap_bullish",
        "forbids_keyword": [r"vwap\s+reclaim", r"vwap\s+(?:bounce|rejection)"],
    },
    "vwap_reclaim_bullish": {
        "phrases": [
            r"vwap\s+reclaim",
            r"price\s+reclaims?\s+vwap",
            r"vwap\s+(?:bullish\s+)?bounce",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "vwap_reclaim_bearish",
    },
    "vwap_reclaim_bearish": {
        "phrases": [
            r"vwap\s+rejection",
            r"vwap\s+bearish\s+(?:bounce|reject)",
            r"price\s+rejects?\s+vwap",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "vwap_reclaim_bullish",
    },
    "vwap_zscore_overbought": {
        "phrases": [
            r"vwap\s+(?:z[\s\-]*score|deviation)\s+(?:is\s+|above|>)\s*(\d+(?:\.\d+)?)",
            r"price\s+stretched\s+above\s+vwap",
        ],
        "acts_as": "exit_trigger",
        "mirrors_to": "vwap_zscore_oversold",
        "extract_params": {
            "threshold": {"from_group": 1, "default": 2, "kind": "float"},
        },
    },
    "vwap_zscore_oversold": {
        "phrases": [
            r"vwap\s+(?:z[\s\-]*score|deviation)\s+(?:is\s+|below|<)\s*-?\s*(\d+(?:\.\d+)?)",
            r"price\s+stretched\s+below\s+vwap",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "vwap_zscore_overbought",
        "extract_params": {
            "threshold": {"from_group": 1, "default": 2, "kind": "float"},
        },
    },

    # ── Z-score (generic) ─────────────────────────────────────────────────
    "zscore_overbought": {
        "phrases": [
            # "z-score above 2", "z-score is above 2", "z-score > 2"
            r"z[\s\-]*score\s+(?:is\s+)?(?:above|over|>=|>)\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s+standard\s+deviations?\s+above",
        ],
        "acts_as": "exit_trigger",
        "mirrors_to": "zscore_oversold",
        "extract_params": {
            "threshold": {"from_group": 1, "default": 2, "kind": "float"},
            "window":    {"from_group": None, "default": 20, "kind": "int"},
        },
        # Require the user to explicitly mention z-score or std dev
        "requires_keyword": [r"z[\s\-]*score|standard\s+deviation"],
    },
    "zscore_oversold": {
        "phrases": [
            # "z-score below -2", "z-score is below -2", "z-score < -2"
            r"z[\s\-]*score\s+(?:is\s+)?(?:below|under|<=|<)\s*-?\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s+standard\s+deviations?\s+below",
        ],
        "acts_as": "entry_trigger",
        "mirrors_to": "zscore_overbought",
        "extract_params": {
            "threshold": {"from_group": 1, "default": 2, "kind": "float"},
            "window":    {"from_group": None, "default": 20, "kind": "int"},
        },
        "requires_keyword": [r"z[\s\-]*score|standard\s+deviation"],
    },
}


# ── Migration runner ──────────────────────────────────────────────────────────


def _format_yaml_addition(spec: SignalSpec) -> str:
    """Render a match_when block as a YAML string ready to append."""
    # Use yaml.safe_dump for consistency; quote phrases (regex) as scalars.
    out_dict = {"match_when": spec}
    text = yaml.safe_dump(out_dict, sort_keys=False, allow_unicode=True, width=1000)
    return "\n" + text


def migrate(overwrite: bool = True) -> None:
    """Apply MATCH_SPECS to every signal YAML.

    `overwrite=True` (default) replaces any existing match_when block —
    treat MATCH_SPECS as the single source of truth.
    """
    if not SIGNALS_DIR.exists():
        print(f"ERROR: signals dir not found: {SIGNALS_DIR}")
        sys.exit(1)

    counted = {"added": 0, "updated": 0, "skipped": 0, "missing": 0}
    for yaml_path in sorted(SIGNALS_DIR.glob("*.yaml")):
        name = yaml_path.stem
        if name.startswith("_"):
            continue
        spec = MATCH_SPECS.get(name)
        if spec is None:
            counted["missing"] += 1
            print(f"  no match_when spec for: {name}")
            continue
        existing = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        had_block = bool(existing.get("match_when"))
        if had_block and not overwrite:
            counted["skipped"] += 1
            continue
        # Re-write the file with the new block.
        existing["match_when"] = spec
        new_text = yaml.safe_dump(existing, sort_keys=False, allow_unicode=True, width=1000)
        yaml_path.write_text(new_text, encoding="utf-8")
        counted["updated" if had_block else "added"] += 1
    total = sum(counted.values())
    print(
        f"\nDone — {counted['added']} added, {counted['updated']} updated, "
        f"{counted['skipped']} skipped, {counted['missing']} missing-spec "
        f"(out of {total} examined)"
    )


if __name__ == "__main__":
    migrate()
