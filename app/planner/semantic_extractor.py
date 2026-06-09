"""
app/planner/semantic_extractor.py — Extract advanced execution-level requirements from natural language.

Parses trader-style prompts to detect and structure:
  1. Multi-timeframe conditions (HTF rules)
  2. Cross-symbol / reference-symbol logic
  3. Structural stop-loss extraction
  4. Trailing stop extraction
  5. Risk:Reward enforcement
  6. Session and time-window filters
  7. Dual-direction strategies
  8. Volume and momentum confirmation
  9. Semantic strategy preservation
  10. Full execution orchestration

Outputs SemanticInstructions which bridge user intent to execution config.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from app.kb.market_aliases import BENCHMARK_ALIASES, normalise_benchmark
from app.kb.execution_schemas import (
    CandleConfirmation,
    HTFCondition,
    HTFRoleType,
    MomentumConfirmation,
    MomentumFilterType,
    PaddingMethod,
    ReferenceSymbolCondition,
    RelationshipType,
    RiskRewardSpec,
    RiskRewardType,
    SemanticInstructions,
    SessionFilter,
    StopLossPadding,
    StopLossType,
    StructuralStopLoss,
    TimeWindow,
    TrailingStopConfig,
    TrailingStopType,
    VolumeAndMomentumFilters,
    VolumeConfirmation,
    VolumeFilterType,
)

logger = logging.getLogger(__name__)


class SemanticExtractor:
    """Parse natural language strategy prompts into structured execution instructions."""

    # Strategy family keywords mapped to family names
    STRATEGY_FAMILIES: dict[str, str] = {}  # Removed — framework detection is now KB-driven via SemanticIntentNormalizer

    # HTF patterns: phrases that indicate higher-timeframe conditions
    HTF_PATTERNS = [
        (r"(?:trend\s+should\s+remain|higher\s+timeframe\s+trend|timeframe.*?trend)\s+(?:bullish|bearish|up|down)", "generic_htf_trend"),
        (r"(?:higher\s+highs|higher\s+lows|bullish\s+structure|bearish\s+structure).*?(?:should|continue|forming)", "structural_trend"),
        (r"(\d+[mh]|daily|weekly|hourly)\s+(?:trend|ema|rsi|macd|momentum).*?(?:bullish|bearish|up|down|above|below)", "trend_condition"),
        (r"(?:higher\s+timeframe|htf|upper\s+tf|h\.?t\.?f\.?).*?(?:bullish|bearish|up|down|should)", "htf_direction"),
        (r"(\d+h|daily)\s+(?:trend|close|ema)\s+(?:should\s+be|must\s+be|should\s+remain)\s+(bullish|bearish|up|down)", "htf_requirement"),
        (r"(?:trade\s+only\s+with|trade\s+when|only\s+trade\s+if|when).*?(\d+[mh]|daily)\s+(.+?)(?:\.|,|$)", "conditional_htf"),
        # Simple patterns: "1h ema above", "daily trend bullish", "above 1h EMA"
        (r"(?:above|below)\s+(?:the\s+)?(\d+[mh]|daily|weekly)\s+(?:ema|sma|trend|vwap)", "above_below_htf"),
        (r"(\d+[mh]|daily|weekly)\s+(?:ema|sma|trend|vwap)\s+(?:should\s+be\s+)?(?:above|below|bullish|bearish|rising|falling)", "htf_indicator_direction"),
        # "Bank Nifty direction should support", "index should be bullish"
        (r"(?:bank\s*nifty|nifty|index|market)\s+(?:direction|trend)\s+(?:should|must)\s+(?:support|be\s+(?:bullish|bearish|up|down))", "index_direction"),
        # "trade only in direction of daily trend"
        (r"(?:direction of|aligned with|in line with)\s+(?:the\s+)?(\d+[mh]|daily|weekly)\s+(?:trend|ema|structure)", "aligned_with_htf"),
    ]

    # SL anchor patterns (more specific patterns first).
    # Patterns are ordered: forward-order ("anchor as stop") before
    # backward-order ("stop at anchor") so the most natural phrasing wins.
    SL_PATTERNS = [
        # ── Forward-order patterns: "anchor as [structural] stop [loss]" ──────
        # e.g. "use opening range low as structural stop loss"
        (r"opening\s+range\s+low\s+as\s+(?:(?:structural|the|a)\s+)?stop(?:\s+loss)?", "orb_low"),
        (r"(?:orb|opening\s+range)\s+low\s+(?:as|is|=)\s+(?:the\s+)?(?:structural\s+)?stop", "orb_low"),
        (r"(?:candle|bar|breakout\s+candle)\s+low\s+as\s+(?:(?:structural|the|a)\s+)?stop(?:\s+loss)?", "candle_low"),
        (r"swing\s+low\s+as\s+(?:(?:structural|the|a)\s+)?stop(?:\s+loss)?", "swing_low"),
        (r"vwap\s+(?:level\s+)?as\s+(?:the\s+)?stop(?:\s+loss)?", "vwap_deviation"),
        (r"opposite\s+(?:end|side)\s+of\s+(?:the\s+)?(?:orb|range|candle)\s+as\s+(?:the\s+)?stop", "opposite_side"),
        # "Stop Loss: Opposite side of the range" — colon-delimited structural format
        (r"stop(?:\s*loss)?\s*:\s*opposite\s+(?:side|end)\s+of\s+(?:the\s+)?range", "opposite_side"),
        # "stop at/is the opposite side of the range"
        (r"stop(?:\s+loss)?\s+(?:at|is|=)\s+(?:the\s+)?opposite\s+(?:side|end)", "opposite_side"),
        # "opposite side of the range" alone (context: following "stop loss:")
        (r"opposite\s+side\s+of\s+(?:the\s+)?(?:orb|range)", "opposite_side"),
        # ── Backward-order patterns: "stop at/below anchor" ──────────────────
        (r"below\s+(?:\w+\s+)*reclaim\s+candle\s+low", "candle_low"),
        (r"below\s+(?:the\s+)?(?:vwap\s+)?reclaim\s+candle", "candle_low"),
        (r"below\s+(?:recent\s+)?swing\s+low", "swing_low"),
        (r"below\s+(?:the\s+)?recent\s+(?:swing\s+)?low", "swing_low"),
        (r"below\s+(?:the\s+)?orb\s+low", "orb_low"),
        (r"opposite\s+side\s+(?:of\s+)?(?:the\s+)?candle", "opposite_side"),
        (r"below\s+(?:the\s+)?wick\s+low", "candle_low"),
        (r"below\s+vwap", "vwap_deviation"),
        # "below breakout candle low" / "below entry candle low" / "below previous candle low"
        (r"below\s+(?:the\s+)?(?:breakout|entry|previous|prior|key)\s+candle\s+low", "candle_low"),
        # "below the candle low" (generic)
        (r"below\s+(?:the\s+)?candle(?:\s+low)?", "candle_low"),
        # "structure-based stop loss" / "structural stop loss" / "use structure as stop"
        (r"struct(?:ure|ural)\s*[- ]?based\s+stop(?:\s+loss)?", "swing_low"),
        (r"struct(?:ure|ural)\s+stop(?:\s+loss)?", "swing_low"),
        (r"use\s+struct(?:ure|ural)\s+(?:as|for)\s+(?:the\s+)?stop", "swing_low"),
        # "previous candle low" / "prior candle low" as implicit SL
        (r"(?:stop|sl)\s+(?:at\s+)?(?:the\s+)?(?:previous|prior|last)\s+candle\s+low", "candle_low"),
        (r"(?:stop\s+loss|sl)\s+(?:at\s+)?(?:below\s+)?(?:recent\s+)?swing\s+low", "swing_low"),
        (r"(\d+\.?\d*)\s*(?:atr|x\s*atr)\s+(?:below|padding|padded)", "atr_padded"),
        (r"(?:with\s+)?atr\s+padding", "atr_padded"),
        (r"(?:sl|stop(?:\s*loss)?)\s+(?:at\s+)?(?:the\s+)?opening\s+range\s+low", "orb_low"),
        (r"(?:sl|stop(?:\s*loss)?)\s+(?:at\s+)?(?:the\s+)?9[:h]15\s+candle\s+low", "candle_low"),
        (r"(?:sl|stop(?:\s*loss)?)\s+(?:at\s+)?(?:the\s+)?first\s+candle\s+low", "candle_low"),
        (r"below\s+(?:the\s+)?(?:9[:h]15|opening)\s+candle", "candle_low"),
        (r"(?:sl|stop)\s+=\s*(?:opening\s+range|orb)\s+low", "orb_low"),
        (r"sl\s+at\s+(?:the\s+)?(?:opposite|other)\s+end\s+of\s+(?:the\s+)?(?:orb|range|candle)", "opposite_side"),
    ]

    # Trailing stop patterns
    TRAILING_PATTERNS = [
        (r"ema\s+trailing\s+(?:stop|exit)", "ema_based"),
        (r"atr\s+trailing\s+(?:stop|exit)", "atr_based"),
        (r"chandelier\s+(?:exit|stop)", "chandelier"),
        # Combined: "percent trailing stop after X% gain" / "trailing X% stop after Y% move"
        # Captures activation threshold in group(1)
        (r"percent\s+trailing\s+stop\s+after\s+(\d+\.?\d*)\s*%", "percent_with_activation"),
        (r"trail(?:ing)?\s+(?:stop|exit)\s+after\s+(\d+\.?\d*)\s*%", "percent_with_activation"),
        (r"trail(?:ing)?\s+(?:stop|exit).*?(?:after|once)\s+(?:a\s+)?(\d+\.?\d*)\s*%\s*(?:gain|move|profit|up)?", "percent_with_activation"),
        # Standalone type patterns (no activation threshold)
        (r"percent\s+trailing", "percent_based"),
        (r"dynamic\s+(?:trail|trailing)", "dynamic"),
    ]

    # RR patterns — CONVENTION: ratio is always reward/risk (2.0 means 2:1 reward:risk).
    # "2:1 Risk Reward" means 2 reward for 1 risk → ratio = 2.0
    # "1:2 Risk Reward" means 1 reward for 2 risk → ratio = 0.5 (rare, but possible)
    RR_PATTERNS = [
        # ── Reward-first notation (N:1) — "2:1 RR", "2:1 Risk Reward", "3:1 reward" ──
        # The FIRST number is the REWARD, second is RISK (1). ratio = N
        (r"(\d+\.?\d*)\s*:\s*1\s+(?:r(?:isk)?[\s:\-]*r(?:eward)?|reward|rr)\b",  "rr_reward_first"),
        (r"(\d+\.?\d*)\s*:\s*1\s+(?:risk[\s\-]*reward|reward[\s\-]*risk)",        "rr_reward_first"),
        (r"(\d+\.?\d*)\s*[x×]\s*(?:rr|risk[\s\-]*reward|reward)",                 "rr_reward_first"),
        # "take profit: 2:1" with no trailing word — assume reward:risk
        (r"(?:take[\s\-]*profit|tp|target)\s*:?\s*(\d+\.?\d*)\s*:\s*1",           "rr_reward_first"),
        # ── Risk-first notation "1:N" — "1:2 RR", "risk:reward 1:2" ──────────────
        # The first number is risk (1), second is reward (N). ratio = N
        (r"(?:risk[:\s]+)?reward\s*[:\s=]+\s*1\s*:\s*(\d+\.?\d*)",               "rr_ratio"),
        (r"rr\s+of\s+1:(\d+\.?\d*)",                                              "rr_ratio"),
        (r"\b1\s*[:/]\s*(\d+\.?\d*)\s+(?:rr|r:r|risk[:\s]*reward|reward)",       "rr_ratio"),
        (r"(?:r:r|rr|risk[\s:]+reward)\s+1\s*[:/]\s*(\d+\.?\d*)",                "rr_ratio"),
        (r"(?:minimum|min|at\s+least)\s+(?:rr\s+)?1\s*[:/]\s*(\d+\.?\d*)",      "minimum_rr"),
        (r"minimum\s+(?:1\s*:\s*)?(\d+\.?\d*)\s+(?:rr|risk[:\s]+reward)",        "minimum_rr"),
        # ── R-multiple notation: "2R target", "1.5R profit" ──────────────────────
        (r"\b(\d+(?:\.\d+)?)\s*r\b(?:\s+(?:target|profit|booking|exit))?",       "rr_ratio"),
        # ── LEGACY (was causing the 2:1 → 0.5 inversion bug) ─────────────────────
        # "N:1 risk" alone (no "reward" word) — ratio = 1/N (rarely written this way)
        (r"(\d+\.?\d*)\s*:\s*1\s+risk\b(?!\s*[\-:]?\s*reward)",                  "rr_inverted"),
    ]

    # Reference symbol patterns
    REF_SYMBOL_PATTERNS = [
        # Explicit uppercase tickers: "outperforming BANKNIFTY", "vs NIFTY50"
        (r"(?:outperforming|stronger than|vs\.?|relative to|compared to)\s+([A-Z][A-Z0-9_]{2,})", "relative_strength"),
        # Lowercase common benchmarks: "outperforming nifty", "relative to bank nifty"
        (r"(?:outperforming|stronger than|vs\.?|relative to|compared to|beats?)\s+(nifty\s*50|nifty\s*bank|bank\s*nifty|nifty|sensex|banknifty|finnifty|nifty\s*it|nifty\s*pharma|nifty\s*midcap)", "relative_strength"),
        # "stock outperforms nifty", "outperforms the index"
        (r"outperform(?:s|ing)?\s+(?:the\s+)?(?:nifty|nifty\s*50|bank\s*nifty|banknifty|index|market|benchmark)", "relative_strength"),
        # Index confirmation: "BANKNIFTY should be bullish"
        (r"([A-Z]+(?:NIFTY|BANK)?)\s+(?:should be|must be|direction)\s+(bullish|bearish)", "index_confirmation"),
        # "in line with nifty", "with NIFTY"
        (r"(?:in line with|align(?:ed)? with|with)\s+(nifty\s*50|nifty|NIFTY50|NIFTY\s*50|banknifty|bank\s*nifty)", "benchmark"),
        # "relative strength to/vs nifty"
        (r"relative\s+strength\s+(?:to|vs\.?|compared to|against)\s+([A-Z0-9a-z\s]+?)(?:\s|$|,|\.)", "rs_explicit"),
    ]

    # Session/timing patterns
    SESSION_PATTERNS = [
        (r"(?:after|trade after|only after|from)\s+(\d{1,2}):?(\d{2})?\s*(?:am|a\.m\.|AM)", "start_time"),
        (r"(?:before|trade before|until)\s+(\d{1,2}):?(\d{2})?\s*(?:pm|p\.m\.|PM)", "end_time"),
        (r"(?:first|initial)\s+(\d+)\s+(?:minutes?|min)", "duration_from_open"),
        # "first 1-hour range", "first 2-hour range", "initial 30-minute range" — ORB definition
        (r"(?:first|initial|opening)\s+(\d+)[\s\-]*hour\s+(?:range|candle|session|period)", "orb_duration_hours"),
        (r"(?:first|initial|opening)\s+(\d+)[\s\-]*(?:minute|min)\s+(?:range|candle|session)", "orb_duration_minutes"),
        (r"first\s+hour\s+(?:range|candle|session|period)", "orb_duration_1h"),
        (r"(?:avoid|skip)\s+(?:market\s+)?open.*?(\d+)\s+(?:minutes?|min)", "blackout_from_open"),
        (r"(?:during\s+)?(?:high\s+)?liquidity\s+(?:market\s+)?hours?", "session_type"),
        (r"(?:morning|afternoon)\s+session", "session_type"),
        (r"(?:market\s+)?timing.*?(?:high\s+)?liquidity", "session_type"),
    ]

    # ORB duration in minutes — for constraint_compiler to compute opening_bars
    # Stored as a separate field on SemanticInstructions via _extract_orb_duration
    ORB_DURATION_PATTERNS = [
        (r"(?:first|initial|opening)\s+(\d+)[\s\-]*hour\s+(?:range|candle|session|period)", "hours"),
        (r"(?:first|initial|opening)\s+(\d+)[\s\-]*(?:minute|min)\s+(?:range|candle|session)", "minutes"),
        (r"first\s+hour\s+(?:range|candle|session|period)", "1h"),
        (r"(?:first|initial)\s+(\d+)\s+(?:minutes?|min)\s+(?:range|candle|session|period)", "minutes"),
    ]

    # Volume patterns
    VOLUME_PATTERNS = [
        (r"volume\s+(?:should|must)\s+(?:increase|rise|spike)", "spike"),
        (r"(?:strong|high)\s+(?:participation|volume|institutional\s+volume)", "spike"),
        (r"volume\s+(?:should\s+)?remain\s+strong", "spike"),  # "Volume should remain strong"
        (r"volume\s+above\s+(?:average|sma)", "above_average"),
        (r"obv\s+(?:rising|increasing)", "obv_rising"),
        (r"chaikin\s+(?:positive|bullish)", "chaikin_positive"),
        (r"(?:strong|high|bullish)\s+(?:participation|volume)", "spike"),
    ]

    # Candle confirmation patterns
    CANDLE_PATTERNS = [
        (r"(?:bullish|green)\s+candle\s+(?:should\s+)?(?:close|confirm|closes|closes.*?above)", "bullish_confirmation"),
        (r"(?:confirmation|confirming)\s+(?:candle|bar)", "bullish_confirmation"),
        (r"bullish.*?candle.*?(?:close|confirm)", "bullish_confirmation"),
        (r"(?:green|bullish)\s+(?:close|closing)", "bullish_confirmation"),
        (r"candle\s+(?:should\s+)?(?:close|confirm)\s+above", "bullish_confirmation"),
        (r"(?:bearish|red)\s+candle.*?(?:close|break)", "bearish_confirmation"),
    ]

    # Direction patterns (Phase 10)
    DIRECTION_PATTERNS = [
        (r"\blong[- ]only\b",                                                  "long_only"),
        (r"\bshort[- ]only\b",                                                 "short_only"),
        (r"\bbuy[- ]only\b",                                                   "long_only"),
        (r"\bsell[- ]only\b",                                                  "short_only"),
        (r"\bonly\s+(?:take|go)\s+long",                                       "long_only"),
        (r"\bonly\s+(?:take|go)\s+short",                                      "short_only"),
        (r"\b(?:both\s+)?longs?\s+and\s+shorts?\b",                            "both"),
        (r"\bboth\s+(?:sides|directions)\b",                                   "both"),
        (r"\bdirectional\s+trade(?:s)?\s+(?:on\s+)?(?:both|either)\b",        "both"),
        # Structured prompts: "Long Entry: ... Short Entry: ..." — two explicit legs
        (r"long\s+entry\s*:.*short\s+entry\s*:",                               "both"),
        (r"short\s+entry\s*:.*long\s+entry\s*:",                               "both"),
    ]

    # Gap filter patterns (Phase 10)
    GAP_FILTER_PATTERNS = [
        (r"\bignore\s+(?:large\s+)?gap(?:s)?\s+up\b",                    "ignore_gap_up"),
        (r"\bavoid\s+(?:large\s+)?gap(?:s)?\s+up\b",                     "ignore_gap_up"),
        (r"\bskip\s+(?:large\s+)?gap(?:s)?\s+up\b",                      "ignore_gap_up"),
        (r"\bno\s+(?:large\s+)?gap(?:s)?\s+up\s+entries?\b",             "ignore_gap_up"),
        (r"\bignore\s+(?:large\s+)?gap(?:s)?\s+down\b",                  "ignore_gap_down"),
        (r"\bavoid\s+(?:large\s+)?gap(?:s)?\s+down\b",                   "ignore_gap_down"),
        (r"\bskip\s+(?:large\s+)?gap(?:s)?\s+down\b",                    "ignore_gap_down"),
        (r"\bignore\s+(?:gap(?:s)?|gapping)\s+(?:days?|sessions?)\b",    "ignore_both"),
        (r"\bno\s+trade\s+on\s+gap(?:s)?\b",                             "ignore_both"),
        (r"\bfilter\s+out\s+gap(?:s)?\b",                                 "ignore_both"),
        (r"\bskip\s+gap(?:s)?\b",                                         "ignore_both"),
    ]

    # Consecutive-loss patterns (Phase 10)
    CONSECUTIVE_LOSS_PATTERNS = [
        r"(?:stop|pause|halt)\s+after\s+(\d+)\s+(?:consecutive\s+)?(?:losses?|losing\s+trades?)",
        r"max(?:imum)?\s+(\d+)\s+consecutive\s+(?:losses?|losing\s+trades?)",
        r"(\d+)\s+(?:straight|consecutive)\s+(?:losses?|losing\s+trades?)\s+(?:and\s+)?stop",
        r"no\s+more\s+than\s+(\d+)\s+(?:consecutive\s+)?(?:losses?|losing\s+trades?)",
    ]

    # Cooldown patterns (Phase 10)
    COOLDOWN_PATTERNS = [
        r"(?:wait|cooldown|rest)\s+(?:for\s+)?(\d+)\s+(?:bars?|candles?)\s+after\s+(?:a\s+)?(?:loss|losing\s+trade|stop)",
        r"(\d+)[- ]bar\s+cooldown\s+(?:after|post)[- ](?:loss|stop)",
        r"after\s+(?:a\s+)?(?:loss|stop)\s+wait\s+(\d+)\s+(?:bars?|candles?)",
    ]

    # Spread filter patterns (Phase 10)
    SPREAD_PATTERNS = [
        r"(?:max(?:imum)?\s+)?spread\s+(?:of|below|under|<|≤)\s*(\d+\.?\d*)\s*(?:bps?|basis\s+points?)",
        r"spread\s+(?:filter|gate|limit)\s+(\d+\.?\d*)\s*(?:bps?|basis\s+points?)",
        r"(\d+\.?\d*)\s*(?:bps?|basis\s+points?)\s+(?:max(?:imum)?\s+)?spread",
    ]

    # Entry confirmation patterns (Phase 10)
    CONFIRMATION_PATTERNS = [
        r"(\d+)[- ](?:bar|candle)\s+(?:close|confirmation|confirm)",
        r"(?:wait\s+for\s+|need\s+)?(\d+)\s+(?:consecutive\s+)?(?:bars?|candles?)\s+(?:to\s+)?confirm",
        r"confirmation\s+on\s+(\d+)\s+(?:bars?|candles?)",
        r"(\d+)\s+bars?\s+(?:of\s+)?(?:confirmation|signal)",
    ]

    # RSI band patterns (Phase 10)
    RSI_BAND_PATTERNS = [
        r"rsi\s+between\s+(\d+\.?\d*)\s+(?:and|-)\s+(\d+\.?\d*)",
        r"rsi\s+(?:in\s+)?(?:the\s+)?range\s+(?:of\s+)?(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)",
        r"rsi\s+(?:above|>|≥)\s*(\d+\.?\d*)\s+(?:and|but)\s+(?:below|<|≤)\s*(\d+\.?\d*)",
        r"rsi\s+(?:between|from)\s+(\d+\.?\d*)\s+to\s+(\d+\.?\d*)",
    ]

    # Single-sided RSI thresholds. Captures "RSI > 60", "RSI above 65",
    # "RSI crosses above 30", "RSI below 30", "RSI reaches 65". Each pattern
    # returns (op, value): op ∈ {"above", "below"} so the planner can decide
    # whether it's an entry filter (e.g. above) or an exit trigger (e.g.
    # reaches 65).
    RSI_THRESHOLD_PATTERNS = [
        (r"rsi\s+(?:crosses\s+)?(?:above|over|>|≥|greater\s+than|reaches?(?:\s+above)?)\s+(\d+\.?\d*)", "above"),
        (r"rsi\s+(?:crosses\s+)?(?:below|under|<|≤|less\s+than)\s+(\d+\.?\d*)", "below"),
        (r"rsi\s+is\s+(?:above|over|>|≥)\s+(\d+\.?\d*)", "above"),
        (r"rsi\s+is\s+(?:below|under|<|≤)\s+(\d+\.?\d*)", "below"),
    ]

    # MACD-state patterns. We capture state (rather than periods, which are
    # handled in indicator extraction). State drives signal selection
    # (macd_bullish_cross, macd_positive, macd_histogram_positive, …).
    # Each direction qualifier is REQUIRED — patterns that only see "macd
    # histogram" without a direction would otherwise match both arms.
    MACD_STATE_PATTERNS = [
        (r"macd\s+(?:histogram\s+)?(?:turns?\s+positive|is\s+positive|>\s*0|above\s+(?:zero|0)|above\s+signal|bullish)", "histogram_positive"),
        (r"macd\s+(?:histogram\s+)?(?:turns?\s+negative|is\s+negative|<\s*0|below\s+(?:zero|0)|below\s+signal|bearish)", "histogram_negative"),
        (r"macd\s+(?:bullish|positive)\s+cross", "bullish_cross"),
        (r"macd\s+(?:bearish|negative)\s+cross", "bearish_cross"),
        (r"macd\s+(?:line\s+)?crosses?\s+above\s+signal", "bullish_cross"),
        (r"macd\s+(?:line\s+)?crosses?\s+below\s+signal", "bearish_cross"),
        (r"macd\s+divergence", "divergence"),
    ]

    # Exit-on-opposite phrases. When the user says "exit on opposite
    # crossover" / "exit on opposite signal", the planner should mirror the
    # entry trigger's direction.  Captured as a single boolean flag so the
    # consumer can flip the entry trigger when emitting the exit.
    OPPOSITE_EXIT_PATTERNS = [
        r"\bexit\s+on\s+opposite\s+(?:cross(?:over)?|signal)",
        r"\bopposite\s+cross(?:over)?\s+(?:as\s+)?exit",
        r"\bexit\s+when\s+(?:it\s+|the\s+)?(?:reverses?|flips?)",
        r"\bcross(?:over)?\s+reversal\s+(?:as\s+)?exit",
    ]

    # Price-vs-VWAP relation patterns. We capture the relation as an entry
    # filter rule so the planner can use the appropriate KB signal
    # (price_above_vwap, vwap_reclaim, etc).
    VWAP_RELATION_PATTERNS = [
        (r"(?:price|close|candle)s?\s+(?:closes?\s+|stays?\s+|are\s+|is\s+)?(?:above|over|>)\s+vwap", "price_above_vwap"),
        (r"(?:price|close|candle)s?\s+(?:closes?\s+|stays?\s+|are\s+|is\s+)?(?:below|under|<)\s+vwap", "price_below_vwap"),
        (r"vwap\s+reclaim", "vwap_reclaim"),
        (r"vwap\s+(?:bounce|rejection)", "vwap_bounce"),
    ]

    # Price-vs-SMA / Price-vs-EMA relation patterns. These let the planner
    # pick the dedicated price-cross-SMA/EMA KB signal instead of an
    # unrelated preset trigger. Each pattern allows optional verb forms
    # ("price is above", "price stays above", "price closes above").
    # The optional verb section uses (?:...)? to cover all common phrasings.
    MA_RELATION_PATTERNS = [
        # ── SMA ──────────────────────────────────────────────────────────
        # "price crosses above 20 SMA"
        (r"(?:price|close)s?\s+crosses?\s+above\s+(?:the\s+)?(\d+)\s*sma", ("sma", "above", "cross")),
        (r"(?:price|close)s?\s+crosses?\s+below\s+(?:the\s+)?(\d+)\s*sma", ("sma", "below", "cross")),
        # "price (is|stays|closes|breaks) above N SMA"  →  regime
        (r"(?:price|close)s?\s+(?:is\s+|stays?\s+|closes?\s+|breaks?\s+|are\s+)?(?:above|over|>)\s+(?:the\s+)?(\d+)\s*sma", ("sma", "above", "regime")),
        (r"(?:price|close)s?\s+(?:is\s+|stays?\s+|closes?\s+|breaks?\s+|are\s+)?(?:below|under|<)\s+(?:the\s+)?(\d+)\s*sma", ("sma", "below", "regime")),
        # SMA-leading forms
        (r"sma\s*\(?(\d+)\)?\s+(?:above|>)\s+(?:price|close)", ("sma", "below", "regime")),
        (r"sma\s*\(?(\d+)\)?\s+(?:below|<)\s+(?:price|close)", ("sma", "above", "regime")),

        # ── EMA ──────────────────────────────────────────────────────────
        (r"(?:price|close)s?\s+crosses?\s+above\s+(?:the\s+)?(\d+)\s*ema", ("ema", "above", "cross")),
        (r"(?:price|close)s?\s+crosses?\s+below\s+(?:the\s+)?(\d+)\s*ema", ("ema", "below", "cross")),
        (r"(?:price|close)s?\s+(?:is\s+|stays?\s+|closes?\s+|breaks?\s+|are\s+)?(?:above|over|>)\s+(?:the\s+)?(\d+)\s*ema", ("ema", "above", "regime")),
        (r"(?:price|close)s?\s+(?:is\s+|stays?\s+|closes?\s+|breaks?\s+|are\s+)?(?:below|under|<)\s+(?:the\s+)?(\d+)\s*ema", ("ema", "below", "regime")),
        (r"ema\s*\(?(\d+)\)?\s+(?:above|>)\s+(?:price|close)", ("ema", "below", "regime")),
        (r"ema\s*\(?(\d+)\)?\s+(?:below|<)\s+(?:price|close)", ("ema", "above", "regime")),
    ]

    # Volume ratio patterns (Phase 10)
    VOLUME_RATIO_PATTERNS = [
        r"volume\s+(?:at\s+least\s+|≥|>|above\s+)?(\d+\.?\d*)(?:x|×)\s+(?:the\s+)?(?:average|avg|mean)",
        r"(\d+\.?\d*)(?:x|×)\s+(?:average|avg)\s+volume",
        r"volume\s+(?:spike|surge)\s+(?:of\s+)?(\d+\.?\d*)(?:x|×)",
        r"volume\s+must\s+be\s+(\d+\.?\d*)(?:x|×)\s+(?:or\s+more\s+)?(?:above|than|the)?\s*(?:average|avg)?",
    ]

    # Position sizing mode patterns (Phase 10)
    POSITION_SIZING_PATTERNS = [
        (r"risk[- ]based\s+(?:position\s+)?sizing",         "risk_based"),
        (r"size\s+(?:based\s+on|by)\s+risk",                "risk_based"),
        (r"fixed[- ](?:fractional|fraction)\s+sizing",      "fixed_fractional"),
        (r"(?:fixed|flat)\s+lot\s+size",                    "fixed_units"),
        (r"fixed\s+units?\s+(?:per\s+trade)?",              "fixed_units"),
        (r"risk\s+(\d+\.?\d*)\s*%\s+(?:of\s+capital|per\s+trade)", "risk_based"),
    ]

    # Momentum patterns
    MOMENTUM_PATTERNS = [
        (r"adx\s+(?:should|must|remain)?\s*(?:above|>|>|be)?\s*(\d+\.?\d*)", "adx_threshold"),
        (r"adx.*?(\d+\.?\d*)", "adx_threshold"),  # Fallback: capture any number after ADX
        (r"(?:momentum|adx)\s+(?:strong|bullish)", "adx_strong"),
        (r"mfi\s+(?:rising|increasing)", "mfi_rising"),
        (r"momentum\s+(?:confirmation|bullish)", "momentum_bullish"),
        (r"macd\s+(?:bullish|positive|above\s+signal)", "macd_bullish"),
        # EMA slope — common in breakout and trend strategies
        (r"(?:bullish|rising|positive|upward)\s+ema\s+slope", "ema_slope_bullish"),
        (r"ema\s+(?:slope|direction|trend)\s+(?:bullish|rising|positive|upward|up)", "ema_slope_bullish"),
        (r"(?:bearish|falling|negative|downward)\s+ema\s+slope", "ema_slope_bearish"),
        (r"ema\s+(?:slope|direction|trend)\s+(?:bearish|falling|negative|downward|down)", "ema_slope_bearish"),
        (r"slope\s+of\s+(?:the\s+)?ema\s+(?:is\s+)?(?:bullish|rising|positive|upward)", "ema_slope_bullish"),
        (r"slope\s+of\s+(?:the\s+)?ema\s+(?:is\s+)?(?:bearish|falling|negative|downward)", "ema_slope_bearish"),
        # Price vs EMA — entry-level (no HTF timeframe prefix)
        (r"price\s+(?:is\s+)?above\s+(?:the\s+)?ema", "ema_bullish"),
        (r"price\s+(?:is\s+)?below\s+(?:the\s+)?ema", "ema_bearish"),
        (r"close\s+(?:is\s+|should\s+be\s+)?above\s+(?:the\s+)?ema", "ema_bullish"),
        (r"close\s+(?:is\s+|should\s+be\s+)?below\s+(?:the\s+)?ema", "ema_bearish"),
    ]

    def __init__(self):
        pass

    def extract(self, prompt: str, *, session_id: str | None = None) -> SemanticInstructions:
        """Parse a user's strategy prompt into structured SemanticInstructions.

        When `session_id` is supplied, every captured field is written to the
        extraction audit trail so the chat layer / UI can show provenance for
        each value.
        """
        if not prompt or not prompt.strip():
            return SemanticInstructions(original_prompt=prompt)

        normalized = self._normalize_text(prompt)
        instructions = SemanticInstructions(original_prompt=prompt)

        # Extract each component
        # NOTE: strategy_family detection removed — the normalizer uses
        # KB.detect_primary_framework_in_text(source_prompt) instead, which
        # finds the framework directly from KB preset keywords without any
        # hardcoded family name table in the extractor.
        instructions.strategy_family = None
        instructions.htf_rules = self._extract_htf_rules(normalized)
        instructions.stop_loss = self._extract_stop_loss(normalized)
        instructions.trailing_stop = self._extract_trailing_stop(normalized)
        instructions.risk_reward = self._extract_risk_reward(normalized)
        instructions.reference_symbols = self._extract_reference_symbols(normalized)
        instructions.session_filters = self._extract_session_filters(normalized)
        instructions.volume_momentum = self._extract_volume_momentum(normalized)
        instructions.candle_confirmation = self._extract_candle_confirmation(normalized)

        # Store indicator windows extracted from prompt (do NOT add defaults)
        instructions.indicators = self._extract_indicators(normalized)

        # Phase 10 — new execution parameter extraction
        instructions.direction               = self._extract_direction(normalized)
        instructions.gap_filter              = self._extract_gap_filter(normalized)
        instructions.max_consecutive_losses  = self._extract_consecutive_losses(normalized)
        cooldown                             = self._extract_cooldown(normalized)
        instructions.cooldown_bars_after_loss = cooldown
        instructions.max_spread_bps          = self._extract_spread_bps(normalized)
        instructions.entry_confirmation_bars = self._extract_confirmation_bars(normalized)
        rsi_band                             = self._extract_rsi_band(normalized)
        if rsi_band:
            instructions.rsi_entry_band_min, instructions.rsi_entry_band_max = rsi_band
        instructions.volume_ratio_threshold  = self._extract_volume_ratio(normalized)
        instructions.position_sizing_mode    = self._extract_position_sizing_mode(normalized)

        # Phase 12 — fine-grained semantic captures the planner can act on
        instructions.rsi_thresholds   = self._extract_rsi_thresholds(normalized)
        instructions.macd_states      = self._extract_macd_states(normalized)
        instructions.vwap_relations   = self._extract_vwap_relations(normalized)
        instructions.regime_preference = self._extract_regime_preference(normalized)
        instructions.sl_type_hint     = self._extract_sl_type_hint(normalized)
        instructions.partial_exits    = self._extract_partial_exits(normalized)
        instructions.ma_relations     = self._extract_ma_relations(normalized)
        instructions.exit_on_opposite = self._extract_exit_on_opposite(normalized)
        instructions.orb_opening_bars_minutes = self._extract_orb_duration_minutes(normalized)

        # Calculate extraction quality
        instructions.extraction_quality_score = self._calculate_quality_score(instructions)

        logger.info(
            "semantic_extractor|extracted|family=%s|htf_rules=%d|sl=%s|rr=%s|indicators=%s|quality=%.2f",
            instructions.strategy_family,
            len(instructions.htf_rules),
            instructions.stop_loss is not None,
            instructions.risk_reward is not None,
            str(instructions.indicators),
            instructions.extraction_quality_score,
        )

        if session_id:
            self._write_audit_trail(session_id, prompt, instructions)

        return instructions

    def _write_audit_trail(
        self,
        session_id: str,
        prompt: str,
        instructions: SemanticInstructions,
    ) -> None:
        """Record every captured field to the per-session extraction audit."""
        from app.core.extraction_audit import record_extraction
        from app.planner.unsupported_indicators import detect_unsupported

        confidence = float(instructions.extraction_quality_score or 0.0)
        extractor_id = "semantic_extractor"

        if instructions.stop_loss:
            record_extraction(
                session_id, "stop_loss", instructions.stop_loss.model_dump(),
                source="semantic", confidence=confidence, extractor=extractor_id,
            )
        if instructions.risk_reward:
            record_extraction(
                session_id, "risk_reward", instructions.risk_reward.model_dump(),
                source="semantic", confidence=confidence, extractor=extractor_id,
            )
        if instructions.trailing_stop and instructions.trailing_stop.enabled:
            record_extraction(
                session_id, "trailing_stop", instructions.trailing_stop.model_dump(),
                source="semantic", confidence=confidence, extractor=extractor_id,
            )
        if instructions.direction:
            record_extraction(
                session_id, "direction", instructions.direction,
                source="semantic", confidence=confidence, extractor=extractor_id,
            )
        if instructions.gap_filter:
            record_extraction(
                session_id, "gap_filter", instructions.gap_filter,
                source="semantic", confidence=confidence, extractor=extractor_id,
            )
        if instructions.rsi_entry_band_min is not None or instructions.rsi_entry_band_max is not None:
            record_extraction(
                session_id, "rsi_entry_band",
                {"min": instructions.rsi_entry_band_min, "max": instructions.rsi_entry_band_max},
                source="semantic", confidence=confidence, extractor=extractor_id,
            )
        if instructions.volume_ratio_threshold is not None:
            record_extraction(
                session_id, "volume_ratio_threshold", instructions.volume_ratio_threshold,
                source="semantic", confidence=confidence, extractor=extractor_id,
            )
        for ind_name, periods in (instructions.indicators or {}).items():
            record_extraction(
                session_id, f"indicator.{ind_name}", periods,
                source="semantic", confidence=confidence, extractor=extractor_id,
            )
        for idx, rule in enumerate(instructions.htf_rules or []):
            record_extraction(
                session_id, f"htf_rule[{idx}]",
                rule.model_dump() if hasattr(rule, "model_dump") else str(rule),
                source="semantic", confidence=confidence, extractor=extractor_id,
            )

        unsupported = detect_unsupported(prompt)
        for mention in unsupported:
            record_extraction(
                session_id, f"unsupported.{mention.canonical_name}",
                {
                    "matched_phrase": mention.matched_phrase,
                    "category": mention.category,
                    "suggested_alternative": mention.suggested_alternative,
                },
                source="semantic",
                confidence=1.0,
                extractor="unsupported_indicators",
                note="user mentioned a concept the KB cannot model yet",
            )

    # ── Strategy Family Detection (removed) ────────────────────────────────────
    # Framework detection has been moved to SemanticIntentNormalizer which uses
    # KB.detect_primary_framework_in_text(source_prompt).  The KB preset keyword
    # lists are the single source of truth — no parallel table here.

    # ── Multi-Timeframe Rules ──────────────────────────────────────────────────

    def _extract_htf_rules(self, text: str) -> list[HTFCondition]:
        """Extract higher-timeframe entry gating and confirmation rules."""
        rules = []
        seen = set()

        for pattern, rule_type in self.HTF_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Avoid duplicates
                match_str = match.group(0)
                if match_str in seen:
                    continue
                seen.add(match_str)

                condition = self._parse_htf_match(match, rule_type)
                if condition:
                    rules.append(condition)

        return rules

    def _parse_htf_match(self, match: re.Match, rule_type: str) -> HTFCondition | None:
        """Convert regex match into HTFCondition."""
        try:
            text = match.group(0)

            # Extract timeframe
            timeframe_match = re.search(r"(\d+m|\d+h|daily|weekly|hourly|4h|1h)", text, re.IGNORECASE)
            timeframe = self._normalize_timeframe(timeframe_match.group(1)) if timeframe_match else "1h"

            # Determine role
            role: HTFRoleType = "gating"
            if "confirmation" in text.lower() or "confirm" in text.lower():
                role = "confirmation"
            elif "optional" in text.lower():
                role = "optional_filter"

            # Extract condition
            condition = self._extract_condition_text(text)

            return HTFCondition(
                timeframe=timeframe,
                condition=condition,
                role=role,
                description=text,
            )
        except Exception as e:
            logger.debug(f"Failed to parse HTF match: {e}")
            return None

    def _normalize_timeframe(self, tf: str) -> str:
        """Normalize timeframe strings."""
        tf_lower = tf.lower()
        if tf_lower in ("daily", "1d", "d"):
            return "1d"
        if tf_lower in ("weekly", "1w", "w"):
            return "1w"
        if tf_lower in ("hourly", "1h", "h"):
            return "1h"
        if tf_lower in ("4h",):
            return "4h"
        return tf_lower

    def _extract_condition_text(self, text: str) -> str | None:
        """Extract the condition part from a sentence."""
        # Look for keywords that indicate conditions
        keywords = ["bullish", "bearish", "up", "down", "above", "below", "above sma", "below ema"]
        for kw in keywords:
            if kw.lower() in text.lower():
                # Extract from keyword onwards
                idx = text.lower().index(kw.lower())
                return text[idx:].strip(".,;:")
        return None

    # ── Structural Stop-Loss Extraction ────────────────────────────────────────

    def _extract_stop_loss(self, text: str) -> StructuralStopLoss | None:
        """Extract structural stop-loss specification."""
        for pattern, sl_type in self.SL_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._parse_sl_match(match, sl_type, text)
        return None

    def _parse_sl_match(self, match: re.Match, sl_type: str, full_text: str) -> StructuralStopLoss:
        """Convert SL pattern match into StructuralStopLoss."""
        text = match.group(0)

        # Check for padding (ATR, percent, points)
        padding = self._extract_sl_padding(full_text, match.start())

        # Derive anchor from the matched text first, then fall back to sl_type.
        # The canonical anchor names used downstream (builder, normalizer, YAML):
        #   opening_range_low, opening_range_high, swing_low_recent,
        #   orb_candle, candle_low, reclaim_candle, vwap_deviation, opposite_side
        text_lower = text.lower()
        if "reclaim" in text_lower:
            sl_anchor = "reclaim_candle"
        elif "opening range low" in text_lower or "orb low" in text_lower or sl_type == "orb_low":
            sl_anchor = "opening_range_low"
        elif "opening range high" in text_lower:
            sl_anchor = "opening_range_high"
        elif "swing" in text_lower:
            sl_anchor = "swing_low_recent"
        elif "candle" in text_lower or "wick" in text_lower or "bar" in text_lower:
            sl_anchor = "candle_low"
        elif "vwap" in text_lower or sl_type == "vwap_deviation":
            sl_anchor = "vwap_deviation"
        elif "opposite" in text_lower or sl_type == "opposite_side":
            sl_anchor = "opposite_side"
        elif "orb" in text_lower:
            sl_anchor = "opening_range_low"
        else:
            # Use sl_type as anchor when text doesn't have explicit anchor keywords
            sl_anchor = sl_type if sl_type not in ("atr_padded",) else None

        return StructuralStopLoss(
            type=sl_type,
            anchor=sl_anchor,
            padding=padding,
            description=text,
        )

    def _extract_sl_padding(self, text: str, pos: int) -> StopLossPadding:
        """Extract ATR/percent/point padding from context around SL match."""
        # Look nearby for padding indicators (expand context for better coverage)
        context = text[max(0, pos - 150) : min(len(text), pos + 150)]

        # ATR padding (with or without specific multiple)
        atr_match = re.search(r"(\d+\.?\d*)\s*(?:atr|x\s*atr)", context, re.IGNORECASE)
        if atr_match:
            return StopLossPadding(
                method="atr",
                atr_multiple=float(atr_match.group(1)),
            )

        # ATR padding (just "with ATR padding", no specific multiple)
        if re.search(r"(?:with\s+)?atr\s+padding", context, re.IGNORECASE):
            return StopLossPadding(
                method="atr",
                atr_multiple=1.0,  # Default 1x ATR if not specified
            )

        # Percent padding
        pct_match = re.search(r"(\d+\.?\d*)\s*%", context)
        if pct_match:
            return StopLossPadding(
                method="percent",
                percent=float(pct_match.group(1)),
            )

        # Points padding
        pts_match = re.search(r"(\d+\.?\d*)\s*(?:points?|pips?)", context, re.IGNORECASE)
        if pts_match:
            return StopLossPadding(
                method="points",
                points=float(pts_match.group(1)),
            )

        return StopLossPadding()

    # ── Trailing Stop Extraction ───────────────────────────────────────────────

    def _extract_trailing_stop(self, text: str) -> TrailingStopConfig | None:
        """Extract trailing stop specification."""
        for pattern, ts_type in self.TRAILING_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                config = self._parse_trailing_match(match, ts_type, text)
                if config:
                    return config
        return None

    def _parse_trailing_match(self, match: re.Match, ts_type: str, full_text: str) -> TrailingStopConfig | None:
        """Convert trailing stop pattern into TrailingStopConfig."""
        text = match.group(0)

        config = TrailingStopConfig(
            enabled=True,
            description=text,
        )

        if "ema" in ts_type:
            config.type = "ema_based"
            period_match = re.search(r"ema\s*\(?(\d+)\)?", full_text, re.IGNORECASE)
            if period_match:
                config.ema_period = int(period_match.group(1))
        elif "atr" in ts_type:
            config.type = "atr_based"
        elif "chandelier" in ts_type:
            config.type = "chandelier"
        elif "dynamic" in ts_type:
            config.type = "dynamic"
        else:
            # percent_based, percent_with_activation, activate_after_pct — all are percent trailing
            config.type = "percent_based"

        # Extract activation threshold from group(1) if the pattern captured it,
        # or fall back to searching the matched text for "after X%"
        if "activation" in ts_type or "activate" in ts_type or "percent_with" in ts_type:
            activate_pct: float | None = None
            try:
                activate_pct = float(match.group(1))
            except (IndexError, ValueError, TypeError):
                pct_m = re.search(r"(?:after|once)\s+(?:a\s+)?(\d+\.?\d*)\s*%", text, re.IGNORECASE)
                if pct_m:
                    activate_pct = float(pct_m.group(1))
            config.activate_after_pct = activate_pct

        return config if config.type else None

    # ── Risk:Reward Extraction ────────────────────────────────────────────────

    def _extract_risk_reward(self, text: str) -> RiskRewardSpec | None:
        """Extract risk:reward specification."""
        for pattern, rr_type in self.RR_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._parse_rr_match(match, rr_type, text)
        return None

    def _parse_rr_match(self, match: re.Match, rr_type: str, full_text: str) -> RiskRewardSpec:
        """Convert RR pattern match into RiskRewardSpec.

        Convention: ratio = reward / risk.
          "2:1 RR"       → ratio = 2.0  (reward-first, rr_reward_first)
          "1:2 RR"       → ratio = 2.0  (risk-first, rr_ratio, group1=2)
          "2R target"    → ratio = 2.0  (rr_ratio, group1=2)
          "N:1 risk"     → ratio = 1/N  (rr_inverted — risk-first without reward word)
        """
        text = match.group(0)
        ratio = None

        try:
            if "rr_reward_first" in rr_type:
                # "N:1 Risk Reward" — first number IS the reward multiple
                ratio = float(match.group(1))
            elif "rr_ratio" in rr_type:
                ratio = float(match.group(1))
            elif "rr_inverted" in rr_type:
                # "N:1 risk" (no reward word) — first number is risk, invert
                val = float(match.group(1))
                ratio = 1.0 / val if val > 0 else None
            elif "minimum_rr" in rr_type:
                ratio = float(match.group(1))
        except (ValueError, IndexError):
            ratio = None

        rr_type_enum: RiskRewardType = "minimum" if "minimum" in rr_type else "fixed"
        logger.debug(
            "semantic_extractor|rr_parsed|type=%s|ratio=%s|text=%r",
            rr_type, ratio, text[:40],
        )
        return RiskRewardSpec(
            type=rr_type_enum,
            ratio=ratio,
            description=text,
        )

    # ── Reference Symbol Extraction ────────────────────────────────────────────

    def _extract_reference_symbols(self, text: str) -> list[ReferenceSymbolCondition]:
        """Extract reference symbol and relative strength conditions."""
        conditions = []
        seen = set()

        for pattern, ref_type in self.REF_SYMBOL_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                match_str = match.group(0)
                if match_str in seen:
                    continue
                seen.add(match_str)

                try:
                    # group(1) may be a benchmark name or None for patterns
                    # that don't capture it (e.g. "outperforms the index")
                    raw_sym = None
                    try:
                        raw_sym = match.group(1)
                    except IndexError:
                        pass

                    if raw_sym is None:
                        # Patterns that don't capture a specific symbol — use
                        # the default benchmark from market_aliases
                        raw_sym = "NIFTY50"

                    # Normalise via the shared market_aliases module
                    ref_symbol = normalise_benchmark(raw_sym)

                    condition = self._parse_reference_match(match, ref_type, text)
                    if ref_symbol and condition:
                        condition.reference_symbol = ref_symbol
                        conditions.append(condition)
                except (IndexError, Exception) as exc:
                    logger.debug("Failed to parse reference symbol: %s", exc)

        return conditions

    def _parse_reference_match(self, match: re.Match, ref_type: str, full_text: str) -> ReferenceSymbolCondition | None:
        """Convert reference symbol pattern into ReferenceSymbolCondition."""
        try:
            text = match.group(0)

            relation: RelationshipType = "rs"
            if "index_confirmation" in ref_type:
                relation = "index_confirmation"
            elif "benchmark" in ref_type:
                relation = "benchmark"

            # reference_symbol is required by the schema; pass a placeholder
            # that the caller immediately overwrites with the resolved symbol.
            return ReferenceSymbolCondition(
                reference_symbol="NIFTY50",  # placeholder — overwritten by caller
                relation=relation,
                condition=text,
                description=text,
            )
        except Exception as e:
            logger.debug(f"Failed to parse reference symbol: {e}")
            return None

    # ── Session & Time-Window Extraction ────────────────────────────────────────

    def _extract_session_filters(self, text: str) -> SessionFilter | None:
        """Extract session and time-window filters."""
        session = SessionFilter(enabled=False)
        found_any = False

        for pattern, timing_type in self.SESSION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                found_any = True

                if "start_time" in timing_type:
                    hour = match.group(1)
                    minute = match.group(2) or "00"
                    session.valid_windows.append(
                        TimeWindow(start_time=f"{hour}:{minute}")
                    )
                elif "end_time" in timing_type:
                    hour = match.group(1)
                    minute = match.group(2) or "00"
                    session.valid_windows.append(
                        TimeWindow(end_time=f"{hour}:{minute}")
                    )
                elif "duration_from_open" in timing_type:
                    duration = int(match.group(1))
                    session.valid_windows.append(
                        TimeWindow(duration_minutes=duration, from_open=True)
                    )
                elif "blackout_from_open" in timing_type:
                    duration = int(match.group(1))
                    session.blackout_windows.append(
                        TimeWindow(duration_minutes=duration, from_open=True)
                    )
                elif "session_type" in timing_type:
                    match_text = match.group(0).lower()
                    if "morning" in match_text:
                        session.session = "morning"
                    elif "afternoon" in match_text:
                        session.session = "afternoon"
                    elif "high" in match_text or "liquidity" in match_text:
                        session.session = "high_liquidity"  # Mark as high liquidity hours

        # Enable if anything was found
        if found_any:
            session.enabled = True

        return session if found_any else None

    # ── Volume & Momentum Extraction ───────────────────────────────────────────

    def _extract_volume_momentum(self, text: str) -> VolumeAndMomentumFilters | None:
        """Extract volume and momentum confirmation specifications."""
        volume = None
        momentum = None

        # Volume extraction
        for pattern, vol_type in self.VOLUME_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                volume = VolumeConfirmation(filter_type=vol_type)
                break

        # Momentum extraction
        for pattern, mom_type in self.MOMENTUM_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if mom_type == "adx_threshold" and match.groups():
                    # Extract specific ADX threshold value
                    try:
                        threshold = float(match.group(1))
                        momentum = MomentumConfirmation(
                            filter_type="adx_threshold",
                            adx_threshold=threshold
                        )
                    except (ValueError, IndexError):
                        momentum = MomentumConfirmation(filter_type="adx_strong")
                else:
                    momentum = MomentumConfirmation(filter_type=mom_type)
                break

        if volume or momentum:
            return VolumeAndMomentumFilters(volume=volume, momentum=momentum)

        return None

    # ── Candle Confirmation Extraction ────────────────────────────────────────────

    def _extract_candle_confirmation(self, text: str) -> CandleConfirmation | None:
        """Extract candle pattern confirmation (bullish/bearish)."""
        for pattern, candle_type in self.CANDLE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return CandleConfirmation(
                    filter_type=candle_type,
                    description=match.group(0)
                )
        return None

    # ── Indicator Extraction ──────────────────────────────────────────────────────

    def _extract_indicators(self, text: str) -> dict[str, list[int]]:
        """Extract indicators explicitly mentioned in prompt.

        Only returns indicators the user actually mentioned, never adds defaults.
        This prevents the mystery EMA(29) problem.
        """
        indicators = {}

        # EMA windows — prefer explicit cross pattern "9 EMA > 21 EMA"
        ema_cross = re.search(
            r"(\d+)\s*ema\s*(?:>|above|over|≥|>=)\s*(\d+)\s*ema",
            text,
            re.IGNORECASE,
        )
        if ema_cross:
            fast, slow = int(ema_cross.group(1)), int(ema_cross.group(2))
            if fast > slow:
                fast, slow = slow, fast
            indicators["EMA"] = [fast, slow]
            logger.debug("semantic_extractor|extracted_ema_cross|windows=%s", indicators["EMA"])
        else:
            ema_patterns = [
                r"ema\s*\(?(\d+)\)?",  # EMA(20) or EMA 20
                r"(\d+)\s*ema\b",      # 20 EMA
                r"(?:exponential|exp)\s+moving\s+average\s+(?:of\s+)?(\d+)",
            ]
            ema_matches: list[str] = []
            for pattern in ema_patterns:
                ema_matches.extend(re.findall(pattern, text, re.IGNORECASE))
            if ema_matches:
                ema_windows = sorted(set(int(m) for m in ema_matches if str(m).isdigit()))
                if ema_windows:
                    indicators["EMA"] = ema_windows
                    logger.debug("semantic_extractor|extracted_ema|windows=%s", ema_windows)

        # SMA windows — accept "SMA(20)", "SMA 20", "20 SMA", "simple moving average 20"
        sma_patterns = [
            r"sma\s*\(?(\d+)\)?",                                       # SMA(20), SMA 20, SMA20
            r"(\d+)\s*sma\b",                                           # 20 SMA
            r"(?:simple)\s+moving\s+average\s+(?:of\s+)?(\d+)",         # simple moving average 20
        ]
        sma_matches: list[str] = []
        for pattern in sma_patterns:
            sma_matches.extend(re.findall(pattern, text, re.IGNORECASE))
        if sma_matches:
            sma_windows = sorted({int(m) for m in sma_matches if str(m).isdigit()})
            if sma_windows:
                indicators["SMA"] = sma_windows

        # ATR — explicit period if given, otherwise empty-list so downstream
        # knows the user mentioned ATR (we never invent a period).
        atr_match = (
            re.search(r"atr\s*\(?(\d+)\)?", text, re.IGNORECASE)
            or re.search(r"(\d+)\s*atr\b", text, re.IGNORECASE)
        )
        if atr_match:
            indicators["ATR"] = [int(atr_match.group(1))]
        elif re.search(r"\batr\b", text, re.IGNORECASE):
            indicators["ATR"] = []
            logger.info(
                "semantic_extractor|indicator_period_missing|indicator=ATR"
                "|action=capture_family|reason=user_mentioned_without_period"
            )

        # ADX — same treatment as ATR.
        adx_match = (
            re.search(r"adx\s*\(?(\d+)\)?", text, re.IGNORECASE)
            or re.search(r"(\d+)\s*adx\b", text, re.IGNORECASE)
        )
        if adx_match:
            indicators["ADX"] = [int(adx_match.group(1))]
        elif re.search(r"\badx\b", text, re.IGNORECASE):
            indicators["ADX"] = []
            logger.info(
                "semantic_extractor|indicator_period_missing|indicator=ADX"
                "|action=capture_family|reason=user_mentioned_without_period"
            )

        # RSI — same treatment as ATR/ADX.
        rsi_patterns = [
            r"rsi\s*\(?(\d+)\)?",                                           # RSI(14), RSI 14, RSI14
            r"(\d+)\s*rsi\b",                                               # 14 RSI
            r"relative\s+strength(?:\s+index)?\s+(?:of\s+)?(\d+)",          # relative strength index 14
        ]
        rsi_matches: list[str] = []
        for pattern in rsi_patterns:
            rsi_matches.extend(re.findall(pattern, text, re.IGNORECASE))
        if rsi_matches:
            rsi_windows = sorted({int(m) for m in rsi_matches if str(m).isdigit()})
            if rsi_windows:
                indicators["RSI"] = rsi_windows
        elif re.search(r"\brsi\b|\brelative\s+strength\b", text, re.IGNORECASE):
            indicators["RSI"] = []
            logger.info(
                "semantic_extractor|indicator_period_missing|indicator=RSI"
                "|action=capture_family|reason=user_mentioned_without_period"
            )

        # MACD — same treatment. Capture the family even without periods so
        # the planner can still select a MACD signal. Periods stay empty when
        # the user didn't give all three.
        macd_full = re.search(
            r"macd\s*\(?\s*(\d+)\s*[,/]\s*(\d+)\s*[,/]\s*(\d+)\s*\)?",
            text,
            re.IGNORECASE,
        )
        if macd_full:
            indicators["MACD"] = [
                int(macd_full.group(1)),
                int(macd_full.group(2)),
                int(macd_full.group(3)),
            ]
        elif re.search(r"\bmacd\b", text, re.IGNORECASE):
            indicators["MACD"] = []
            logger.info(
                "semantic_extractor|indicator_period_missing|indicator=MACD"
                "|action=capture_family|reason=user_mentioned_without_period"
            )

        # VWAP — no period (it's session-anchored), so the presence/absence is
        # all we need to capture. Always record when mentioned.
        if re.search(r"\bvwap\b|volume[\s\-]+weighted\s+average\s+price", text, re.IGNORECASE):
            indicators["VWAP"] = []

        # Bollinger Bands — period if given, else empty.
        bb_period = re.search(r"bollinger(?:\s+bands?)?\s*\(?\s*(\d+)", text, re.IGNORECASE)
        if bb_period:
            indicators["BB"] = [int(bb_period.group(1))]
        elif re.search(r"\bbollinger(?:\s+bands?)?\b|\bbb\s+bands?\b", text, re.IGNORECASE):
            indicators["BB"] = []

        # Supertrend — capture family when mentioned (params are downstream).
        if re.search(r"\bsuper\s*trend\b", text, re.IGNORECASE):
            indicators["SUPERTREND"] = []

        # Stochastic — capture family when mentioned.
        if re.search(r"\bstoch(?:astic)?\b", text, re.IGNORECASE):
            indicators["STOCHASTIC"] = []

        # OBV — capture family when mentioned.
        if re.search(r"\bobv\b|on[\s\-]+balance\s+volume", text, re.IGNORECASE):
            indicators["OBV"] = []

        # CCI / MFI — capture family when mentioned.
        if re.search(r"\bcci\b", text, re.IGNORECASE):
            indicators["CCI"] = []
        if re.search(r"\bmfi\b|money[\s\-]+flow", text, re.IGNORECASE):
            indicators["MFI"] = []

        return indicators

    # ── Utilities ──────────────────────────────────────────────────────────────

    def _normalize_text(self, text: str) -> str:
        """Normalize text for pattern matching."""
        # Lowercase, extra whitespace, remove special chars
        return re.sub(r"\s+", " ", text.lower().strip())

    def _calculate_quality_score(self, instructions: SemanticInstructions) -> float:
        """Calculate extraction quality (0.0 to 1.0)."""
        score = 0.0
        components = 0

        # Family detection
        if instructions.strategy_family:
            score += 0.1
        components += 0.1

        # HTF rules
        if instructions.htf_rules:
            score += 0.15
        components += 0.15

        # SL extraction
        if instructions.stop_loss:
            score += 0.15
        components += 0.15

        # TP / RR extraction
        if instructions.risk_reward:
            score += 0.15
        components += 0.15

        # Reference symbols
        if instructions.reference_symbols:
            score += 0.1
        components += 0.1

        # Session filters
        if instructions.session_filters and instructions.session_filters.enabled:
            score += 0.1
        components += 0.1

        # Volume/momentum
        if instructions.volume_momentum:
            score += 0.1
        components += 0.1

        # Trailing stop
        if instructions.trailing_stop and instructions.trailing_stop.enabled:
            score += 0.1
        components += 0.1

        return score / components if components > 0 else 0.0

    # ── Phase 10 extractors ───────────────────────────────────────────────────

    def _extract_direction(self, text: str) -> str | None:
        # 1. Explicit phrasings first (high precision)
        for pattern, direction in self.DIRECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return direction
        # 2. Implicit long signals: "buy when X", "enter long when Y",
        #    "go long on Z", "long entry on …", "enter only when …" paired
        #    with bullish-context indicators. Sell/exit clauses in the same
        #    sentence are usually the EXIT of a long trade.
        has_long = bool(re.search(
            r"\b(?:buy|enter[\s\-]+long|go[\s\-]+long|long[\s\-]+entry"
            r"|take[\s\-]+long|open[\s\-]+long|accumulate)\b",
            text, re.IGNORECASE,
        ))
        has_short = bool(re.search(
            r"\b(?:short(?:[\s\-]+(?:only|entry|the|when|side|trade))?"
            r"|sell[\s\-]+short|enter[\s\-]+short|go[\s\-]+short"
            r"|short[\s\-]+entry|open[\s\-]+short)\b",
            text, re.IGNORECASE,
        ))
        # "enter only when X" + bullish indicator clues → long_only inference
        if not (has_long or has_short):
            enter_only = re.search(r"\benter\s+only\s+when\b", text, re.IGNORECASE)
            bullish_hints = re.search(
                r"\b(?:above|rising|positive|crosses?\s+above|breakout(?:\s+up)?"
                r"|bullish|momentum|uptrend)\b",
                text, re.IGNORECASE,
            )
            if enter_only and bullish_hints:
                return "long_only"
        if has_long and has_short:
            return "both"           # BUG FIX: was returning None, silently dropping short leg
        if has_long:
            return "long_only"
        if has_short:
            return "short_only"
        return None

    def _extract_gap_filter(self, text: str) -> str | None:
        for pattern, filter_type in self.GAP_FILTER_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return filter_type
        return None

    def _extract_consecutive_losses(self, text: str) -> int | None:
        for pattern in self.CONSECUTIVE_LOSS_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    return int(m.group(1))
                except (IndexError, ValueError):
                    pass
        return None

    def _extract_cooldown(self, text: str) -> int | None:
        for pattern in self.COOLDOWN_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    return int(m.group(1))
                except (IndexError, ValueError):
                    pass
        return None

    def _extract_spread_bps(self, text: str) -> float | None:
        for pattern in self.SPREAD_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except (IndexError, ValueError):
                    pass
        return None

    def _extract_confirmation_bars(self, text: str) -> int | None:
        for pattern in self.CONFIRMATION_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    n = int(m.group(1))
                    if 1 <= n <= 10:   # sanity bounds
                        return n
                except (IndexError, ValueError):
                    pass
        return None

    def _extract_rsi_band(self, text: str) -> tuple[float, float] | None:
        for pattern in self.RSI_BAND_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    lo, hi = float(m.group(1)), float(m.group(2))
                    if lo > hi:
                        lo, hi = hi, lo
                    if 0 <= lo < hi <= 100:
                        return lo, hi
                except (IndexError, ValueError):
                    pass
        return None

    def _extract_rsi_thresholds(self, text: str) -> list[dict]:
        """Capture every single-sided RSI threshold the user wrote.

        Returns a list of {op: "above"|"below", value: float} preserving order
        so the planner can map first to entry filter and subsequent to exit
        triggers when appropriate.
        """
        out: list[dict] = []
        seen: set[tuple[str, float]] = set()
        for pattern, op in self.RSI_THRESHOLD_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    val = float(m.group(1))
                    if 0 <= val <= 100 and (op, val) not in seen:
                        out.append({"op": op, "value": val})
                        seen.add((op, val))
                except (IndexError, ValueError):
                    pass
        return out

    def _extract_macd_states(self, text: str) -> list[str]:
        out: list[str] = []
        for pattern, state in self.MACD_STATE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE) and state not in out:
                out.append(state)
        return out

    def _extract_vwap_relations(self, text: str) -> list[str]:
        out: list[str] = []
        for pattern, relation in self.VWAP_RELATION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE) and relation not in out:
                out.append(relation)
        return out

    def _extract_regime_preference(self, text: str) -> str | None:
        """Detect explicit regime asks: 'avoid sideways', 'only trending'.

        Tolerates intervening words (e.g. "avoid trades during sideways
        market conditions" — "trades during" sits between "avoid" and
        "sideways").
        """
        # "avoid ... sideways/chop/range/consolidation" → trending
        if re.search(
            r"\bavoid\b[\s\S]{0,40}\b(?:sideways|chop(?:py)?|range[\s\-]+bound|consolidation)\b",
            text, re.IGNORECASE,
        ):
            return "trending"
        if re.search(
            r"\b(?:only|prefer)\b[\s\S]{0,20}\btrend(?:ing|s)?\b",
            text, re.IGNORECASE,
        ):
            return "trending"
        # Inverse: only ranging / mean-reverting
        if re.search(
            r"\bavoid\b[\s\S]{0,40}\btrend(?:ing|s)?\b"
            r"|\b(?:only|prefer)\b[\s\S]{0,20}\b(?:range[\s\-]+bound|sideways|chop|mean[\s\-]+revert)",
            text, re.IGNORECASE,
        ):
            return "ranging"
        # Volatile preference
        if re.search(
            r"\b(?:only|prefer)\b[\s\S]{0,20}\bvolatile\b|\bhigh[\s\-]+volatility\b",
            text, re.IGNORECASE,
        ):
            return "volatile"
        return None

    def _extract_sl_type_hint(self, text: str) -> str | None:
        """Decide which SL family the user is asking for, in priority order."""
        if re.search(r"\batr[\s\-]+based\s+stop|atr\s+stop\s+loss?|stop\s+loss\s+based\s+on\s+atr",
                     text, re.IGNORECASE):
            return "atr"
        if re.search(r"\bstop\s+loss\s+(?:below|above|at)\s+(?:previous\s+)?candle\s+(?:low|high)"
                     r"|\bstop\s+(?:loss\s+)?below\s+(?:the\s+)?(?:breakout|signal|setup)\s+candle"
                     r"|\bswing\s+(?:low|high)\s+(?:as\s+)?(?:stop|sl)",
                     text, re.IGNORECASE):
            return "structural"
        if re.search(r"\bstop\s+(?:loss\s+)?(?:of\s+|at\s+)?\d+(?:\.\d+)?\s*%",
                     text, re.IGNORECASE):
            return "percent"
        return None

    def _extract_exit_on_opposite(self, text: str) -> bool:
        return any(
            re.search(pat, text, re.IGNORECASE)
            for pat in self.OPPOSITE_EXIT_PATTERNS
        )

    def _extract_orb_duration_minutes(self, text: str) -> int | None:
        """Extract the user's explicitly stated opening-range duration in minutes.

        "first 1-hour range" → 60
        "first 30-minute range" → 30
        "first hour range" → 60 (implied 1 hour)
        Returns None when the user did not specify a range duration.
        """
        # "first N-hour range" / "opening N-hour range"
        m = re.search(
            r"(?:first|initial|opening)\s+(\d+(?:\.\d+)?)\s*[\s\-]*hour\s+(?:range|candle|session|period)",
            text, re.IGNORECASE,
        )
        if m:
            try:
                return int(float(m.group(1)) * 60)
            except (ValueError, IndexError):
                pass

        # "first N-minute range"
        m = re.search(
            r"(?:first|initial|opening)\s+(\d+)\s*[\s\-]*(?:minute|min)\s+(?:range|candle|session)",
            text, re.IGNORECASE,
        )
        if m:
            try:
                return int(m.group(1))
            except (ValueError, IndexError):
                pass

        # "first hour range" (plain "hour" = 1h = 60 min)
        if re.search(
            r"(?:first|initial)\s+hour\s+(?:range|candle|session|period)",
            text, re.IGNORECASE,
        ):
            return 60

        return None

    def _extract_ma_relations(self, text: str) -> list[dict]:
        """Capture user phrases like 'price crosses above 20 SMA' so the
        planner can pick price_cross_above_sma instead of a generic preset
        trigger that swaps SMA for EMA. Returns ordered, deduplicated dicts.
        """
        out: list[dict] = []
        seen: set[tuple] = set()
        for pattern, (family, side, kind) in self.MA_RELATION_PATTERNS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    window = int(m.group(1))
                except (IndexError, ValueError):
                    continue
                key = (family, side, kind, window)
                if key in seen:
                    continue
                seen.add(key)
                out.append({
                    "family": family,
                    "side": side,
                    "window": window,
                    "kind": kind,
                })
        return out

    def _extract_partial_exits(self, text: str) -> list[dict]:
        """Capture partial profit booking instructions: '1.5R', '50% at 2R'."""
        out: list[dict] = []
        # "partial profit booking at 1.5R", "book 50% at 2R", "exit half at 1R"
        for m in re.finditer(
            r"(?:partial\s+(?:profit\s+)?booking|book(?:ing)?\s+(?:profit\s+)?|exit\s+(?:half|partial))"
            r"\s+(?:of\s+)?(?:(\d+)\s*%\s+)?(?:at|after|on)\s+(\d+(?:\.\d+)?)\s*r\b",
            text, re.IGNORECASE,
        ):
            entry: dict = {"trigger": "rr_multiple", "value": float(m.group(2))}
            if m.group(1):
                entry["size_pct"] = int(m.group(1))
            out.append(entry)
        # Simple "at 1.5R" without explicit "partial"
        if not out:
            for m in re.finditer(r"\bat\s+(\d+(?:\.\d+)?)\s*r\b", text, re.IGNORECASE):
                out.append({"trigger": "rr_multiple", "value": float(m.group(1))})
        return out

    def _extract_volume_ratio(self, text: str) -> float | None:
        for pattern in self.VOLUME_RATIO_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    ratio = float(m.group(1))
                    if 0.1 <= ratio <= 20.0:   # sanity bounds
                        return ratio
                except (IndexError, ValueError):
                    pass
        return None

    def _extract_position_sizing_mode(self, text: str) -> str | None:
        for pattern, mode in self.POSITION_SIZING_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return mode
        return None
