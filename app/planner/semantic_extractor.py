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

    # RR patterns
    RR_PATTERNS = [
        (r"(?:risk[:\s]+)?reward\s*[:\s=]+\s*1\s*:\s*(\d+\.?\d*)", "rr_ratio"),
        (r"(\d+\.?\d*)\s*:\s*1\s+(?:risk|rr)", "rr_inverted"),
        (r"minimum\s+(?:1\s*:\s*)?(\d+\.?\d*)\s+(?:rr|risk[:\s]+reward)", "minimum_rr"),
        (r"rr\s+of\s+1:(\d+\.?\d*)", "rr_ratio"),
        # Suffix patterns: "1:2 RR", "1:3 r:r", "1:2 risk reward"
        (r"\b1\s*[:/]\s*(\d+\.?\d*)\s+(?:rr|r:r|risk[:\s]*reward|reward)", "rr_ratio"),
        # "R:R 1:2", "risk reward 1:2"
        (r"(?:r:r|rr|risk[\s:]+reward)\s+1\s*[:/]\s*(\d+\.?\d*)", "rr_ratio"),
        # "minimum 1:2", "at least 1:2", "min RR 1:3"
        (r"(?:minimum|min|at\s+least)\s+(?:rr\s+)?1\s*[:/]\s*(\d+\.?\d*)", "minimum_rr"),
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
        (r"(?:avoid|skip)\s+(?:market\s+)?open.*?(\d+)\s+(?:minutes?|min)", "blackout_from_open"),
        (r"(?:during\s+)?(?:high\s+)?liquidity\s+(?:market\s+)?hours?", "session_type"),  # "high liquidity market hours"
        (r"(?:morning|afternoon)\s+session", "session_type"),
        (r"(?:market\s+)?timing.*?(?:high\s+)?liquidity", "session_type"),
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
        (r"\blong[- ]only\b",                                             "long_only"),
        (r"\bshort[- ]only\b",                                            "short_only"),
        (r"\bbuy[- ]only\b",                                              "long_only"),
        (r"\bsell[- ]only\b",                                             "short_only"),
        (r"\bonly\s+(?:take|go)\s+long",                                  "long_only"),
        (r"\bonly\s+(?:take|go)\s+short",                                 "short_only"),
        (r"\b(?:both\s+)?longs?\s+and\s+shorts?\b",                       "both"),
        (r"\bboth\s+(?:sides|directions)\b",                              "both"),
        (r"\bdirectional\s+trade(?:s)?\s+(?:on\s+)?(?:both|either)\b",   "both"),
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

    def extract(self, prompt: str) -> SemanticInstructions:
        """Parse a user's strategy prompt into structured SemanticInstructions."""
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

        return instructions

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
        """Convert RR pattern match into RiskRewardSpec."""
        text = match.group(0)
        ratio = None

        try:
            if "rr_ratio" in rr_type:
                ratio = float(match.group(1))
            elif "rr_inverted" in rr_type:
                ratio = 1.0 / float(match.group(1))
            elif "minimum_rr" in rr_type:
                ratio = float(match.group(1))
        except (ValueError, IndexError):
            ratio = None

        rr_type_enum: RiskRewardType = "minimum" if "minimum" in rr_type else "fixed"

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

        # SMA windows
        sma_matches = re.findall(r"sma\s*\(?(\d+)\)?", text, re.IGNORECASE)
        if sma_matches:
            sma_windows = sorted(set(int(m) for m in sma_matches))
            indicators["SMA"] = sma_windows

        # ATR (usually period 14 by default, but record if mentioned)
        if re.search(r"\batr\b", text, re.IGNORECASE):
            atr_match = re.search(r"atr\s*\(?(\d+)\)?", text, re.IGNORECASE)
            if atr_match:
                indicators["ATR"] = [int(atr_match.group(1))]
            else:
                # Only include if explicitly mentioned
                indicators["ATR"] = [14]  # Default ATR period

        # ADX (usually period 14 by default)
        if re.search(r"\badx\b", text, re.IGNORECASE):
            adx_match = re.search(r"adx\s*\(?(\d+)\)?", text, re.IGNORECASE)
            if adx_match:
                indicators["ADX"] = [int(adx_match.group(1))]
            else:
                indicators["ADX"] = [14]  # Default ADX period

        # RSI
        rsi_matches = re.findall(r"rsi\s*\(?(\d+)\)?", text, re.IGNORECASE)
        if rsi_matches:
            rsi_windows = sorted(set(int(m) for m in rsi_matches))
            indicators["RSI"] = rsi_windows

        # MACD
        if re.search(r"\bmacd\b", text, re.IGNORECASE):
            indicators["MACD"] = [12, 26, 9]  # Standard MACD periods

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
        for pattern, direction in self.DIRECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return direction
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
