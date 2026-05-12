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
    STRATEGY_FAMILIES = {
        "orb": "ORB",
        "opening range breakout": "ORB",
        "vwap": "VWAP_RECLAIM",
        "vwap reclaim": "VWAP_RECLAIM",
        "ema pullback": "EMA_PULLBACK",
        "ema retracement": "EMA_PULLBACK",
        "momentum": "MOMENTUM",
        "reversal": "REVERSAL",
        "ict": "ICT_BOS_FVG",
        "break of structure": "ICT_BOS_FVG",
        "bos fvg": "ICT_BOS_FVG",
        "mean reversion": "MEAN_REVERSION",
        "breakout": "BREAKOUT",
        "scalping": "SCALPING",
    }

    # HTF patterns: phrases that indicate higher-timeframe conditions
    HTF_PATTERNS = [
        (r"(?:trend\s+should\s+remain|higher\s+timeframe\s+trend|timeframe.*?trend)\s+(?:bullish|bearish|up|down)", "generic_htf_trend"),
        (r"(?:higher\s+highs|higher\s+lows|bullish\s+structure|bearish\s+structure).*?(?:should|continue|forming)", "structural_trend"),  # "Higher highs should continue forming"
        (r"(\d+[mh]|daily|weekly|hourly)\s+(?:trend|ema|rsi|macd|momentum).*?(?:bullish|bearish|up|down|above|below)", "trend_condition"),
        (r"(?:higher\s+timeframe|htf|upper\s+tf|h\.?t\.?f\.?).*?(?:bullish|bearish|up|down|should)", "htf_direction"),
        (r"(\d+h|daily)\s+(?:trend|close|ema)\s+(?:should\s+be|must\s+be|should\s+remain)\s+(bullish|bearish|up|down)", "htf_requirement"),
        (r"(?:trade\s+only\s+with|trade\s+when|only\s+trade\s+if|when).*?(\d+[mh]|daily)\s+(.+?)(?:\.|,|$)", "conditional_htf"),
    ]

    # SL anchor patterns (more specific patterns first)
    SL_PATTERNS = [
        (r"below\s+(?:\w+\s+)*reclaim\s+candle\s+low", "candle_low"),  # "below VWAP reclaim candle low"
        (r"below\s+(?:the\s+)?(?:vwap\s+)?reclaim\s+candle", "candle_low"),  # alternate format
        (r"below\s+(?:recent\s+)?swing\s+low", "swing_low"),  # "below recent swing low" or "below swing low"
        (r"below\s+(?:the\s+)?recent\s+(?:swing\s+)?low", "swing_low"),  # alternate
        (r"below\s+(?:the\s+)?orb\s+low", "orb_low"),
        (r"opposite\s+side\s+(?:of\s+)?(?:the\s+)?candle", "opposite_side"),
        (r"below\s+(?:the\s+)?wick\s+low", "candle_low"),
        (r"below\s+vwap", "vwap_deviation"),
        (r"(?:stop\s+loss|sl)\s+(?:at\s+)?(?:below\s+)?(?:recent\s+)?swing\s+low", "swing_low"),  # explicit SL reference
        (r"(\d+\.?\d*)\s*(?:atr|x\s*atr)\s+(?:below|padding|padded)", "atr_padded"),  # ATR padding
        (r"(?:with\s+)?atr\s+padding", "atr_padded"),  # ATR padding standalone
    ]

    # Trailing stop patterns
    TRAILING_PATTERNS = [
        (r"ema\s+trailing\s+(?:stop|exit)", "ema_based"),
        (r"atr\s+trailing\s+(?:stop|exit)", "atr_based"),
        (r"chandelier\s+(?:exit|stop)", "chandelier"),
        (r"trail(?:ing)?\s+(?:stop|exit).*?(?:after|once)\s+(\d+\.?\d*)\s*%", "activate_after_pct"),
        (r"dynamic\s+(?:trail|trailing)", "dynamic"),
    ]

    # RR patterns
    RR_PATTERNS = [
        (r"(?:risk[:\s]+)?reward\s*[:\s=]+\s*1\s*:\s*(\d+\.?\d*)", "rr_ratio"),
        (r"(\d+\.?\d*)\s*:\s*1\s+(?:risk|rr)", "rr_inverted"),
        (r"minimum\s+(?:1\s*:\s*)?(\d+\.?\d*)\s+(?:rr|risk[:\s]+reward)", "minimum_rr"),
        (r"rr\s+of\s+1:(\d+\.?\d*)", "rr_ratio"),
    ]

    # Reference symbol patterns
    REF_SYMBOL_PATTERNS = [
        (r"(?:outperforming|stronger than|vs\.?|relative to|compared to)\s+([A-Z]+(?:NIFTY|BANK|INDEX)?)", "relative_strength"),
        (r"([A-Z]+(?:NIFTY|BANK)?)\s+(?:should be|must be|direction)\s+(bullish|bearish)", "index_confirmation"),
        (r"(?:in line with|with)\s+([A-Z]+NIFTY)", "benchmark"),
        (r"relative\s+strength\s+(?:to|vs\.?|compared to)\s+([A-Z0-9]+)", "rs_explicit"),
    ]

    # Session/timing patterns
    SESSION_PATTERNS = [
        (r"(?:after|trade after|only after|from)\s+(\d{1,2}):?(\d{2})?\s*(?:am|a\.m\.|AM)", "start_time"),
        (r"(?:before|trade before|until)\s+(\d{1,2}):?(\d{2})?\s*(?:pm|p\.m\.|PM)", "end_time"),
        (r"(?:first|initial)\s+(\d+)\s+(?:minutes?|min)", "duration_from_open"),
        (r"(?:avoid|skip)\s+(?:market\s+)?open.*?(\d+)\s+(?:minutes?|min)", "blackout_from_open"),
        (r"(?:avoid|skip)\s+(?:the\s+)?first\s+(\d+)\s+(?:minutes?|min)\s+(?:from\s+|of\s+|after\s+)?(?:open)?", "blackout_from_open"),
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

    # Momentum patterns
    MOMENTUM_PATTERNS = [
        (r"adx\s+(?:should|must|remain)?\s*(?:above|>|>|be)?\s*(\d+\.?\d*)", "adx_threshold"),
        (r"adx.*?(\d+\.?\d*)", "adx_threshold"),  # Fallback: capture any number after ADX
        (r"(?:momentum|adx)\s+(?:strong|bullish)", "adx_strong"),
        (r"mfi\s+(?:rising|increasing)", "mfi_rising"),
        (r"momentum\s+(?:confirmation|bullish)", "momentum_bullish"),
        (r"macd\s+(?:bullish|positive|above\s+signal)", "macd_bullish"),
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
        instructions.strategy_family = self._detect_strategy_family(normalized)
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

    # ── Strategy Family Detection ──────────────────────────────────────────────

    def _detect_strategy_family(self, text: str) -> str | None:
        """Detect strategy family from keywords."""
        text_lower = text.lower()
        best_match = None
        best_len = 0

        for keyword, family in self.STRATEGY_FAMILIES.items():
            if keyword in text_lower:
                # Prefer longer matches to avoid "momentum" matching "breakout momentum"
                if len(keyword) > best_len:
                    best_match = family
                    best_len = len(keyword)

        return best_match

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

        sl_anchor = None
        if "reclaim" in text.lower():
            sl_anchor = "reclaim_candle"
        elif "orb" in text.lower():
            sl_anchor = "orb_candle"
        elif "swing" in text.lower():
            sl_anchor = "swing_low_recent"

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

        if "ema" in ts_type.lower():
            config.type = "ema_based"
            # Look for EMA period
            period_match = re.search(r"ema\s*\(?(\d+)\)?", full_text, re.IGNORECASE)
            if period_match:
                config.ema_period = int(period_match.group(1))
        elif "atr" in ts_type.lower():
            config.type = "atr_based"
        elif "chandelier" in ts_type.lower():
            config.type = "chandelier"
        elif "dynamic" in ts_type.lower():
            config.type = "dynamic"

        # Look for activation percentage
        if "activate_after_pct" in ts_type:
            activate_match = re.search(r"(\d+\.?\d*)\s*%", text)
            if activate_match:
                config.activate_after_pct = float(activate_match.group(1))

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
                    ref_symbol = match.group(1).upper()
                    condition = self._parse_reference_match(match, ref_type, text)
                    if ref_symbol and condition:
                        condition.reference_symbol = ref_symbol
                        conditions.append(condition)
                except IndexError:
                    pass

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

            return ReferenceSymbolCondition(
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

        Phase 2: walks the full indicator catalog. For each catalog entry,
        scans the prompt for the canonical name and every alias. Periodic
        indicators look for an explicit window (RSI(14), 20 EMA, EMA 20) and
        fall back to the catalog default if a name appears without a window.
        Multi-param indicators (Supertrend, Keltner, …) are recorded with
        their default param tuple unless the user typed an explicit
        argument list.
        """
        from app.kb.indicator_catalog import CATALOG, IndicatorSpec

        indicators: dict[str, list] = {}
        lowered = text.lower()
        for spec in CATALOG.values():
            tokens = {spec.name.lower(), *spec.aliases}
            mentioned = False
            for tok in tokens:
                if tok and re.search(r"\b" + re.escape(tok) + r"\b", lowered):
                    mentioned = True
                    break
            if not mentioned:
                continue
            indicators[spec.name] = self._indicator_periods_for_spec(spec, text, lowered)

        # Multi-pattern EMA / SMA window detection — the catalog walk above
        # records the canonical token; this loop captures the natural
        # variants ("20 EMA", "Exponential moving average of 50") that
        # users tend to type. Merges into the catalog-extracted windows so
        # we don't lose anything.
        ema_extra: list[int] = []
        for pattern in (r"(\d+)\s*ema\b", r"(?:exponential|exp)\s+moving\s+average\s+(?:of\s+)?(\d+)"):
            ema_extra.extend(int(m) for m in re.findall(pattern, lowered) if m.isdigit())
        if ema_extra:
            current = set(indicators.get("EMA", []))
            current.update(ema_extra)
            indicators["EMA"] = sorted(current)

        sma_extra = [int(m) for m in re.findall(r"(\d+)\s*sma\b", lowered) if m.isdigit()]
        if sma_extra:
            current = set(indicators.get("SMA", []))
            current.update(sma_extra)
            indicators["SMA"] = sorted(current)

        return indicators

    def _indicator_periods_for_spec(self, spec: Any, text: str, lowered: str) -> list:
        """Best-effort period extraction for a single catalog spec.

        • Single-param indicators with `int` first param: returns a list of
          int windows found in the prompt (e.g. RSI(14), 20 RSI) — falling
          back to the catalog default if no number is paired with the name.
        • Multi-param indicators (Supertrend, Keltner, …): returns a list
          containing one tuple with the catalog default values. Detailed
          two-arg parsing is left to threshold_conditions / explicit user
          configuration.
        • Zero-param indicators: returns an empty list.
        """
        if not spec.params:
            return []
        first_param = spec.params[0]
        names = [spec.name.lower(), *spec.aliases]
        # Try every alias form: NAME(N), NAME N, N NAME.
        windows: set[int] = set()
        for alias in names:
            esc = re.escape(alias)
            for pattern in (
                rf"\b{esc}\s*\(\s*(\d+)\s*\)",   # NAME(20)
                rf"\b{esc}\s*(\d+)\b",           # NAME 20
                rf"\b(\d+)\s+{esc}\b",           # 20 NAME
            ):
                for m in re.finditer(pattern, lowered, re.IGNORECASE):
                    try:
                        windows.add(int(m.group(1)))
                    except (TypeError, ValueError):
                        continue
        if len(spec.params) == 1:
            if windows:
                return sorted(windows)
            return [first_param.default] if first_param.default is not None else []
        # Multi-param: anchor to the catalog defaults; user can override via
        # explicit threshold expressions captured elsewhere in the snapshot.
        default_tuple = tuple(p.default for p in spec.params if p.default is not None)
        return [default_tuple] if default_tuple else []

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
