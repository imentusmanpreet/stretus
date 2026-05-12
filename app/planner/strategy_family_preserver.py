"""
app/planner/strategy_family_preserver.py — Preserve strategy family intent and semantic meaning.

Prevents signal collapse where all strategies degrade to generic EMA-crossover logic.
Maps detected strategy families to:
  - Canonical signal specs (which signals define the family)
  - Structural requirements (e.g., "ORB must anchor to opening range low")
  - Contraindicated signals (e.g., don't mix momentum scalping signals into structural ORB)
  - Intent preservation invariants

Usage:
  1. SemanticExtractor detects family from prompt
  2. StrategyFamilyPreserver validates and enriches with canonical specs
  3. SignalSelector ranks signals considering family affinity
  4. ExecutionConfigurator applies family-specific requirements
"""
from __future__ import annotations

import logging
from typing import Optional

from app.kb.execution_schemas import (
    CanonicalSignalSpec,
    StopLossType,
    StrategyFamilySpec,
    StrategyFamilyType,
)
from app.kb.schemas import SignalRole

logger = logging.getLogger(__name__)


class StrategyFamilyPreserver:
    """Preserve strategy family intent and prevent signal collapse."""

    # Define canonical signals for each strategy family
    FAMILY_SPECS: dict[StrategyFamilyType, StrategyFamilySpec] = {
        "ORB": StrategyFamilySpec(
            family="ORB",
            canonical_signals=[
                CanonicalSignalSpec(
                    signal_name="breakout_above_orb_high",
                    role="entry_trigger",
                    mandatory=True,
                    description="Entry when price breaks above opening range high",
                ),
                CanonicalSignalSpec(
                    signal_name="volume_spike",
                    role="entry_filter",
                    mandatory=False,
                    description="Volume confirmation on breakout",
                ),
            ],
            sl_anchor_type="orb_low",
            preserved_invariants=[
                "SL must be below opening range low",
                "Must measure opening range (first N minutes)",
                "Prevents mean-reversion logic (opposite of ORB philosophy)",
                "No counter-trend entry on ORB failure",
            ],
            contraindicated_signals=[
                "vwap_reclaim_bullish",  # VWAP is mean-reversion, ORB is breakout
                "rsi_oversold",  # Oversold is contrarian to breakout momentum
                "ema_pullback",  # Pullback waiting contradicts breakout urgency
            ],
            description="Opening Range Breakout — trade breaks of first N-minute range with structural SL",
        ),
        "VWAP_RECLAIM": StrategyFamilySpec(
            family="VWAP_RECLAIM",
            canonical_signals=[
                CanonicalSignalSpec(
                    signal_name="vwap_reclaim_bullish",
                    role="entry_trigger",
                    mandatory=True,
                    description="Entry when price reclaims VWAP after dip below",
                ),
                CanonicalSignalSpec(
                    signal_name="bullish_rejection_candle",
                    role="entry_filter",
                    mandatory=False,
                    description="Bullish candle confirming reclaim",
                ),
            ],
            sl_anchor_type="candle_low",
            preserved_invariants=[
                "SL anchors to reclaim candle low (NOT VWAP level)",
                "Entry must have price close > VWAP",
                "Prior candle close <= VWAP (dip requirement)",
                "Bullish wick rejection shows strength",
                "Emphasizes quick reclaim (institutional defense of VWAP)",
            ],
            contraindicated_signals=[
                "breakout_above_orb_high",  # ORB is structural, VWAP is level-based
                "momentum_extreme",  # VWAP reclaim is mean-reversion, not momentum chasing
            ],
            description="VWAP Reclaim — institutional traders defend VWAP; entry on quick reclaim after dip",
        ),
        "EMA_PULLBACK": StrategyFamilySpec(
            family="EMA_PULLBACK",
            canonical_signals=[
                CanonicalSignalSpec(
                    signal_name="ema_pullback_support",
                    role="entry_trigger",
                    mandatory=True,
                    description="Price pulls back to EMA as support, then bounces",
                ),
                CanonicalSignalSpec(
                    signal_name="ema_stack_confirmation",
                    role="entry_filter",
                    mandatory=False,
                    description="Faster EMA above slower EMA confirms trend",
                ),
            ],
            sl_anchor_type="candle_low",
            preserved_invariants=[
                "EMA must act as support (price cannot close below it)",
                "Trend must be intact (EMA stack or EMA above price N candles ago)",
                "Entry on first bounce candle closing above EMA",
                "SL below pullback candle low",
            ],
            contraindicated_signals=[
                "vwap_reclaim_bullish",  # VWAP reclaim works at any trend, EMA pullback requires active uptrend
                "rsi_oversold",  # Pullback is NOT about oversold; about support test
            ],
            description="EMA Pullback — trade pullbacks to EMA as support in established trend",
        ),
        "MOMENTUM": StrategyFamilySpec(
            family="MOMENTUM",
            canonical_signals=[
                CanonicalSignalSpec(
                    signal_name="rsi_extreme",
                    role="entry_trigger",
                    mandatory=False,
                    description="RSI > 70 (overbought) for bullish momentum",
                ),
                CanonicalSignalSpec(
                    signal_name="adx_strong",
                    role="entry_filter",
                    mandatory=False,
                    description="ADX > threshold confirms strong directional momentum",
                ),
            ],
            sl_anchor_type="percent",
            preserved_invariants=[
                "Momentum is trend-following, NOT contrarian",
                "Entry on momentum extremes (overbought/oversold)",
                "Ride the trend, don't fade it",
                "Higher timeframe trend confirmation critical",
            ],
            contraindicated_signals=[
                "rsi_oversold",  # Momentum fades oversold; doesn't enter there
                "mean_reversion_zscore",  # Reversion contradicts momentum",
            ],
            description="Momentum — ride extreme RSI/MACD signals with trend confirmation",
        ),
        "REVERSAL": StrategyFamilySpec(
            family="REVERSAL",
            canonical_signals=[
                CanonicalSignalSpec(
                    signal_name="bullish_rejection_candle",
                    role="entry_trigger",
                    mandatory=True,
                    description="Rejection candle showing reversal intent",
                ),
                CanonicalSignalSpec(
                    signal_name="rsi_oversold",
                    role="entry_filter",
                    mandatory=False,
                    description="Extreme RSI confirms reversal",
                ),
            ],
            sl_anchor_type="candle_low",
            preserved_invariants=[
                "Entry requires rejection/reversal candle (wick-based)",
                "Oversold RSI, momentum divergence, or volume climax context",
                "SL below the rejection candle",
                "Target is swing high (structured TP anchor)",
            ],
            contraindicated_signals=[
                "momentum_extreme",  # Reversal bets against momentum
                "adx_strong",  # Strong ADX means trend is WEAK (no reversal likely)
            ],
            description="Reversal — trade exhaustion patterns with rejection candles and oversold indicators",
        ),
        "ICT_BOS_FVG": StrategyFamilySpec(
            family="ICT_BOS_FVG",
            canonical_signals=[
                CanonicalSignalSpec(
                    signal_name="break_of_structure",
                    role="entry_trigger",
                    mandatory=True,
                    description="Break of the prior structure (swing high/low)",
                ),
                CanonicalSignalSpec(
                    signal_name="fair_value_gap_fill",
                    role="entry_filter",
                    mandatory=False,
                    description="Entry if FVG is not yet filled",
                ),
            ],
            sl_anchor_type="swing_low",
            preserved_invariants=[
                "BOS (Break of Structure) defines entry: break of recent swing",
                "FVG (Fair Value Gap) retention: don't enter if FVG already filled",
                "Sweep confirmation: BOS often followed by sweep before reversal",
                "No reversal entry; BOS continuation only",
            ],
            contraindicated_signals=[
                "vwap_reclaim_bullish",  # ICT is structural, VWAP is level-based
                "rsi_oversold",  # BOS doesn't care about oversold; structural only
            ],
            description="ICT Break of Structure / FVG — trade structural breaks with FVG logic",
        ),
        "MEAN_REVERSION": StrategyFamilySpec(
            family="MEAN_REVERSION",
            canonical_signals=[
                CanonicalSignalSpec(
                    signal_name="vwap_reclaim_bullish",
                    role="entry_trigger",
                    mandatory=False,
                    description="Reclaim from VWAP deviation",
                ),
                CanonicalSignalSpec(
                    signal_name="zscore_extreme",
                    role="entry_trigger",
                    mandatory=False,
                    description="Price > 2σ (2 standard deviations) from mean",
                ),
            ],
            sl_anchor_type="percent",
            preserved_invariants=[
                "Entry on extreme deviation from mean (Z-score > 2.0)",
                "Reversion to mean is the thesis (not trend following)",
                "Targets the mean or swing high",
                "Works in ranges; fails in strong trends",
                "Risk: strong trend can extend deviation further",
            ],
            contraindicated_signals=[
                "momentum_extreme",  # Momentum confirms the move, contradicts reversion thesis
                "breakout_above_orb_high",  # ORB is breakout; reversion fades breakouts
            ],
            description="Mean Reversion — trade Z-score extremes back toward mean",
        ),
        "BREAKOUT": StrategyFamilySpec(
            family="BREAKOUT",
            canonical_signals=[
                CanonicalSignalSpec(
                    signal_name="breakout_above_resistance",
                    role="entry_trigger",
                    mandatory=True,
                    description="Break above resistance level",
                ),
                CanonicalSignalSpec(
                    signal_name="volume_spike",
                    role="entry_filter",
                    mandatory=False,
                    description="Volume confirms breakout",
                ),
            ],
            sl_anchor_type="percent",
            preserved_invariants=[
                "Entry only on actual break, not approaching level",
                "Volume must confirm (heavy participation)",
                "SL below breakout candle or support level",
                "Targets are profit-taking levels (structure-based or TP%)",
            ],
            contraindicated_signals=[
                "rsi_oversold",  # Breakout is momentum, not contrarian
                "mean_reversion_zscore",  # Reversion fades breakouts
            ],
            description="Breakout — trade breaks of resistance with volume confirmation",
        ),
        "SCALPING": StrategyFamilySpec(
            family="SCALPING",
            canonical_signals=[
                CanonicalSignalSpec(
                    signal_name="ema_cross_up",
                    role="entry_trigger",
                    mandatory=False,
                    description="Quick EMA crossover signals",
                ),
                CanonicalSignalSpec(
                    signal_name="rsi_extreme",
                    role="entry_filter",
                    mandatory=False,
                    description="Extreme RSI for quick turns",
                ),
            ],
            sl_anchor_type="percent",
            preserved_invariants=[
                "Ultra-short hold horizon (seconds to 5 min)",
                "Tight SL (0.5-1.0%)",
                "Quick profit taking (0.5-1.5% TP)",
                "No HTF gating; scalp off intraday noise",
                "High frequency, small R:R per trade, volume through frequency",
            ],
            contraindicated_signals=[
                "swing_low_structural",  # Structural plays are longer-term than scalps
                "ema_pullback_support",  # Pullback waiting is too slow for scalps
            ],
            description="Scalping — rapid entries/exits on intraday noise with tight SL/TP",
        ),
    }

    def __init__(self):
        pass

    def get_family_spec(self, family: StrategyFamilyType | str | None) -> StrategyFamilySpec | None:
        """Retrieve specification for a strategy family."""
        if not family:
            return None

        # Normalize to standard format
        if isinstance(family, str):
            normalized = family.upper().replace(" ", "_")
            # Direct lookup (keys are strings like "ORB", "VWAP_RECLAIM")
            if normalized in self.FAMILY_SPECS:
                return self.FAMILY_SPECS[normalized]
            # Fallback: try all keys with normalized comparison
            for key in self.FAMILY_SPECS.keys():
                if isinstance(key, str) and key.upper() == normalized:
                    return self.FAMILY_SPECS[key]
            return None

        return self.FAMILY_SPECS.get(family)

    def get_canonical_signals(
        self, family: StrategyFamilyType | str | None
    ) -> list[CanonicalSignalSpec]:
        """Get canonical signals for a family."""
        spec = self.get_family_spec(family)
        return spec.canonical_signals if spec else []

    def get_mandatory_signals(
        self, family: StrategyFamilyType | str | None
    ) -> list[CanonicalSignalSpec]:
        """Get mandatory (non-optional) canonical signals."""
        signals = self.get_canonical_signals(family)
        return [s for s in signals if s.mandatory]

    def is_contraindicated(self, family: StrategyFamilyType | str | None, signal_name: str) -> bool:
        """Check if a signal contradicts the family's intent."""
        spec = self.get_family_spec(family)
        if not spec:
            return False
        return signal_name in spec.contraindicated_signals

    def get_preserved_invariants(self, family: StrategyFamilyType | str | None) -> list[str]:
        """Get the key invariants that define the family."""
        spec = self.get_family_spec(family)
        return spec.preserved_invariants if spec else []

    def get_recommended_sl_type(self, family: StrategyFamilyType | str | None) -> StopLossType | None:
        """Get the recommended SL anchor type for a family."""
        spec = self.get_family_spec(family)
        return spec.sl_anchor_type if spec else None

    def validate_signal_fit(self, family: StrategyFamilyType | str | None, signal_name: str) -> dict:
        """Validate how well a signal fits the strategy family."""
        result = {
            "fits": False,
            "canonical": False,
            "contraindicated": False,
            "reasoning": "",
        }

        if not family:
            result["reasoning"] = "No family specified"
            return result

        spec = self.get_family_spec(family)
        if not spec:
            result["reasoning"] = f"Unknown family: {family}"
            return result

        # Check if canonical
        canonical_names = [s.signal_name for s in spec.canonical_signals]
        if signal_name in canonical_names:
            result["canonical"] = True
            result["fits"] = True
            result["reasoning"] = f"Canonical signal for {family}"
            return result

        # Check if contraindicated
        if signal_name in spec.contraindicated_signals:
            result["contraindicated"] = True
            result["fits"] = False
            result["reasoning"] = f"Contradicts {family} intent: {spec.contraindicated_signals}"
            return result

        # Default: not canonical but not contraindicated
        result["fits"] = True
        result["reasoning"] = f"Acceptable but not canonical for {family}"
        return result

    def log_family_preservation(self, family: StrategyFamilyType | str | None, picked_signals: list[str]) -> None:
        """Log preservation metrics for a strategy."""
        spec = self.get_family_spec(family)
        if not spec:
            logger.warning(f"No spec found for family: {family}")
            return

        canonical_names = [s.signal_name for s in spec.canonical_signals]
        canonical_picked = [s for s in picked_signals if s in canonical_names]
        contraindicated_picked = [s for s in picked_signals if s in spec.contraindicated_signals]

        logger.info(
            "strategy_family_preserver|family=%s|canonical_picked=%d/%d|contraindicated=%d|invariants=%d",
            family,
            len(canonical_picked),
            len(canonical_names),
            len(contraindicated_picked),
            len(spec.preserved_invariants),
        )

        if contraindicated_picked:
            logger.warning(
                "strategy_family_preserver|contraindicated_signals_picked=%s|family=%s",
                contraindicated_picked,
                family,
            )

    def apply_family_requirements_to_config(self, family: StrategyFamilyType | str | None, config: dict) -> dict:
        """Apply family-specific requirements to execution config."""
        spec = self.get_family_spec(family)
        if not spec:
            return config

        # Enforce SL anchor type if present
        if spec.sl_anchor_type:
            if "structural_sl" not in config:
                config["structural_sl"] = {}
            config["structural_sl"]["type"] = spec.sl_anchor_type

        # Tag family
        config["strategy_family"] = spec.family

        # Add invariants to trace
        if "semantic_extraction_trace" not in config:
            config["semantic_extraction_trace"] = {}
        config["semantic_extraction_trace"]["preserved_invariants"] = spec.preserved_invariants

        return config
