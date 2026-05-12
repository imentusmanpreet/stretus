"""
app/planner/execution_configurator.py — Assemble execution-ready strategy configs.

Combines:
  1. SemanticInstructions (extracted from prompt)
  2. StrategyPlan (signal selection from planner)
  3. StrategyFamilyPreserver (family preservation and validation)

Outputs ExecutionConfig ready for backtesting/live trading, with all 10
advanced execution layers properly configured.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.kb.execution_schemas import ExecutionConfig
from app.kb.schemas import StrategyPlan
from app.planner.semantic_extractor import SemanticExtractor, SemanticInstructions
from app.planner.strategy_family_preserver import StrategyFamilyPreserver

logger = logging.getLogger(__name__)


class ExecutionConfigurator:
    """Assemble complete execution config from semantic + signal layers."""

    def __init__(
        self,
        semantic_extractor: SemanticExtractor | None = None,
        family_preserver: StrategyFamilyPreserver | None = None,
    ):
        self.semantic_extractor = semantic_extractor or SemanticExtractor()
        self.family_preserver = family_preserver or StrategyFamilyPreserver()

    async def configure(
        self,
        prompt: str,
        strategy_plan: StrategyPlan,
    ) -> ExecutionConfig:
        """
        Assemble ExecutionConfig from semantic extraction + strategy plan.

        Args:
            prompt: User's natural language strategy description
            strategy_plan: Output from planner pipeline (signal selection, SL/TP %)

        Returns:
            ExecutionConfig with all 10 execution layers properly configured
        """
        # Step 1: Extract semantic instructions from prompt
        semantic = self.semantic_extractor.extract(prompt)

        # Step 2: Validate strategy family against selected signals
        self._validate_family_signal_fit(semantic, strategy_plan)

        # Step 3: Apply family-specific requirements
        self._apply_family_requirements(semantic, strategy_plan)

        # Step 4: Derive TP from RR if needed
        self._derive_tp_from_rr(semantic, strategy_plan)

        # Step 5: Assemble full execution config
        config = ExecutionConfig(
            signal_logic=self._build_signal_logic(strategy_plan),
            structural_sl=semantic.stop_loss,
            trailing_stop=semantic.trailing_stop,
            risk_reward=semantic.risk_reward,
            htf_rules=semantic.htf_rules,
            reference_symbols=semantic.reference_symbols,
            session_filters=semantic.session_filters,
            volume_momentum=semantic.volume_momentum,
            strategy_family=semantic.strategy_family,
            semantic_extraction_trace=self._build_trace(semantic, strategy_plan),
        )

        # Step 6: Validate config invariants
        self._validate_execution_invariants(config)

        logger.info(
            "execution_configurator|assembled|family=%s|htf_rules=%d|structural_sl=%s|"
            "trailing_stop=%s|rr=%s|session_filters=%s",
            config.strategy_family,
            len(config.htf_rules),
            config.structural_sl is not None,
            config.trailing_stop and config.trailing_stop.enabled,
            config.risk_reward is not None,
            config.session_filters and config.session_filters.enabled,
        )

        return config

    # ── Validation ─────────────────────────────────────────────────────────────

    def _validate_family_signal_fit(self, semantic: SemanticInstructions, plan: StrategyPlan) -> None:
        """Check if picked signals align with detected strategy family."""
        if not semantic.strategy_family:
            return

        picked_signals = [plan.entry_trigger.name, plan.exit_trigger.name]
        picked_signals.extend([f.name for f in plan.entry_filters])

        # Log validation
        self.family_preserver.log_family_preservation(semantic.strategy_family, picked_signals)

        # Warn if contraindicated signals were picked
        for signal_name in picked_signals:
            validation = self.family_preserver.validate_signal_fit(semantic.strategy_family, signal_name)
            if validation["contraindicated"]:
                logger.warning(
                    "execution_configurator|contraindicated_signal|family=%s|signal=%s|reason=%s",
                    semantic.strategy_family,
                    signal_name,
                    validation["reasoning"],
                )

    def _apply_family_requirements(self, semantic: SemanticInstructions, plan: StrategyPlan) -> None:
        """Apply family-specific SL anchor and invariants to semantic instructions."""
        if not semantic.strategy_family:
            return

        spec = self.family_preserver.get_family_spec(semantic.strategy_family)
        if not spec or not spec.sl_anchor_type:
            return

        # If SL wasn't extracted from prompt, use family default
        if not semantic.stop_loss and spec.sl_anchor_type:
            logger.info(
                "execution_configurator|applying_family_sl_default|family=%s|sl_type=%s",
                semantic.strategy_family,
                spec.sl_anchor_type,
            )
            from app.kb.execution_schemas import StructuralStopLoss
            semantic.stop_loss = StructuralStopLoss(type=spec.sl_anchor_type)

    def _derive_tp_from_rr(self, semantic: SemanticInstructions, plan: StrategyPlan) -> None:
        """Derive TP from RR specification if present."""
        if not semantic.risk_reward or not semantic.risk_reward.ratio:
            return

        if semantic.risk_reward.type != "fixed":
            return

        # TP = Entry + (RR_ratio * abs(Entry - SL))
        # For now, we'll preserve this as a formula to be evaluated at backtest time
        # since SL may be structural (not a fixed number yet)
        rr_ratio = semantic.risk_reward.ratio
        semantic.risk_reward.tp_formula = f"entry + ({rr_ratio} * abs(entry - sl))"
        semantic.risk_reward.sl_formula = None  # SL is already in semantic.stop_loss

        logger.info(
            "execution_configurator|derived_tp_from_rr|ratio=%s|formula=%s",
            rr_ratio,
            semantic.risk_reward.tp_formula,
        )

    def _validate_execution_invariants(self, config: ExecutionConfig) -> None:
        """Validate that the config satisfies strategy family invariants."""
        if not config.strategy_family:
            return

        invariants = self.family_preserver.get_preserved_invariants(config.strategy_family)
        if not invariants:
            return

        # Log invariants for debug/audit trail
        for inv in invariants:
            logger.debug(
                "execution_configurator|checking_invariant|family=%s|invariant=%s",
                config.strategy_family,
                inv,
            )

        # Note: Full validation requires market data and signal evaluation
        # These are semantic invariants documented for executor/backtest layer

    # ── Config assembly ───────────────────────────────────────────────────────

    def _build_signal_logic(self, plan: StrategyPlan) -> dict:
        """Extract signal logic from StrategyPlan into execution config."""
        return {
            "entry_trigger": plan.entry_trigger.model_dump(),
            "entry_filters": [f.model_dump() for f in plan.entry_filters],
            "exit_trigger": plan.exit_trigger.model_dump(),
            "entry_condition": plan.trace.get("entry_condition"),
            "exit_condition": plan.trace.get("exit_condition"),
        }

    def _build_trace(self, semantic: SemanticInstructions, plan: StrategyPlan) -> dict:
        """Build extraction trace for debugging and audit."""
        return {
            "semantic_extraction_quality": semantic.extraction_quality_score,
            "strategy_family": semantic.strategy_family,
            "strategy_family_confidence": semantic.family_confidence,
            "htf_rules_extracted": len(semantic.htf_rules),
            "structural_sl_extracted": semantic.stop_loss is not None,
            "trailing_stop_extracted": semantic.trailing_stop and semantic.trailing_stop.enabled,
            "rr_specification_extracted": semantic.risk_reward is not None,
            "reference_symbols_extracted": len(semantic.reference_symbols),
            "session_filters_extracted": semantic.session_filters and semantic.session_filters.enabled,
            "volume_momentum_filters_extracted": semantic.volume_momentum is not None,
            "original_prompt": semantic.original_prompt[:200] if semantic.original_prompt else None,
            "signal_entry_trigger": plan.entry_trigger.name,
            "signal_entry_filters": [f.name for f in plan.entry_filters],
            "signal_exit_trigger": plan.exit_trigger.name,
        }


def create_execution_config_from_prompt_and_plan(
    prompt: str,
    strategy_plan: StrategyPlan,
    semantic_extractor: SemanticExtractor | None = None,
    family_preserver: StrategyFamilyPreserver | None = None,
) -> ExecutionConfig:
    """
    Convenience function to create ExecutionConfig in a single call.

    Usage:
        config = create_execution_config_from_prompt_and_plan(
            prompt="Please create VWAP Reversal Strategy...",
            strategy_plan=pipeline_output,
        )
    """
    configurator = ExecutionConfigurator(semantic_extractor, family_preserver)
    # Note: In async context, wrap in asyncio.run()
    import asyncio
    return asyncio.run(configurator.configure(prompt, strategy_plan))
