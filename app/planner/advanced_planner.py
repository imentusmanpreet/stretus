"""
app/planner/advanced_planner.py — Enhanced planner with semantic extraction.

Wraps the existing pipeline to add advanced execution-layer extraction:
  - Semantic instructions from user prompt (10 capabilities)
  - Strategy family preservation
  - Complete execution config assembly

Usage:
    planner = AdvancedPlanner()
    config = await planner.plan_with_execution_config(builder, ohlcv)

The output is an ExecutionConfig that includes all 10 advanced layers,
ready to send to backtest or live execution engine.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from app.kb.schemas import StrategyPlan
from app.planner.execution_configurator import ExecutionConfigurator
from app.planner.pipeline import Pipeline, PlannerError
from app.planner.semantic_extractor import SemanticExtractor
from app.planner.strategy_family_preserver import StrategyFamilyPreserver

logger = logging.getLogger(__name__)


class AdvancedPlanner:
    """Planner enhanced with semantic extraction and execution config."""

    def __init__(
        self,
        base_pipeline: Pipeline | None = None,
        semantic_extractor: SemanticExtractor | None = None,
        family_preserver: StrategyFamilyPreserver | None = None,
        execution_configurator: ExecutionConfigurator | None = None,
    ):
        self.pipeline = base_pipeline or Pipeline()
        self.semantic_extractor = semantic_extractor or SemanticExtractor()
        self.family_preserver = family_preserver or StrategyFamilyPreserver()
        self.execution_configurator = (
            execution_configurator
            or ExecutionConfigurator(self.semantic_extractor, self.family_preserver)
        )

    async def plan_with_execution_config(
        self,
        builder: Any,
        ohlcv: pd.DataFrame | None = None,
    ) -> dict:
        """
        Run complete planning including semantic extraction and execution config.

        Args:
            builder: StrategyBuilder with user inputs
            ohlcv: Optional OHLCV data for param resolution

        Returns:
            Dict with:
                - strategy_plan: StrategyPlan (signals, SL/TP %)
                - execution_config: ExecutionConfig (all 10 advanced layers)
                - semantic_instructions: SemanticInstructions (raw extraction)
                - quality_metrics: Quality assessment
        """
        # Step 1: Run standard planner pipeline
        logger.info("advanced_planner|step1_run_pipeline")
        strategy_plan = await self.pipeline.plan(builder, ohlcv)

        # Step 2: Extract semantic instructions from user goal/prompt
        user_goal = getattr(builder, "goal", "") or ""
        logger.info("advanced_planner|step2_semantic_extraction|goal=%s", user_goal[:60])
        semantic_instructions = self.semantic_extractor.extract(user_goal)

        # Step 3: Validate family signal fit
        if semantic_instructions.strategy_family:
            self._log_family_validation(semantic_instructions, strategy_plan)

        # Step 4: Assemble execution config
        logger.info("advanced_planner|step3_assemble_execution_config")
        execution_config = await self.execution_configurator.configure(
            user_goal,
            strategy_plan,
        )

        # Step 5: Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            semantic_instructions,
            strategy_plan,
            execution_config,
        )

        result = {
            "strategy_plan": strategy_plan,
            "execution_config": execution_config,
            "semantic_instructions": semantic_instructions,
            "quality_metrics": quality_metrics,
        }

        logger.info(
            "advanced_planner|complete|family=%s|quality=%.2f|htf_rules=%d|structural_sl=%s",
            execution_config.strategy_family,
            quality_metrics["overall_quality_score"],
            len(execution_config.htf_rules),
            execution_config.structural_sl is not None,
        )

        return result

    def _log_family_validation(
        self,
        semantic: Any,  # SemanticInstructions
        plan: StrategyPlan,
    ) -> None:
        """Log strategy family validation results."""
        picked_signals = [plan.entry_trigger.name, plan.exit_trigger.name]
        picked_signals.extend([f.name for f in plan.entry_filters])

        logger.info(
            "advanced_planner|family_validation|family=%s|confidence=%.2f|signals_picked=%s",
            semantic.strategy_family,
            semantic.family_confidence,
            picked_signals,
        )

        # Validate each picked signal against family
        for signal_name in picked_signals:
            validation = self.family_preserver.validate_signal_fit(
                semantic.strategy_family,
                signal_name,
            )
            if validation["canonical"]:
                logger.info(
                    "advanced_planner|signal_validation|signal=%s|fit=canonical",
                    signal_name,
                )
            elif validation["contraindicated"]:
                logger.warning(
                    "advanced_planner|signal_validation|signal=%s|fit=contraindicated|"
                    "reason=%s",
                    signal_name,
                    validation["reasoning"],
                )

    def _calculate_quality_metrics(
        self,
        semantic: Any,  # SemanticInstructions
        plan: StrategyPlan,
        config: Any,  # ExecutionConfig
    ) -> dict:
        """Calculate comprehensive quality metrics."""
        metrics = {
            # Semantic extraction quality (0.0-1.0)
            "semantic_extraction_score": semantic.extraction_quality_score,

            # Execution completeness (0.0-1.0)
            "execution_completeness": self._calc_execution_completeness(config),

            # Family preservation (0.0-1.0)
            "family_preservation_score": self._calc_family_preservation(semantic, plan),

            # Coverage (how many of 10 layers are utilized)
            "layers_covered": self._count_layers_covered(semantic, config),
            "total_layers": 10,

            # Overall quality (average of components)
            "overall_quality_score": 0.0,
        }

        # Calculate overall as weighted average
        scores = [
            metrics["semantic_extraction_score"] * 0.3,
            metrics["execution_completeness"] * 0.4,
            metrics["family_preservation_score"] * 0.3,
        ]
        metrics["overall_quality_score"] = sum(scores)

        return metrics

    def _calc_execution_completeness(self, config: Any) -> float:
        """Score execution config completeness (0.0-1.0)."""
        score = 0.0

        # Signal logic (always present)
        if config.signal_logic:
            score += 0.15

        # Structural SL
        if config.structural_sl:
            score += 0.15

        # Trailing stop
        if config.trailing_stop and config.trailing_stop.enabled:
            score += 0.1

        # Risk:Reward
        if config.risk_reward:
            score += 0.15

        # HTF rules
        if config.htf_rules:
            score += 0.15

        # Reference symbols
        if config.reference_symbols:
            score += 0.1

        # Session filters
        if config.session_filters and config.session_filters.enabled:
            score += 0.1

        # Volume/momentum
        if config.volume_momentum:
            score += 0.1

        return min(score, 1.0)

    def _calc_family_preservation(self, semantic: Any, plan: StrategyPlan) -> float:
        """Score how well family intent is preserved (0.0-1.0)."""
        if not semantic.strategy_family:
            return 0.5  # Neutral if no family detected

        picked_signals = [plan.entry_trigger.name, plan.exit_trigger.name]
        picked_signals.extend([f.name for f in plan.entry_filters])

        # Count canonical signals
        canonical_signals = self.family_preserver.get_canonical_signals(
            semantic.strategy_family
        )
        canonical_names = {s.signal_name for s in canonical_signals}
        canonical_picked = [s for s in picked_signals if s in canonical_names]

        # Count contraindicated signals
        spec = self.family_preserver.get_family_spec(semantic.strategy_family)
        contraindicated_count = (
            sum(1 for s in picked_signals if s in spec.contraindicated_signals)
            if spec
            else 0
        )

        # Score: high for canonical, penalized for contraindicated
        canonical_score = min(len(canonical_picked) / max(len(canonical_names), 1), 1.0)
        contraindicated_penalty = contraindicated_count * 0.25

        return max(canonical_score - contraindicated_penalty, 0.0)

    def _count_layers_covered(self, semantic: Any, config: Any) -> int:
        """Count how many of 10 execution layers are covered."""
        count = 0

        # 1. Signal logic (always)
        if config.signal_logic:
            count += 1

        # 2. Multi-timeframe rules
        if config.htf_rules:
            count += 1

        # 3. Reference symbols
        if config.reference_symbols:
            count += 1

        # 4. Structural SL
        if config.structural_sl:
            count += 1

        # 5. Trailing stop
        if config.trailing_stop and config.trailing_stop.enabled:
            count += 1

        # 6. Risk:Reward
        if config.risk_reward:
            count += 1

        # 7. Session filters
        if config.session_filters and config.session_filters.enabled:
            count += 1

        # 8. Volume confirmation
        if config.volume_momentum and config.volume_momentum.volume:
            count += 1

        # 9. Momentum confirmation
        if config.volume_momentum and config.volume_momentum.momentum:
            count += 1

        # 10. Semantic preservation (family tag)
        if config.strategy_family:
            count += 1

        return count


async def plan_with_advanced_features(
    builder: Any,
    ohlcv: pd.DataFrame | None = None,
    base_pipeline: Pipeline | None = None,
) -> dict:
    """
    Convenience function to run complete advanced planning in one call.

    Usage:
        result = await plan_with_advanced_features(
            builder,
            ohlcv=ohlcv_data,
        )
        # result contains:
        # - strategy_plan: StrategyPlan
        # - execution_config: ExecutionConfig with all 10 layers
        # - semantic_instructions: Extracted instructions
        # - quality_metrics: Quality assessment
    """
    planner = AdvancedPlanner(base_pipeline=base_pipeline)
    return await planner.plan_with_execution_config(builder, ohlcv)
