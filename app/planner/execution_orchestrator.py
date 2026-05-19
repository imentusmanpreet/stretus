"""
app/planner/execution_orchestrator.py — Apply semantic instructions to signal plans.

Bridges the gap between SemanticInstructions (extracted from user prompts) and
StrategyPlan (generated from signal library). Applies:
- Multi-timeframe entry gates (HTF rules)
- Momentum filters (ADX thresholds, etc.)
- Structural stop-loss anchoring
- Risk:Reward validation
- Candle confirmation filters
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from app.kb.execution_schemas import (
    HTFCondition,
    MomentumConfirmation,
    RiskRewardSpec,
    SemanticInstructions,
    StructuralStopLoss,
)

logger = logging.getLogger(__name__)


class ExecutionOrchestrator:
    """Apply semantic instructions to signal-based strategy plans."""

    def apply_semantic_gates(
        self,
        signal_plan: dict[str, Any],
        semantic_instructions: SemanticInstructions,
    ) -> dict[str, Any]:
        """
        Apply all semantic rules to the signal plan.

        Returns enhanced signal plan with:
        - HTF entry gates added as filters
        - Momentum filters added as filters
        - Structural SL added as exit signal
        - Audit trail of what was added
        """
        if not semantic_instructions:
            return signal_plan

        enhanced_plan = self._deep_copy_plan(signal_plan)
        audit_trail: list[str] = []

        # Apply HTF gates
        if semantic_instructions.htf_rules:
            self._apply_htf_gates(
                enhanced_plan, semantic_instructions.htf_rules, audit_trail
            )

        # Apply momentum filters (ADX, etc.)
        if semantic_instructions.volume_momentum and semantic_instructions.volume_momentum.momentum:
            self._apply_momentum_filters(
                enhanced_plan,
                semantic_instructions.volume_momentum.momentum,
                audit_trail,
            )

        # Apply candle confirmation filters
        if semantic_instructions.candle_confirmation:
            self._apply_candle_confirmation(
                enhanced_plan,
                semantic_instructions.candle_confirmation,
                audit_trail,
            )

        # Apply structural SL
        if semantic_instructions.stop_loss:
            self._apply_structural_sl(
                enhanced_plan, semantic_instructions.stop_loss, audit_trail
            )

        # ── Bridge semantic results into plan underscore keys ────────────────
        # These keys are consumed by StrategyBuilder.apply_signal_plan so that
        # semantic-extracted specs are persisted on the builder (single source
        # of truth), not just injected as loose "entry_filters" / "exit" items.

        # HTF rules: merge semantic conditions into the existing _htf_rules list
        # (which may already carry rules from the KB preset leg).
        if semantic_instructions.htf_rules:
            existing_htf: list[dict] = list(enhanced_plan.get("_htf_rules") or [])
            existing_keys = {
                (r.get("timeframe", ""), r.get("condition", ""))
                for r in existing_htf
                if isinstance(r, dict)
            }
            for htf_rule in semantic_instructions.htf_rules:
                tf = (htf_rule.timeframe or "").strip()
                cond = (htf_rule.condition or "").strip()
                if tf and cond and (tf, cond) not in existing_keys:
                    existing_htf.append({"timeframe": tf, "condition": cond})
                    existing_keys.add((tf, cond))
            enhanced_plan["_htf_rules"] = existing_htf

        # Reference symbol: first RS condition wins; never overwrite preset value.
        if semantic_instructions.reference_symbols and not enhanced_plan.get("_reference_symbol"):
            first_rs = semantic_instructions.reference_symbols[0]
            if first_rs.reference_symbol:
                enhanced_plan["_reference_symbol"] = first_rs.reference_symbol.upper()
                audit_trail.append(f"ref_symbol:{first_rs.reference_symbol}")

        # Structural SL: only set if the planner didn't already attach one
        # (builder.stop_loss_spec already takes priority in the pipeline).
        if semantic_instructions.stop_loss and not enhanced_plan.get("_stop_loss_spec"):
            sl = semantic_instructions.stop_loss
            enhanced_plan["_stop_loss_spec"] = {
                "type": sl.type,
                "anchor": sl.anchor,
                "source": "semantic",
                "description": sl.description,
            }
            audit_trail.append(f"structural_sl:{sl.type}")

        # Trailing stop: only set if not already provided by the preset.
        if (
            semantic_instructions.trailing_stop
            and semantic_instructions.trailing_stop.enabled
            and not enhanced_plan.get("_trailing_stop_spec")
        ):
            ts = semantic_instructions.trailing_stop
            enhanced_plan["_trailing_stop_spec"] = {
                "type": ts.type,
                "activate_after_pct": ts.activate_after_pct,
                "ema_period": ts.ema_period,
                "source": "semantic",
            }
            audit_trail.append(f"trailing:{ts.type}")

        # Risk:Reward: expose in plan so that the chat layer can sync it back
        # to builder.risk_execution_config without reading SemanticInstructions.
        if semantic_instructions.risk_reward and semantic_instructions.risk_reward.ratio:
            enhanced_plan.setdefault("_semantic_risk_reward", {
                "ratio": semantic_instructions.risk_reward.ratio,
                "type": semantic_instructions.risk_reward.type,
            })
            audit_trail.append(f"rr:{semantic_instructions.risk_reward.ratio}")

        # Log audit trail
        enhanced_plan["_semantic_gates_applied"] = audit_trail
        logger.info(
            "execution_orchestrator|gates_applied|count=%d|gates=%s",
            len(audit_trail),
            ", ".join(audit_trail),
        )

        return enhanced_plan

    def validate_rr(
        self,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        rr_spec: Optional[RiskRewardSpec],
    ) -> tuple[bool, float, str]:
        """
        Validate that actual RR meets specification.

        Returns:
            (is_valid, actual_rr, reason)
        """
        if not rr_spec or not rr_spec.ratio:
            return True, 0.0, "No RR specification"

        if entry_price == stop_loss_price:
            return False, 0.0, "Entry equals stop loss"

        sl_distance = abs(entry_price - stop_loss_price)
        tp_distance = abs(take_profit_price - entry_price)

        if sl_distance == 0:
            return False, 0.0, "SL distance is zero"

        actual_rr = tp_distance / sl_distance

        if rr_spec.type == "minimum":
            # Minimum RR: actual must be >= specified
            is_valid = actual_rr >= rr_spec.ratio
            reason = (
                f"RR {actual_rr:.2f}:1 {'meets' if is_valid else 'below'} "
                f"minimum {rr_spec.ratio}:1"
            )
        else:  # "fixed" or "derived"
            # Fixed RR: actual must equal (within 0.01 tolerance)
            is_valid = abs(actual_rr - rr_spec.ratio) < 0.01
            reason = (
                f"RR {actual_rr:.2f}:1 "
                f"{'matches' if is_valid else 'differs from'} "
                f"target {rr_spec.ratio}:1"
            )

        return is_valid, actual_rr, reason

    # ── Private Helpers ──────────────────────────────────────────────────────────

    def _apply_htf_gates(
        self,
        signal_plan: dict[str, Any],
        htf_rules: list[HTFCondition],
        audit_trail: list[str],
    ) -> None:
        """Apply higher-timeframe confirmation/gating rules as entry filters."""
        if "entry_filters" not in signal_plan:
            signal_plan["entry_filters"] = []

        for rule in htf_rules:
            if rule.role == "gating":
                # Add as mandatory entry filter
                gate_signal = {
                    "name": f"htf_{rule.timeframe}_gate",
                    "params": {
                        "timeframe": rule.timeframe,
                        "condition": rule.condition or "bullish",
                    },
                    "signal_type": "GATING",
                    "source": "SEMANTIC",
                    "description": rule.description or f"{rule.timeframe} must be {rule.condition}",
                }

                # Avoid duplicates
                if not any(
                    s.get("name") == gate_signal["name"]
                    for s in signal_plan["entry_filters"]
                ):
                    signal_plan["entry_filters"].append(gate_signal)
                    audit_trail.append(f"HTF gate: {rule.timeframe} {rule.condition}")

            elif rule.role == "confirmation":
                # Add as optional entry filter
                confirm_signal = {
                    "name": f"htf_{rule.timeframe}_confirm",
                    "params": {
                        "timeframe": rule.timeframe,
                        "condition": rule.condition or "bullish",
                    },
                    "signal_type": "FILTER",
                    "source": "SEMANTIC",
                    "description": rule.description or f"{rule.timeframe} should be {rule.condition}",
                }

                if not any(
                    s.get("name") == confirm_signal["name"]
                    for s in signal_plan["entry_filters"]
                ):
                    signal_plan["entry_filters"].append(confirm_signal)
                    audit_trail.append(f"HTF confirm: {rule.timeframe} {rule.condition}")

    def _apply_momentum_filters(
        self,
        signal_plan: dict[str, Any],
        momentum: MomentumConfirmation,
        audit_trail: list[str],
    ) -> None:
        """Apply momentum confirmation filters (ADX > threshold, etc.)."""
        if "entry_filters" not in signal_plan:
            signal_plan["entry_filters"] = []

        if momentum.filter_type == "adx_threshold" and momentum.adx_threshold:
            # Add ADX threshold as entry filter
            adx_signal = {
                "name": f"adx_above_{int(momentum.adx_threshold)}",
                "params": {
                    "threshold": momentum.adx_threshold,
                    "period": 14,
                },
                "signal_type": "FILTER",
                "source": "SEMANTIC",
                "description": f"ADX > {momentum.adx_threshold}",
            }

            if not any(s.get("name") == adx_signal["name"] for s in signal_plan["entry_filters"]):
                signal_plan["entry_filters"].append(adx_signal)
                audit_trail.append(f"Momentum: ADX > {momentum.adx_threshold}")

        elif momentum.filter_type in ("adx_strong", "momentum_bullish"):
            # Generic strong momentum filter
            mom_signal = {
                "name": "momentum_strong",
                "params": {"period": 14},
                "signal_type": "FILTER",
                "source": "SEMANTIC",
                "description": "Strong momentum confirmation",
            }

            if not any(s.get("name") == mom_signal["name"] for s in signal_plan["entry_filters"]):
                signal_plan["entry_filters"].append(mom_signal)
                audit_trail.append("Momentum: Strong momentum required")

    def _apply_candle_confirmation(
        self,
        signal_plan: dict[str, Any],
        candle_confirmation: Any,  # CandleConfirmation type
        audit_trail: list[str],
    ) -> None:
        """Apply candle pattern confirmation as entry filter."""
        if "entry_filters" not in signal_plan:
            signal_plan["entry_filters"] = []

        candle_signal_name = f"candle_{candle_confirmation.filter_type}"
        candle_signal = {
            "name": candle_signal_name,
            "params": {},
            "signal_type": "FILTER",
            "source": "SEMANTIC",
            "description": candle_confirmation.description or f"{candle_confirmation.filter_type} candle confirmation",
        }

        if not any(s.get("name") == candle_signal_name for s in signal_plan["entry_filters"]):
            signal_plan["entry_filters"].append(candle_signal)
            audit_trail.append(f"Candle: {candle_confirmation.filter_type}")

    def _apply_structural_sl(
        self,
        signal_plan: dict[str, Any],
        sl_spec: StructuralStopLoss,
        audit_trail: list[str],
    ) -> None:
        """Apply structural stop-loss as exit signal."""
        if "exit" not in signal_plan:
            signal_plan["exit"] = []

        # Build SL signal name based on anchor and padding
        anchor_name = (sl_spec.anchor or sl_spec.type).replace("_", "_").lower()
        padding_suffix = (
            f"_{sl_spec.padding.method}"
            if sl_spec.padding and sl_spec.padding.method != "none"
            else ""
        )
        sl_signal_name = f"structural_sl_{anchor_name}{padding_suffix}"

        # Build SL signal
        sl_signal = {
            "name": sl_signal_name,
            "params": {
                "anchor": sl_spec.anchor or sl_spec.type,
                "lookback": 20,  # Default lookback for swing detection
            },
            "signal_type": "TRIGGER",
            "source": "SEMANTIC",
            "description": sl_spec.description or f"SL at {sl_spec.anchor}",
        }

        # Add padding params if specified
        if sl_spec.padding:
            if sl_spec.padding.method == "atr" and sl_spec.padding.atr_multiple:
                sl_signal["params"]["atr_multiple"] = sl_spec.padding.atr_multiple
            elif sl_spec.padding.method == "percent" and sl_spec.padding.percent:
                sl_signal["params"]["padding_percent"] = sl_spec.padding.percent
            elif sl_spec.padding.method == "points" and sl_spec.padding.points:
                sl_signal["params"]["padding_points"] = sl_spec.padding.points

        # Avoid duplicate SL signals
        if not any(
            s.get("name") == sl_signal_name or "structural_sl" in s.get("name", "")
            for s in signal_plan["exit"]
        ):
            signal_plan["exit"] = [s for s in signal_plan["exit"] if "price_below" not in s.get("name", "")]
            signal_plan["exit"].append(sl_signal)
            audit_trail.append(f"SL: {sl_spec.anchor} {sl_spec.padding.method if sl_spec.padding else 'no padding'}")

    def _deep_copy_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Create a deep copy of the signal plan."""
        import copy

        return copy.deepcopy(plan)
