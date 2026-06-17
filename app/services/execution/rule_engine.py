"""
app/services/execution/rule_engine.py
───────────────────────────────────────
Signal evaluation engine for the execution service.

Evaluation contract:
  entry_signal()  → True when ALL of the following hold on the latest candle:
                      • entry trigger fires
                      • every entry filter fires  (AND logic)

  exit_signal()   → True when ANY of the following hold on the latest candle:
                      • exit trigger fires
                      • every exit filter fires  (AND logic)

  Both methods return a (result: bool, messages: List[str]) tuple so the
  caller can log every decision step for auditability.

The actual indicator computation is handled by stretus_kb.RuleRegistry.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

# Importing indicator_engine ensures signals are registered before first call
from app.services.execution import indicator_engine as _  # noqa: F401
from stretus_knowledge_base.stretus_kb.registry import RuleRegistry

logger = logging.getLogger(__name__)


class RuleEngine:
    """Stateless evaluation engine; instantiate once per request."""

    # ── Entry ─────────────────────────────────────────────────────────────────

    def evaluate_entry(
        self,
        df: pd.DataFrame,
        entry_block: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Evaluate the entry block against the last row of `df`.

        entry_block shape:
            {
              "trigger": {"type": "...", "params": {...}},
              "filters": [{"type": "...", "params": {...}}, ...]
            }

        Returns (True, messages) only when trigger AND all filters pass.
        """
        messages: List[str] = []

        trigger = entry_block.get("trigger", {})
        filters = entry_block.get("filters", [])

        trigger_result, trigger_msgs = self._eval_rule(df, trigger, label="ENTRY trigger")
        messages.extend(trigger_msgs)

        if not trigger_result:
            messages.append(
                f"  ↳ Trigger did not fire — filters skipped (AND logic requires trigger first)."
            )
            return False, messages

        for i, filt in enumerate(filters, start=1):
            ok, msgs = self._eval_rule(df, filt, label=f"ENTRY filter[{i}]")
            messages.extend(msgs)
            if not ok:
                messages.append(
                    f"  ↳ Filter[{i}] ({filt.get('type')}) blocked entry — AND logic failed."
                )
                return False, messages

        messages.append("✅ All entry conditions passed — signal confirmed.")
        return True, messages

    # ── Exit ──────────────────────────────────────────────────────────────────

    def evaluate_exit(
        self,
        df: pd.DataFrame,
        exit_block: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Evaluate the exit block against the last row of `df`.

        Uses the same AND logic as entry (trigger + all filters must fire).
        """
        messages: List[str] = []

        trigger = exit_block.get("trigger", {})
        filters = exit_block.get("filters", [])

        trigger_result, trigger_msgs = self._eval_rule(df, trigger, label="EXIT trigger")
        messages.extend(trigger_msgs)

        if not trigger_result:
            messages.append("  ↳ Exit trigger did not fire — holding position.")
            return False, messages

        for i, filt in enumerate(filters, start=1):
            ok, msgs = self._eval_rule(df, filt, label=f"EXIT filter[{i}]")
            messages.extend(msgs)
            if not ok:
                messages.append(
                    f"  ↳ Exit filter[{i}] ({filt.get('type')}) did not pass — holding position."
                )
                return False, messages

        messages.append("🚪 Exit signal confirmed — all exit conditions met.")
        return True, messages

    # ── SL / TP price check (stateless, no signals) ───────────────────────────

    def check_stop_loss(self, ltp: float, stop_loss_price: float) -> bool:
        """Return True if the current price has breached the stop loss level."""
        return ltp <= stop_loss_price

    def check_take_profit(self, ltp: float, take_profit_price: float) -> bool:
        """Return True if the current price has reached the take profit target."""
        return ltp >= take_profit_price

    # ── Internal ──────────────────────────────────────────────────────────────

    def _eval_rule(
        self,
        df: pd.DataFrame,
        rule: Dict[str, Any],
        label: str,
    ) -> Tuple[bool, List[str]]:
        """Evaluate a single rule dict against the DataFrame."""
        rule_type = rule.get("type", "")
        params = rule.get("params", {})
        messages: List[str] = []

        if not rule_type:
            messages.append(f"  ⚠️  {label}: missing rule type — treating as False.")
            return False, messages

        # Formula-string trigger synthesised by the Go scheduler when a strategy
        # has only an entry_condition / exit_condition and no structured KB signal.
        # The formula is evaluated via the same engine used by the backtest simulator
        # so live execution stays consistent with backtested results.
        if rule_type == "condition":
            formula = str(params.get("formula", "")).strip()
            if not formula:
                messages.append(f"  ⚠️  {label} [condition]: no formula param — treating as False.")
                return False, messages
            try:
                from engine.conditions import evaluate_condition, explain_condition  # noqa: PLC0415
                bar_i  = len(df) - 1
                result = bool(evaluate_condition(formula, df, bar_i))
                icon   = "✅" if result else "❌"
                status = "PASS" if result else "FAIL"
                messages.append(f"  {icon} {label} [condition] → {status}  (formula={formula!r})")
                # Always emit per-comparison breakdown so discrepancies with chart are diagnosable.
                for detail in explain_condition(formula, df, bar_i):
                    messages.append(detail)
                return result, messages
            except Exception as exc:
                messages.append(f"  💥 {label} [condition]: formula evaluation error — {exc}")
                logger.error("Error evaluating condition formula %r: %s", formula, exc, exc_info=True)
                return False, messages

        try:
            result = RuleRegistry.evaluate(
                {"name": rule_type, "params": params},
                df,
            )
            result = bool(result)
            icon   = "✅" if result else "❌"
            status = "PASS" if result else "FAIL"
            param_str = "  ".join(f"{k}={v}" for k, v in params.items()) if params else "no params"
            messages.append(f"  {icon} {label} [{rule_type}] → {status}  ({param_str})")
            return result, messages

        except ValueError:
            # Signal not in RuleRegistry — try rendering its KB formula and evaluating
            # it as a condition string. This covers YAML-only signals (e.g.
            # price_cross_above_sma) that have no Python-registered implementation.
            from app.planner.formulas import render_formula  # noqa: PLC0415
            formula = render_formula(rule_type, params)
            if formula:
                try:
                    from engine.conditions import evaluate_condition, explain_condition  # noqa: PLC0415
                    bar_i  = len(df) - 1
                    result = bool(evaluate_condition(formula, df, bar_i))
                    icon   = "✅" if result else "❌"
                    status = "PASS" if result else "FAIL"
                    messages.append(
                        f"  {icon} {label} [{rule_type}] → {status}  "
                        f"(kb formula: {formula!r})"
                    )
                    for detail in explain_condition(formula, df, bar_i):
                        messages.append(detail)
                    return result, messages
                except Exception as exc:
                    messages.append(
                        f"  💥 {label} [{rule_type}]: kb formula evaluation error — {exc}"
                    )
                    logger.error(
                        "Error evaluating KB formula for signal %r: %s", rule_type, exc, exc_info=True
                    )
                    return False, messages
            messages.append(
                f"  ⚠️  {label} [{rule_type}]: unknown signal — not in registry and no KB formula found."
            )
            logger.warning("Unknown signal '%s': not in RuleRegistry and no KB formula.", rule_type)
            return False, messages

        except Exception as exc:
            messages.append(f"  💥 {label} [{rule_type}]: evaluation error — {exc}")
            logger.error("Error evaluating signal '%s': %s", rule_type, exc, exc_info=True)
            return False, messages
