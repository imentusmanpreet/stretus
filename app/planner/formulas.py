"""
app/planner/formulas.py — Render the formula string for a signal.

Resolution order:
  1. Computed-param signals (gap_up_open / gap_down_open) — derived values
     that cannot be expressed as a plain str.format template.
  2. Card-driven template from the new KB (preferred).
  3. RuleRegistry-driven template (legacy fallback for any signal not yet
     migrated to a SignalCard).
  4. None — log a warning; caller decides to skip or substitute a safe exit.
"""
from __future__ import annotations

import logging

from app.kb import kb
from stretus_knowledge_base.stretus_kb.registry import RuleRegistry

logger = logging.getLogger(__name__)


def render_formula(name: str, params: dict) -> str | None:
    if name == "gap_up_open":
        gap_factor = round(1 + float(params.get("gap_pct", 0.5)) / 100.0, 6)
        return f"OPEN > PREV(CLOSE, 1) * {gap_factor}"
    if name == "gap_down_open":
        gap_factor = round(1 - float(params.get("gap_pct", 0.5)) / 100.0, 6)
        return f"OPEN < PREV(CLOSE, 1) * {gap_factor}"

    card = kb.signals.get(name)
    template = card.formula if card else RuleRegistry.get_formula(name)

    if template:
        try:
            return template.format(**params)
        except KeyError as exc:
            logger.warning(
                "signal_formula|format_error|signal=%s|missing_key=%s",
                name, exc,
            )

    logger.error(
        "signal_formula|no_template|signal=%s — cannot render condition string; "
        "it will be SKIPPED.",
        name,
    )
    return None
