"""
app/planner/formulas.py — Render the formula string for a signal.

Resolution order:
  1. Computed-param signals (gap_up_open / gap_down_open) — derived values
     that cannot be expressed as a plain str.format template.
  2. Card-driven template from the new KB (preferred).
  3. RuleRegistry-driven template (legacy fallback for any signal not yet
     migrated to a SignalCard).
  4. None — log a warning; caller decides to skip or substitute a safe exit.

When a template has placeholders (e.g. {window}) that are absent from `params`,
the function fills them from the card's `params_by_timeframe[timeframe]` defaults
before giving up, so a signal picked without explicit param extraction still
renders correctly instead of being silently dropped.
"""
from __future__ import annotations

import logging
import re

from app.kb import kb
from stretus_knowledge_base.stretus_kb.registry import RuleRegistry

logger = logging.getLogger(__name__)

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _fill_defaults(template: str, params: dict, card, timeframe: str | None) -> dict:
    """Return a copy of params with any missing template keys filled from card defaults."""
    needed = set(_PLACEHOLDER_RE.findall(template))
    missing = needed - params.keys()
    if not missing:
        return params

    # Build a flat defaults dict: prefer the exact timeframe, then any timeframe.
    defaults: dict = {}
    if card is not None:
        by_tf: dict = card.params_by_timeframe or {}
        if timeframe and timeframe in by_tf:
            defaults.update(by_tf[timeframe])
        else:
            # Fall back to the first available timeframe entry.
            for tf_params in by_tf.values():
                defaults.update(tf_params)
                break

    filled = {**params}
    for key in missing:
        if key in defaults:
            filled[key] = defaults[key]
    return filled


def render_formula(name: str, params: dict, timeframe: str | None = None) -> str | None:
    if name == "gap_up_open":
        gap_factor = round(1 + float(params.get("gap_pct", 0.5)) / 100.0, 6)
        return f"OPEN > PREV(CLOSE, 1) * {gap_factor}"
    if name == "gap_down_open":
        gap_factor = round(1 - float(params.get("gap_pct", 0.5)) / 100.0, 6)
        return f"OPEN < PREV(CLOSE, 1) * {gap_factor}"

    card = kb.signals.get(name)
    template = card.formula if card else RuleRegistry.get_formula(name)

    if template:
        merged = _fill_defaults(template, params, card, timeframe)
        try:
            return template.format(**merged)
        except KeyError as exc:
            logger.warning(
                "signal_formula|format_error|signal=%s|missing_key=%s|timeframe=%s",
                name, exc, timeframe,
            )

    logger.error(
        "signal_formula|no_template|signal=%s — cannot render condition string; "
        "it will be SKIPPED.",
        name,
    )
    return None
