"""
app/planner/intent_extractor.py — Free-text goal → structured Intent.

The ONLY job of this module is natural-language understanding.
The LLM must emit values from the closed taxonomy in intent_taxonomy.yaml;
anything else is rejected and the planner falls back to taxonomy.defaults.

This is the single LLM call in the entire planner pipeline. Everything
downstream of here is deterministic.
"""
from __future__ import annotations

import json
import logging
import re

from app.kb.schemas import Intent, IntentTaxonomy

logger = logging.getLogger(__name__)


class IntentExtractor:
    """Extract structured Intent from a user's free-text goal."""

    def __init__(self, llm_service=None):
        # Late-imported to avoid circular import at module load
        if llm_service is None:
            from app.services.ai.llm import LLMService
            llm_service = LLMService()
        self._llm = llm_service

    # ── Public API ───────────────────────────────────────────────────────────

    async def extract(self, goal: str | None, taxonomy: IntentTaxonomy) -> Intent:
        """Return a validated Intent. On any failure, returns taxonomy.defaults."""
        defaults = Intent(**taxonomy.defaults)

        if not goal or not goal.strip():
            logger.info("intent_extractor|empty_goal|using_defaults=%s", defaults.model_dump())
            return defaults

        try:
            payload = await self._call_llm(goal, taxonomy)
        except Exception as exc:
            logger.warning(
                "intent_extractor|llm_call_failed|goal=%r|err=%s — using defaults",
                goal[:60], exc,
            )
            return defaults

        sanitized = self._sanitize(payload, taxonomy, defaults)
        try:
            intent = Intent(**sanitized)
        except Exception as exc:
            logger.warning(
                "intent_extractor|invalid_intent|raw=%r|err=%s — using defaults",
                payload, exc,
            )
            return defaults

        logger.info(
            "intent_extractor|extracted|goal=%r|intent=%s",
            goal[:60], intent.model_dump(),
        )
        return intent

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _system_prompt(self, taxonomy: IntentTaxonomy) -> str:
        field_lines = "\n".join(
            f"- {name}: one of {options}"
            for name, options in taxonomy.fields.items()
        )
        return (
            "You extract structured trading intent from a user's free-text goal.\n"
            "Return ONLY valid JSON with these exact fields:\n\n"
            f"{field_lines}\n\n"
            "Rules:\n"
            "- Use ONLY values from the lists above. Never invent new values.\n"
            "- The word 'short' means short-duration (quick) UNLESS the user\n"
            "  explicitly says 'short selling' or 'go short'. 'Short profit' = small profit.\n"
            "- 'Quick trades' / 'scalping' / 'fast' → frequency: high or very_high, hold_horizon: minutes or seconds.\n"
            "- 'Hold for days' / 'long term' → hold_horizon: days or weeks.\n"
            "- If the goal is vague, pick reasonable defaults.\n"
            'Return JSON exactly like: {"hold_horizon": "minutes", "frequency": "high",'
            ' "profit_size": "small", "style": "scalping", "risk_appetite": "conservative"}'
        )

    async def _call_llm(self, goal: str, taxonomy: IntentTaxonomy) -> dict:
        messages = [
            {"role": "system", "content": self._system_prompt(taxonomy)},
            {"role": "user",   "content": f"User goal: {goal}"},
        ]
        raw = await self._llm.chat(messages)
        return self._parse_json(raw)

    # ── Parsing & validation ─────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict:
        if not text:
            return {}
        stripped = text.strip()
        # Strip ```json ... ``` fences if present
        if stripped.startswith("```"):
            stripped = re.sub(
                r"^```(?:json)?\s*|\s*```$", "", stripped,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        return {}

    @staticmethod
    def _sanitize(
        payload: dict,
        taxonomy: IntentTaxonomy,
        defaults: Intent,
    ) -> dict:
        """Replace out-of-vocab values with defaults — Intent constructor would
        otherwise reject the whole payload for one bad field."""
        sanitized: dict = {}
        defaults_dict = defaults.model_dump()
        for field, allowed in taxonomy.fields.items():
            value = payload.get(field)
            if isinstance(value, str) and value in allowed:
                sanitized[field] = value
            else:
                sanitized[field] = defaults_dict[field]
        return sanitized
