"""
app/services/discovery/llm_extractor.py
───────────────────────────────────────
Phase 9l — LLM-based primitive extractor for discovery prompts.

The chat layer's regex extractor (`_extract_discovery_conditions`) is
fast and deterministic for common trader phrasings, but it can't
realistically cover every variant a real user might type. This module
fills the long tail by asking the configured LLM (Groq / Ollama via
`LLMService`) to map the user's prose onto the same primitive library
the regex uses — so both extractors produce structurally identical
output and the orchestrator doesn't care which path produced it.

Design:

  • The primitive library (`app.services.discovery.primitives.PRIMITIVES`)
    is the single source of truth. The LLM gets it as a JSON-schema
    tool definition with the primitive names as an enum, so it can
    only emit names that actually exist.

  • A small system prompt frames the task: "extract trading conditions",
    "use only the listed primitives", "include params when the user
    typed numbers", "return [] if the message is unrelated".

  • The LLM is asked to call ONE tool with the parsed list. We don't
    use multi-turn chat — one shot, one extraction.

  • Output is validated against the library before it's returned.
    Unknown primitive names are dropped, non-numeric params are
    dropped (so a hallucinated "rsi_above with threshold='high'"
    becomes a silent no-op rather than crashing the scanner).

  • All failures (no API key configured, network error, tool didn't
    fire, JSON malformed) return [] so the caller can fall back to
    the regex path. NEVER raises.

NO SIDE EFFECTS — pure async function. Caching is the caller's
responsibility (chat layer caches by message text).
"""
from __future__ import annotations

import json
import logging
from typing import Any

from app.services.discovery.primitives import PRIMITIVES

logger = logging.getLogger(__name__)


# Re-exported at module level so tests can patch
# `app.services.discovery.llm_extractor.LLMService` directly without
# stubbing the entire `app.services.ai.llm` import path. The runtime
# code below references this name, falling back to a fresh import if
# the module attribute is missing for any reason.
try:
    from app.services.ai.llm import LLMService as _LLMServiceImport
    LLMService = _LLMServiceImport
except Exception:
    LLMService = None  # type: ignore[assignment]


# ── System prompt + tool definition ─────────────────────────────────────────


_EXTRACTION_SYSTEM_PROMPT = """You extract trading conditions from a user's natural-language description of a stock-screening or strategy idea.

Your ONLY job is to call the `set_discovery_conditions` tool with a list of structured conditions matching the primitives in the tool schema.

Rules:
1. Use ONLY the primitive names listed in the tool's enum. Do not invent new names.
2. When the user mentions a numeric threshold ("1.2x volume", "RSI above 60", "within 5% of 52-week high", "20-day high", "above 50 EMA"), include it in the primitive's params.
3. When the user mentions a primitive but no number, OMIT the params (the orchestrator applies the primitive's default).
4. Distinguish "near 52-week high" (`near_52w_high`) from "breaking 52-week high" (`above_52w_high`). The former is proximity, the latter is a fresh breakout.
5. "low pullback", "shallow pullback", "pullback < 1%" all map to `shallow_pullback_long` UNLESS the surrounding context is bearish/short ("bearish", "short", "weak recovery"), in which case use `shallow_pullback_short`.
6. "high volume", "strong volume", "volume spike" without a number → `volume_spike` (default 2× will be applied).
7. Return an EMPTY list when the message is not about trading conditions ("what is RSI?", "show me the tutorial", "hi").
8. Do not include conditions the user did not mention. If the user only typed about volume, do NOT add 52-week or pullback conditions.
9. The order of conditions in the list does not matter — every condition must hold simultaneously (logical AND).

Examples:
- "create intraday strategy on NSE stock whose volume spike up today 1.5x" → [{"name":"volume_spike","params":{"multiplier":1.5}}]
- "stocks with RSI above 70 and price above VWAP" → [{"name":"rsi_above","params":{"threshold":70}}, {"name":"above_vwap"}]
- "near 20-day high with 2x volume" → [{"name":"near_52w_high","params":{"window":20}}, {"name":"volume_spike","params":{"multiplier":2.0}}]
- "what is the difference between EMA and SMA?" → []
"""


def build_extraction_tool() -> dict:
    """Build the JSONSchema tool definition fed to the LLM. The
    primitive library is the source of truth: the enum and the
    parameter docs are both derived from PRIMITIVES so adding a new
    primitive automatically extends the LLM's vocabulary."""
    primitive_choices = sorted(PRIMITIVES.keys())
    primitive_help = "\n".join(
        f"  • {name}: {p.description}"
        + (
            "  default params: "
            + ", ".join(f"{k}={v}" for k, v in p.default_params.items())
            if p.default_params
            else ""
        )
        for name, p in PRIMITIVES.items()
    )
    return {
        "type": "function",
        "function": {
            "name": "set_discovery_conditions",
            "description": (
                "Record the structured trading conditions extracted from "
                "the user's prose. Each condition references one primitive "
                "name and may include numeric params. Available primitives:\n"
                + primitive_help
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "conditions": {
                        "type": "array",
                        "description": (
                            "Ordered list of primitives the user explicitly "
                            "mentioned. Empty when the message is not about "
                            "trading conditions."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "enum": primitive_choices,
                                    "description": (
                                        "The primitive identifier (must be one "
                                        "of the enum values)."
                                    ),
                                },
                                "params": {
                                    "type": "object",
                                    "description": (
                                        "Numeric parameters for the primitive "
                                        "(e.g. multiplier=1.5, threshold=60, "
                                        "window=20). Omit when the user did "
                                        "not specify numbers."
                                    ),
                                    "additionalProperties": {"type": "number"},
                                },
                            },
                            "required": ["name"],
                        },
                    }
                },
                "required": ["conditions"],
            },
        },
    }


# ── Public extractor ────────────────────────────────────────────────────────


async def extract_via_llm(message: str) -> list[dict[str, Any]]:
    """Map `message` onto a list of `{name, params}` dicts using the
    configured LLM. Returns [] on any failure (no API key, network
    error, malformed tool output, etc.) so the caller can fall back
    to the regex extractor without special-casing.
    """
    text = (message or "").strip()
    if not text:
        return []

    if LLMService is None:
        logger.warning("llm_extractor|llm_service_unavailable")
        return []

    tool = build_extraction_tool()
    messages = [
        {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    try:
        llm = LLMService()
    except Exception as exc:
        logger.warning("llm_extractor|init_failed|err=%s", exc)
        return []

    try:
        result = await llm.chat_with_tools(
            messages, [tool], tool_choice={"type": "function",
                                           "function": {"name": "set_discovery_conditions"}},
            fast=True,
        )
    except Exception as exc:
        # AppError, network, rate limit — caller decides whether to retry.
        # We just say "the LLM didn't help this turn" so the regex
        # extractor's result is what gets used.
        logger.info("llm_extractor|llm_call_failed|err=%s", exc)
        return []

    raw_calls = (result or {}).get("tool_calls") or []
    if not raw_calls:
        logger.info("llm_extractor|llm_returned_no_tool_call")
        return []

    args = raw_calls[0].get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (TypeError, ValueError):
            logger.warning("llm_extractor|tool_args_not_json")
            return []
    raw_conditions = args.get("conditions") if isinstance(args, dict) else None
    if not isinstance(raw_conditions, list):
        logger.warning("llm_extractor|conditions_not_a_list")
        return []

    return validate_conditions(raw_conditions)


def validate_conditions(raw: list[Any]) -> list[dict[str, Any]]:
    """Strip out anything the LLM might emit that doesn't match the
    primitive library exactly. Defensive: returns a clean list ready
    for the orchestrator without raising on malformed input."""
    cleaned: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name or name not in PRIMITIVES:
            continue
        raw_params = item.get("params") if isinstance(item.get("params"), dict) else {}
        params: dict[str, float] = {}
        for k, v in raw_params.items():
            if not isinstance(k, str) or not k.strip():
                continue
            try:
                params[k.strip()] = float(v)
            except (TypeError, ValueError):
                continue
        cleaned.append({"name": name, "params": params})
    return cleaned
