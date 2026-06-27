
"""
app/strategy/generator.py — turn conversation into a validated StrategySpec (🧠 propose + 🔧 repair).

One LLM call proposes a StrategySpec JSON from the full conversation; the strict
validator checks it; if it fails, the precise, machine-readable errors are fed back
and the model is asked to repair — up to ``max_repairs`` times. Output quality comes
from this loop + the validator, not from canned prompt examples.

Returns ``(spec, ValidationResult)``:
  * success → (StrategySpec, result.ok == True) with any non-blocking ``notes``
    (assumed values) for the chat layer to surface.
  * failure → (None, result) carrying the last blocking ``errors`` so the caller can
    ask the user to clarify instead of guessing.
"""
from __future__ import annotations

import json
import logging
import re
import time

from pydantic import ValidationError as PydanticValidationError

from app.core import step_logger
from app.strategy.spec import StrategySpec
from app.strategy.validator import ValidationError, ValidationResult, validate_spec

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)


def _extract_json_object(text: str) -> dict:
    """Pull a single JSON object out of an LLM reply (tolerates ``` fences / prose).

    Raises ValueError if nothing JSON-shaped is found or it doesn't parse.
    """
    if not text or not text.strip():
        raise ValueError("LLM returned an empty response.")
    candidate = text.strip()
    fenced = _JSON_FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    # Fall back to the outermost { ... } span if there is surrounding prose.
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in the LLM response.")
        candidate = candidate[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response was not valid JSON: {exc}") from exc


def _parse_spec(raw_text: str) -> tuple[StrategySpec | None, ValidationError | None]:
    """Parse LLM text → StrategySpec. Returns (spec, None) or (None, structured_error)."""
    try:
        data = _extract_json_object(raw_text)
    except ValueError as exc:
        return None, ValidationError("_output", "not_json", str(exc))
    try:
        return StrategySpec.model_validate(data), None
    except PydanticValidationError as exc:
        # Compact, model-readable summary of which fields are malformed.
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        return None, ValidationError("_schema", "schema_invalid", details)


def _repair_message(errors: list[ValidationError]) -> dict:
    """A user-turn that hands the blocking errors back to the model to fix."""
    body = "\n".join(f"- [{e.code}] {e.field}: {e.message}" for e in errors)
    return {
        "role": "user",
        "content": (
            "Your previous StrategySpec was rejected for these reasons. Fix ALL of them "
            "and return the corrected StrategySpec JSON only (no prose):\n"
            f"{body}"
        ),
    }


async def generate_strategy(
    history: list[dict],
    *,
    llm,
    system_prompt: str,
    max_repairs: int = 2,
    max_output_tokens: int = 8192,
    reasoning_effort: str | None = None,
    session_id: str | None = None,
    turn: int = 0,
) -> tuple[StrategySpec | None, ValidationResult]:
    """Propose a StrategySpec from ``history`` and repair until valid or exhausted.

    Args:
        history: chat messages [{role, content}, ...] (system prompt is prepended).
        llm: an object exposing ``async chat(messages, session_id, max_tokens) -> str``.
        system_prompt: the generated strategy system prompt.
        max_repairs: how many times to feed validation errors back for a fix.
        max_output_tokens: output-token budget — the spec JSON is large and reasoning
            models burn tokens thinking, so the default chat cap (2048) truncates.
    """
    messages: list[dict] = [{"role": "system", "content": system_prompt}, *history]
    last_result = ValidationResult()

    for attempt in range(max_repairs + 1):
        # ── propose ───────────────────────────────────────────────────────────
        logger.info(
            "🧠 generator | requesting spec | attempt=%d/%d | msgs=%d | budget=%d tok",
            attempt, max_repairs, len(messages), max_output_tokens,
        )
        _t0 = time.monotonic()
        try:
            raw = await llm.chat(
                messages, session_id=session_id,
                max_tokens=max_output_tokens, reasoning_effort=reasoning_effort,
            )
        except Exception as exc:  # noqa: BLE001 — LLM/transport failure: surface, don't crash the turn
            _elapsed = (time.monotonic() - _t0) * 1000.0
            logger.error(
                "❌ generator | LLM call FAILED | attempt=%d | elapsed=%.0fms | %r",
                attempt, _elapsed, exc,
            )
            last_result = ValidationResult(
                errors=[ValidationError("_llm", "llm_call_failed", f"LLM request failed: {exc}")]
            )
            step_logger.log_step(
                step_logger.BLOCK, step_logger.PROPOSE, session_id, turn,
                "llm_call_failed", attempt=attempt,
            )
            break

        # ── KEY DIAGNOSTIC ────────────────────────────────────────────────────
        # elapsed≈0ms ⇒ no real API round-trip (code/cache path), seconds ⇒ a real
        # call that came back empty. raw_len=0 pinpoints the empty-response case.
        _elapsed = (time.monotonic() - _t0) * 1000.0
        _raw_len = len(raw) if raw else 0
        _preview = (raw or "").strip().replace("\n", " ")
        if _raw_len == 0:
            logger.warning(
                "📭 generator | EMPTY reply | attempt=%d | elapsed=%.0fms | raw_len=0 "
                "(elapsed≈0 ⇒ no API round-trip; seconds ⇒ model returned empty)",
                attempt, _elapsed,
            )
        else:
            logger.info(
                "📥 generator | reply received | attempt=%d | elapsed=%.0fms | raw_len=%d",
                attempt, _elapsed, _raw_len,
            )
            # FULL, untrimmed AI response — so the exact StrategySpec JSON the model
            # produced is visible during testing (no preview truncation).
            logger.info(
                "\n┌─🧠 AI RESPONSE (attempt %d) ─────────────────────────────────────────\n"
                "%s\n"
                "└──────────────────────────────────────────────────────────────────────",
                attempt, raw,
            )

        spec, parse_error = _parse_spec(raw)
        if parse_error is not None:
            logger.warning("⚠️ generator | unparseable spec | attempt=%d | %s", attempt, parse_error.code)
            step_logger.log_step(
                step_logger.REPAIR if attempt < max_repairs else step_logger.BLOCK,
                step_logger.PROPOSE, session_id, turn,
                "parse_failed", attempt=attempt, code=parse_error.code,
            )
            last_result = ValidationResult(errors=[parse_error])
            if attempt < max_repairs:
                messages += [{"role": "assistant", "content": raw}, _repair_message([parse_error])]
                continue
            break

        # ── verify ────────────────────────────────────────────────────────────
        result = validate_spec(spec, session_id=session_id, turn=turn)
        last_result = result
        if result.ok:
            step_logger.log_step(
                step_logger.PASS, step_logger.PROPOSE, session_id, turn,
                "spec_generated", attempt=attempt, notes=len(result.notes), symbol=spec.symbol,
            )
            _kind = "🌌 DYNAMIC UNIVERSE" if spec.is_dynamic else "📈 single-symbol"
            logger.info(
                "✅ generator | valid spec on attempt=%d | %s | symbol=%s tf=%s dir=%s",
                attempt, _kind, spec.symbol, spec.timeframe, spec.direction,
            )
            # FULL validated StrategySpec — the exact contract handed downstream.
            logger.info(
                "\n┌─📋 STRATEGY SPEC (validated) ────────────────────────────────────────\n"
                "%s\n"
                "└──────────────────────────────────────────────────────────────────────",
                spec.model_dump_json(indent=2, exclude_none=True),
            )
            return spec, result

        # ── repair ────────────────────────────────────────────────────────────
        if attempt < max_repairs:
            logger.info("🔧 generator | repairing | attempt=%d | errors=%s",
                        attempt, ",".join(e.code for e in result.errors))
            step_logger.log_step(
                step_logger.REPAIR, step_logger.VERIFY, session_id, turn,
                "repair", attempt=attempt, errors=len(result.errors),
            )
            messages += [{"role": "assistant", "content": raw}, _repair_message(result.errors)]
        else:
            logger.warning("⚠️ generator | exhausted repairs | errors=%s",
                           ",".join(e.code for e in result.errors))

    return None, last_result
