"""
Guardrails for dynamic LLM-generated assistant replies.
"""
from __future__ import annotations

import re

from app.core.assistant_responses import compose_assistant_response

_LEADING_CASUAL_RE = re.compile(
    r"^\s*(?:sure|absolutely|of course|certainly|got it|gotcha|yep|yeah|no problem|happy to help)"
    r"[\s,!.:-]+",
    re.IGNORECASE,
)
_UNSAFE_ADVICE_RE = re.compile(
    r"\b(?:you\s+should\s+buy|you\s+should\s+sell|buy\s+this|sell\s+this|"
    r"go\s+long|go\s+short|take\s+(?:a\s+)?position|strong\s+buy|strong\s+sell)\b",
    re.IGNORECASE,
)
_OVERPROMISE_RE = re.compile(
    r"\b(?:guaranteed?|risk[-\s]*free|sure[-\s]*shot|certain\s+profit|"
    r"cannot\s+lose|can't\s+lose|100%\s*(?:win|profit|return)|double\s+your\s+money)\b",
    re.IGNORECASE,
)


def _normalise_reply_text(reply_text: str) -> str:
    text = re.sub(r"\s+", " ", str(reply_text or "")).strip()
    text = _LEADING_CASUAL_RE.sub("", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _trim_sentences(text: str, max_sentences: int = 2, max_chars: int = 320) -> str:
    compact = text.strip()
    if len(compact) <= max_chars:
        return compact

    sentences = re.split(r"(?<=[.!?])\s+", compact)
    shortened = " ".join(sentences[:max_sentences]).strip()
    if shortened and len(shortened) <= max_chars:
        return shortened

    truncated = compact[: max_chars - 3].rstrip(" ,;:")
    return truncated + "..."


def guard_dynamic_assistant_reply(route_intent: str, reply_text: str | None) -> str | None:
    if not reply_text:
        return None

    guarded = _normalise_reply_text(reply_text)
    if not guarded:
        return None

    if route_intent in {"general_chat", "clarification"}:
        if _UNSAFE_ADVICE_RE.search(guarded):
            return compose_assistant_response("safety.stock_advice_boundary")
        if _OVERPROMISE_RE.search(guarded):
            return None
        return _trim_sentences(guarded)

    return guarded
