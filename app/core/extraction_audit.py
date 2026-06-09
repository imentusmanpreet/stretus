"""
app/core/extraction_audit.py — Structured audit trail for slot/field extraction.

Every place that captures a user-mentioned value (symbol, timeframe, indicator
period, RMS field, etc.) should call `record_extraction()` so downstream
consumers (UI, debugger, post-hoc accuracy reviews) can see:

  * what was extracted
  * what it replaced (if anything)
  * who claims authorship (user / semantic / agent / ai_default)
  * how confident the extractor is
  * the raw evidence span from the original message

Storage: in-memory per session. The session_id is the caller's responsibility
(usually the chat session_id). The trail is exposed via `get_trail(session_id)`
for inclusion in API responses or debug payloads.

This module is intentionally framework-free — no FastAPI, no Pydantic, no DB.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_MAX_EVENTS_PER_SESSION = 500


@dataclass
class ExtractionEvent:
    """One captured field-extraction decision."""

    field: str
    new_value: Any
    source: str
    confidence: float | None = None
    old_value: Any = None
    evidence: str | None = None
    extractor: str | None = None
    ts: float = field(default_factory=time.time)
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ts_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.ts))
        return d


class _AuditStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, deque[ExtractionEvent]] = defaultdict(
            lambda: deque(maxlen=_MAX_EVENTS_PER_SESSION)
        )

    def record(self, session_id: str, event: ExtractionEvent) -> None:
        with self._lock:
            self._events[session_id].append(event)

    def trail(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [e.to_dict() for e in self._events.get(session_id, ())]

    def latest_for_field(self, session_id: str, field_name: str) -> ExtractionEvent | None:
        with self._lock:
            for evt in reversed(self._events.get(session_id, ())):
                if evt.field == field_name:
                    return evt
        return None

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._events.pop(session_id, None)


_store = _AuditStore()


def record_extraction(
    session_id: str | None,
    field: str,
    new_value: Any,
    *,
    source: str,
    confidence: float | None = None,
    old_value: Any = None,
    evidence: str | None = None,
    extractor: str | None = None,
    note: str | None = None,
) -> None:
    """Record a single field extraction.

    `source` must be one of: "user", "semantic", "agent", "preset",
    "ai_default", "user_confirmed_default", "kb", "ohlcv_estimator".

    `confidence` is on [0.0, 1.0] when known; None when not estimable.
    `evidence` is the raw substring from the user message that produced the
    value, when available.

    Conflict detection (new differs from prior latest of same field) is logged
    at WARNING level so the chat layer can surface a clarification prompt to
    the user.
    """
    if not session_id:
        return

    evt = ExtractionEvent(
        field=field,
        new_value=new_value,
        source=source,
        confidence=confidence,
        old_value=old_value,
        evidence=evidence,
        extractor=extractor,
        note=note,
    )
    _store.record(session_id, evt)

    if old_value is not None and old_value != new_value:
        logger.warning(
            "extraction_audit|conflict|session=%s|field=%s|old=%r|new=%r"
            "|source=%s|extractor=%s|evidence=%r",
            session_id, field, old_value, new_value, source, extractor, evidence,
        )
    else:
        logger.info(
            "extraction_audit|recorded|session=%s|field=%s|value=%r"
            "|source=%s|confidence=%s|extractor=%s",
            session_id, field, new_value, source, confidence, extractor,
        )


def get_trail(session_id: str | None) -> list[dict[str, Any]]:
    """Return the full extraction trail for a session (oldest first)."""
    if not session_id:
        return []
    return _store.trail(session_id)


def latest_for_field(session_id: str | None, field: str) -> ExtractionEvent | None:
    """Return the most recent extraction event for a given field, or None."""
    if not session_id:
        return None
    return _store.latest_for_field(session_id, field)


def clear_session(session_id: str | None) -> None:
    """Drop all recorded events for a session (e.g. when the chat is reset)."""
    if not session_id:
        return
    _store.clear(session_id)
