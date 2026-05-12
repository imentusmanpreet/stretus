"""
app/planner/trace.py — Structured decision log for one planner run.

Every plan() call produces one DecisionTrace. The trace records:
  - the structured intent extracted from the user's free-text goal
  - which candidate signals were eliminated by hard filters and why
  - the soft-rank score breakdown for surviving candidates
  - the final picks
  - param resolution source (card / ohlcv estimator / perf cache)
  - validation outcome

The trace is logged at INFO and exposed at /api/v1/strategy/debug/last-plan
for production debugging without log scraping.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EliminationRecord:
    signal: str
    reason: str


@dataclass
class ScoreRecord:
    signal: str
    score: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class DecisionTrace:
    intent: dict | None = None
    candidates_initial: dict[str, list[str]] = field(default_factory=dict)
    eliminated: dict[str, list[EliminationRecord]] = field(default_factory=dict)
    rankings: dict[str, list[ScoreRecord]] = field(default_factory=dict)
    picks: dict[str, str] = field(default_factory=dict)
    params_source: dict[str, str] = field(default_factory=dict)
    sl_tp: dict[str, float] = field(default_factory=dict)
    validation: str = "pending"
    errors: list[str] = field(default_factory=list)

    # ── Mutators ─────────────────────────────────────────────────────────────

    def set_intent(self, intent: Any) -> None:
        self.intent = intent.model_dump() if hasattr(intent, "model_dump") else dict(intent)

    def record_initial(self, role: str, names: list[str]) -> None:
        self.candidates_initial[role] = list(names)

    def record_eliminations(self, role: str, eliminations: list[tuple[str, str]]) -> None:
        self.eliminated[role] = [EliminationRecord(s, r) for s, r in eliminations]

    def record_ranking(self, role: str, ranking: list[ScoreRecord]) -> None:
        self.rankings[role] = list(ranking)

    def record_pick(self, role: str, signal_name: str) -> None:
        self.picks[role] = signal_name

    def record_param_source(self, signal_name: str, source: str) -> None:
        self.params_source[signal_name] = source

    def record_sl_tp(self, sl_pct: float, tp_pct: float, vol_mult: float | None = None) -> None:
        self.sl_tp = {
            "sl_pct": round(sl_pct, 4),
            "tp_pct": round(tp_pct, 4),
        }
        if vol_mult is not None:
            self.sl_tp["vol_mult"] = round(vol_mult, 4)

    def mark_validated(self) -> None:
        self.validation = "passed"

    def mark_failed(self, reason: str) -> None:
        self.validation = "failed"
        self.errors.append(reason)

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def log(self, level: int = logging.INFO) -> None:
        logger.log(level, "planner_trace=%s", self.to_json())


# ── Per-session trace store (for /api/v1/strategy/debug/last-plan) ───────────


from collections import OrderedDict
from threading import Lock

_TRACE_STORE_MAX = 256
_trace_store: "OrderedDict[str, dict]" = OrderedDict()
_trace_lock = Lock()


def remember_trace(session_id: str | None, trace_dict: dict) -> None:
    """Store the most recent trace for a session. Bounded LRU; thread-safe."""
    if not session_id:
        return
    with _trace_lock:
        _trace_store[session_id] = trace_dict
        _trace_store.move_to_end(session_id)
        while len(_trace_store) > _TRACE_STORE_MAX:
            _trace_store.popitem(last=False)


def recall_trace(session_id: str) -> dict | None:
    with _trace_lock:
        return _trace_store.get(session_id)
