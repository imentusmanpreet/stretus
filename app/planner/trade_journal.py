"""
app/planner/trade_journal.py — Phase 4 trade journaling (rule #25).

Two journal-entry shapes:

  • signal_generated     — written every time the planner produces or
                           updates a signal plan. Carries the strategy
                           context (what indicators triggered, the setup
                           grade, the entry condition, the SL/TP plan).
  • trade_outcome        — written when a trade actually completes
                           (or, in the backtest path, when the simulator
                           records the exit). Carries entry / exit prices,
                           P&L, outcome label, time stamps, and any
                           mistakes the post-mortem step flagged.

The journal lives on the StrategyBuilder under `builder.trade_journal`
(a chronological list). It also gets pushed into the ChatMessage
strategy_draft on every turn so it survives across sessions, and the
final strategy_json carries it too — so post-trade reviews can read
straight from the message history without a separate datastore.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ── Entry data classes ────────────────────────────────────────────────────────


@dataclass
class JournalEntry:
    id:           str
    kind:         str               # "signal_generated" | "trade_outcome"
    created_at:   str               # ISO-8601 UTC
    payload:      dict[str, Any]    # see helpers below for shape
    notes:        str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Signal-generation entry ───────────────────────────────────────────────────


def build_signal_entry(
    *,
    builder: Any,
    plan: dict[str, Any],
    setup_grade: str | None = None,
    confirmation_factors: Iterable[str] = (),
    notes: str = "",
) -> JournalEntry:
    """Compose the entry written when the planner produces a new signal
    plan. We capture everything needed to reconstruct WHY the signal was
    generated — entry condition, exit condition, SL/TP, mirror plan,
    trade-management overrides, signal-filter status, and the user-
    selected setup grade."""
    payload: dict[str, Any] = {
        "symbol":                 getattr(builder, "symbol", None),
        "timeframe":              getattr(builder, "timeframe", None),
        "objective":              getattr(builder, "objective", None),
        "sentiment":              getattr(builder, "sentiment", None),
        "goal":                   getattr(builder, "goal", None),
        "entry_condition":        plan.get("entry_condition"),
        "exit_condition":         plan.get("exit_condition"),
        "signals_used":           plan.get("signals_used"),
        "stop_loss_spec":         plan.get("_stop_loss_spec"),
        "trailing_stop_spec":     plan.get("_trailing_stop_spec"),
        "rr_ratio":               plan.get("_rr_ratio"),
        "htf_rules":              plan.get("_htf_rules"),
        "reference_symbol":       plan.get("_reference_symbol"),
        "required_indicators":    plan.get("_required_indicators"),
        "signal_filters":         plan.get("_signal_filters"),
        "trade_management":       plan.get("_trade_management"),
        "market_context":         plan.get("_market_context"),
        "mirror_plan":            plan.get("_mirror_plan"),
        "setup_grade":            setup_grade,
        "confirmation_factors":   list(confirmation_factors),
    }
    return JournalEntry(
        id=str(uuid.uuid4()),
        kind="signal_generated",
        created_at=_now_iso(),
        payload=payload,
        notes=notes,
    )


# ── Trade-outcome entry ───────────────────────────────────────────────────────


@dataclass
class TradeOutcomeInput:
    """The minimum shape a caller has to provide when logging a completed
    trade. Mirrors the fields the spec called out:

      - reason for entry
      - setup grade
      - indicators that triggered
      - entry / exit price
      - P&L
      - outcome
      - time of entry / exit
      - mistakes observed
    """
    entry_time:   str
    entry_price: float
    exit_time:    str
    exit_price:  float
    pnl:          float
    outcome:      str                          # "win" | "loss" | "scratch"
    direction:    str = "long"
    reason_for_entry:     str = ""
    setup_grade:          str | None = None
    indicators_triggered: list[str] = field(default_factory=list)
    mistakes_observed:    list[str] = field(default_factory=list)
    extra:                dict[str, Any] = field(default_factory=dict)


def build_outcome_entry(
    *,
    builder: Any,
    outcome: TradeOutcomeInput,
    notes: str = "",
) -> JournalEntry:
    payload = {
        "symbol":      getattr(builder, "symbol", None),
        "timeframe":   getattr(builder, "timeframe", None),
        "direction":   outcome.direction,
        "entry": {
            "time":  outcome.entry_time,
            "price": outcome.entry_price,
            "reason": outcome.reason_for_entry,
            "indicators_triggered": list(outcome.indicators_triggered),
            "setup_grade": outcome.setup_grade,
        },
        "exit": {
            "time":  outcome.exit_time,
            "price": outcome.exit_price,
        },
        "pnl":               outcome.pnl,
        "outcome":           outcome.outcome,
        "mistakes_observed": list(outcome.mistakes_observed),
        "extra":             dict(outcome.extra or {}),
    }
    return JournalEntry(
        id=str(uuid.uuid4()),
        kind="trade_outcome",
        created_at=_now_iso(),
        payload=payload,
        notes=notes,
    )


# ── Builder-facing helpers ────────────────────────────────────────────────────


def append_to_builder(builder: Any, entry: JournalEntry) -> None:
    """Push a journal entry onto builder.trade_journal — creating the
    attribute if the builder didn't have it yet. Caps the in-memory list
    at 200 entries so the persisted draft stays a sensible size."""
    journal: list[dict[str, Any]] = list(getattr(builder, "trade_journal", []) or [])
    journal.append(entry.to_dict())
    if len(journal) > 200:
        journal = journal[-200:]
    builder.trade_journal = journal
    logger.info(
        "trade_journal|event=entry_logged|kind=%s|id=%s|symbol=%s",
        entry.kind,
        entry.id,
        entry.payload.get("symbol"),
    )


def confirmation_factors_from_plan(plan: dict[str, Any]) -> list[str]:
    """Best-effort enumeration of the confirmation factors active in a plan
    — used by the setup-grade logic and by the journal entry's
    `confirmation_factors` field."""
    factors: list[str] = []
    entry = (plan.get("entry_condition") or "").upper()
    if "ADX(" in entry:
        factors.append("trending market (ADX)")
    if "VOLUME" in entry:
        factors.append("volume confirmation")
    if "CHOPPINESS(" in entry:
        factors.append("volatility cap")
    if "RSI(" in entry:
        factors.append("RSI threshold")
    if "STOCH_K(" in entry:
        factors.append("stochastic confirmation")
    if "EMA(" in entry:
        factors.append("EMA alignment")
    if plan.get("_htf_rules"):
        factors.append("higher-timeframe trend confirmation")
    mc = plan.get("_market_context") or {}
    if (mc.get("broader_market_direction") or {}).get("enabled"):
        factors.append("broader-market direction match")
    if (mc.get("sector_strength") or {}).get("enabled"):
        factors.append("sector strength match")
    if (mc.get("candlestick_pattern") or {}).get("enabled"):
        factors.append("candlestick pattern confirmation")
    if plan.get("_stop_loss_spec"):
        factors.append("structural stop loss anchor")
    if plan.get("_rr_ratio"):
        factors.append(f"R:R = 1:{plan['_rr_ratio']}")
    return factors


def grade_setup(plan: dict[str, Any]) -> str:
    """Map the count of confirmation factors to a coarse grade.

      6+ confirmations → A+
      4-5             → A
      2-3             → B
      0-1             → C (skip or take with reduced size)
    """
    n = len(confirmation_factors_from_plan(plan))
    if n >= 6:
        return "A+"
    if n >= 4:
        return "A"
    if n >= 2:
        return "B"
    return "C"


# ── Summary rendering ────────────────────────────────────────────────────────


def render_journal_for_summary(builder: Any) -> list[str]:
    """Return a short summary of the most recent journal entries — used in
    the comprehensive prompt summary so the user can see the audit trail
    starting to take shape."""
    journal = getattr(builder, "trade_journal", []) or []
    if not journal:
        return [
            "── Trade journal ──",
            "Auto-journaling is enabled. Every signal generated and every "
            "trade completed will be logged with full context (indicators, "
            "setup grade, prices, P&L, observed mistakes).",
        ]
    lines: list[str] = ["── Trade journal (latest entries) ──"]
    for entry in journal[-3:]:
        ts = entry.get("created_at", "")
        kind = entry.get("kind", "?")
        payload = entry.get("payload") or {}
        if kind == "signal_generated":
            grade = payload.get("setup_grade") or "(ungraded)"
            factors = ", ".join(payload.get("confirmation_factors") or []) or "(no factors recorded)"
            lines.append(f"  {ts} — signal_generated, grade {grade}")
            lines.append(f"      factors: {factors}")
        elif kind == "trade_outcome":
            outcome = payload.get("outcome", "?")
            pnl = payload.get("pnl")
            lines.append(f"  {ts} — trade_outcome={outcome}, P&L={pnl}")
    return lines


__all__ = [
    "JournalEntry",
    "TradeOutcomeInput",
    "append_to_builder",
    "build_outcome_entry",
    "build_signal_entry",
    "confirmation_factors_from_plan",
    "grade_setup",
    "render_journal_for_summary",
]
