"""
app/core/step_logger.py — the single, icon-led step logger for the
Propose → Verify → Render → Backtest → Reply pipeline (design §11).

ONE structured, greppable line per *meaningful* step so a tester can read a whole
turn top-to-bottom and see exactly what happened and why.

    <icon> <stage> | sid=<sid8> t=<turn> | <event> | k=v k=v

Example:

    🛡️ verify  | sid=d0881d27 t=1 | repair.market | crypto -> equity_cash reason=from_symbol

Rules (enforced by convention, not code):
  * Every meaningful step in the new pipeline calls `log_step` — no raw
    `print` / ad-hoc `logger.info(f"...")` in the Propose→Verify→Render path.
  * Logs and the user readback derive from the SAME coverage object, so they
    can never disagree.

Verbosity is gated by the STEP_LOG_LEVEL env var (a standard logging level
name: DEBUG/INFO/WARNING/...). Default INFO. All step logs emit at INFO.
"""
from __future__ import annotations

import logging
import os
from typing import Final

logger = logging.getLogger("stretus.steps")

# ── Stage icons (design §11) ──────────────────────────────────────────────────
PROPOSE: Final = "🧠"
VERIFY: Final = "🛡️"
RENDER: Final = "🏗️"
BACKTEST: Final = "📊"
REPLY: Final = "💬"

# ── Action icons ──────────────────────────────────────────────────────────────
PASS: Final = "✅"
REPAIR: Final = "🔧"
INFERRED: Final = "💡"
DEFAULT: Final = "⚙️"
CLARIFY: Final = "❓"
SELF_HEAL: Final = "♻️"
BLOCK: Final = "⛔"

# Canonical stage label that follows the icon. Kept fixed-width-ish so columns
# line up when scanning a turn.
_STAGE_LABELS: Final[dict[str, str]] = {
    PROPOSE: "propose",
    VERIFY: "verify",
    RENDER: "render",
    BACKTEST: "backtest",
    REPLY: "reply",
}


def _resolve_level() -> int:
    raw = (os.environ.get("STEP_LOG_LEVEL") or "INFO").strip().upper()
    return logging.getLevelName(raw) if isinstance(logging.getLevelName(raw), int) else logging.INFO


def _short_sid(sid: str | None) -> str:
    s = str(sid or "").strip()
    return s[:8] if s else "--------"


def _format_kv(kv: dict) -> str:
    """Render keyword pairs as `k=v` space-joined, in call order.

    Values are stringified compactly; spaces inside a value are preserved (the
    line is meant to be human-read, not machine-parsed field-by-field).
    """
    parts = []
    for key, value in kv.items():
        parts.append(f"{key}={value}")
    return " ".join(parts)


def log_step(icon: str, stage: str, sid: str | None, turn: int, event: str, **kv) -> None:
    """Emit one pipeline step line.

    Args:
        icon: an action/stage icon constant from this module (e.g. REPAIR).
        stage: a stage label, normally one of the stage-icon labels. If `icon`
            is itself a stage icon and `stage` is omitted/blank, the canonical
            label is used.
        sid: session id (truncated to 8 chars in the output).
        turn: 1-based turn number.
        event: dotted event name (e.g. "repair.market", "gate_pass").
        **kv: structured key=value context appended to the line.
    """
    level = _resolve_level()
    if not logger.isEnabledFor(level):
        return

    stage_label = (stage or "").strip() or _STAGE_LABELS.get(icon, "")
    head = f"{icon} {stage_label:<8} | sid={_short_sid(sid)} t={turn} | {event}"
    tail = _format_kv(kv)
    line = f"{head} | {tail}" if tail else head
    logger.log(level, line)
