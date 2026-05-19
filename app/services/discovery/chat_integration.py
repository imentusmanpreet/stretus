"""
app/services/discovery/chat_integration.py
──────────────────────────────────────────
Phase 9b — thin glue between the discovery orchestrator and the chat
service. Exists so chat_service.py only needs ~10 lines of conditional
dispatch instead of a sprawling state machine.

Both helpers return a `DiscoveryStepResult` so the caller can short-circuit
its normal flow with a small if-block:

    step = await maybe_dispatch_discovery(builder)
    if step.handled:
        assistant_text = step.assistant_text
        assistant_state = step.assistant_state
        draft = builder.to_draft_json(mode_override=step.draft_mode, ...)
        # normal response building follows; no other branching needed

Tested in isolation in tests/test_discovery/test_chat_integration.py so
chat_service-level changes stay tiny and provably-correct logic lives
here.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable

from app.services.chat.strategy_flow import (
    build_discovery_choice_prompt,
    build_discovery_no_match_reply,
    build_discovery_resolved_reply,
)
from app.services.discovery.orchestrator import resolve_tie_break, run_discovery

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryStepResult:
    """One step in the discovery sub-flow.

    `handled` tells the caller whether the discovery layer fully owned this
    turn (caller should short-circuit and return the assistant_text). False
    means "no discovery work to do — continue with your normal flow".

    When handled=True:
      • symbol_resolved=True  → builder.symbol is set; caller MAY continue
        with normal signal-planning flow OR emit a confirmation reply first.
        We return a confirmation `assistant_text` so the user knows which
        stock got picked.
      • symbol_resolved=False → emit `assistant_text` + state and wait for
        the user's next message (which will be a tie-break choice or a
        criteria-relax instruction).
    """
    handled: bool = False
    symbol_resolved: bool = False
    assistant_text: str | None = None
    assistant_state: str | None = None
    draft_mode: str = "collect_user_input"


# ── Step 1 — tie-break-reply handler ────────────────────────────────────────


async def handle_pending_tie_break(builder, user_content: str) -> DiscoveryStepResult:
    """When `builder.discovery_pending` is True, the user's message is a
    method choice — NOT new strategy input. This routes the message to the
    orchestrator and returns the appropriate next-turn instruction."""
    if not builder.discovery_pending:
        return DiscoveryStepResult(handled=False)

    text = (user_content or "").strip()
    if not text:
        # Empty user message — re-prompt with the same options
        return DiscoveryStepResult(
            handled=True,
            assistant_text=_rebuild_choice_prompt(builder),
            assistant_state="discovery_pending",
            draft_mode="collect_user_input",
        )

    # Capture the symbol→display-name map BEFORE resolve_tie_break clears
    # builder.discovery_candidates, otherwise the confirmation reply only
    # has the bare symbol to work with.
    display_by_symbol = {
        c.get("symbol"): c.get("display_name")
        for c in (builder.discovery_candidates or [])
        if isinstance(c, dict)
    }

    ok, message = resolve_tie_break(builder, text)
    if not ok:
        # Parser couldn't recognise the choice — message is the re-prompt
        return DiscoveryStepResult(
            handled=True,
            assistant_text=message,
            assistant_state="discovery_pending",
            draft_mode="collect_user_input",
        )

    # Successfully picked a symbol. Build a confirmation reply so the user
    # sees which stock got chosen before the strategy build kicks off.
    display = display_by_symbol.get(builder.discovered_symbol) or builder.discovered_symbol
    confirmation = build_discovery_resolved_reply(
        symbol=builder.discovered_symbol,
        display_name=display,
        method=builder.discovery_chosen_method or "alphabetical",
    )
    logger.info(
        "discovery_chat|tie_break_resolved|symbol=%s|method=%s",
        builder.discovered_symbol, builder.discovery_chosen_method,
    )
    return DiscoveryStepResult(
        handled=True,
        symbol_resolved=True,
        assistant_text=confirmation,
        assistant_state="collect_user_input",
        draft_mode="collect_user_input",
    )


# ── Step 2 — fresh-discovery dispatch ────────────────────────────────────────


async def maybe_dispatch_discovery(
    builder,
    *,
    fetch_ohlcv: Callable | None = None,
) -> DiscoveryStepResult:
    """Run the discovery scan when the builder needs it. Idempotent — does
    nothing when discovery has already produced a result this session."""
    if not builder.requires_discovery():
        return DiscoveryStepResult(handled=False)

    # Don't re-run while waiting for the user's tie-break choice — handled
    # separately by handle_pending_tie_break above.
    if builder.discovery_pending:
        return DiscoveryStepResult(handled=False)

    # If a previous scan already produced a "no match" response and the
    # user hasn't changed the criteria, re-emit the same message rather
    # than burning the universe again.
    if builder.discovery_no_match:
        return DiscoveryStepResult(
            handled=True,
            assistant_text=build_discovery_no_match_reply(
                parameters_used=getattr(builder, "discovery_parameters_used", None),
                conditions_used=getattr(builder, "discovery_conditions", None),
                scan_diagnostics=getattr(builder, "discovery_last_scan_diagnostics", None),
            ),
            assistant_state="discovery_no_match",
            draft_mode="collect_user_input",
        )

    result = await run_discovery(builder, fetch_ohlcv=fetch_ohlcv)
    if result is None:
        # Preset has no discovery block, OR builder.timeframe is unset.
        # Either way we can't run a scan; let the normal flow continue.
        return DiscoveryStepResult(handled=False)

    if result.status == "single":
        # builder.symbol was set inside run_discovery. Emit a confirmation
        # so the user sees which stock got picked, then let normal flow
        # continue — the next turn will start signal planning.
        confirmation = build_discovery_resolved_reply(
            symbol=builder.discovered_symbol,
            display_name=result.candidates[0].display_name,
            method="auto (only one match)",
        )
        return DiscoveryStepResult(
            handled=True,
            symbol_resolved=True,
            assistant_text=confirmation,
            assistant_state="collect_user_input",
            draft_mode="collect_user_input",
        )

    if result.status == "none":
        return DiscoveryStepResult(
            handled=True,
            assistant_text=build_discovery_no_match_reply(
                parameters_used=getattr(builder, "discovery_parameters_used", None),
                conditions_used=getattr(builder, "discovery_conditions", None),
                scan_diagnostics=getattr(builder, "discovery_last_scan_diagnostics", None),
            ),
            assistant_state="discovery_no_match",
            draft_mode="collect_user_input",
        )

    # status == "multiple" — emit the candidate prompt and wait
    return DiscoveryStepResult(
        handled=True,
        assistant_text=build_discovery_choice_prompt(result.to_dict()),
        assistant_state="discovery_pending",
        draft_mode="collect_user_input",
    )


# ── Internals ────────────────────────────────────────────────────────────────


def _candidate_display_for(builder, symbol: str | None) -> str:
    """Look up a candidate's display name in the saved discovery state.
    Falls back to the bare symbol when state is missing."""
    if not symbol or not builder.discovery_candidates:
        return symbol or ""
    for c in builder.discovery_candidates:
        if c.get("symbol") == symbol:
            return c.get("display_name") or symbol
    return symbol


def _rebuild_choice_prompt(builder) -> str:
    """Reconstruct the same prompt the chat originally emitted, from the
    saved candidate + option dicts on the builder. Used when the user's
    reply is empty / unparseable so we re-show the choices."""
    fake_scan = {
        "candidates":          list(builder.discovery_candidates or []),
        "tie_break_options":   list(builder.discovery_tie_break_options or []),
    }
    return build_discovery_choice_prompt(fake_scan)
