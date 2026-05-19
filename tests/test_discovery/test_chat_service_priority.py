"""Phase 9e — chat_service.py priority + early-preset-detection guards.

These tests stay at the source-text level (the chat_service.py module
can't be imported in the local test env because of the asyncpg gap, and
patching the agent router for an end-to-end async test would require
hundreds of lines of fixtures). The two source-level invariants we
enforce here both directly correspond to bugs the user surfaced:

  1. Preset detection must happen BEFORE the AgentRouter call so the
     agent sees the full builder state and doesn't ask the user for a
     stock that discovery would supply.
  2. The discovery_step elif must precede grounded_clarification_reply
     and direct_llm_reply in the response-building elif chain so the
     agent's "ask for symbol" clarification doesn't short-circuit the
     discovery prompt.

If either invariant breaks, the user's "create intraday strategy on NSE
stock whose volume spike up..." prompt gets a "Which specific NSE stock?"
reply instead of the discovery flow — exactly the bug we fixed.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_CHAT_SERVICE_PATH = (
    Path(__file__).resolve().parents[2]
    / "app" / "services" / "chat" / "chat_service.py"
)


@pytest.fixture(scope="module")
def chat_service_source() -> str:
    return _CHAT_SERVICE_PATH.read_text(encoding="utf-8")


def _line_index(text: str, marker: str) -> int:
    """Return the 0-based line index of the first occurrence of `marker`,
    or raise an explicit error pointing the developer at the broken
    invariant."""
    for i, line in enumerate(text.splitlines()):
        if marker in line:
            return i
    raise AssertionError(
        f"chat_service.py source does not contain expected marker {marker!r}"
    )


# ── Invariant 1: preset detection runs before AgentRouter ────────────────────


def test_early_preset_detection_runs_before_agent_router(chat_service_source):
    """Phase 9e fix (a). The detect_preset_in_text call must appear in the
    source BEFORE the AgentRouter().decide(...) call. Otherwise the agent
    sees no preset, treats symbol as missing, and asks the user for one."""
    detect_idx = _line_index(chat_service_source, "detect_preset_in_text(user_content)")
    decide_idx = _line_index(chat_service_source, "await AgentRouter().decide(")
    assert detect_idx < decide_idx, (
        "detect_preset_in_text must appear before AgentRouter().decide("
        f" (detect at line {detect_idx + 1}, decide at line {decide_idx + 1}). "
        "Phase 9e fix (a) regression."
    )


def test_early_preset_detection_writes_to_main_builder(chat_service_source):
    """The early detection must write into the main builder.strategy_preset
    field so the agent (which receives the builder) sees it."""
    # Find the early-detection block and confirm it assigns to
    # builder.strategy_preset (not parsed_builder.strategy_preset).
    src = chat_service_source
    # The block lives between the risk-config-hydrated log and the agent
    # router call — search there.
    anchor = "event=preset_detected_pre_router"
    assert anchor in src, (
        "Phase 9e early-preset-detection log marker missing — fix (a) regressed"
    )
    # The assignment must mutate the main `builder`, not a local stand-in.
    assert "builder.strategy_preset = _early_preset.name" in src, (
        "Phase 9e early preset assignment must write to builder.strategy_preset"
    )


# ── Invariant 2: discovery_step elif precedes clarification + direct LLM ────


def test_discovery_step_elif_precedes_grounded_clarification_reply(chat_service_source):
    """Phase 9e fix (b) — discovery wins over the agent's clarification."""
    discovery_idx = _line_index(
        chat_service_source,
        "elif discovery_step.handled and not discovery_step.symbol_resolved:",
    )
    grounded_idx = _line_index(
        chat_service_source, "elif grounded_clarification_reply:"
    )
    assert discovery_idx < grounded_idx, (
        "discovery_step elif must precede grounded_clarification_reply "
        f"(discovery at line {discovery_idx + 1}, grounded at line "
        f"{grounded_idx + 1}). Phase 9e fix (b) regression."
    )


def test_discovery_step_elif_precedes_direct_llm_reply(chat_service_source):
    """The same priority must hold against direct_llm_reply — the agent's
    raw reply text would otherwise short-circuit discovery."""
    discovery_idx = _line_index(
        chat_service_source,
        "elif discovery_step.handled and not discovery_step.symbol_resolved:",
    )
    direct_idx = _line_index(chat_service_source, "elif direct_llm_reply:")
    assert discovery_idx < direct_idx, (
        "discovery_step elif must precede direct_llm_reply "
        f"(discovery at line {discovery_idx + 1}, direct at line "
        f"{direct_idx + 1}). Phase 9e fix (b) regression."
    )


# ── Sanity guards on the helper plumbing ────────────────────────────────────


def test_chat_service_imports_discovery_helpers(chat_service_source):
    """Whatever else changes, the discovery helpers must still be imported."""
    assert "from app.services.discovery.chat_integration import (" in chat_service_source
    assert "handle_pending_tie_break" in chat_service_source
    assert "maybe_dispatch_discovery" in chat_service_source


def test_discovery_step_is_computed_before_elif_chain(chat_service_source):
    """The two `await` calls that compute discovery_step must run BEFORE
    the elif chain references it. Source-level check that the variable is
    actually set when the elif evaluates."""
    src = chat_service_source
    compute_idx = _line_index(
        src, "discovery_step = await handle_pending_tie_break(builder, user_content)"
    )
    use_idx = _line_index(
        src, "elif discovery_step.handled and not discovery_step.symbol_resolved:"
    )
    assert compute_idx < use_idx, (
        "discovery_step must be computed BEFORE it is referenced in the "
        f"elif chain (compute at line {compute_idx + 1}, use at line {use_idx + 1})"
    )
