"""
app/planner/sdl_flow.py — SDL pipeline entry point for chat_service.py.

Single public function:
  try_sdl_plan(prompt, builder, session_id) → SDLFlowResult

If the prompt is a strategy description, runs the full SDL pipeline:
  detect → compile_to_sdl (LLM once) → validate → compile → readback

Returns an SDLFlowResult; caller decides what to show and whether to
fall back to the legacy pipeline based on `result.used_sdl`.

Design rules
────────────
* No UI logic here. The caller owns the response text and state machine.
* Never raises. All failures return used_sdl=False so the caller can fall
  back to the legacy pipeline.
* Runs synchronously (awaited) — the caller is already in an async context.
* DESIGN-TIME LLM ONLY. No LLM is called during evaluation / backtest.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SDLFlowResult:
    """Result of try_sdl_plan."""
    used_sdl: bool = False            # False → fall back to legacy pipeline

    # SDL pipeline outputs (set when used_sdl=True)
    sdl: Any = None                   # SDL ticket
    artifact: Any = None              # StrategyArtifact
    signal_plan: dict = field(default_factory=dict)   # legacy shape for builder
    readback_text: str = ""           # plain-English summary to show the user
    match_pct: float = 0.0            # built / (built + missed) × 100
    validation_ok: bool = True
    validation_errors: list = field(default_factory=list)
    engine_gaps: list = field(default_factory=list)

    # Verification gate (WS2) — set when the gate ran.
    coverage: dict = field(default_factory=dict)      # CoverageReport.as_dict()
    gate_clarification: dict | None = None            # one pre-filled question
    gate_blocked: str | None = None                   # rare friendly hard-stop
    gate_notes: list = field(default_factory=list)    # non-blocking adjust chips

    # Failure info (set when used_sdl=False)
    skip_reason: str = ""


async def try_sdl_plan(
    prompt: str,
    builder: Any,
    *,
    session_id: str | None = None,
    force: bool = False,
) -> SDLFlowResult:
    """Run the SDL pipeline for the given prompt.

    Args:
        prompt:     The user's raw message (full strategy description).
        builder:    StrategyBuilder — read for symbol/timeframe hints;
                    written with artifact fields on success.
        session_id: For logging only.
        force:      Skip the strategy-description detector; always run SDL.
                    Use in tests.

    Returns SDLFlowResult.  Never raises.
    """
    try:
        return await _try_sdl_plan_inner(prompt, builder, session_id=session_id, force=force)
    except Exception as exc:
        logger.warning(
            "⚠️ sdl_flow | UNEXPECTED ERROR | session=%s | err=%s",
            session_id, exc, exc_info=True,
        )
        return SDLFlowResult(used_sdl=False, skip_reason=f"unexpected_error:{exc}")


async def _try_sdl_plan_inner(
    prompt: str,
    builder: Any,
    *,
    session_id: str | None,
    force: bool,
) -> SDLFlowResult:
    from app.planner.sdl_bridge import is_strategy_description
    from app.planner.sdl_selector import compile_to_sdl
    from app.planner.catalog_schema import build_menu

    # ── 1. Detection gate ──────────────────────────────────────────────────────
    if not force and not is_strategy_description(prompt):
        logger.info(
            "🔍 sdl_flow | SKIP | not a strategy description | session=%s | "
            "prompt=%.60s…",
            session_id, prompt,
        )
        return SDLFlowResult(used_sdl=False, skip_reason="not_strategy_description")

    logger.info(
        "🧠 sdl_flow | START | session=%s | prompt=%.80s…",
        session_id, prompt,
    )

    # ── 2. Build catalog menu (cached) ─────────────────────────────────────────
    try:
        menu = build_menu()
    except Exception as exc:
        logger.warning("⚠️ sdl_flow | catalog build failed | %s", exc)
        return SDLFlowResult(used_sdl=False, skip_reason=f"catalog_error:{exc}")

    # ── 3. SDL Selector (LLM call — once) ────────────────────────────────────
    try:
        sdl = await compile_to_sdl(prompt, menu, skip_flag=True)
        logger.info(
            "✅ sdl_flow | SDL CREATED | v%d | legs=%d | hash=%.8s",
            sdl.version, len(sdl.legs or []), sdl.content_hash,
        )
    except Exception as exc:
        logger.warning("⚠️ sdl_flow | selector failed | %s", exc)
        return SDLFlowResult(used_sdl=False, skip_reason=f"selector_error:{exc}")

    # ── 4-7. Validate → compile → readback → bridge (shared with modify) ───────
    return await _finalize_sdl(sdl, menu, builder, session_id=session_id)


async def _finalize_sdl(
    sdl: Any,
    menu: Any,
    builder: Any,
    *,
    session_id: str | None,
) -> SDLFlowResult:
    """Validate → compile → readback → bridge an already-built SDL.

    Shared by the create path (`_try_sdl_plan_inner`) and the modify path
    (`try_sdl_modify`) so a modify produces the SAME artifact/plan/readback as a
    fresh create — just from the merged SDL. Persists the SDL on the builder
    (`builder._sdl`) so the NEXT modify turn has something to edit, even when
    validation fails (the user can fix it with another change).
    """
    from app.planner.sdl_bridge import artifact_to_signal_plan, populate_builder_from_artifact
    from app.planner.sdl_validator import validate_sdl
    from app.planner.compiler import compile_sdl
    from app.planner.readback import build_readback, compute_match_pct

    # ── Verification gate (WS2) — runs after PROPOSE, before RENDER ─────────────
    # Repairs market/asset/anchor mismatches, enforces contradictions, normalises
    # the stop to an engine-legal form, and emits the honest coverage report.
    # Behind STRATEGY_GATE_ENABLED so it can be rolled out safely.
    gate_result = _run_gate(sdl, menu, builder, session_id=session_id)
    if gate_result is not None:
        if gate_result.blocked:
            # Rare hard-stop: show the friendly "closest I can do" message.
            setattr(builder, "_sdl", gate_result.sdl or sdl)
            return SDLFlowResult(
                used_sdl=True, sdl=gate_result.sdl or sdl, artifact=None,
                readback_text=gate_result.blocked, validation_ok=False,
                coverage=gate_result.coverage.as_dict(),
                gate_blocked=gate_result.blocked,
                gate_notes=list(gate_result.notes),
            )
        sdl = gate_result.sdl or sdl   # use the repaired, engine-safe SDL downstream

    _gate_cov = gate_result.coverage.as_dict() if gate_result else {}
    _gate_clar = gate_result.clarification if gate_result else None
    _gate_notes = list(gate_result.notes) if gate_result else []

    # ── Validation ────────────────────────────────────────────────────────────
    try:
        vr = validate_sdl(sdl, menu)
    except Exception as exc:
        logger.warning("⚠️ sdl_flow | validator failed | %s", exc)
        return SDLFlowResult(used_sdl=False, skip_reason=f"validator_error:{exc}")

    # Persist the SDL for the next modify turn regardless of validation outcome.
    setattr(builder, "_sdl", sdl)

    if not vr.ok:
        # Surface errors via readback — the caller shows them and asks the user
        # to fix the strategy. Do NOT compile.
        rb = build_readback(sdl, vr)
        built, missed, pct = compute_match_pct(sdl)
        logger.warning(
            "❌ sdl_flow | VALIDATION FAILED | errors=%d | session=%s",
            len(vr.errors), session_id,
        )
        return SDLFlowResult(
            used_sdl=True,
            sdl=sdl,
            artifact=None,
            readback_text=rb.format(),
            match_pct=pct,
            validation_ok=False,
            validation_errors=[{"field": e.field, "code": e.code, "message": e.message} for e in vr.errors],
            engine_gaps=vr.engine_gaps,
            coverage=_gate_cov,
            gate_clarification=_gate_clar,
            gate_notes=_gate_notes,
        )

    # ── Compile → Artifact ─────────────────────────────────────────────────────
    try:
        artifact = compile_sdl(sdl)
        logger.info(
            "📦 sdl_flow | ARTIFACT | id=%.8s | sym=%s | tf=%s | sl=%.2f%% | tp=%.2f%%",
            artifact.artifact_id, artifact.symbol or "dynamic",
            artifact.timeframe, artifact.stop_loss_pct, artifact.take_profit_pct,
        )
    except Exception as exc:
        logger.warning("⚠️ sdl_flow | compiler failed | %s", exc)
        return SDLFlowResult(used_sdl=False, skip_reason=f"compiler_error:{exc}")

    # ── Read-back ──────────────────────────────────────────────────────────────
    rb = build_readback(sdl, vr)
    built, missed, pct = compute_match_pct(sdl)
    readback_text = rb.format()

    logger.info(
        "📊 sdl_flow | READBACK | match=%d/%d (%.0f%%) | clarifications=%d | "
        "engine_gaps=%d | session=%s",
        built, built + missed, pct,
        len(sdl.provenance.clarifications_needed or []),
        len(vr.engine_gaps),
        session_id,
    )

    # ── Bridge artifact → legacy plan + builder ────────────────────────────────
    signal_plan = artifact_to_signal_plan(artifact, sdl)
    populate_builder_from_artifact(builder, artifact, sdl, readback_text)

    logger.info(
        "✅ sdl_flow | DONE | session=%s | match=%.0f%% | signals=%s",
        session_id, pct, signal_plan.get("signals_used", []),
    )

    return SDLFlowResult(
        used_sdl=True,
        sdl=sdl,
        artifact=artifact,
        signal_plan=signal_plan,
        readback_text=readback_text,
        match_pct=pct,
        validation_ok=True,
        engine_gaps=vr.engine_gaps,
        coverage=_gate_cov,
        gate_clarification=_gate_clar,
        gate_notes=_gate_notes,
    )


# ── Verification gate plumbing (WS2) ───────────────────────────────────────────

_GATE_FLAG_ENV = "STRATEGY_GATE_ENABLED"


def _gate_enabled() -> bool:
    # Default ON; set STRATEGY_GATE_ENABLED=0 to disable during rollout.
    return os.environ.get(_GATE_FLAG_ENV, "1").strip().lower() not in ("0", "false", "no", "")


def _run_gate(sdl: Any, menu: Any, builder: Any, *, session_id: str | None):
    """Run strategy_gate.verify behind the flag. Never raises — a gate failure
    falls through to the ungated path (returns None) so a bug can't break chat."""
    if not _gate_enabled():
        return None
    try:
        from app.planner.strategy_gate import verify, SymbolContext
        ctx = SymbolContext.from_builder(builder)
        return verify(
            sdl,
            symbol_ctx=ctx,
            catalog=menu,
            sid=(session_id or getattr(sdl, "content_hash", "") or ""),
            turn=getattr(sdl, "version", 1),
        )
    except Exception as exc:
        logger.warning("⚠️ sdl_flow | gate failed (skipping) | %s", exc, exc_info=True)
        return None


# ── Modify turn ───────────────────────────────────────────────────────────────

async def try_sdl_modify(
    change_msg: str,
    builder: Any,
    *,
    session_id: str | None = None,
) -> SDLFlowResult:
    """Run a modify turn on an existing SDL stored on the builder.

    Returns SDLFlowResult with the updated SDL.  If builder has no SDL,
    falls back gracefully (used_sdl=False).
    """
    existing_sdl = getattr(builder, "_sdl", None)
    if existing_sdl is None:
        return SDLFlowResult(used_sdl=False, skip_reason="no_existing_sdl")

    try:
        from app.planner.sdl_selector import modify_sdl
        from app.planner.catalog_schema import build_menu
        menu = build_menu()
        # Apply the change to the EXISTING SDL (the LLM edits the current ticket
        # and returns the full updated SDL). Crucially we finalize THIS merged
        # SDL — not a fresh compile of the change message, which would throw away
        # the rest of the strategy and rebuild from just "change RSI to 20".
        updated_sdl = await modify_sdl(existing_sdl, change_msg, menu, skip_flag=True)
        logger.info(
            "🔄 sdl_flow | MODIFY | v%d→v%d | hash %.8s→%.8s | session=%s",
            existing_sdl.version, updated_sdl.version,
            existing_sdl.content_hash, updated_sdl.content_hash, session_id,
        )
        return await _finalize_sdl(updated_sdl, menu, builder, session_id=session_id)
    except Exception as exc:
        logger.warning("⚠️ sdl_flow | modify failed | %s", exc, exc_info=True)
        return SDLFlowResult(used_sdl=False, skip_reason=f"modify_error:{exc}")
