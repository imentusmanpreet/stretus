"""
app/services/discovery/orchestrator.py
──────────────────────────────────────
Phase 9 — bridge between the scanner and the chat / builder layers.

Two public functions:

  • run_discovery(builder, *, asof_utc=None, fetch_ohlcv=None)
        Looks up the builder's strategy_preset, runs scan_universe with the
        preset's discovery config, and writes the result onto the builder.
        Status side-effects:
          single   → builder.symbol = candidates[0].symbol
                      (no further user interaction needed)
          none     → builder.discovery_no_match = True
          multiple → builder.discovery_pending = True
                      builder.discovery_candidates = [...]
                      builder.discovery_tie_break_options = [...]
        Returns the ScanResult so the caller can compose a chat reply.

  • resolve_tie_break(builder, user_reply)
        When builder.discovery_pending is True, parse the user's reply as
        a tie-break choice, apply the chosen method, set builder.symbol,
        and clear the pending state. Returns (success: bool, message: str).
"""
from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Callable

from app.services.discovery.scanner import scan_universe
from app.services.discovery.tie_break import (
    apply_tie_break,
    available_tie_break_options,
    parse_user_tie_break_reply,
)
from app.services.discovery.types import (
    Candidate,
    DiscoveryConfig,
    ScanResult,
    TieBreakOption,
)

logger = logging.getLogger(__name__)


def _merge_parameters(
    defaults: dict[str, Any] | None,
    overrides: dict[str, Any] | None,
) -> dict[str, float]:
    """Merge preset defaults with user-supplied overrides. Overrides
    win, but only for keys that exist in the defaults (so a typo or
    stale override key can't silently introduce an unknown parameter).
    Values are coerced to float."""
    merged: dict[str, float] = {}
    for k, v in (defaults or {}).items():
        try:
            merged[str(k)] = float(v)
        except (TypeError, ValueError):
            logger.warning("discovery|drop_non_numeric_default|key=%s|value=%r", k, v)
    if overrides:
        for k, v in overrides.items():
            if k not in merged:
                logger.warning(
                    "discovery|drop_unknown_override|key=%s|value=%r|valid_keys=%s",
                    k, v, sorted(merged.keys()),
                )
                continue
            try:
                merged[str(k)] = float(v)
            except (TypeError, ValueError):
                logger.warning("discovery|drop_non_numeric_override|key=%s|value=%r", k, v)
    return merged


def _apply_parameters_to_condition(condition: str, params: dict[str, float]) -> str:
    """Substitute {placeholder} tokens in `condition` using `params`.
    Raises ValueError when the condition references a placeholder that
    isn't in params (so a typo blows up loudly instead of silently
    producing an invalid AST).

    Phase 9i — integer-valued floats are coerced to int before format
    so e.g. lookback_window_bars=252.0 substitutes as `MAX(HIGH, 252)`
    not `MAX(HIGH, 252.0)`. The AST accepts both forms but the integer
    form reads cleanly in logs and tests.
    """
    if not params:
        return condition
    cleaned = {
        k: (int(v) if isinstance(v, float) and v.is_integer() else v)
        for k, v in params.items()
    }
    try:
        return condition.format(**cleaned)
    except KeyError as missing:
        raise ValueError(
            f"Discovery condition references unknown parameter {missing}: "
            f"{condition!r}. Known keys: {sorted(params.keys())}"
        )


def _preset_discovery_config(builder) -> DiscoveryConfig | None:
    """Resolve builder.strategy_preset → DiscoveryConfig, or None if the
    preset has no discovery block (caller should not have invoked us).

    Phase 9h — merges builder.discovery_parameter_overrides into the
    preset's `parameters` defaults and substitutes {placeholder} tokens
    in `conditions` before constructing the DiscoveryConfig so the
    scanner sees concrete numeric thresholds.
    """
    name = (builder.strategy_preset or "").strip().lower()
    if not name:
        return None
    try:
        from app.kb import kb as _kb
    except Exception:
        return None
    preset = _kb.presets.get(name)
    if preset is None or not preset.discovery:
        return None
    raw = dict(preset.discovery)

    # Phase 9h — merge defaults + user overrides, then substitute
    # placeholders in the condition strings. The effective parameter
    # set is also stashed on the builder so the no-match message can
    # show the threshold actually used.
    params = _merge_parameters(
        raw.get("parameters"),
        getattr(builder, "discovery_parameter_overrides", None),
    )
    builder.discovery_parameters_used = dict(params) if params else None

    # Phase 9k — when the chat layer parsed compositional primitives
    # from the user's prose, run the scanner with EXACTLY those
    # constraints (no implicit 52-week / pullback / etc. that the
    # preset would otherwise force). Falls back to the preset's
    # hardcoded conditions when no primitives were captured.
    user_conditions = getattr(builder, "discovery_conditions", None)
    if user_conditions:
        from app.services.discovery.primitives import render_conditions
        try:
            raw["conditions"] = render_conditions(user_conditions)
            logger.info(
                "discovery|using_user_primitives|count=%d|names=%s",
                len(user_conditions),
                [c.get("name") for c in user_conditions if isinstance(c, dict)],
            )
        except ValueError as exc:
            logger.warning(
                "discovery|invalid_user_conditions|err=%s|fallback=preset_defaults",
                exc,
            )
            user_conditions = None
    if not user_conditions and params:
        # Original 9h path — substitute placeholders in preset conditions.
        raw["conditions"] = [
            _apply_parameters_to_condition(c, params)
            for c in (raw.get("conditions") or [])
        ]

    if params:
        # Phase 9i — when the user picks a longer high/low window
        # (e.g. "2-year high" → 504 bars), bump the lookback_days so
        # the scanner fetches enough OHLCV history to evaluate the
        # condition. +30 days covers weekends/holidays for daily bars.
        # Also runs when user primitives reference a longer window
        # (the override populates lookback_window_bars in params).
        window = params.get("lookback_window_bars")
        if window is not None:
            try:
                required_days = int(window) + 30
            except (TypeError, ValueError):
                required_days = 0
            existing_days = 0
            try:
                existing_days = int(raw.get("lookback_days") or 0)
            except (TypeError, ValueError):
                pass
            raw["lookback_days"] = max(existing_days, required_days)
        # Phase 9i — write the merged params back so the scanner can
        # read tunables like `lookback_window_bars` when computing
        # candidate metrics (distance to high/low, etc.).
        raw["parameters"] = params

    # Materialise tie-break options: a preset can supply just method ids and
    # the orchestrator will look up the labels from the framework defaults.
    raw_options = raw.get("tie_break_options") or []
    options: list[dict] = []
    for entry in raw_options:
        if isinstance(entry, str):
            # Method id only — look up label/desc from the framework.
            opts = available_tie_break_options([entry])
            if opts:
                options.append(opts[0].model_dump())
        elif isinstance(entry, dict) and entry.get("method"):
            options.append({
                "method":      str(entry["method"]).strip(),
                "label":       str(entry.get("label") or entry["method"]).strip(),
                "description": str(entry.get("description") or "").strip(),
            })

    # Phase 9i — when the active lookback window differs from the
    # preset's "52-week" default, rewrite the tie-break label so the
    # user sees the actual window they chose ("Closest to 20-day high"
    # rather than the misleading "Closest to 52-week high"). The
    # method ID stays stable for backend lookups.
    window_bars = None
    try:
        window_bars = int(float(params.get("lookback_window_bars", 252)))
    except (TypeError, ValueError):
        window_bars = 252
    if window_bars and window_bars != 252:
        window_phrase = _format_window_phrase(window_bars)
        for opt in options:
            label = opt["label"]
            opt["label"] = (
                label.replace("52-week", window_phrase)
                     .replace("52 week", window_phrase)
                     .replace("52w", window_phrase)
            )
            desc = opt["description"]
            opt["description"] = (
                desc.replace("52-week", window_phrase)
                    .replace("52 week", window_phrase)
                    .replace("52w", window_phrase)
            )

    raw["tie_break_options"] = [TieBreakOption(**o) for o in options]
    return DiscoveryConfig(**raw)


def _format_window_phrase(bars: int) -> str:
    """Render a bar count as a user-friendly window phrase ("20-day",
    "26-week", "6-month", "1-year"). Used for tie-break label rewrites
    when the user chose a non-default lookback window.

    Heuristic prefers larger units only when they read naturally:
      • year  — exact multiples of 252 starting at 1 year
      • month — exact multiples of 21 starting at 2 months (so 21
                doesn't become the awkward "1-month")
      • week  — exact multiples of 5 starting at 5 weeks (so small
                windows like 15 bars stay as "15-day" rather than
                "3-week")
      • day   — everything else
    """
    if bars % 252 == 0 and bars >= 252:
        years = bars // 252
        return f"{years}-year" if years > 1 else "1-year"
    if bars % 21 == 0 and bars >= 42:
        months = bars // 21
        return f"{months}-month"
    if bars % 5 == 0 and bars >= 25:
        weeks = bars // 5
        return f"{weeks}-week"
    return f"{bars}-day"


def _reset_discovery_state(builder) -> None:
    """Wipe per-scan discovery flags so a re-run starts clean."""
    builder.discovery_pending           = False
    builder.discovery_no_match          = False
    builder.discovery_candidates        = None
    builder.discovery_tie_break_options = None
    builder.discovery_chosen_method     = None


async def run_discovery(
    builder,
    *,
    asof_utc: datetime | None = None,
    fetch_ohlcv: Callable | None = None,
) -> ScanResult | None:
    """Run the scanner for the builder's preset. Mutates the builder with
    the result. Returns the ScanResult, or None when the preset has no
    discovery block (caller should not call us in that case but we degrade
    gracefully)."""
    discovery = _preset_discovery_config(builder)
    if discovery is None:
        logger.info("discovery|skip|reason=no_preset_or_no_discovery_config")
        return None
    if not builder.timeframe:
        # The scanner needs a timeframe to know what bars to fetch. Without
        # one we can't proceed — caller should make sure the user has given
        # a timeframe before invoking us.
        logger.info("discovery|skip|reason=builder.timeframe_unset")
        return None

    _reset_discovery_state(builder)
    # Issue 2 — honour an explicit user-supplied universe list (set by
    # the chat layer when the user types "choose among these stocks:
    # TCS, INFY, ..."). When None, the scanner uses the preset's
    # default universe.
    universe_override = getattr(builder, "discovery_universe_override", None)
    result = await scan_universe(
        discovery,
        interval=builder.timeframe,
        asof_utc=asof_utc,
        fetch_ohlcv=fetch_ohlcv,
        universe_override=universe_override,
    )

    if result.status == "single":
        chosen = result.candidates[0]
        builder.discovered_symbol = chosen.symbol
        builder.symbol = chosen.symbol
        logger.info(
            "discovery|single_match|symbol=%s|asof=%s",
            chosen.symbol, result.asof_iso,
        )
    elif result.status == "none":
        builder.discovery_no_match = True
        # Phase 9m — stash diagnostics on the builder so the chat
        # layer's no-match reply can show per-condition failure counts
        # + asof + universe size. Without these the user has no way to
        # tell whether nothing matched because the constraints are too
        # tight, the data is stale, or the universe is too small.
        builder.discovery_last_scan_diagnostics = {
            "asof_iso":                   result.asof_iso,
            "scanned_count":              result.scanned_count,
            "failed_fetches":             result.failed_fetches,
            "fail_counts":                dict(result.fail_counts or {}),
            "fetch_errors":               dict(result.fetch_errors or {}),
            "scanned_symbols":            list(result.scanned_symbols or []),
            "passed_symbols":             list(result.passed_symbols or []),
            "failed_fetch_symbols":       list(result.failed_fetch_symbols or []),
            "failed_condition_by_symbol": dict(result.failed_condition_by_symbol or {}),
        }
        logger.info(
            "discovery|no_match|asof=%s|scanned=%d|failed_fetches=%d|"
            "fail_counts=%s|fetch_errors=%s",
            result.asof_iso, result.scanned_count, result.failed_fetches,
            dict(result.fail_counts or {}),
            dict(result.fetch_errors or {}),
        )
    else:  # multiple
        builder.discovery_pending = True
        builder.discovery_candidates = [c.to_dict() for c in result.candidates]
        builder.discovery_tie_break_options = [o.model_dump() for o in result.tie_break_options]
        logger.info(
            "discovery|awaiting_tie_break|count=%d|methods=%s",
            len(result.candidates),
            [o.method for o in result.tie_break_options],
        )
    return result


def resolve_tie_break(builder, user_reply: str) -> tuple[bool, str]:
    """Apply a user's tie-break choice. Returns (ok, message).

    Caller must have verified `builder.discovery_pending is True` before
    calling. On success, builder.symbol is set and the discovery_pending
    flag is cleared so the chat flow can proceed normally.
    """
    if not builder.discovery_pending:
        return False, "There is no pending discovery choice to resolve."

    raw_options = builder.discovery_tie_break_options or []
    raw_candidates = builder.discovery_candidates or []
    if not raw_options or not raw_candidates:
        # Should never happen but degrade gracefully rather than crashing.
        _reset_discovery_state(builder)
        return False, "Internal error: discovery state is incomplete; please retry the scan."

    options = [TieBreakOption(**o) for o in raw_options]
    method = parse_user_tie_break_reply(user_reply, options)
    if method is None:
        # Build a friendlier prompt listing the choices again.
        choices = "\n".join(f"  {i + 1}. {o.label}" for i, o in enumerate(options))
        return False, (
            f"I didn't recognise that choice. Please reply with one of:\n{choices}"
        )

    candidates = [
        Candidate(
            symbol=c["symbol"],
            display_name=c["display_name"],
            sector=c["sector"],
            metrics=dict(c.get("metrics") or {}),
        )
        for c in raw_candidates
    ]
    ranked = apply_tie_break(method, candidates)
    if not ranked:
        _reset_discovery_state(builder)
        return False, "Internal error: tie-break produced no candidates."

    chosen = ranked[0]
    builder.discovered_symbol      = chosen.symbol
    builder.symbol                 = chosen.symbol
    builder.discovery_chosen_method = method
    builder.discovery_pending       = False
    builder.discovery_candidates        = None
    builder.discovery_tie_break_options = None

    logger.info(
        "discovery|resolved|method=%s|symbol=%s|metrics=%s",
        method, chosen.symbol, chosen.metrics,
    )
    return True, f"Picked {chosen.display_name} ({chosen.symbol}) via {method}."
