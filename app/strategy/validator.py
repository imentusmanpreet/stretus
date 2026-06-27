"""
app/strategy/validator.py — the single, strict, fail-closed strategy gate (🛡️ verify).

A ``StrategySpec`` is already structurally valid by the time it reaches here (Pydantic
enforced shape/enums/ranges at construction). This module adds the checks that need
the live quant engine, and it is the ONLY thing that decides whether a strategy is
allowed through:

  1. engine availability  — fail-closed: no engine ⇒ hard error, never a silent pass
  2. condition compile    — every formula string is parsed by the engine's OWN parser
                            (unknown function ⇒ reject), plus a bare-identifier walk
                            (the engine resolves unknown identifiers to NaN, so we
                            backstop it here)
  3. safety + coherence   — a stop exists, an exit exists, direction/legs agree
  4. enums + risk vocab   — timeframe/market are supported; SL/trailing types/anchors
                            are engine-legal

The old behavioural "does it ever fire?" probe (800 synthetic bars) was removed: it
was redundant with the compile check, frequently false-positive on legitimately
selective strategies, and the real backtest is the ground truth on whether a
strategy actually trades.

Blocking problems go in ``errors`` (the generator feeds them back to the LLM to
repair). Non-blocking observations — chiefly assumed-value reductions — go in
``notes`` so the chat layer can surface "I assumed X" to the user. Nothing is ever
silently substituted.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.core import step_logger
from app.core.config import get_settings
from app.strategy import engine_bridge, vocab
from app.strategy.enums import is_supported_market, is_supported_timeframe, supported_timeframes
from app.strategy.spec import StrategySpec

# Ranking metrics whose data feeds are not yet wired (Phase A). Accepted by the
# model (intent preserved — LLM-only rule), surfaced here as an informational note.
_RANK_METRICS_NEEDING_FEEDS: frozenset[str] = frozenset({"delivery_volume", "oi_change"})

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationError:
    """One validation finding. ``code`` is stable/machine-usable; ``message`` is human."""

    field: str
    code: str
    message: str
    severity: str = "error"  # "error" (blocking) | "warning" (note)


@dataclass
class ValidationResult:
    errors: list[ValidationError] = field(default_factory=list)
    notes: list[ValidationError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_repair_text(self) -> str:
        """Compact bullet list of blocking errors, fed back to the LLM for repair."""
        return "\n".join(f"- [{e.code}] {e.field}: {e.message}" for e in self.errors)


# Conditions in the engine grammar that must be evaluable for the chosen direction.
def _active_condition_fields(spec: StrategySpec) -> list[tuple[str, str]]:
    """(field_name, condition_string) pairs that must compile for this spec."""
    pairs: list[tuple[str, str]] = [("entry_condition", spec.entry_condition)]
    if spec.exit_condition:
        pairs.append(("exit_condition", spec.exit_condition))
    if spec.direction == "both":
        pairs.append(("short_entry_condition", spec.short_entry_condition))
        if spec.short_exit_condition:
            pairs.append(("short_exit_condition", spec.short_exit_condition))
    for i, rule in enumerate(spec.htf_rules):
        pairs.append((f"htf_rules[{i}].condition", rule.condition))
    return pairs


def _check_engine_available(result: ValidationResult) -> bool:
    """Fail-closed gate: without the engine we cannot trust ANY condition."""
    if engine_bridge.is_available():
        return True
    result.errors.append(
        ValidationError(
            field="_engine",
            code="engine_unavailable",
            message=(
                "The backtest engine grammar is not importable in this environment, "
                "so conditions cannot be validated. Refusing to pass an unvalidated "
                "strategy (install TA-Lib / ensure quant_engine is present)."
            ),
        )
    )
    logger.error("❌ validator | engine grammar unavailable — failing closed")
    return False


def _compile_and_check_condition(
    field_name: str, condition: str, result: ValidationResult
) -> None:
    """Compile ONE condition with the engine parser and run the identifier/function
    backstops, appending any findings to ``result``.

    Shared by both the strategy's active conditions (entry/exit/short/htf) and the
    dynamic universe's ``screen`` conditions — they use the SAME engine grammar, so
    they get the SAME compile + identifier + function checks (Invariant: KB-free,
    screen reuses the formula compiler).
    """
    compile_condition = engine_bridge.get_compile_condition()
    parse_error = engine_bridge.get_parse_error_type()

    if not condition or not condition.strip():
        result.errors.append(
            ValidationError(
                field=field_name,
                code="condition_empty",
                message=f"{field_name} is required but is empty.",
            )
        )
        return
    try:
        compiled = compile_condition(condition)
    except parse_error as exc:
        result.errors.append(
            ValidationError(
                field=field_name,
                code="condition_invalid",
                message=(
                    f"The engine cannot parse this condition (so it cannot be "
                    f"backtested): {exc}. Condition: `{condition}`"
                ),
            )
        )
        logger.warning("⚠️ validator | %s did not compile | %r", field_name, exc)
        return
    except Exception as exc:  # noqa: BLE001 — unexpected parser failure, surface it
        result.errors.append(
            ValidationError(
                field=field_name,
                code="condition_compile_error",
                message=f"Unexpected error compiling {field_name}: {exc!r}. Condition: `{condition}`",
            )
        )
        logger.error("❌ validator | unexpected compile error | %s | %r", field_name, exc)
        return
    if compiled is None:
        return
    # Bare-identifier backstop: the engine resolves unknowns to NaN, we reject them.
    used = engine_bridge.collect_identifier_names(compiled.tree)
    unknown = sorted(n for n in used if not vocab.is_legal_identifier(n))
    if unknown:
        result.errors.append(
            ValidationError(
                field=field_name,
                code="unknown_identifier",
                message=(
                    f"{field_name} uses identifier(s) the engine does not know: "
                    f"{', '.join(unknown)}. Use only documented fields/indicators."
                ),
            )
        )
        logger.warning("⚠️ validator | %s unknown identifiers | %s", field_name, unknown)
    # Function-name backstop: the parser may accept unknown function names
    # syntactically but fail at evaluation time (e.g. SUPERTREND). Catch them here.
    used_funcs = engine_bridge.collect_function_names(compiled.tree)
    unknown_funcs = sorted(n for n in used_funcs if n not in engine_bridge.known_functions())
    if unknown_funcs:
        result.errors.append(
            ValidationError(
                field=field_name,
                code="unknown_function",
                message=(
                    f"{field_name} uses function(s) the engine cannot run: "
                    f"{', '.join(unknown_funcs)}. "
                    f"Use only: {', '.join(sorted(engine_bridge.known_functions()))}."
                ),
            )
        )
        logger.warning("⚠️ validator | %s unknown functions | %s", field_name, unknown_funcs)


def _check_conditions(spec: StrategySpec, result: ValidationResult) -> None:
    """Compile each active condition with the engine parser; walk for bad identifiers."""
    for field_name, condition in _active_condition_fields(spec):
        _compile_and_check_condition(field_name, condition, result)


def _check_safety_and_coherence(spec: StrategySpec, result: ValidationResult) -> None:
    """A stop must exist, and direction/legs must agree."""
    # Stop: the model always resolves a percent > 0, but guard explicitly.
    if spec.resolved_stop_loss_pct() <= 0:
        result.errors.append(
            ValidationError(
                field="stop_loss",
                code="no_stop_loss",
                message="A positive stop-loss is required; none could be resolved.",
            )
        )
    # A positive static target is required UNLESS a trailing exit supplies the profit
    # exit — a trailing take-profit OR a trailing stop (which ratchets above entry and
    # books the gain on a reversal). Then the static target is intentionally absent so
    # it can't fire before the trail engages. The "no profit exit at all" case is
    # caught separately as `missing_profit_exit`.
    if (
        spec.resolved_take_profit_pct() <= 0
        and spec.trailing_take_profit is None
        and spec.trailing_stop is None
    ):
        result.errors.append(
            ValidationError(
                field="take_profit",
                code="no_take_profit",
                message="A positive take-profit is required; none could be resolved.",
            )
        )
    # Direction coherence: a "both" strategy needs an explicit short leg.
    if spec.direction == "both" and not (spec.short_entry_condition or "").strip():
        result.errors.append(
            ValidationError(
                field="short_entry_condition",
                code="missing_short_leg",
                message=(
                    "direction is 'both' but no short_entry_condition was given. Provide "
                    "the short leg, or set direction to 'long_only'/'short_only'."
                ),
            )
        )


def _check_enums_and_risk_vocab(spec: StrategySpec, result: ValidationResult) -> None:
    """Timeframe/market are supported; SL/trailing types + anchors are engine-legal."""
    if not is_supported_timeframe(spec.timeframe):
        result.errors.append(
            ValidationError(
                field="timeframe",
                code="unsupported_timeframe",
                message=(
                    f"timeframe {spec.timeframe!r} is not supported. "
                    f"Supported: {', '.join(supported_timeframes())}."
                ),
            )
        )
    if not is_supported_market(spec.market):
        # Non-blocking: the engine normalises markets and the data feed is the real
        # gate on the symbol; surface as a note rather than block.
        result.notes.append(
            ValidationError(
                field="market",
                code="unrecognised_market",
                message=f"market {spec.market!r} did not map to a known market id.",
                severity="warning",
            )
        )

    # Risk vocab: only meaningful when the rich SL maps to an engine spec.
    sl_spec = spec.stop_loss_engine_spec()
    if sl_spec is not None:
        sl_type = sl_spec.get("type")
        if sl_type not in vocab.stop_loss_types():
            result.errors.append(
                ValidationError(
                    field="stop_loss.type",
                    code="unsupported_stop_type",
                    message=f"stop-loss type {sl_type!r} is not engine-legal "
                    f"({', '.join(sorted(vocab.stop_loss_types()))}).",
                )
            )
        anchor = sl_spec.get("anchor")
        if anchor is not None and anchor not in vocab.stop_loss_anchors():
            result.errors.append(
                ValidationError(
                    field="stop_loss.anchor",
                    code="unsupported_stop_anchor",
                    message=f"stop-loss anchor {anchor!r} is not engine-legal "
                    f"({', '.join(sorted(vocab.stop_loss_anchors()))}).",
                )
            )
    if spec.trailing_stop is not None and spec.trailing_stop.type not in vocab.trailing_stop_types():
        result.errors.append(
            ValidationError(
                field="trailing_stop.type",
                code="unsupported_trailing_type",
                message=f"trailing-stop type {spec.trailing_stop.type!r} is not engine-legal "
                f"({', '.join(sorted(vocab.trailing_stop_types()))}).",
            )
        )
    if (
        spec.trailing_take_profit is not None
        and spec.trailing_take_profit.type not in vocab.trailing_take_profit_types()
    ):
        result.errors.append(
            ValidationError(
                field="trailing_take_profit.type",
                code="unsupported_trailing_take_profit_type",
                message=f"trailing-take-profit type {spec.trailing_take_profit.type!r} is not "
                f"engine-legal ({', '.join(sorted(vocab.trailing_take_profit_types()))}).",
            )
        )
    # A position MAY trail BOTH its stop and its take-profit. They don't conflict —
    # the engine runs both lines and exits on whichever a reversal hits first, which
    # (with different distances/activations) gives a staged trail. The only degenerate
    # case is a DOMINATED line: if one line is always at least as tight AND activates
    # no later than the other, the other can never fire. Surface that as a non-blocking
    # warning so the user can widen/move it instead of being silently surprised.
    ts, ttp = spec.trailing_stop, spec.trailing_take_profit
    if (
        ts is not None and ttp is not None
        and ts.type == "percent" and ttp.type == "percent"
        and ts.distance_pct is not None and ttp.distance_pct is not None
    ):
        as_, at_ = (ts.activate_after_pct or 0.0), (ttp.activate_after_pct or 0.0)
        ds, dt = ts.distance_pct, ttp.distance_pct
        dead = None
        if ds <= dt and as_ <= at_:            # trailing stop dominates → TP never fires
            dead = ("trailing_take_profit", dt, at_, "trailing stop", ds, as_)
        elif dt <= ds and at_ <= as_:          # trailing TP dominates → stop never fires
            dead = ("trailing_stop", ds, as_, "trailing take-profit", dt, at_)
        if dead is not None:
            field, dead_d, dead_a, win_name, win_d, win_a = dead
            result.notes.append(
                ValidationError(
                    field=field,
                    code="dominated_trailing_line",
                    message=(
                        f"The {field} ({dead_d}% give-back after +{dead_a}% profit) can never "
                        f"trigger: the {win_name} ({win_d}% after +{win_a}%) is always tighter and "
                        f"activates no later, so it fires first. Widen or delay this line, or drop it."
                    ),
                    severity="warning",
                )
            )
    # Every strategy needs SOME exit that can realise a gain: a static take_profit,
    # a trailing take-profit, OR a trailing stop (which ratchets above entry and
    # books the gain on a reversal — a "ride until stopped out" trend strategy is
    # legitimate). take_profit may be omitted whenever a trailing exit supplies it.
    if (
        spec.take_profit is None
        and spec.trailing_take_profit is None
        and spec.trailing_stop is None
    ):
        result.errors.append(
            ValidationError(
                field="take_profit",
                code="missing_profit_exit",
                message="A strategy needs an exit that can take a gain: set a static "
                "take_profit, a trailing_take_profit, or a trailing_stop. Omit take_profit "
                "only when a trailing exit supplies it.",
            )
        )


def _check_universe(spec: StrategySpec, result: ValidationResult) -> None:
    """Validate the dynamic-universe block (only present on the dynamic path).

    * Each ``screen`` condition must compile against the SAME engine grammar as the
      entry condition (reuses ``_compile_and_check_condition`` — KB-free).
    * ``scan_timeframe`` (when set) must be a supported timeframe.
    * ``take`` / ``max_positions`` ABOVE the platform cap are an INFORMATIONAL note,
      never a hard reject (§7.1): the resolver clamps at runtime, so the cap can change
      via config without re-validating saved strategies.
    * A watchlist source must list symbols; an index/sector source needs a name.
    * Ranking metrics with no wired feed yet (delivery/oi) are noted, not rejected.
    Nothing here imports the KB / planner (Invariant 11).
    """
    universe = spec.universe
    if universe is None:
        return

    for i, condition in enumerate(universe.screen):
        _compile_and_check_condition(f"universe.screen[{i}]", condition, result)

    if universe.scan_timeframe and not is_supported_timeframe(universe.scan_timeframe):
        result.errors.append(
            ValidationError(
                field="universe.scan_timeframe",
                code="unsupported_scan_timeframe",
                message=(
                    f"universe.scan_timeframe {universe.scan_timeframe!r} is not supported. "
                    f"Supported: {', '.join(supported_timeframes())}."
                ),
            )
        )

    src = universe.source
    if src.kind == "watchlist" and not src.symbols:
        result.errors.append(
            ValidationError(
                field="universe.source.symbols",
                code="empty_watchlist",
                message="A 'watchlist' universe source needs a non-empty `symbols` list.",
            )
        )
    if src.kind in ("index", "sector") and not (src.name and src.name.strip()):
        result.errors.append(
            ValidationError(
                field="universe.source.name",
                code="missing_source_name",
                message=f"A {src.kind!r} universe source needs a `name` (e.g. 'NIFTY500').",
            )
        )

    if universe.rank.by in _RANK_METRICS_NEEDING_FEEDS:
        result.notes.append(
            ValidationError(
                field="universe.rank.by",
                code="rank_metric_needs_feed",
                message=(
                    f"Ranking by {universe.rank.by!r} needs a data feed that is not yet "
                    f"wired; this metric will be unavailable until the feed lands."
                ),
                severity="warning",
            )
        )

    settings = get_settings()
    max_assets = settings.dynamic_universe_max_assets
    max_positions = settings.dynamic_universe_max_positions
    if universe.take > max_assets:
        result.notes.append(
            ValidationError(
                field="universe.take",
                code="take_above_platform_cap",
                message=(
                    f"Requested take={universe.take} exceeds the platform cap "
                    f"({max_assets}); the resolver will clamp to {max_assets} after "
                    f"ranking (top-{max_assets} by {universe.rank.by})."
                ),
                severity="warning",
            )
        )
    eff_positions = universe.effective_max_positions()
    if eff_positions > max_positions:
        result.notes.append(
            ValidationError(
                field="universe.max_positions",
                code="max_positions_above_platform_cap",
                message=(
                    f"Requested max_positions={eff_positions} exceeds the platform cap "
                    f"({max_positions}); the resolver will clamp it at runtime."
                ),
                severity="warning",
            )
        )


def _collect_assumption_notes(spec: StrategySpec, result: ValidationResult) -> None:
    """Surface assumed values + engine reductions so the chat can confirm them."""
    if spec.stop_loss.source == "assumed":
        result.notes.append(
            ValidationError(
                field="stop_loss",
                code="assumed_stop_loss",
                message=f"Assumed stop-loss: {spec.resolved_stop_loss_pct()}% "
                f"({spec.stop_loss.reason or spec.stop_loss.logic or 'no user value given'}).",
                severity="warning",
            )
        )
    if spec.take_profit is not None and spec.take_profit.source == "assumed":
        result.notes.append(
            ValidationError(
                field="take_profit",
                code="assumed_take_profit",
                message=f"Assumed take-profit: {spec.resolved_take_profit_pct()}% "
                f"({spec.take_profit.reason or spec.take_profit.logic or 'no user value given'}).",
                severity="warning",
            )
        )
    if spec.stop_loss.type in ("fixed_points", "indicator_based"):
        result.notes.append(
            ValidationError(
                field="stop_loss",
                code="stop_loss_reduced_to_percent",
                message=(
                    f"The engine has no native {spec.stop_loss.type!r} stop; using "
                    f"{spec.resolved_stop_loss_pct()}% for the backtest."
                ),
                severity="warning",
            )
        )
    if spec.take_profit is not None and spec.take_profit.type in ("atr", "fixed_points", "indicator_based"):
        result.notes.append(
            ValidationError(
                field="take_profit",
                code="take_profit_reduced_to_percent",
                message=(
                    f"The engine consumes a percent target; {spec.take_profit.type!r} "
                    f"take-profit rendered as {spec.resolved_take_profit_pct()}%."
                ),
                severity="warning",
            )
        )


def validate_spec(
    spec: StrategySpec,
    *,
    session_id: str | None = None,
    turn: int = 0,
) -> ValidationResult:
    """Run the full strict gate over ``spec`` and return its result.

    Never raises on a bad strategy — problems become ``errors``. Only the engine
    being unavailable is itself an error (fail-closed).
    """
    result = ValidationResult()

    if not _check_engine_available(result):
        step_logger.log_step(
            step_logger.BLOCK, step_logger.VERIFY, session_id, turn,
            "engine_unavailable", errors=1,
        )
        return result

    _check_conditions(spec, result)
    # NOTE: the behavioural "does it ever fire?" probe (800 synthetic bars) was
    # removed — it is redundant with the grammar/compile check above, has a high
    # false-positive rate on legitimately selective strategies, and the real
    # backtest is the ground truth for whether a strategy actually trades.
    _check_safety_and_coherence(spec, result)
    _check_enums_and_risk_vocab(spec, result)
    _check_universe(spec, result)
    _collect_assumption_notes(spec, result)

    icon = step_logger.PASS if result.ok else step_logger.BLOCK
    step_logger.log_step(
        icon, step_logger.VERIFY, session_id, turn,
        "validate_spec",
        ok=result.ok,
        errors=len(result.errors),
        notes=len(result.notes),
        symbol=spec.symbol or (f"universe:{spec.universe.source.kind}" if spec.universe else "?"),
        tf=spec.timeframe,
        dir=spec.direction,
    )
    if result.errors:
        logger.info(
            "🛡️ validator | %d blocking error(s): %s",
            len(result.errors),
            ", ".join(e.code for e in result.errors),
        )
    return result
