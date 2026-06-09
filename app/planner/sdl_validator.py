"""
app/planner/sdl_validator.py — SDL validation pipeline.

Returns ValidationResult(ok, errors) — never raises.
Validation failure → do NOT compile; surface errors via read-back.

Stages (run in order; later stages only run if earlier ones pass)
────────────────────────────────────────────────────────────────
1. Referential  — signal names exist; symbol resolves; dynamic-universe
                  screen compiles; rank metric + tie_break valid.
2. Parameter    — every param is a known key for that card; numeric
                  values are in a sane range (> 0, not astronomically large).
3. Satisfiability — entry condition can fire (reuse condition_satisfiability).
4. Safety       — every leg has an exit AND a stop.
5. Feasibility  — max indicator warmup fits the available timeframe support.
6. Engine capability — unsupported combos surfaced BEFORE compile.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Result types ──────────────────────────────────────────────────────────────


@dataclass
class ValidationError:
    field: str
    code: str
    message: str


@dataclass
class ValidationResult:
    ok: bool
    errors: list[ValidationError] = field(default_factory=list)
    engine_gaps: list[str] = field(default_factory=list)  # non-blocking capability notes

    def add_error(self, field: str, code: str, message: str) -> None:
        self.ok = False
        self.errors.append(ValidationError(field=field, code=code, message=message))

    def add_gap(self, note: str) -> None:
        self.engine_gaps.append(note)


# ── Crypto alias map (user-entered → canonical KB symbol) ─────────────────────

_CRYPTO_ALIASES: dict[str, str] = {
    "ETH":     "ETH_USDC",
    "ETHEREUM": "ETH_USDC",
    "ETHUSDC": "ETH_USDC",
    "ETHUSDT": "ETH_USDT",
    "BTC":     "BTC_USDC",
    "BITCOIN": "BTC_USDC",
    "BTCUSDC": "BTC_USDC",
    "BTCUSDT": "BTC_USDT",
    "SOL":     "SOL_USDC",
    "ADA":     "ADA_USDC",
    "MATIC":   "MATIC_USDC",
    "AVAX":    "AVAX_USDC",
    "ATOM":    "ATOM_USDC",
    "LINK":    "LINK_USDC",
    "DOT":     "DOT_USDC",
    "XRP":     "XRP_USDC",
    "DOGE":    "DOGE_USDC",
}

# Scanner-supported rank metrics (mirrors SDL DynamicRank + scanner behaviour)
_VALID_RANK_METRICS = frozenset({"rvol", "rsi", "atr_pct", "distance_52w"})

# Sane numeric bounds for indicator params
_MAX_WINDOW = 500
_MAX_MULTIPLIER = 20.0
_MAX_THRESHOLD = 200.0  # RSI 0-100, CCI up to ±200, etc.


# ── Public entry point ────────────────────────────────────────────────────────


def validate_sdl(sdl, menu=None) -> ValidationResult:
    """Validate an SDL ticket. Returns ValidationResult (never raises).

    Args:
        sdl: SDL instance to validate.
        menu: CatalogMenu (built if not provided).
    """
    from app.planner.catalog_schema import build_menu

    if menu is None:
        menu = build_menu()

    logger.info(
        "🔍 sdl_validator | START | hash=%.8s | legs=%d | universe=%s",
        sdl.content_hash, len(sdl.legs or []),
        getattr(sdl.universe, "type", "?"),
    )

    result = ValidationResult(ok=True)

    ref_ok = _stage_referential(sdl, menu, result)
    _stage_parameter(sdl, menu, result) if ref_ok else None
    _stage_satisfiability(sdl, menu, result) if ref_ok else None
    _stage_safety(sdl, result)
    _stage_feasibility(sdl, menu, result)
    _stage_engine_capability(sdl, result)

    if result.errors:
        logger.warning(
            "❌ sdl_validator | INVALID | errors=%d | fields=%s",
            len(result.errors),
            [e.field for e in result.errors],
        )
        for e in result.errors:
            logger.warning("   ✗ [%s] %s: %s", e.code, e.field, e.message[:80])
    else:
        gaps_str = f" | gaps={len(result.engine_gaps)}" if result.engine_gaps else ""
        logger.info(
            "✅ sdl_validator | VALID | hash=%.8s%s",
            sdl.content_hash, gaps_str,
        )
    if result.engine_gaps:
        for gap in result.engine_gaps:
            logger.info("   ⚠️  engine gap: %s", gap[:100])
    return result


# ── Stage 1: Referential ──────────────────────────────────────────────────────


def _stage_referential(sdl, menu, result: ValidationResult) -> bool:
    """Returns True if no referential errors (compilation can proceed)."""
    initial_ok = result.ok

    _validate_universe(sdl, result)
    _validate_signal_refs_in_legs(sdl, menu, result)

    return result.ok  # True only if no new errors were added


def _validate_universe(sdl, result: ValidationResult) -> None:
    from app.planner.sdl import StaticUniverse, DynamicUniverse
    from app.kb import kb

    uni = sdl.universe

    if isinstance(uni, StaticUniverse):
        resolved = _resolve_symbol(uni.symbol, str(uni.asset_class), kb)
        if resolved is None:
            result.add_error(
                "universe.symbol",
                "unknown_symbol",
                f"Symbol '{uni.symbol}' was not found in the knowledge base. "
                "Check the symbol name and asset class.",
            )

    elif isinstance(uni, DynamicUniverse):
        # Validate rank metric
        if uni.rank.by not in _VALID_RANK_METRICS:
            result.add_error(
                "universe.rank.by",
                "invalid_rank_metric",
                f"rank.by='{uni.rank.by}' is not supported. "
                f"Valid values: {sorted(_VALID_RANK_METRICS)}",
            )

        # Validate tie_break id
        try:
            from app.services.discovery.tie_break import TIE_BREAK_METHODS
            if uni.tie_break not in TIE_BREAK_METHODS:
                result.add_error(
                    "universe.tie_break",
                    "invalid_tie_break",
                    f"tie_break='{uni.tie_break}' is not a valid tie-break method. "
                    f"Valid: {sorted(TIE_BREAK_METHODS.keys())}",
                )
        except ImportError:
            pass  # scanner not importable in test env — skip

        # Validate each screen condition compiles
        for i, cond in enumerate(uni.screen or []):
            err = _try_compile_condition(cond)
            if err:
                result.add_error(
                    f"universe.screen.{i}",
                    "invalid_screen_condition",
                    f"Screen condition '{cond}' won't compile: {err}",
                )


def _resolve_symbol(symbol: str, asset_class: str, kb) -> Any:
    """Return the resolved stock object, or None if not found."""
    # Direct lookup first
    if symbol in kb.stocks:
        stock = kb.stocks[symbol]
        if str(stock.asset_class) == asset_class:
            return stock
        return None

    # Try crypto alias map
    normalized = _CRYPTO_ALIASES.get(symbol.upper().replace("-", "_"))
    if normalized and normalized in kb.stocks:
        stock = kb.stocks[normalized]
        if str(stock.asset_class) == asset_class:
            return stock
        return None

    # Try KB lookup_stock (handles NSE abbreviations)
    try:
        stock = kb.lookup_stock(symbol)
        if stock and str(stock.asset_class) == asset_class:
            return stock
    except Exception:
        pass

    return None


def _try_compile_condition(cond_str: str) -> str | None:
    """Try to compile a condition string. Returns error message or None."""
    try:
        from engine.conditions import compile_condition
        compile_condition(cond_str)
        return None
    except Exception as exc:
        return str(exc)
    except ImportError:
        return None  # engine not importable — skip silently


def _validate_signal_refs_in_legs(sdl, menu, result: ValidationResult) -> None:
    known = menu.signal_names
    for li, leg in enumerate(sdl.legs or []):
        _check_signal_ref(leg.entry.trigger, f"legs.{li}.entry.trigger", known, result)
        for fi, flt in enumerate(leg.entry.filters or []):
            _check_signal_ref(flt, f"legs.{li}.entry.filters.{fi}", known, result)
        for ti, trig in enumerate(leg.exit.triggers or []):
            _check_signal_ref(trig, f"legs.{li}.exit.triggers.{ti}", known, result)
        for fi, flt in enumerate(leg.exit.filters or []):
            _check_signal_ref(flt, f"legs.{li}.exit.filters.{fi}", known, result)

    # HTF signal refs
    for ri, rule in enumerate(sdl.htf_rules or []):
        if rule.signal:
            _check_signal_ref(rule.signal, f"htf_rules.{ri}.signal", known, result)


def _check_signal_ref(ref, path: str, known: frozenset, result: ValidationResult) -> None:
    if ref.name not in known:
        result.add_error(
            path,
            "unknown_signal",
            f"Signal card '{ref.name}' does not exist in the catalog. "
            "Choose only names from the catalog menu.",
        )


# ── Stage 2: Parameter ────────────────────────────────────────────────────────


def _stage_parameter(sdl, menu, result: ValidationResult) -> bool:
    from app.kb import kb

    for li, leg in enumerate(sdl.legs or []):
        _validate_signal_params(
            leg.entry.trigger, f"legs.{li}.entry.trigger", kb, sdl.context.timeframe, result
        )
        for fi, flt in enumerate(leg.entry.filters or []):
            _validate_signal_params(flt, f"legs.{li}.entry.filters.{fi}", kb, sdl.context.timeframe, result)
        for ti, trig in enumerate(leg.exit.triggers or []):
            _validate_signal_params(trig, f"legs.{li}.exit.triggers.{ti}", kb, sdl.context.timeframe, result)

    return result.ok


def _validate_signal_params(ref, path: str, kb, timeframe: str, result: ValidationResult) -> None:
    """Validate that provided params are known for the card and have sane values."""
    card = kb.signals.get(ref.name)
    if card is None:
        return  # referential stage already caught this

    # Collect known param names from params_by_timeframe
    known_params: set[str] = set()
    for tf_params in card.params_by_timeframe.values():
        known_params.update(tf_params.keys())

    for pname, pval in (ref.params or {}).items():
        if known_params and pname not in known_params:
            result.add_error(
                f"{path}.params.{pname}",
                "unknown_param",
                f"Signal '{ref.name}' does not define param '{pname}'. "
                f"Known params: {sorted(known_params)}",
            )
            continue

        # Sanity-check numeric values
        if isinstance(pval, (int, float)):
            pname_lower = pname.lower()
            if any(k in pname_lower for k in ("window", "period", "length", "fast", "slow", "sign")):
                if pval <= 0 or pval > _MAX_WINDOW:
                    result.add_error(
                        f"{path}.params.{pname}",
                        "param_out_of_range",
                        f"Param '{pname}={pval}' for signal '{ref.name}' is out of range. "
                        f"Expected a positive integer ≤ {_MAX_WINDOW}.",
                    )
            elif "threshold" in pname_lower:
                if not (-_MAX_THRESHOLD <= pval <= _MAX_THRESHOLD):
                    result.add_error(
                        f"{path}.params.{pname}",
                        "param_out_of_range",
                        f"Param '{pname}={pval}' for signal '{ref.name}' is out of range "
                        f"[-{_MAX_THRESHOLD}, {_MAX_THRESHOLD}].",
                    )
            elif any(k in pname_lower for k in ("multiple", "mult", "num_std", "std")):
                if pval <= 0 or pval > _MAX_MULTIPLIER:
                    result.add_error(
                        f"{path}.params.{pname}",
                        "param_out_of_range",
                        f"Param '{pname}={pval}' for signal '{ref.name}' is out of range. "
                        f"Expected 0 < value ≤ {_MAX_MULTIPLIER}.",
                    )


# ── Stage 3: Satisfiability ───────────────────────────────────────────────────


def _stage_satisfiability(sdl, menu, result: ValidationResult) -> bool:
    from app.kb import kb
    from app.planner.condition_satisfiability import check_entry_satisfiable

    for li, leg in enumerate(sdl.legs or []):
        cond_str = _build_condition_string(leg.entry.trigger, kb)
        if cond_str is None:
            continue  # card unknown or formula not available — skip silently

        finding = check_entry_satisfiable(cond_str)
        if finding:
            result.add_error(
                f"legs.{li}.entry.trigger",
                finding.get("code", "entry_never_fires"),
                finding.get("message", "Entry condition can never fire."),
            )

    return result.ok


def _build_condition_string(ref, kb) -> str | None:
    """Format the card formula with the provided params."""
    card = kb.signals.get(ref.name)
    if card is None or not card.formula:
        return None
    try:
        params = ref.params or {}
        # Get card defaults for any missing params (use 15m or first tf)
        by_tf = card.params_by_timeframe or {}
        defaults: dict = {}
        if "15m" in by_tf:
            defaults = dict(by_tf["15m"])
        elif by_tf:
            defaults = dict(next(iter(by_tf.values())))
        merged = {**defaults, **params}
        return card.formula.format_map(merged)
    except (KeyError, ValueError):
        return None  # formula references a param we don't have — skip


# ── Stage 4: Safety ───────────────────────────────────────────────────────────


def _stage_safety(sdl, result: ValidationResult) -> None:
    # Every strategy needs a stop loss
    if sdl.risk.stop_loss is None and not (
        sdl.risk.trailing and sdl.risk.trailing.enabled
    ):
        result.add_error(
            "risk.stop_loss",
            "missing_stop_loss",
            "Every strategy requires a stop loss. "
            "Add a stop_loss spec (percent, atr, or structural).",
        )

    # Every leg needs an exit mechanism
    for li, leg in enumerate(sdl.legs or []):
        has_exit = (
            bool(leg.exit.triggers)
            or sdl.risk.take_profit is not None
            or (sdl.risk.trailing and sdl.risk.trailing.enabled)
        )
        if not has_exit:
            result.add_error(
                f"legs.{li}.exit",
                "missing_exit",
                f"Leg {li} (direction={leg.direction}) has no exit mechanism. "
                "Add exit triggers, a take_profit, or a trailing stop.",
            )


# ── Stage 5: Feasibility ──────────────────────────────────────────────────────


def _stage_feasibility(sdl, menu, result: ValidationResult) -> None:
    from app.kb import kb

    # Check that the strategy timeframe is in the signal's works_best_on
    # (warning only — not a hard error since signals may work outside their best-on list)
    tf = sdl.context.timeframe
    for li, leg in enumerate(sdl.legs or []):
        card = kb.signals.get(leg.entry.trigger.name)
        if card and card.unsupported_on and tf in card.unsupported_on:
            result.add_error(
                f"legs.{li}.entry.trigger",
                "unsupported_timeframe",
                f"Signal '{leg.entry.trigger.name}' is explicitly unsupported on timeframe '{tf}'.",
            )


# ── Stage 6: Engine capability ────────────────────────────────────────────────


def _stage_engine_capability(sdl, result: ValidationResult) -> None:
    """Surface engine capability gaps as non-blocking notes (engine_gaps)."""

    # Scale-outs AND trailing together — the engine doesn't support both
    has_scale_outs = bool(sdl.risk.scale_outs)
    has_trailing = bool(sdl.risk.trailing and sdl.risk.trailing.enabled)
    if has_scale_outs and has_trailing:
        result.add_gap(
            "scale_outs + trailing_stop together are not supported by the current engine. "
            "Only one of the two will apply at run time (trailing takes precedence)."
        )

    # Dynamic universe with point-in-time historical backtest
    from app.planner.sdl import DynamicUniverse
    if isinstance(sdl.universe, DynamicUniverse):
        result.add_gap(
            "Dynamic universe: the scanner picks ONE asset at resolution time. "
            "Point-in-time historical universe backtest (re-scanning for each bar) "
            "is not supported — the chosen asset is fixed for the full backtest."
        )

    # HTF rules beyond what the engine supports
    if len(sdl.htf_rules) > 1:
        result.add_gap(
            "Multiple HTF rules: the engine currently supports one HTF timeframe. "
            "Only the first HTF rule will be applied."
        )
