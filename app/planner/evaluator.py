"""
app/planner/evaluator.py — Evaluation integration layer.

Connects the SDL pipeline (StrategyArtifact) to the existing quant_engine
backtest runner.  Does NOT rewrite the backtester — it is a thin adapter.

Flow
────
1. artifact_to_yaml(artifact, symbol)    — converts artifact to the YAML format
                                           the engine's load_strategy_from_content
                                           already parses.
2. resolve_dynamic_symbol(artifact)      — for dynamic universes, runs the
                                           discovery scanner to pick ONE symbol.
3. run_evaluation(artifact, ...)         — orchestrates 1+2, calls the engine,
                                           version-stamps the result.

The run_evaluation() function is synchronous for simplicity; the async wrapper
is left to the API layer which already knows how to call the backtest service.

Acceptance #8: output = existing backtest result (in-sample/OOS/per-regime),
               version-stamped with the SDL version.
"""
from __future__ import annotations

import logging
import yaml
from typing import Any

logger = logging.getLogger(__name__)


# ── Artifact → YAML conversion ────────────────────────────────────────────────

def artifact_to_yaml(artifact, symbol_override: str | None = None) -> str:  # noqa: C901
    """Convert a StrategyArtifact to the YAML string the engine loader expects.

    For dynamic universes, `symbol_override` MUST be provided (the discovery
    scanner resolves it before this call).  For static universes it is ignored
    (the artifact already carries the symbol).

    Returns a YAML string with a top-level `strategy:` key.
    """
    symbol = symbol_override if symbol_override else artifact.symbol

    strategy: dict[str, Any] = {
        "name":            artifact.name,
        "symbol":          symbol,
        "market":          artifact.market,
        "timeframe":       artifact.timeframe,
        "objective":       artifact.objective,
        "entry_condition": artifact.entry_condition,
        "exit_condition":  artifact.exit_condition or (
            "PROFIT >= TAKE_PROFIT_TARGET OR LOSS <= -STOP_LOSS_TARGET"
        ),
        # Phase 12 — short leg (only populated for two-sided "both" strategies).
        # The engine runs a separate short pass on these; empty otherwise.
        "short_entry_condition": artifact.short_entry_condition or "",
        "short_exit_condition":  artifact.short_exit_condition or (
            "PROFIT >= TAKE_PROFIT_TARGET OR LOSS <= -STOP_LOSS_TARGET"
            if artifact.short_entry_condition else ""
        ),
        "indicators":      {},
        "stop_loss_pct":   artifact.stop_loss_pct,
        "take_profit_pct": artifact.take_profit_pct,
        "direction":       artifact.direction,
    }

    # Rich stop_loss spec
    if artifact.stop_loss_spec:
        sl = artifact.stop_loss_spec
        if sl["type"] == "percent":
            strategy["stop_loss"] = {"type": "percent", "pct": sl["pct"]}
        elif sl["type"] == "atr":
            strategy["stop_loss"] = {
                "type": "atr",
                "multiplier": sl.get("multiplier", 1.5),
                "window": sl.get("window", 14),
            }
        elif sl["type"] == "structural":
            anchor = sl.get("anchor", "opening_range_low")
            spec: dict[str, Any] = {"type": "structural", "anchor": anchor}
            if "opening" in anchor:
                spec["opening_bars"] = sl.get("opening_bars", 3)
            else:
                spec["window"] = sl.get("window", 5)
            if sl.get("padding_pct"):
                spec["padding_pct"] = sl["padding_pct"]
            strategy["stop_loss"] = spec

    # Trailing stop
    if artifact.trailing_stop_spec:
        strategy["trailing_stop"] = artifact.trailing_stop_spec

    # HTF rules
    if artifact.htf_rules:
        strategy["htf"] = [
            {"timeframe": r["timeframe"], "condition": r["condition"]}
            for r in artifact.htf_rules
        ]

    # Gate fields
    if artifact.vol_filter_metric:
        strategy["volatility_filter"] = {
            "metric": artifact.vol_filter_metric,
            "window": artifact.vol_filter_window,
            "min":    artifact.vol_filter_min,
            "max":    artifact.vol_filter_max,
        }

    if artifact.regime_filter_allowed is not None:
        strategy["regime_filter"] = {"allowed": list(artifact.regime_filter_allowed)}

    if artifact.rs_filter_window is not None:
        strategy["relative_strength_filter"] = {
            "window":    artifact.rs_filter_window,
            "min_ratio": artifact.rs_filter_min_ratio,
        }
        if artifact.reference_symbol:
            strategy["reference_symbol"] = artifact.reference_symbol

    if artifact.event_skip_dates:
        strategy["event_filter"] = {"skip_dates": list(artifact.event_skip_dates)}

    if artifact.entry_window_start_utc is not None or artifact.entry_window_end_utc is not None:
        ew: dict[str, Any] = {}
        if artifact.entry_window_start_utc is not None:
            ew["start_utc_minutes"] = artifact.entry_window_start_utc
        if artifact.entry_window_end_utc is not None:
            ew["end_utc_minutes"] = artifact.entry_window_end_utc
        strategy["entry_window"] = ew

    if artifact.volume_ratio_threshold is not None:
        strategy["volume_ratio_threshold"] = artifact.volume_ratio_threshold

    return yaml.dump({"strategy": strategy}, allow_unicode=True, sort_keys=False)


# ── Dynamic symbol resolution ─────────────────────────────────────────────────

def resolve_dynamic_symbol(artifact, candidates: list | None = None) -> str | None:
    """Resolve the symbol for a dynamic-universe artifact.

    For now, this is a stub — in production the caller runs the discovery
    scanner and passes the chosen candidate.  This function extracts the
    symbol from the first candidate if provided.

    Returns the resolved symbol, or None if resolution is not possible.
    """
    if not artifact.discovery_config:
        return artifact.symbol or None

    if candidates and hasattr(candidates[0], "symbol"):
        return candidates[0].symbol
    if candidates and isinstance(candidates[0], dict):
        return candidates[0].get("symbol")
    if candidates and isinstance(candidates[0], str):
        return candidates[0]

    return None


# ── Version stamping ──────────────────────────────────────────────────────────

def stamp_result(backtest_result: dict[str, Any], artifact) -> dict[str, Any]:
    """Attach version metadata to a backtest result dict."""
    stamped = dict(backtest_result)
    stamped["_sdl_artifact"] = {
        "artifact_id":      artifact.artifact_id,
        "artifact_version": artifact.version,
        "content_hash":     artifact.content_hash,
    }
    trades = (backtest_result.get("metrics") or {}).get("backtest_trades") or []
    n_trades = len(trades) if isinstance(trades, list) else "?"
    logger.info(
        "📈 evaluator | RESULT STAMPED | v%d | hash=%.8s | id=%.8s | trades=%s",
        artifact.version, artifact.content_hash,
        artifact.artifact_id, n_trades,
    )
    return stamped


# ── Run evaluation (thin adapter) ─────────────────────────────────────────────

def run_evaluation(
    artifact,
    ohlcv_data: list[dict[str, Any]] | None,
    run_config: dict[str, Any] | None = None,
    market_data_request: dict[str, Any] | None = None,
    *,
    symbol_override: str | None = None,
) -> dict[str, Any]:
    """Run the backtest for a compiled strategy artifact (synchronous).


    Accepts the existing engine's interface (ohlcv_data + run_config +
    market_data_request) and returns the result dict, version-stamped.

    For dynamic universes, caller must resolve the symbol first (via the
    discovery scanner) and pass it as `symbol_override`.

    This function calls the engine's run_backtest directly (not via HTTP).
    The HTTP path (quant_engine_client) is for production API calls.

    Raises RuntimeError if the engine is not importable (test environments
    without the full quant stack).
    """
    sym = symbol_override or artifact.symbol or "(dynamic)"
    bars = len(ohlcv_data) if ohlcv_data else 0
    logger.info(
        "🚀 evaluator | RUN | sym=%s | tf=%s | v%d | hash=%.8s | bars=%d",
        sym, artifact.timeframe, artifact.version,
        artifact.content_hash, bars,
    )

    try:
        from engine.runner import run_backtest
    except ImportError as exc:
        raise RuntimeError(
            "quant_engine not importable — run_evaluation requires the full engine stack. "
            "In tests, mock this function instead."
        ) from exc

    yaml_str = artifact_to_yaml(artifact, symbol_override=symbol_override)
    result = run_backtest(
        yaml_content=yaml_str,
        ohlcv_data=ohlcv_data or [],
        run_config=run_config or {},
        market_data_request=market_data_request or {},
    )
    return stamp_result(result, artifact)
