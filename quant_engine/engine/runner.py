"""
Orchestrates the backtest pipeline end to end.

Flow:
  1. Load strategy YAML → StrategyConfig
  2. Normalize OHLCV data → DataFrame
  3. Check data sufficiency vs indicator warm-up requirements
  4. Add indicators
  5. Ensure condition-referenced indicators are computed
  6. Simulate trades (objective-aware)
  7. Build result envelope with metrics + diagnostics
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from engine.conditions import CompiledCondition, compile_condition
from engine.config import BACKTEST_MARKET_DATA_FROM_UTC, BACKTEST_MARKET_DATA_TO_UTC
from engine.data import load_ohlcv_data, merge_reference_data
from engine.htf import HtfContext, build_htf_contexts
from engine.indicators import add_all_indicators, max_indicator_warmup
from engine.patterns import (
    add_all_patterns,
    merge_pattern_configs,
    patterns_required_by_identifiers,
)
from engine.loader import load_strategy_from_content
from engine.metrics import build_backtest_result
from engine.kb_signals import estimate_kb_warmup
from engine.simulator import simulate_trades

logger = logging.getLogger(__name__)

# Warm-up safety multiplier: we want at least 2× the indicator warm-up period
# available as eligible candles AFTER warm-up completes.
_WARMUP_BUFFER_MULTIPLIER = 2


def _parse_window_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, utc=True)
    if getattr(timestamp, "tzinfo", None) is not None:
        return timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.tz_localize(None)


def _enforce_market_data_window(
    df: pd.DataFrame,
    market_data_request: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start_ts = _parse_window_timestamp(BACKTEST_MARKET_DATA_FROM_UTC)
    end_ts = _parse_window_timestamp(BACKTEST_MARKET_DATA_TO_UTC)
    trimmed_df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()

    normalized_request = dict(market_data_request)
    requested_from = normalized_request.get("from_utc")
    requested_to = normalized_request.get("to_utc")
    normalized_request["from_utc"] = BACKTEST_MARKET_DATA_FROM_UTC
    normalized_request["to_utc"] = BACKTEST_MARKET_DATA_TO_UTC

    if requested_from != BACKTEST_MARKET_DATA_FROM_UTC or requested_to != BACKTEST_MARKET_DATA_TO_UTC or len(trimmed_df) != len(df):
        logger.info(
            "Enforced backtest market-data window | requested_from=%s requested_to=%s "
            "enforced_from=%s enforced_to=%s input_rows=%s trimmed_rows=%s",
            requested_from,
            requested_to,
            BACKTEST_MARKET_DATA_FROM_UTC,
            BACKTEST_MARKET_DATA_TO_UTC,
            len(df),
            len(trimmed_df),
        )

    if trimmed_df.empty:
        raise ValueError(
            "No OHLCV data is available inside the configured backtest window "
            f"{BACKTEST_MARKET_DATA_FROM_UTC} to {BACKTEST_MARKET_DATA_TO_UTC}."
        )

    return trimmed_df, normalized_request


def _load_strategy_config_from_input(yaml_input: str):
    candidate_path = Path(str(yaml_input))
    if "\n" not in str(yaml_input) and candidate_path.exists():
        return load_strategy_from_content(candidate_path.read_text(encoding="utf-8"))
    return load_strategy_from_content(str(yaml_input))


def run_backtest(
    yaml_content: str,
    ohlcv_data: list[dict[str, Any]] | list[list[Any]] | None,
    run_config: dict[str, Any],
    market_data_request: dict[str, Any],
    backtest_ref_id: str | None = None,
    # Phase 4 — when the strategy declares a reference_symbol, the API layer
    # fetches its OHLCV in parallel and passes it here. The runner joins it
    # onto the main df with REF_-prefixed columns so REF_CLOSE / RS(n) etc.
    # resolve correctly during simulation.
    reference_ohlcv: list[dict[str, Any]] | list[list[Any]] | None = None,
    # Phase 5 — when the strategy declares htf_rules, the API layer fetches
    # OHLCV for each HTF timeframe and passes a {tf: ohlcv} dict here.
    # The runner builds an HtfContext per declared HTF and the simulator
    # uses each as an entry-gate (no-look-ahead).
    htf_ohlcv: dict[str, list[dict[str, Any]] | list[list[Any]]] | None = None,
) -> dict:
    logger.info("🧮 Starting backtest pipeline")

    cfg = _load_strategy_config_from_input(yaml_content)
    df  = load_ohlcv_data(ohlcv_data)
    df, market_data_request = _enforce_market_data_window(df, market_data_request)

    # ── Phase 4: reference symbol merge ───────────────────────────────────────
    if cfg.reference_symbol:
        if not reference_ohlcv:
            raise ValueError(
                f"Strategy declares reference_symbol={cfg.reference_symbol!r} "
                "but no reference_ohlcv was supplied to run_backtest(). "
                "The API layer must fetch the reference series and pass it in."
            )
        ref_df = load_ohlcv_data(reference_ohlcv)
        df = merge_reference_data(df, ref_df)
        logger.info(
            "🔗 Reference symbol merged | reference_symbol=%s ref_rows=%s",
            cfg.reference_symbol, len(ref_df),
        )
    elif reference_ohlcv:
        # Strategy doesn't ask for reference data but caller supplied some.
        # Quietly ignore — surfacing as a hard error would break smooth UX
        # when API callers attach reference data preemptively.
        logger.info(
            "Reference OHLCV supplied but strategy has no reference_symbol — ignoring."
        )

    # ── Phase 5: higher-timeframe context build ───────────────────────────────
    # Strategy declares `htf_rules` → caller must supply OHLCV for each HTF.
    # We build one HtfContext per rule (with its own indicators precomputed
    # and its no-look-ahead main→htf index mapping). Simulator uses these as
    # entry-gates: an LTF entry signal must be confirmed by every HTF rule.
    htf_contexts: list[HtfContext] = []
    if cfg.htf_rules:
        if not htf_ohlcv:
            declared = [r.timeframe for r in cfg.htf_rules]
            raise ValueError(
                f"Strategy declares htf_rules for {declared} but no htf_ohlcv "
                "was supplied to run_backtest(). The API layer must fetch each "
                "HTF series and pass it as {timeframe: ohlcv}."
            )
        # Normalise each HTF payload through the same loader used for the main df.
        htf_dfs: dict[str, pd.DataFrame] = {}
        for tf, payload in htf_ohlcv.items():
            tf_key = str(tf).strip().lower()
            try:
                htf_dfs[tf_key] = load_ohlcv_data(payload)
            except ValueError as exc:
                raise ValueError(f"htf_ohlcv[{tf!r}] failed to load: {exc}") from exc
        htf_contexts = build_htf_contexts(cfg.htf_rules, htf_dfs, df.index)
        logger.info(
            "🔭 HTF gates active | rules=%s",
            [(r.timeframe, r.condition) for r in cfg.htf_rules],
        )
    elif htf_ohlcv:
        logger.info(
            "HTF OHLCV supplied but strategy has no htf_rules — ignoring."
        )

    # ── Data sufficiency check ─────────────────────────────────────────────────
    required_warmup = max_indicator_warmup(cfg.indicators)
    if cfg.entry_evaluation_mode == "registry":
        required_warmup = max(
            required_warmup,
            estimate_kb_warmup(cfg.entry_signal_rules or []),
        )
    if cfg.exit_evaluation_mode == "registry":
        required_warmup = max(
            required_warmup,
            estimate_kb_warmup(cfg.exit_signal_rules or []),
        )
    _check_data_sufficiency(df, required_warmup, cfg.symbol)

    logger.info(
        "📊 Backtest inputs ready | symbol=%s timeframe=%s objective=%s candles=%s warm_up=%s",
        cfg.symbol, cfg.timeframe, cfg.objective, len(df), required_warmup,
    )

    # ── Compile formula conditions once (Fix 2: pre-parse) ────────────────────
    # Parsing the condition string and walking its AST ~210k× per backtest is
    # the dominant CPU cost. compile_condition() does it once and also tells us
    # every indicator the formulas reference, so we can vectorise them below.
    compiled_entry: CompiledCondition | None = None
    compiled_exit:  CompiledCondition | None = None
    if cfg.entry_evaluation_mode == "formula":
        compiled_entry = compile_condition(cfg.entry_condition or "")
    if cfg.exit_evaluation_mode == "formula":
        compiled_exit = compile_condition(cfg.exit_condition or "")

    # ── Compute indicators (Fix 1: precompute once, never again) ──────────────
    # We merge the YAML-declared indicators with anything the compiled conditions
    # refer to. After this block every indicator the simulator/conditions need
    # exists as a column on `df`, so per-bar evaluation is just a column lookup.
    # Phase 3: trailing/structural-SL specs may reference ATR(N)/EMA(N) columns
    # that the formula doesn't otherwise mention. Inject them so add_all_indicators
    # precomputes them in the same vectorised pass.
    sl_indicator_requirements = _stop_spec_indicator_requirements(
        cfg.stop_loss_spec, cfg.trailing_stop_spec,
    )

    enriched_indicator_config = _merge_indicator_requirements(
        cfg.indicators,
        compiled_entry,
        compiled_exit,
        extra=sl_indicator_requirements,
    )
    df = add_all_indicators(df, enriched_indicator_config)
    _ensure_scalar_indicators(df, compiled_entry, compiled_exit)

    # Phase 6 — structural patterns. We auto-discover any IS_* identifier the
    # entry/exit conditions reference and compute the corresponding columns
    # ONCE here, then merge any per-strategy parameter overrides from
    # cfg.patterns. Order matters: detected requirements first (sets defaults),
    # YAML overrides last (so user-specified windows win).
    pattern_idents: set[str] = set()
    for compiled in (compiled_entry, compiled_exit):
        if compiled is not None:
            pattern_idents.update(compiled.pattern_refs)
    if pattern_idents:
        auto_config = patterns_required_by_identifiers(pattern_idents)
        merged_pattern_config = merge_pattern_configs(auto_config, cfg.patterns)
        df = add_all_patterns(df, merged_pattern_config)
        logger.info(
            "🧩 Patterns precomputed | identifiers=%s overrides=%s",
            sorted(pattern_idents), bool(cfg.patterns),
        )

    # ── Resolve simulation parameters ─────────────────────────────────────────
    # run_config values override strategy YAML values (useful for UI overrides)
    objective          = str(run_config.get("objective") or cfg.objective).lower()
    daily_loss_cap_pct = float(run_config.get("daily_loss_cap_pct") or cfg.daily_loss_cap_pct or 0.0)
    max_trades_per_day = int(run_config.get("max_trades_per_day") or cfg.max_trades_per_day or 0)

    # max_holding_candles: run_config > YAML-derived value
    max_holding_candles_override = run_config.get("max_holding_candles")
    if max_holding_candles_override is not None:
        max_holding_candles: int | None = int(max_holding_candles_override) or None
    else:
        max_holding_candles = cfg.max_holding_candles

    logger.info(
        "⚙️ Simulation params | objective=%s max_holding_candles=%s daily_loss_cap_pct=%.2f max_trades_per_day=%s",
        objective, max_holding_candles, daily_loss_cap_pct, max_trades_per_day,
    )

    # ── Run simulation ────────────────────────────────────────────────────────
    trades, diagnostics = simulate_trades(
        df=df,
        symbol=cfg.symbol,
        entry_condition=cfg.entry_condition,
        exit_condition=cfg.exit_condition,
        compiled_entry=compiled_entry,
        compiled_exit=compiled_exit,
        stop_loss_pct=cfg.stop_loss,
        take_profit_pct=cfg.take_profit,
        slippage_bps=float(run_config.get("slippage_bps", 5.0)),
        commission_bps=float(run_config.get("commission_bps", 2.0)),
        warm_up_candles=required_warmup,
        max_holding_candles=max_holding_candles,
        objective=objective,
        daily_loss_cap_pct=daily_loss_cap_pct,
        max_trades_per_day=max_trades_per_day,
        stt_intraday_sell_pct=float(run_config.get("stt_intraday_sell_pct", 0.025)),
        stt_delivery_pct=float(run_config.get("stt_delivery_pct", 0.1)),
        entry_evaluation_mode=cfg.entry_evaluation_mode,
        exit_evaluation_mode=cfg.exit_evaluation_mode,
        entry_signal_rules=cfg.entry_signal_rules,
        exit_signal_rules=cfg.exit_signal_rules,
        stop_loss_spec=cfg.stop_loss_spec,
        trailing_stop_spec=cfg.trailing_stop_spec,
        htf_contexts=htf_contexts,
        time_exit_spec=cfg.time_exit_spec,
        trade_direction="AUTO",  # Auto-detect from entry signals
    )

    # Derive strategy side from the trades produced
    if trades:
        long_count   = sum(1 for t in trades if str(t.side).upper() == "LONG")
        short_count  = sum(1 for t in trades if str(t.side).upper() == "SHORT")
        strategy_side = "LONG" if long_count >= short_count else "SHORT"
    else:
        strategy_side = "LONG"

    # ── Build result ──────────────────────────────────────────────────────────
    result = build_backtest_result(
        trades=trades,
        df=df,
        backtest_ref_id=backtest_ref_id or str(uuid.uuid4()),
        strategy_name=cfg.name,
        strategy_side=strategy_side,
        starting_balance=float(run_config.get("starting_balance", 10000.0)),
        start_utc=str(market_data_request["from_utc"]),
        end_utc=str(market_data_request["to_utc"]),
        objective=objective,
        diagnostics=diagnostics,
        config={
            "symbol":                market_data_request["symbol"],
            "interval":              market_data_request["interval"],
            "from_utc":              market_data_request["from_utc"],
            "to_utc":                market_data_request["to_utc"],
            "starting_balance":      float(run_config.get("starting_balance", 10000.0)),
            "slippage_bps":          float(run_config.get("slippage_bps", 5.0)),
            "commission_bps":        float(run_config.get("commission_bps", 2.0)),
            "stt_intraday_sell_pct": float(run_config.get("stt_intraday_sell_pct", 0.025)),
            "stt_delivery_pct":      float(run_config.get("stt_delivery_pct", 0.1)),
            "warm_up_candles":       required_warmup,
            "max_holding_candles":   max_holding_candles,
            "objective":             objective,
            "daily_loss_cap_pct":    daily_loss_cap_pct,
            "max_trades_per_day":    max_trades_per_day,
        },
    )

    n_trades = int(result["metrics"].get("total_trades") or 0)
    logger.info(
        "quant_engine|event=backtest_complete|ref=%s|objective=%s|trades_executed=%s|"
        "ending_balance=%.2f|pass=%s",
        result["backtest_ref_id"],
        objective,
        n_trades,
        float(result["metrics"].get("ending_balance") or 0.0),
        result["pass"],
    )
    return result


def _check_data_sufficiency(df, required_warmup: int, symbol: str) -> None:
    """Warn if there are too few candles relative to indicator warm-up requirements."""
    if len(df) < 20:
        raise ValueError(
            f"Insufficient data: only {len(df)} candles were supplied for {symbol}. "
            "Need at least 20 candles to run indicator and trade evaluation."
        )

    eligible_after_warmup = len(df) - required_warmup
    if required_warmup > 0 and eligible_after_warmup < required_warmup:
        logger.warning(
            "⚠️  Warm-up concern | symbol=%s total_candles=%s required_warmup=%s "
            "eligible_candles_after_warmup=%s. "
            "Consider requesting a longer data window to get more entry opportunities. "
            "Rule of thumb: fetch at least %s candles (2× the warm-up period).",
            symbol,
            len(df),
            required_warmup,
            eligible_after_warmup,
            required_warmup * _WARMUP_BUFFER_MULTIPLIER,
        )


def _merge_indicator_requirements(
    yaml_indicators: dict,
    *compiled_conditions: CompiledCondition | None,
    extra: dict | None = None,
) -> dict:
    """
    Merge YAML-declared indicators with anything the compiled conditions reference.

    The AST scan in compile_condition() tells us *exactly* which periodic indicators
    (e.g. RSI(14), EMA(20), SMA(50)) the formula uses. Adding them to the indicator
    config means add_all_indicators() computes them in one vectorised pass instead
    of the simulator falling back to per-bar recomputation later.

    `extra` lets callers add side-channel requirements that aren't visible in any
    formula — e.g. an ATR(14) needed only by the trailing-stop spec.
    """
    merged: dict[str, set[int]] = {}
    for name, periods in (yaml_indicators or {}).items():
        merged.setdefault(str(name).upper(), set()).update(int(p) for p in (periods or []))

    for compiled in compiled_conditions:
        if compiled is None:
            continue
        for ref in compiled.indicator_refs:
            merged.setdefault(ref.name, set()).add(ref.period)

    for name, periods in (extra or {}).items():
        merged.setdefault(str(name).upper(), set()).update(int(p) for p in (periods or []))

    # Preserve list shape that add_all_indicators expects.
    return {name: sorted(periods) for name, periods in merged.items()}


def _stop_spec_indicator_requirements(
    stop_loss_spec: dict | None,
    trailing_stop_spec: dict | None,
) -> dict[str, list[int]]:
    """Extract ATR/EMA period requirements implied by the SL specs.

    The simulator's structural-SL and trailing-SL paths read precomputed
    columns like ATR_14 and EMA_20 directly. Surfacing those here so the
    runner can hand them to add_all_indicators avoids per-bar recomputation
    AND avoids silent NaN when the column is missing.
    """
    out: dict[str, set[int]] = {}
    if isinstance(stop_loss_spec, dict) and stop_loss_spec.get("type") == "atr":
        window = int(stop_loss_spec.get("window", 14))
        out.setdefault("ATR", set()).add(window)
    if isinstance(trailing_stop_spec, dict):
        ts_type = trailing_stop_spec.get("type")
        if ts_type in {"atr", "chandelier"}:
            window = int(trailing_stop_spec.get("window", 14))
            out.setdefault("ATR", set()).add(window)
        elif ts_type == "ema":
            window = int(trailing_stop_spec.get("window", 20))
            out.setdefault("EMA", set()).add(window)
    return {name: sorted(periods) for name, periods in out.items()}


def _ensure_scalar_indicators(
    df,
    *compiled_conditions: CompiledCondition | None,
) -> None:
    """Compute scalar (period-less) indicators — MACD, VWAP — that conditions reference."""
    from engine.indicators import macd_line, macd_signal, vwap

    needed: set[str] = set()
    for compiled in compiled_conditions:
        if compiled is not None:
            needed.update(compiled.scalar_refs)

    if not needed:
        return

    close = df["close"]
    if "MACD" in needed and "MACD" not in df.columns:
        df["MACD"] = macd_line(close)
    if "MACD_SIGNAL" in needed and "MACD_SIGNAL" not in df.columns:
        df["MACD_SIGNAL"] = macd_signal(close)
    if "MACD_HIST" in needed and "MACD_HIST" not in df.columns and "MACD" in df.columns and "MACD_SIGNAL" in df.columns:
        df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]
    if "VWAP" in needed and "VWAP" not in df.columns:
        df["VWAP"] = vwap(df)
