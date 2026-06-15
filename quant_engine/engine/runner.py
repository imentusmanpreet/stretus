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
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from engine.conditions import CompiledCondition, compile_condition
from engine.config import BACKTEST_MARKET_DATA_FROM_UTC, BACKTEST_MARKET_DATA_TO_UTC
from engine.data import load_ohlcv_data, merge_reference_data
from engine.htf import HtfContext, build_htf_contexts, timeframe_to_timedelta
from engine.resample import build_subbar_slices, resample_ohlcv
from engine.indicators import add_all_indicators, max_indicator_warmup
from engine.patterns import (
    add_all_patterns,
    merge_pattern_configs,
    patterns_required_by_identifiers,
)
from engine.loader import load_strategy_from_content
from engine.metrics import build_backtest_result
from engine.kb_signals import estimate_kb_warmup
from engine.simulator import build_minute_arrays, simulate_trades

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
    normalized_request = dict(market_data_request)
    from_utc = str(
        normalized_request.get("from_utc") or BACKTEST_MARKET_DATA_FROM_UTC
    )
    to_utc = str(
        normalized_request.get("to_utc") or BACKTEST_MARKET_DATA_TO_UTC
    )
    start_ts = _parse_window_timestamp(from_utc)
    end_ts = _parse_window_timestamp(to_utc)
    trimmed_df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()

    normalized_request["from_utc"] = from_utc
    normalized_request["to_utc"] = to_utc

    if len(trimmed_df) != len(df):
        logger.info(
            "Enforced backtest market-data window | from=%s to=%s "
            "input_rows=%s trimmed_rows=%s",
            from_utc,
            to_utc,
            len(df),
            len(trimmed_df),
        )

    if trimmed_df.empty:
        raise ValueError(
            "No OHLCV data is available inside the configured backtest window "
            f"{from_utc} to {to_utc}."
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
    pipeline_start = time.perf_counter()
    logger.info("🧮 Starting backtest pipeline")

    # Step 1: Load strategy configuration
    step_start = time.perf_counter()
    cfg = _load_strategy_config_from_input(yaml_content)
    logger.info(
        "⏱️  TIMING|step=load_strategy_config|duration=%.4fs",
        time.perf_counter() - step_start,
    )
    
    # Step 2: Load and normalize OHLCV data
    step_start = time.perf_counter()
    df  = load_ohlcv_data(ohlcv_data)
    df, market_data_request = _enforce_market_data_window(df, market_data_request)

    # ── Phase 11: 1-minute execution ──────────────────────────────────────────
    # When run_config["intrabar_execution"] is set, the supplied OHLCV is the
    # 1-minute series. We keep it (`minute_df`) for execution and resample it up
    # to the strategy timeframe so EVERY downstream step — indicator warm-up,
    # condition evaluation, HTF mapping, metrics — sees exactly the strategy-
    # timeframe frame it expects. Only the trade fills and SL/TP/trailing
    # resolution drop to the minute series, by walking each strategy bar's
    # minutes (see simulator.simulate_trades / engine.resample). Default off so
    # callers still supplying pre-aggregated data are completely unaffected.
    intrabar_execution = bool(run_config.get("intrabar_execution"))
    minute_df: pd.DataFrame | None = None
    if intrabar_execution:
        minute_df = df
        df = resample_ohlcv(minute_df, cfg.timeframe)
        logger.info(
            "🕐 1-minute execution | timeframe=%s minute_bars=%s strategy_bars=%s",
            cfg.timeframe, len(minute_df), len(df),
        )
    logger.info(
        "⏱️  TIMING|step=load_normalize_ohlcv|duration=%.4fs|rows=%d",
        time.perf_counter() - step_start,
        len(df),
    )

    # ── Phase 4: reference symbol merge ───────────────────────────────────────
    if cfg.reference_symbol:
        step_start = time.perf_counter()
        if not reference_ohlcv:
            raise ValueError(
                f"Strategy declares reference_symbol={cfg.reference_symbol!r} "
                "but no reference_ohlcv was supplied to run_backtest(). "
                "The API layer must fetch the reference series and pass it in."
            )
        ref_df = load_ohlcv_data(reference_ohlcv)
        # In 1-minute execution the reference series arrives at 1m too; resample
        # it to the strategy timeframe so REF_ columns align with the (resampled)
        # main frame's bars rather than picking a single intrabar minute.
        if intrabar_execution:
            ref_df = resample_ohlcv(ref_df, cfg.timeframe)
        df = merge_reference_data(df, ref_df)
        logger.info(
            "⏱️  TIMING|step=merge_reference_data|duration=%.4fs|reference_symbol=%s|ref_rows=%s",
            time.perf_counter() - step_start,
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
        step_start = time.perf_counter()
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
            "⏱️  TIMING|step=build_htf_contexts|duration=%.4fs|rules=%s",
            time.perf_counter() - step_start,
            [(r.timeframe, r.condition) for r in cfg.htf_rules],
        )
    elif htf_ohlcv:
        logger.info(
            "HTF OHLCV supplied but strategy has no htf_rules — ignoring."
        )

    # ── Data sufficiency check ─────────────────────────────────────────────────
    step_start = time.perf_counter()
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
        "⏱️  TIMING|step=data_sufficiency_check|duration=%.4fs|required_warmup=%d",
        time.perf_counter() - step_start,
        required_warmup,
    )

    logger.info(
        "📊 Backtest inputs ready | symbol=%s timeframe=%s objective=%s candles=%s warm_up=%s",
        cfg.symbol, cfg.timeframe, cfg.objective, len(df), required_warmup,
    )

    # ── Compile formula conditions once (Fix 2: pre-parse) ────────────────────
    # Parsing the condition string and walking its AST ~210k× per backtest is
    # the dominant CPU cost. compile_condition() does it once and also tells us
    # every indicator the formulas reference, so we can vectorise them below.
    step_start = time.perf_counter()
    compiled_entry: CompiledCondition | None = None
    compiled_exit:  CompiledCondition | None = None
    if cfg.entry_evaluation_mode == "formula":
        compiled_entry = compile_condition(cfg.entry_condition or "")
    if cfg.exit_evaluation_mode == "formula":
        compiled_exit = compile_condition(cfg.exit_condition or "")

    # ── Phase 12: resolve direction → which side(s) to simulate ───────────────
    # A "both" strategy runs TWO passes (long + short); the long leg uses the
    # main entry/exit conditions and the short leg uses short_entry/exit. A
    # "short_only" strategy has its single leg in the main conditions, so the
    # short pass just reuses those. Each pass is single-sided in the simulator.
    direction = (cfg.direction or "both").lower().strip()
    run_long  = direction in ("long_only", "both")
    run_short = direction in ("short_only", "both")

    compiled_short_entry: CompiledCondition | None = None
    compiled_short_exit:  CompiledCondition | None = None
    if direction == "short_only":
        # The only leg lives in the main fields — the short pass reuses them.
        short_entry_condition = cfg.entry_condition or ""
        short_exit_condition  = cfg.exit_condition or ""
        compiled_short_entry, compiled_short_exit = compiled_entry, compiled_exit
        run_long = False
    else:
        short_entry_condition = cfg.short_entry_condition or ""
        short_exit_condition  = cfg.short_exit_condition or ""
        # Only run the short pass for "both" when a short leg was actually given.
        if direction == "both" and not short_entry_condition:
            run_short = False
        if run_short and cfg.entry_evaluation_mode == "formula" and short_entry_condition:
            compiled_short_entry = compile_condition(short_entry_condition)
        if run_short and cfg.exit_evaluation_mode == "formula" and short_exit_condition:
            compiled_short_exit = compile_condition(short_exit_condition)

    logger.info(
        "⏱️  TIMING|step=compile_conditions|duration=%.4fs|direction=%s|run_long=%s|run_short=%s",
        time.perf_counter() - step_start, direction, run_long, run_short,
    )

    # ── Compute indicators (Fix 1: precompute once, never again) ──────────────
    # We merge the YAML-declared indicators with anything the compiled conditions
    # refer to. After this block every indicator the simulator/conditions need
    # exists as a column on `df`, so per-bar evaluation is just a column lookup.
    # Phase 3: trailing/structural-SL specs may reference ATR(N)/EMA(N) columns
    # that the formula doesn't otherwise mention. Inject them so add_all_indicators
    # precomputes them in the same vectorised pass.
    step_start = time.perf_counter()
    sl_indicator_requirements = _stop_spec_indicator_requirements(
        cfg.stop_loss_spec, cfg.trailing_stop_spec,
    )

    # Phase 2 — the volatility-band gate reads ATR_{w}/NATR_{w} directly, so inject
    # that column even when no formula mentions it (same shape as the SL specs).
    extra_requirements: dict[str, list[int]] = {
        name: list(periods) for name, periods in sl_indicator_requirements.items()
    }
    if cfg.vol_filter_metric in ("atr", "natr") and (
        cfg.vol_filter_min is not None or cfg.vol_filter_max is not None
    ):
        key = cfg.vol_filter_metric.upper()
        extra_requirements.setdefault(key, []).append(int(cfg.vol_filter_window))

    enriched_indicator_config = _merge_indicator_requirements(
        cfg.indicators,
        compiled_entry,
        compiled_exit,
        compiled_short_entry,
        compiled_short_exit,
        extra=extra_requirements,
    )
    df = add_all_indicators(df, enriched_indicator_config)
    _ensure_scalar_indicators(df, compiled_entry, compiled_exit, compiled_short_entry, compiled_short_exit)
    logger.info(
        "⏱️  TIMING|step=compute_indicators|duration=%.4fs|indicators=%s",
        time.perf_counter() - step_start,
        list(enriched_indicator_config.keys()),
    )

    # Phase 6 — structural patterns. We auto-discover any IS_* identifier the
    # entry/exit conditions reference and compute the corresponding columns
    # ONCE here, then merge any per-strategy parameter overrides from
    # cfg.patterns. Order matters: detected requirements first (sets defaults),
    # YAML overrides last (so user-specified windows win).
    pattern_idents: set[str] = set()
    for compiled in (compiled_entry, compiled_exit, compiled_short_entry, compiled_short_exit):
        if compiled is not None:
            pattern_idents.update(compiled.pattern_refs)
    if pattern_idents:
        step_start = time.perf_counter()
        auto_config = patterns_required_by_identifiers(pattern_idents)
        merged_pattern_config = merge_pattern_configs(auto_config, cfg.patterns)
        df = add_all_patterns(df, merged_pattern_config)
        logger.info(
            "⏱️  TIMING|step=compute_patterns|duration=%.4fs|identifiers=%s|overrides=%s",
            time.perf_counter() - step_start,
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

    # ── Phase 11: build the 1-minute execution arrays for the simulator ───────
    # Map each (resampled) strategy bar to the half-open slice of minute bars
    # inside its window, and extract the minute OHLC arrays. No-look-ahead is
    # guaranteed by build_subbar_slices (a bar only ever sees its own minutes).
    minute_arrays = None
    subbar_starts = None
    subbar_ends = None
    if intrabar_execution and minute_df is not None:
        td = timeframe_to_timedelta(cfg.timeframe)
        subbar_starts, subbar_ends = build_subbar_slices(df.index, minute_df.index, td)
        minute_arrays = build_minute_arrays(minute_df)

    # ── Run simulation ────────────────────────────────────────────────────────
    # Phase 12 — a "both" strategy is two single-sided passes (long + short) that
    # share every gate/cost/risk setting and differ only in `side` and the
    # entry/exit conditions. _run_pass wraps simulate_trades with the shared
    # kwargs so each pass is one call.
    step_start = time.perf_counter()

    def _run_pass(
        side, p_entry_condition, p_exit_condition, p_compiled_entry, p_compiled_exit,
        *, p_entry_mode=None, p_exit_mode=None, p_entry_rules=None, p_exit_rules=None,
    ):
        return simulate_trades(
            df=df,
            symbol=cfg.symbol,
            side=side,
            entry_condition=p_entry_condition,
            exit_condition=p_exit_condition,
            compiled_entry=p_compiled_entry,
            compiled_exit=p_compiled_exit,
            entry_evaluation_mode=p_entry_mode if p_entry_mode is not None else cfg.entry_evaluation_mode,
            exit_evaluation_mode=p_exit_mode if p_exit_mode is not None else cfg.exit_evaluation_mode,
            entry_signal_rules=p_entry_rules if p_entry_mode is not None else cfg.entry_signal_rules,
            exit_signal_rules=p_exit_rules if p_exit_mode is not None else cfg.exit_signal_rules,
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
            stop_loss_spec=cfg.stop_loss_spec,
            trailing_stop_spec=cfg.trailing_stop_spec,
            trailing_take_profit_spec=cfg.trailing_take_profit_spec,
            htf_contexts=htf_contexts,
            time_exit_spec=cfg.time_exit_spec,
            # Phase 10 — new entry gates and circuit breakers
            entry_window_start_utc=cfg.entry_window_start_utc,
            entry_window_end_utc=cfg.entry_window_end_utc,
            max_consecutive_losses=cfg.max_consecutive_losses,
            cooldown_bars_after_loss=cfg.cooldown_bars_after_loss,
            cooldown_bars_after_profit=cfg.cooldown_bars_after_profit,
            max_spread_bps=cfg.max_spread_bps,
            gap_filter=cfg.gap_filter,
            gap_threshold_pct=cfg.gap_threshold_pct,
            entry_confirmation_bars=cfg.entry_confirmation_bars,
            rsi_entry_band_min=cfg.rsi_entry_band_min,
            rsi_entry_band_max=cfg.rsi_entry_band_max,
            volume_ratio_threshold=cfg.volume_ratio_threshold,
            # Volatility / regime filters
            vol_filter_metric=cfg.vol_filter_metric,
            vol_filter_window=cfg.vol_filter_window,
            vol_filter_min=cfg.vol_filter_min,
            vol_filter_max=cfg.vol_filter_max,
            regime_filter_allowed=cfg.regime_filter_allowed,
            rs_filter_window=cfg.rs_filter_window,
            rs_filter_min_ratio=cfg.rs_filter_min_ratio,
            event_skip_dates=cfg.event_skip_dates,
            lunch_lull_start_utc=cfg.lunch_lull_start_utc,
            lunch_lull_end_utc=cfg.lunch_lull_end_utc,
            # Phase 11 — 1-minute execution (None unless intrabar_execution is set)
            minute_arrays=minute_arrays,
            subbar_starts=subbar_starts,
            subbar_ends=subbar_ends,
        )

    trades: list = []
    diagnostics: list = []
    if run_long:
        long_trades, diagnostics = _run_pass(
            "LONG", cfg.entry_condition, cfg.exit_condition, compiled_entry, compiled_exit,
        )
        trades.extend(long_trades)
    if run_short:
        if direction == "short_only":
            # The single short leg reuses the main conditions AND their eval
            # mode/rules (which may be registry).
            short_kwargs = {}
        else:
            # "both": the short leg comes from the SDL compiler as a FORMULA
            # string, so force formula evaluation regardless of the long leg's
            # mode (there are no separate short registry rules).
            short_kwargs = dict(
                p_entry_mode="formula", p_exit_mode="formula",
                p_entry_rules=None, p_exit_rules=None,
            )
        short_trades, short_diag = _run_pass(
            "SHORT", short_entry_condition, short_exit_condition,
            compiled_short_entry, compiled_short_exit, **short_kwargs,
        )
        trades.extend(short_trades)
        # Diagnostics are per-bar and single-sided; keep the long pass's as the
        # primary view and fall back to the short pass's when long didn't run.
        if not diagnostics:
            diagnostics = short_diag
    # Interleave both passes' trades back into chronological order.
    trades.sort(key=lambda t: str(t.entry_date))

    simulation_duration = time.perf_counter() - step_start
    logger.info(
        "⏱️  TIMING|step=simulate_trades|duration=%.4fs|trades=%d|candles=%d",
        simulation_duration,
        len(trades),
        len(df),
    )

    # Derive strategy side from the trades produced (LONG if majority are long)
    if trades:
        long_count   = sum(1 for t in trades if str(t.side).upper() == "LONG")
        strategy_side = "LONG" if long_count >= len(trades) / 2 else "SHORT"
    else:
        strategy_side = "LONG"

    # ── Build result ──────────────────────────────────────────────────────────
    step_start = time.perf_counter()
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
    metrics_duration = time.perf_counter() - step_start
    logger.info(
        "⏱️  TIMING|step=build_metrics|duration=%.4fs",
        metrics_duration,
    )

    pipeline_duration = time.perf_counter() - pipeline_start
    n_trades = int(result["metrics"].get("total_trades") or 0)
    logger.info(
        "⏱️  TIMING|step=COMPLETE_PIPELINE|duration=%.4fs|"
        "quant_engine|event=backtest_complete|ref=%s|objective=%s|trades_executed=%s|"
        "ending_balance=%.2f|pass=%s",
        pipeline_duration,
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
    """Compute scalar (period-less) studies — MACD, VWAP, OBV, SAR, STOCH_*,
    CDL_*, … — that conditions reference, so the simulator's fast array path can
    read them as columns. Single source of truth: delegates to conditions, which
    builds each via the same engine.indicators code add_all_indicators() uses."""
    from engine.conditions import ensure_scalar_columns

    needed: set[str] = set()
    for compiled in compiled_conditions:
        if compiled is not None:
            needed.update(compiled.scalar_refs)

    if needed:
        ensure_scalar_columns(df, needed)
