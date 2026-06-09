"""
app/planner/param_resolver.py — Param + SL/TP resolution.

Resolution order for signal params (highest authority first):
  1. Performance cache  — learned-best params for (signal, symbol, timeframe)
  2. OHLCV estimator    — statistical estimate from recent candles
  3. Card defaults      — params_by_timeframe[tf] from the SignalCard

For SL/TP:
  base values from RiskTier are scaled by realized ATR%, so a high-vol stock
  like ADANIENT gets a wider SL automatically without any per-stock config.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from app.core.signal_param_estimator import estimate_params_for_signal
from app.core.signal_performance_cache import get_best_params
from app.kb.schemas import RiskTier, SignalCard

logger = logging.getLogger(__name__)

# Volatility baseline used when scaling SL/TP. Roughly the median ATR% of large-cap
# Indian stocks on the daily timeframe; the multiplier is what matters, not the absolute.
_BASELINE_ATR_PCT = 1.5
_VOL_MULT_MIN = 0.5
_VOL_MULT_MAX = 3.0
_MIN_BARS_FOR_VOL = 20


# ── Param resolution ──────────────────────────────────────────────────────────


def resolve_params(
    card: SignalCard,
    *,
    timeframe: str,
    symbol: str,
    objective: str = "intraday",
    ohlcv: pd.DataFrame | None = None,
) -> tuple[dict[str, float | int], str]:
    """Resolve params for a signal. Returns (params, source_label)."""
    card_default = dict(card.params_by_timeframe.get(timeframe, {}))

    # 1. Performance cache (learned over backtests)
    if symbol:
        cached = get_best_params(card.name, symbol, timeframe, card_default)
        if cached and cached != card_default:
            logger.info(
                "param_resolver|signal=%s|symbol=%s|tf=%s|source=perf_cache|params=%s",
                card.name, symbol, timeframe, cached,
            )
            return cached, "perf_cache"

    # 2. OHLCV-based estimator
    if isinstance(ohlcv, pd.DataFrame) and len(ohlcv) >= _MIN_BARS_FOR_VOL:
        try:
            estimated = estimate_params_for_signal(
                signal_name=card.name,
                df=ohlcv,
                objective=objective,
                timeframe=timeframe,
                yaml_params=card_default,
            )
            if estimated and estimated != card_default:
                logger.info(
                    "param_resolver|signal=%s|symbol=%s|tf=%s|source=ohlcv_estimator|params=%s",
                    card.name, symbol, timeframe, estimated,
                )
                return estimated, "ohlcv_estimator"
        except Exception as exc:
            logger.warning(
                "param_resolver|estimator_failed|signal=%s|err=%s",
                card.name, exc,
            )

    # 3. Card defaults
    logger.info(
        "param_resolver|signal=%s|symbol=%s|tf=%s|source=card_default|params=%s",
        card.name, symbol, timeframe, card_default,
    )
    return card_default, "card_default"


# ── SL/TP volatility scaling ──────────────────────────────────────────────────


def _atr_pct(ohlcv: pd.DataFrame, window: int = 14) -> float | None:
    """Return ATR as a percentage of last close. None if not computable."""
    if not isinstance(ohlcv, pd.DataFrame) or len(ohlcv) < window + 1:
        return None
    cols = {c.lower(): c for c in ohlcv.columns}
    hi = ohlcv[cols.get("high", "High")]
    lo = ohlcv[cols.get("low", "Low")]
    cl = ohlcv[cols.get("close", "Close")]
    prev_cl = cl.shift(1)
    tr = pd.concat([(hi - lo).abs(), (hi - prev_cl).abs(), (lo - prev_cl).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window).mean().iloc[-1]
    last_close = float(cl.iloc[-1])
    if last_close <= 0 or pd.isna(atr):
        return None
    return float(atr) / last_close * 100.0


def parse_risk_reward(raw: Any) -> float | None:
    """Parse a user-supplied risk:reward into the TP/SL multiple.

    Accepted forms (returns the TP/SL ratio):
        "1:2"  → 2.0     "1:3"  → 3.0     "3:1"  → 3.0     "2:1"  → 2.0
        "2"    → 2.0     2.5    → 2.5     "  "   → None
    Returns None for blank, malformed, or non-positive values so the caller
    can fall back to ATR-scaling.

    Note on ``N:1`` vs ``1:N``: traders write a risk:reward ratio in either
    order but almost always mean the SAME thing — "N reward per 1 unit of risk".
    So "3:1 risk-reward" means 3.0 (3x reward), not 0.33. Whenever one side of
    the ratio is exactly 1, the other side IS the reward-to-risk multiple,
    regardless of which side it's on. Only when neither side is 1 do we fall
    back to the literal risk-first reading (reward/risk).
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        if ":" in text:
            a_str, b_str = text.split(":", 1)
            a = float(a_str.strip())
            b = float(b_str.strip())
            if a <= 0 or b <= 0:
                return None
            # "N:1" and "1:N" both mean N-to-1 reward:risk.
            if a == 1.0:
                return b
            if b == 1.0:
                return a
            return b / a   # ambiguous "A:B" — literal risk-first reading
        value = float(text)
        return value if value > 0 else None
    except (TypeError, ValueError):
        return None


# Regime-based SL multiplier adjustments.
# In ranging markets, tighter SL is viable (less noise, clearer structure).
# In volatile/trending markets, wider SL needed to avoid getting stopped out.
_REGIME_SL_MULT: dict[str, float] = {
    "trending_up":   1.1,
    "trending_down": 1.1,
    "ranging":       0.85,
    "volatile":      1.4,
}
# Regime-based TP multiplier adjustments.
# In trending markets, let winners run (wider TP); in ranging, take profit sooner.
_REGIME_TP_MULT: dict[str, float] = {
    "trending_up":   1.2,
    "trending_down": 1.2,
    "ranging":       0.9,
    "volatile":      1.0,
}


# A take-profit must never be tighter than the stop-loss: that is a risk:reward
# below 1:1 (you'd risk more than you aim to gain), which the plan validator
# forbids outright. If volatility/regime scaling or a sub-1 user ratio would
# produce tp < sl, we floor the TP at the SL (a 1:1 minimum) instead of letting
# the planner raise and crash the whole chat flow.
_MIN_RISK_REWARD = 1.0


def _enforce_min_rr(sl_pct: float, tp_pct: float, user_rr: Any) -> float:
    """Clamp tp_pct up to sl_pct when it would otherwise fall below a 1:1 RR."""
    floor = round(sl_pct * _MIN_RISK_REWARD, 2)
    if tp_pct < floor:
        logger.warning(
            "param_resolver|tp_floored_to_min_rr|sl=%.2f%%|tp_in=%.2f%%|tp_out=%.2f%%"
            "|user_rr=%r|reason=risk_reward_below_1:1_is_not_allowed",
            sl_pct, tp_pct, floor, user_rr,
        )
        return floor
    return tp_pct


def resolve_sl_tp(
    risk_tier: RiskTier,
    *,
    ohlcv: pd.DataFrame | None = None,
    user_risk_reward: Any = None,
    regime: dict | None = None,
) -> tuple[float, float, float]:
    """Return (sl_pct, tp_pct, vol_mult). vol_mult==1.0 if OHLCV unavailable.

    When `user_risk_reward` is a parseable RR ("1:2", "3", 2.5), TP is set to
    SL × parsed_ratio so the plan honors the user's request instead of using
    the ATR-scaled base_tp. SL itself is still ATR-scaled so it adapts to
    each stock's realized volatility.

    When `regime` is provided (from classify_regime()), both SL and TP are
    additionally scaled by a regime-specific multiplier — wider in volatile/trending,
    tighter in ranging markets.
    """
    base_sl = risk_tier.base_sl_pct
    base_tp = risk_tier.base_tp_pct
    user_rr = parse_risk_reward(user_risk_reward)

    # Regime-aware multipliers (applied on top of volatility scaling)
    regime_type = (regime or {}).get("type", "ranging")
    regime_sl_mult = _REGIME_SL_MULT.get(regime_type, 1.0)
    regime_tp_mult = _REGIME_TP_MULT.get(regime_type, 1.0)

    atr_pct = _atr_pct(ohlcv) if ohlcv is not None else None
    if atr_pct is None or atr_pct <= 0:
        sl_pct = round(base_sl * regime_sl_mult, 2)
        if user_rr is not None:
            tp_pct = round(sl_pct * user_rr, 2)
            tp_source = "user_rr"
        else:
            tp_pct = round(base_tp * regime_tp_mult, 2)
            tp_source = "ai_default"
            logger.warning(
                "param_resolver|tp_ai_defaulted|sl=%.2f%%|tp=%.2f%%|regime=%s"
                "|reason=user_did_not_provide_risk_reward_or_take_profit"
                "|action_required=chat_layer_should_gate_planner_until_rms_complete",
                sl_pct, tp_pct, regime_type,
            )
        tp_pct = _enforce_min_rr(sl_pct, tp_pct, user_rr)
        logger.info(
            "param_resolver|sl_tp|regime=%s|sl=%.2f%%|tp=%.2f%%|tp_source=%s"
            "|atr_unavailable=true",
            regime_type, sl_pct, tp_pct, tp_source,
        )
        return sl_pct, tp_pct, 1.0

    raw_mult = atr_pct / _BASELINE_ATR_PCT
    vol_mult = max(_VOL_MULT_MIN, min(_VOL_MULT_MAX, raw_mult))
    sl_pct   = round(base_sl * vol_mult * regime_sl_mult, 2)
    if user_rr is not None:
        tp_pct = round(sl_pct * user_rr, 2)
        tp_source = "user_rr"
    else:
        tp_pct = round(base_tp * vol_mult * regime_tp_mult, 2)
        tp_source = "ai_default"
        logger.warning(
            "param_resolver|tp_ai_defaulted|sl=%.2f%%|tp=%.2f%%|regime=%s"
            "|reason=user_did_not_provide_risk_reward_or_take_profit"
            "|action_required=chat_layer_should_gate_planner_until_rms_complete",
            sl_pct, tp_pct, regime_type,
        )

    tp_pct = _enforce_min_rr(sl_pct, tp_pct, user_rr)
    logger.info(
        "param_resolver|sl_tp|regime=%s|regime_sl_mult=%.2f|regime_tp_mult=%.2f"
        "|vol_mult=%.2f|sl=%.2f%%|tp=%.2f%%|tp_source=%s",
        regime_type, regime_sl_mult, regime_tp_mult, vol_mult, sl_pct, tp_pct, tp_source,
    )
    return sl_pct, tp_pct, round(vol_mult, 3)


# ── Risk dict (for execution layer) ───────────────────────────────────────────


def build_risk_dict(risk_tier: RiskTier) -> dict[str, Any]:
    """Build the `risk` block consumed by /strategy/evaluate/execute."""
    return {
        "max_risk_per_trade_pct":    risk_tier.per_trade_risk_pct,
        "max_open_positions":        risk_tier.max_open_positions,
        "cash_reserve_pct":          0.1,
        "cooldown_bars_after_loss":  risk_tier.cooldown_bars_after_loss,
        "min_trade_value":           risk_tier.min_trade_value,
        # Phase 10 — new tier defaults surfaced to execution layer
        "max_consecutive_losses":    risk_tier.max_consecutive_losses,
        "cooldown_bars_after_profit": risk_tier.cooldown_bars_after_profit,
        "max_spread_bps":            risk_tier.max_spread_bps,
        "entry_confirmation_bars":   risk_tier.entry_confirmation_bars,
        "max_capital_allocation_pct": risk_tier.max_capital_allocation_pct,
        "position_sizing_mode":      risk_tier.position_sizing_mode,
    }
