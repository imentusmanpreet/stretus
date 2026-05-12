"""
app/services/execution/risk_manager.py
───────────────────────────────────────
Computes stop-loss price, take-profit price, position size, and principal
amount for a potential entry.

Inputs  (RiskInputs dataclass):
  entry_price           — LTP or last close
  risk_config           — StrategyRiskConfig ORM row  (stop_loss_pct, take_profit_pct,
                          daily_loss_cap_pct, per_trade_risk_pct)
  exec_state            — ExecutionState ORM row  (capital, max_risk_per_trade_pct,
                          min_trade_value, cash_reserve_pct)
  instrument            — InstrumentDefaults dataclass (tick_size, lot_size,
                          upper_circuit, lower_circuit) built from ref_data
  circuit_threshold_pct — fraction of circuit limit at which we refuse to trade
                          (default: settings.market_data_circuit_threshold_pct)

Outputs (RiskOutput dataclass):
  stop_loss_price   — rounded to tick_size
  take_profit_price — rounded to tick_size
  position_size     — quantity (rounded to lot_size multiple)
  principal_amount  — entry_price * position_size
  ok                — False when any hard check fails (min_trade, circuit, etc.)
  messages          — log-friendly decision trail

All calculations are percent-based (the only SL/TP mode for this service).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Safe defaults used when ORM rows are None (fallback only — DB rows preferred)
_DEFAULT_STOP_LOSS_PCT       = 2.0
_DEFAULT_TAKE_PROFIT_PCT     = 5.0
_DEFAULT_TICK_SIZE           = 0.05
_DEFAULT_LOT_SIZE            = 1
_DEFAULT_CAPITAL             = 100_000.0
_DEFAULT_MAX_RISK_PCT        = 2.0
_DEFAULT_MIN_TRADE_VALUE     = 500.0
# Max single-position size as % of capital.
# Prevents tight-SL + risk% combos from generating over-sized positions
# that exceed available margin (e.g. SL=1.5% + risk=2% on ₹1353 stock → 492 shares).
_DEFAULT_MAX_POSITION_PCT    = 20.0   # 20% of capital per position


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RiskInput:
    entry_price: float
    # ORM row references (None triggers safe defaults + warning)
    risk_config: Optional[object] = None     # StrategyRiskConfig
    exec_state: Optional[object] = None      # ExecutionState
    instrument: Optional[object] = None      # InstrumentDefaults (from ref_data_service)
    circuit_threshold_pct: float = field(
        default_factory=lambda: settings.market_data_circuit_threshold_pct
    )


@dataclass
class RiskOutput:
    stop_loss_price: float
    take_profit_price: float
    position_size: int
    principal_amount: float
    ok: bool
    messages: List[str] = field(default_factory=list)


# ── Main class ────────────────────────────────────────────────────────────────

class RiskManager:
    """Stateless; instantiate once per request."""

    def calculate(self, inp: RiskInput) -> RiskOutput:
        """
        Run all risk calculations and normalisations.

        Returns a RiskOutput.  When ok=False, the evaluator must NOT
        generate a bracket order.
        """
        messages: List[str] = []
        entry = inp.entry_price

        # ── 1. Extract parameters (prefer DB rows; fall back to defaults) ──────
        stop_loss_pct, take_profit_pct = self._extract_sl_tp(inp.risk_config, messages)
        capital, max_risk_pct, min_trade_value = self._extract_exec_params(
            inp.exec_state, messages
        )
        tick_size, lot_size, upper_circuit, lower_circuit = self._extract_instrument(
            inp.instrument, messages
        )

        # ── 2. Circuit limit guard ─────────────────────────────────────────────
        if upper_circuit is not None:
            threshold = upper_circuit * inp.circuit_threshold_pct
            if entry >= threshold:
                msg = (
                    f"  🔴 Near upper circuit | entry=₹{entry:.2f} ≥ threshold=₹{threshold:.2f}"
                    f"  (upper_circuit=₹{upper_circuit:.2f}, guard={inp.circuit_threshold_pct*100:.0f}%)"
                    f" — trade blocked."
                )
                messages.append(msg)
                logger.warning(msg)
                return self._blocked(entry, stop_loss_pct, take_profit_pct, tick_size, messages)

        if lower_circuit is not None and entry <= lower_circuit:
            msg = (
                f"  🔴 At/below lower circuit | entry=₹{entry:.2f} ≤ lower=₹{lower_circuit:.2f}"
                f" — trade blocked."
            )
            messages.append(msg)
            logger.warning(msg)
            return self._blocked(entry, stop_loss_pct, take_profit_pct, tick_size, messages)

        messages.append(
            f"  ✅ Circuit check passed | entry=₹{entry:.2f}"
            f"  upper=₹{upper_circuit:.2f}  lower=₹{lower_circuit:.2f}"
        )

        # ── 3. SL / TP price calculation ──────────────────────────────────────
        raw_sl = entry * (1 - stop_loss_pct / 100)
        raw_tp = entry * (1 + take_profit_pct / 100)

        sl_price = self._round_tick(raw_sl, tick_size)
        tp_price = self._round_tick(raw_tp, tick_size)

        messages.append(
            f"  📉 SL | ₹{entry:.2f} × (1 - {stop_loss_pct}%) = ₹{raw_sl:.4f}"
            f" → tick-rounded ₹{sl_price:.4f}"
        )
        messages.append(
            f"  📈 TP | ₹{entry:.2f} × (1 + {take_profit_pct}%) = ₹{raw_tp:.4f}"
            f" → tick-rounded ₹{tp_price:.4f}"
        )

        # ── 4. Position size ──────────────────────────────────────────────────
        # Step A: risk-based sizing (how many shares to lose exactly risk_amount if SL hit)
        risk_amount    = capital * max_risk_pct / 100
        risk_per_share = entry - sl_price

        if risk_per_share <= 0:
            msg = f"  ❌ risk_per_share ₹{risk_per_share:.4f} ≤ 0 — cannot size position."
            messages.append(msg)
            logger.error(msg)
            return RiskOutput(
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
                position_size=0,
                principal_amount=0.0,
                ok=False,
                messages=messages,
            )

        raw_qty  = risk_amount / risk_per_share
        quantity = self._round_lot(raw_qty, lot_size)

        messages.append(
            f"  🔢 Position size | risk_amount=₹{risk_amount:.2f}"
            f"  risk/share=₹{risk_per_share:.4f}"
            f"  raw_qty={raw_qty:.2f}  lot_rounded={quantity}"
        )

        # Step B: capital cap — clamp so principal never exceeds max_position_capital_pct
        # of total capital.  This prevents tight SL + risk% from generating over-sized
        # positions that exceed available margin.
        # Default cap: 20% of capital.  Override via exec_state.max_position_capital_pct.
        max_pos_pct = float(
            getattr(inp.exec_state, "max_position_capital_pct", _DEFAULT_MAX_POSITION_PCT)
            or _DEFAULT_MAX_POSITION_PCT
        )
        max_principal   = capital * max_pos_pct / 100
        max_qty_by_cap  = self._round_lot(max_principal / entry, lot_size)

        if quantity > max_qty_by_cap and max_qty_by_cap > 0:
            messages.append(
                f"  ⚠️  Capital cap applied | risk-based qty={quantity}"
                f" → capped to {max_qty_by_cap}"
                f" ({max_pos_pct:.0f}% of ₹{capital:,.0f} = ₹{max_principal:,.0f})"
            )
            quantity = max_qty_by_cap

        if quantity <= 0:
            messages.append("  ❌ Quantity is 0 after lot rounding — trade blocked.")
            return RiskOutput(
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
                position_size=0,
                principal_amount=0.0,
                ok=False,
                messages=messages,
            )

        # ── 5. Min trade value check ───────────────────────────────────────────
        principal = entry * quantity
        if principal < min_trade_value:
            msg = (
                f"  ❌ Trade value ₹{principal:,.2f} < min_trade_value ₹{min_trade_value:,.2f}"
                f" — trade blocked."
            )
            messages.append(msg)
            logger.warning(msg)
            return RiskOutput(
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
                position_size=quantity,
                principal_amount=principal,
                ok=False,
                messages=messages,
            )

        messages.append(
            f"  ✅ Risk OK | qty={quantity}  principal=₹{principal:,.2f}"
            f"  SL=₹{sl_price:.2f}  TP=₹{tp_price:.2f}"
        )

        return RiskOutput(
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            position_size=quantity,
            principal_amount=principal,
            ok=True,
            messages=messages,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_sl_tp(
        self, risk_config: Optional[object], messages: List[str]
    ) -> tuple[float, float]:
        if risk_config is None:
            messages.append(
                f"  ⚠️  No risk config — using fallback SL={_DEFAULT_STOP_LOSS_PCT}%"
                f"  TP={_DEFAULT_TAKE_PROFIT_PCT}%"
            )
            logger.warning("RiskManager: falling back to default SL/TP (no risk_config)")
            return _DEFAULT_STOP_LOSS_PCT, _DEFAULT_TAKE_PROFIT_PCT

        sl = float(getattr(risk_config, "stop_loss_pct", _DEFAULT_STOP_LOSS_PCT) or _DEFAULT_STOP_LOSS_PCT)
        tp = float(getattr(risk_config, "take_profit_pct", _DEFAULT_TAKE_PROFIT_PCT) or _DEFAULT_TAKE_PROFIT_PCT)
        messages.append(f"  💡 Risk config | SL={sl}%  TP={tp}%")
        return sl, tp

    def _extract_exec_params(
        self, exec_state: Optional[object], messages: List[str]
    ) -> tuple[float, float, float]:
        if exec_state is None:
            messages.append(
                f"  ⚠️  No exec state — using fallback capital=₹{_DEFAULT_CAPITAL:,.0f}"
                f"  max_risk={_DEFAULT_MAX_RISK_PCT}%  min_trade=₹{_DEFAULT_MIN_TRADE_VALUE:,.0f}"
            )
            return _DEFAULT_CAPITAL, _DEFAULT_MAX_RISK_PCT, _DEFAULT_MIN_TRADE_VALUE

        raw_capital = getattr(exec_state, "capital", _DEFAULT_CAPITAL)
        raw_max_risk = getattr(exec_state, "max_risk_per_trade_pct", _DEFAULT_MAX_RISK_PCT)
        raw_min_trade = getattr(exec_state, "min_trade_value", _DEFAULT_MIN_TRADE_VALUE)

        capital = float(_DEFAULT_CAPITAL if raw_capital is None else raw_capital)
        max_risk = float(_DEFAULT_MAX_RISK_PCT if raw_max_risk is None else raw_max_risk)
        min_trade = float(_DEFAULT_MIN_TRADE_VALUE if raw_min_trade is None else raw_min_trade)
        messages.append(
            f"  💡 Exec state | capital=₹{capital:,.0f}"
            f"  max_risk={max_risk}%  min_trade=₹{min_trade:,.0f}"
        )
        return capital, max_risk, min_trade

    def _extract_instrument(
        self, instrument: Optional[object], messages: List[str]
    ) -> tuple[float, int, Optional[float], Optional[float]]:
        if instrument is None:
            messages.append(
                f"  ⚠️  No instrument defaults — using fallback tick=₹{_DEFAULT_TICK_SIZE}"
                f"  lot={_DEFAULT_LOT_SIZE}"
            )
            logger.warning("RiskManager: no instrument defaults; using fallback values")
            return _DEFAULT_TICK_SIZE, _DEFAULT_LOT_SIZE, None, None

        tick  = float(getattr(instrument, "tick_size", _DEFAULT_TICK_SIZE) or _DEFAULT_TICK_SIZE)
        lot   = int(getattr(instrument, "lot_size", _DEFAULT_LOT_SIZE) or _DEFAULT_LOT_SIZE)
        upper = getattr(instrument, "upper_circuit", None)
        lower = getattr(instrument, "lower_circuit", None)
        upper = float(upper) if upper is not None else None
        lower = float(lower) if lower is not None else None
        messages.append(
            f"  💡 Instrument | tick=₹{tick}  lot={lot}"
            f"  upper_circuit=₹{upper:.2f}  lower_circuit=₹{lower:.2f}"
        )
        return tick, lot, upper, lower

    @staticmethod
    def _round_tick(price: float, tick_size: float) -> float:
        """Round price to the nearest tick_size multiple."""
        if tick_size <= 0:
            return round(price, 4)
        return round(round(price / tick_size) * tick_size, 4)

    @staticmethod
    def _round_lot(qty: float, lot_size: int) -> int:
        """Floor qty to the nearest lot_size multiple (minimum 1 lot)."""
        if lot_size <= 1:
            return max(1, math.floor(qty))
        lots = math.floor(qty / lot_size)
        return max(lot_size, lots * lot_size)

    def _blocked(
        self,
        entry: float,
        sl_pct: float,
        tp_pct: float,
        tick_size: float,
        messages: List[str],
    ) -> RiskOutput:
        sl = self._round_tick(entry * (1 - sl_pct / 100), tick_size)
        tp = self._round_tick(entry * (1 + tp_pct / 100), tick_size)
        return RiskOutput(
            stop_loss_price=sl,
            take_profit_price=tp,
            position_size=0,
            principal_amount=0.0,
            ok=False,
            messages=messages,
        )
