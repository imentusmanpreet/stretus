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
                          upper_circuit, lower_circuit, qty_step_size,
                          min_notional) built from ref_data
  circuit_threshold_pct — fraction of circuit limit at which we refuse to trade
                          (default: settings.market_data_circuit_threshold_pct)
  asset_class           — equity_cash or crypto_spot. Drives:
                            • position-size rounding (integer lot for equity
                              vs fractional qty_step for crypto)
                            • min-trade value vs Binance min_notional

Outputs (RiskOutput dataclass):
  stop_loss_price   — rounded to tick_size
  take_profit_price — rounded to tick_size
  position_size     — Decimal quantity (lot-rounded for equity, qty_step-rounded
                      for crypto). Equity values are integer-valued Decimals.
  principal_amount  — entry_price * position_size
  ok                — False when any hard check fails (min_trade, circuit, etc.)
  messages          — log-friendly decision trail

All calculations are percent-based (the only SL/TP mode for this service).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from decimal import ROUND_DOWN, Decimal
from typing import List, Optional

from app.core.config import get_settings
from app.schemas.execution import AssetClass

logger = logging.getLogger(__name__)
settings = get_settings()

# Safe defaults used when ORM rows are None (fallback only — DB rows preferred)
_DEFAULT_STOP_LOSS_PCT       = 0.25
_DEFAULT_TAKE_PROFIT_PCT     = 0.5
_DEFAULT_TICK_SIZE           = 0.05
_DEFAULT_LOT_SIZE            = 1
_DEFAULT_CAPITAL             = 100_000.0
_DEFAULT_MAX_RISK_PCT        = 2.0
_DEFAULT_MIN_TRADE_VALUE     =10.0
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
    asset_class: AssetClass = AssetClass.equity_cash


@dataclass
class RiskOutput:
    stop_loss_price: float
    take_profit_price: float
    # ``Decimal`` so equity (int-valued) and crypto (fractional) sizing share
    # one return shape. The TradeManager normalises before constructing
    # OrderLeg, and the RiskSnapshot exposes the same Decimal in the API.
    position_size: Decimal
    principal_amount: float
    ok: bool
    messages: List[str] = field(default_factory=list)


# ── Main class ────────────────────────────────────────────────────────────────

class RiskManager:
    """Stateless; instantiate once per request."""

    def calculate(self, inp: RiskInput) -> RiskOutput:
        """
        Run all risk calculations and normalisations.

        Returns a RiskOutput. When ok=False, the evaluator must NOT
        generate a bracket order.
        """
        messages: List[str] = []
        entry = inp.entry_price
        asset_class = inp.asset_class
        is_crypto = asset_class == AssetClass.crypto_spot
        currency_sym = "$" if is_crypto else "₹"

        # ── 1. Extract parameters (prefer DB rows; fall back to defaults) ──────
        stop_loss_pct, take_profit_pct = self._extract_sl_tp(inp.risk_config, messages)
        capital, max_risk_pct, min_trade_value = self._extract_exec_params(
            inp.exec_state, messages, currency_sym,
        )
        tick_size, lot_size, upper_circuit, lower_circuit, qty_step, min_notional = (
            self._extract_instrument(inp.instrument, messages, currency_sym)
        )

        # For crypto we honour the Binance min_notional (denominated in the
        # quote asset, e.g. USDT) on top of the user's min_trade_value.
        effective_min_trade = max(min_trade_value, min_notional or 0.0)

        # ── 2. Circuit / price-band guard ──────────────────────────────────────
        # Binance 24h bands are not exchange circuit limits — skip for crypto.
        if upper_circuit is not None and not is_crypto:
            threshold = upper_circuit * inp.circuit_threshold_pct
            if entry >= threshold:
                msg = (
                    f"  🔴 Near upper bound | entry={currency_sym}{entry:.4f} ≥ "
                    f"threshold={currency_sym}{threshold:.4f}"
                    f"  (upper={currency_sym}{upper_circuit:.4f}, "
                    f"guard={inp.circuit_threshold_pct*100:.0f}%) — trade blocked."
                )
                messages.append(msg)
                logger.warning(msg)
                return self._blocked(
                    entry, stop_loss_pct, take_profit_pct, tick_size, messages,
                )

        if lower_circuit is not None and not is_crypto and entry <= lower_circuit:
            msg = (
                f"  🔴 At/below lower bound | entry={currency_sym}{entry:.4f} ≤ "
                f"lower={currency_sym}{lower_circuit:.4f} — trade blocked."
            )
            messages.append(msg)
            logger.warning(msg)
            return self._blocked(
                entry, stop_loss_pct, take_profit_pct, tick_size, messages,
            )

        messages.append(
            f"  ✅ Price-band check passed | entry={currency_sym}{entry:.4f}"
            f"  upper={currency_sym}{(upper_circuit or 0):.4f}"
            f"  lower={currency_sym}{(lower_circuit or 0):.4f}"
        )

        # ── 3. SL / TP price calculation ──────────────────────────────────────
        raw_sl = entry * (1 - stop_loss_pct / 100)
        raw_tp = entry * (1 + take_profit_pct / 100)

        sl_price = self._round_tick(raw_sl, tick_size)
        tp_price = self._round_tick(raw_tp, tick_size)

        messages.append(
            f"  📉 SL | {currency_sym}{entry:.4f} × (1 - {stop_loss_pct}%) = "
            f"{currency_sym}{raw_sl:.6f} → tick-rounded {currency_sym}{sl_price:.6f}"
        )
        messages.append(
            f"  📈 TP | {currency_sym}{entry:.4f} × (1 + {take_profit_pct}%) = "
            f"{currency_sym}{raw_tp:.6f} → tick-rounded {currency_sym}{tp_price:.6f}"
        )

        # ── 4. Position size ──────────────────────────────────────────────────
        risk_amount   = capital * max_risk_pct / 100
        risk_per_unit = entry - sl_price

        if risk_per_unit <= 0:
            msg = f"  ❌ risk_per_unit {currency_sym}{risk_per_unit:.6f} ≤ 0 — cannot size."
            messages.append(msg)
            logger.error(msg)
            return RiskOutput(
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
                position_size=Decimal(0),
                principal_amount=0.0,
                ok=False,
                messages=messages,
            )

        raw_qty = risk_amount / risk_per_unit
        quantity = self._round_quantity(
            raw_qty,
            asset_class=asset_class,
            lot_size=lot_size,
            qty_step=qty_step,
        )

        messages.append(
            f"  🔢 Position size | risk_amount={currency_sym}{risk_amount:.2f}"
            f"  risk/unit={currency_sym}{risk_per_unit:.6f}"
            f"  raw_qty={raw_qty:.8f}  rounded={quantity}"
        )

        # Capital cap — same logic for both asset classes.
        max_pos_pct = float(
            getattr(inp.exec_state, "max_position_capital_pct", _DEFAULT_MAX_POSITION_PCT)
            or _DEFAULT_MAX_POSITION_PCT
        )
        max_principal  = capital * max_pos_pct / 100
        max_qty_by_cap = self._round_quantity(
            max_principal / entry,
            asset_class=asset_class,
            lot_size=lot_size,
            qty_step=qty_step,
        )

        if quantity > max_qty_by_cap and max_qty_by_cap > 0:
            messages.append(
                f"  ⚠️  Capital cap applied | risk-based qty={quantity}"
                f" → capped to {max_qty_by_cap}"
                f" ({max_pos_pct:.0f}% of {currency_sym}{capital:,.2f} = "
                f"{currency_sym}{max_principal:,.2f})"
            )
            quantity = max_qty_by_cap

        if quantity <= 0:
            messages.append("  ❌ Quantity is 0 after rounding — trade blocked.")
            return RiskOutput(
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
                position_size=Decimal(0),
                principal_amount=0.0,
                ok=False,
                messages=messages,
            )

        # ── 5. Min trade value check ───────────────────────────────────────────
        principal = float(entry * float(quantity))
        if principal < effective_min_trade:
            msg = (
                f"  ❌ Trade value {currency_sym}{principal:,.4f} < "
                f"min_trade={currency_sym}{effective_min_trade:,.4f} — trade blocked."
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
            f"  ✅ Risk OK | qty={quantity}  principal={currency_sym}{principal:,.4f}"
            f"  SL={currency_sym}{sl_price:.6f}  TP={currency_sym}{tp_price:.6f}"
        )

        return RiskOutput(
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            position_size=quantity,
            principal_amount=principal,
            ok=True,
            messages=messages,
        )

    def compute_tp_ladder(
        self,
        entry_price: float,
        stop_loss_price: float,
        total_quantity: Decimal,
        tp_targets: List[dict],
        asset_class: AssetClass,
        tick_size: float = 0.05,
        lot_size: int = 1,
        qty_step: Optional[float] = None,
    ) -> "TpLadder":
        """Resolve multi-TP rung specs to absolute prices and per-rung quantities.

        Resolves prices at entry time so R-multiple rungs honour the actual fill and
        SL distance rather than any pre-computed approximation.
        """
        from app.schemas.execution import TpLadder, TpRungState  # local to avoid circular import

        risk_per_unit = abs(entry_price - stop_loss_price) or (entry_price * 0.01)
        is_short = stop_loss_price > entry_price
        rungs: list[TpRungState] = []

        for rung in tp_targets:
            tp_type     = str(rung.get("type", "risk_reward")).lower()
            value       = float(rung.get("value", 1.0))
            frac        = float(rung.get("exit_fraction", 0.0))
            risk_action = str(rung.get("risk_action", "none"))
            rung_index  = int(rung.get("rung_index", len(rungs)))

            if tp_type == "risk_reward":
                raw_price = (
                    entry_price - value * risk_per_unit if is_short
                    else entry_price + value * risk_per_unit
                )
            else:  # percent
                raw_price = (
                    entry_price * (1.0 - value / 100.0) if is_short
                    else entry_price * (1.0 + value / 100.0)
                )
            rounded_price = self._round_tick(raw_price, tick_size)
            raw_qty  = float(total_quantity) * frac
            exit_qty = self._round_quantity(raw_qty, asset_class=asset_class, lot_size=lot_size, qty_step=qty_step)

            rungs.append(TpRungState(
                rung_index=rung_index,
                price=rounded_price,
                exit_fraction=frac,
                exit_qty=exit_qty,
                hit=False,
                risk_action=risk_action,
            ))

        return TpLadder(rungs=rungs, total_quantity=total_quantity)

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
        self,
        exec_state: Optional[object],
        messages: List[str],
        currency_sym: str = "₹",
    ) -> tuple[float, float, float]:
        if exec_state is None:
            messages.append(
                f"  ⚠️  No exec state — using fallback capital={currency_sym}{_DEFAULT_CAPITAL:,.2f}"
                f"  max_risk={_DEFAULT_MAX_RISK_PCT}%  min_trade={currency_sym}{_DEFAULT_MIN_TRADE_VALUE:,.2f}"
            )
            return _DEFAULT_CAPITAL, _DEFAULT_MAX_RISK_PCT, _DEFAULT_MIN_TRADE_VALUE

        raw_capital = getattr(exec_state, "capital", _DEFAULT_CAPITAL)
        raw_max_risk = getattr(exec_state, "max_risk_per_trade_pct", _DEFAULT_MAX_RISK_PCT)
        raw_min_trade = getattr(exec_state, "min_trade_value", _DEFAULT_MIN_TRADE_VALUE)

        capital = float(_DEFAULT_CAPITAL if raw_capital is None else raw_capital)
        max_risk = float(_DEFAULT_MAX_RISK_PCT if raw_max_risk is None else raw_max_risk)
        min_trade = float(_DEFAULT_MIN_TRADE_VALUE if raw_min_trade is None else raw_min_trade)
        messages.append(
            f"  💡 Exec state | capital={currency_sym}{capital:,.2f}"
            f"  max_risk={max_risk}%  min_trade={currency_sym}{min_trade:,.2f}"
        )
        return capital, max_risk, min_trade

    def _extract_instrument(
        self,
        instrument: Optional[object],
        messages: List[str],
        currency_sym: str = "₹",
    ) -> tuple[float, int, Optional[float], Optional[float], Optional[float], Optional[float]]:
        if instrument is None:
            messages.append(
                f"  ⚠️  No instrument defaults — using fallback tick={currency_sym}{_DEFAULT_TICK_SIZE}"
                f"  lot={_DEFAULT_LOT_SIZE}"
            )
            logger.warning("RiskManager: no instrument defaults; using fallback values")
            return _DEFAULT_TICK_SIZE, _DEFAULT_LOT_SIZE, None, None, None, None

        tick  = float(getattr(instrument, "tick_size", _DEFAULT_TICK_SIZE) or _DEFAULT_TICK_SIZE)
        lot   = int(getattr(instrument, "lot_size", _DEFAULT_LOT_SIZE) or _DEFAULT_LOT_SIZE)
        upper = getattr(instrument, "upper_circuit", None)
        lower = getattr(instrument, "lower_circuit", None)
        qty_step = getattr(instrument, "qty_step_size", None)
        min_notional = getattr(instrument, "min_notional", None)
        upper = float(upper) if upper is not None else None
        lower = float(lower) if lower is not None else None
        qty_step = float(qty_step) if qty_step is not None else None
        min_notional = float(min_notional) if min_notional is not None else None
        messages.append(
            f"  💡 Instrument | tick={currency_sym}{tick}  lot={lot}"
            f"  upper={currency_sym}{(upper or 0):.4f}  lower={currency_sym}{(lower or 0):.4f}"
            f"  qty_step={qty_step}  min_notional={min_notional}"
        )
        return tick, lot, upper, lower, qty_step, min_notional

    @staticmethod
    def _round_tick(price: float, tick_size: float) -> float:
        """Round price to the nearest tick_size multiple."""
        if tick_size <= 0:
            return round(price, 8)
        # Use Decimal to avoid float-arithmetic drift at small tick sizes
        # (e.g. crypto 0.00000001 ticks).
        d_price = Decimal(str(price))
        d_tick  = Decimal(str(tick_size))
        steps   = (d_price / d_tick).quantize(Decimal("1"), rounding=ROUND_DOWN)
        result  = (steps * d_tick)
        return float(result)

    @staticmethod
    def _round_lot(qty: float, lot_size: int) -> int:
        """Floor qty to the nearest lot_size multiple (minimum 1 lot)."""
        if lot_size <= 1:
            return max(1, math.floor(qty))
        lots = math.floor(qty / lot_size)
        return max(lot_size, lots * lot_size)

    @classmethod
    def _round_quantity(
        cls,
        raw_qty: float,
        *,
        asset_class: AssetClass,
        lot_size: int,
        qty_step: Optional[float],
    ) -> Decimal:
        """
        Floor a raw quantity to the venue-allowed step.

          equity_cash : nearest integer lot (Decimal-valued integer).
          crypto_spot : floor to the qty_step multiple (e.g. 0.00001 BTC).

        Returns a ``Decimal`` so the caller can compare / multiply without
        re-introducing float drift.
        """
        if asset_class == AssetClass.crypto_spot:
            step = Decimal(str(qty_step)) if qty_step and qty_step > 0 else Decimal("0.00000001")
            d_raw = Decimal(str(raw_qty))
            if step <= 0:
                return d_raw.quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
            steps = (d_raw / step).quantize(Decimal("1"), rounding=ROUND_DOWN)
            return steps * step

        # equity_cash
        return Decimal(cls._round_lot(raw_qty, lot_size))

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
            position_size=Decimal(0),
            principal_amount=0.0,
            ok=False,
            messages=messages,
        )
