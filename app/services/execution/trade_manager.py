"""
app/services/execution/trade_manager.py
────────────────────────────────────────
Builds the final bracket order (entry + SL + TP legs) and exit instructions.

The backend is a STATELESS decision engine — it generates instructions only.
No orders are submitted here.

Multi-asset notes
─────────────────
``build_bracket_order`` and ``build_exit_instruction`` both take an
``asset_class`` parameter which selects the broker / venue conventions
applied to each leg:

  equity_cash (Indian retail brokers, Upstox/Kite family)
    product_type   : MIS (intraday) | CNC (positional)
    order_type     : LIMIT (entry, TP), SL-M (SL)
    validity       : DAY
    quantity       : integer share count

  crypto_spot (Binance spot)
    product_type   : SPOT
    order_type     : LIMIT (entry, TP), STOP_LOSS (SL trigger order)
    validity       : GTC
    quantity       : fractional Decimal in base asset (e.g. 0.00125 BTC)

``idempotency_key`` (on BracketOrder) = SHA-256 fingerprint of
strategy_id + symbol + bar time + asset_class, hex-truncated. OMS / clients
use it as the industry-standard idempotency token so the same bar evaluated
twice does not place duplicate orders.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Union

from app.schemas.execution import (
    AssetClass,
    BracketOrder,
    BracketOrderLegRole,
    ExchangeOrderType,
    ExitInstruction,
    OpenPosition,
    OrderLeg,
    OrderValidity,
    ProductType,
)

logger = logging.getLogger(__name__)


# Quantities arrive as either ``int`` (equity shares) or ``Decimal`` (crypto
# base-asset amount). ``OrderLeg.quantity`` is typed ``Decimal`` so we always
# normalise to that. ``float`` is rejected at the public boundary to keep the
# precision contract obvious.
QuantityInput = Union[int, Decimal]


def _as_decimal(qty: QuantityInput) -> Decimal:
    if isinstance(qty, Decimal):
        return qty
    if isinstance(qty, int):
        return Decimal(qty)
    raise TypeError(
        f"Order quantity must be int or Decimal (got {type(qty).__name__}). "
        "Floats are rejected to preserve precision — convert via Decimal(str(x))."
    )


# ── Asset-class → venue conventions ───────────────────────────────────────────

def _product_type(
    strategy_type: str,
    asset_class: AssetClass,
) -> ProductType:
    if asset_class == AssetClass.crypto_spot:
        return ProductType.spot
    # equity_cash (default)
    return ProductType.mis if strategy_type.lower() == "intraday" else ProductType.cnc


def _entry_order_type(asset_class: AssetClass) -> ExchangeOrderType:
    # Both venue families support LIMIT for entry; we keep the same conservative
    # entry leg across asset classes so risk is bounded by the limit price.
    return ExchangeOrderType.limit


def _stop_loss_order_type(asset_class: AssetClass) -> ExchangeOrderType:
    if asset_class == AssetClass.crypto_spot:
        # Binance Spot ``STOP_LOSS`` = market order triggered at stopPrice.
        # Mirrors the SL-M semantics used on the equity path.
        return ExchangeOrderType.stop_loss
    return ExchangeOrderType.sl_m


def _take_profit_order_type(asset_class: AssetClass) -> ExchangeOrderType:
    # Both venue families: a resting LIMIT order at the TP price is the
    # cleanest representation of a guaranteed-price take profit.
    return ExchangeOrderType.limit


def _default_validity(asset_class: AssetClass) -> OrderValidity:
    if asset_class == AssetClass.crypto_spot:
        return OrderValidity.gtc
    return OrderValidity.day


# ── TradeManager ──────────────────────────────────────────────────────────────

class TradeManager:
    """Stateless bracket order builder; instantiate once per request."""

    def build_bracket_order(
        self,
        symbol: str,
        quantity: QuantityInput,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        strategy_type: str,
        strategy_id: Optional[str],
        mode: str,
        bar_datetime: Optional[str] = None,
        *,
        asset_class: AssetClass = AssetClass.equity_cash,
    ) -> BracketOrder:
        """
        Construct a long (BUY) bracket order.

        Equity defaults preserved exactly when ``asset_class=equity_cash``:
          product=MIS/CNC, order_type=LIMIT/SL-M, validity=DAY, qty=int.

        Crypto path:
          product=SPOT, entry/TP=LIMIT, SL=STOP_LOSS, validity=GTC,
          qty=Decimal (fractional base asset).
        """
        qty_decimal     = _as_decimal(quantity)
        product         = _product_type(strategy_type, asset_class)
        entry_type      = _entry_order_type(asset_class)
        sl_type         = _stop_loss_order_type(asset_class)
        tp_type         = _take_profit_order_type(asset_class)
        validity        = _default_validity(asset_class)
        idempotency_key = self._idempotency_key(strategy_id, symbol, bar_datetime, asset_class)

        entry_id = str(uuid.uuid4())
        sl_id    = str(uuid.uuid4())
        tp_id    = str(uuid.uuid4())

        # Price rounding: 4 decimals is fine for equity (paise level) and a
        # safe upper bound for most crypto pairs. Per-instrument tick-size
        # rounding has already been applied upstream by RiskManager.
        entry_px = round(float(entry_price), 8 if asset_class == AssetClass.crypto_spot else 4)
        sl_px    = round(float(stop_loss_price),   8 if asset_class == AssetClass.crypto_spot else 4)
        tp_px    = round(float(take_profit_price), 8 if asset_class == AssetClass.crypto_spot else 4)

        entry_order = OrderLeg(
            order_id=entry_id,
            parent_order_id=None,
            order_role=BracketOrderLegRole.entry,
            symbol=symbol,
            side="BUY",
            quantity=qty_decimal,
            price=entry_px,
            order_type=entry_type,
            product_type=product,
            validity=validity,
        )

        sl_order = OrderLeg(
            order_id=sl_id,
            parent_order_id=entry_id,
            order_role=BracketOrderLegRole.stop_loss_exit,
            symbol=symbol,
            side="SELL",
            quantity=qty_decimal,
            # SL leg is trigger-driven: trigger_price = stopPrice on Binance
            # and Upstox alike; the order fires as market once touched.
            trigger_price=sl_px,
            price=sl_px,
            order_type=sl_type,
            product_type=product,
            validity=validity,
        )

        tp_order = OrderLeg(
            order_id=tp_id,
            parent_order_id=entry_id,
            order_role=BracketOrderLegRole.take_profit_exit,
            symbol=symbol,
            side="SELL",
            quantity=qty_decimal,
            price=tp_px,
            order_type=tp_type,
            product_type=product,
            validity=validity,
        )

        metadata = {
            "strategy_id":  strategy_id,
            "mode":         mode,
            "asset_class":  asset_class.value,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "product_type": product.value,
            "validity":     validity.value,
        }

        logger.info(
            "Bracket order built | asset_class=%s symbol=%s qty=%s "
            "entry=%s sl=%s tp=%s product=%s validity=%s key=%s",
            asset_class.value, symbol, qty_decimal,
            entry_px, sl_px, tp_px,
            product.value, validity.value, idempotency_key,
        )

        return BracketOrder(
            idempotency_key=idempotency_key,
            entry_order=entry_order,
            stop_loss_order=sl_order,
            take_profit_order=tp_order,
            metadata=metadata,
        )

    def build_exit_instruction(
        self,
        position: OpenPosition,
        reason: str,
        exit_price: float,
        strategy_type: str,
        *,
        asset_class: AssetClass = AssetClass.equity_cash,
    ) -> ExitInstruction:
        """Return an exit instruction for an existing open position."""
        rounded = round(
            float(exit_price),
            8 if asset_class == AssetClass.crypto_spot else 4,
        )
        return ExitInstruction(
            position_id=position.position_id,
            symbol=position.symbol,
            reason=reason,  # type: ignore[arg-type]
            exit_price=rounded,
            quantity=position.quantity,
            product_type=_product_type(strategy_type, asset_class),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _product_type(strategy_type: str) -> ProductType:
        """Back-compat shim — equity-only call sites continue to work."""
        return _product_type(strategy_type, AssetClass.equity_cash)

    @staticmethod
    def _idempotency_key(
        strategy_id: Optional[str],
        symbol: str,
        bar_datetime: Optional[str],
        asset_class: AssetClass = AssetClass.equity_cash,
    ) -> str:
        raw = (
            f"{strategy_id or 'none'}::{asset_class.value}::{symbol}::"
            f"{bar_datetime or datetime.now(timezone.utc).isoformat()}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()[:32]
