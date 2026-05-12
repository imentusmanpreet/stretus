"""
app/services/execution/trade_manager.py
────────────────────────────────────────
Builds the final bracket order (entry + SL + TP legs) and exit instructions.

The backend is a STATELESS decision engine — it generates instructions only.
No orders are submitted here.

`idempotency_key` (on BracketOrder) = SHA-256 fingerprint of
strategy_id + symbol + bar time, hex-truncated. OMS / clients use it as the
industry-standard idempotency token (same as many REST `Idempotency-Key` flows)
so the same bar evaluated twice does not place duplicate orders.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.schemas.execution import (
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


class TradeManager:
    """Stateless bracket order builder; instantiate once per request."""

    def build_bracket_order(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        strategy_type: str,
        strategy_id: Optional[str],
        mode: str,
        bar_datetime: Optional[str] = None,
    ) -> BracketOrder:
        """
        Construct a bracket order for a long (BUY) entry.

        product_type: MIS for intraday, CNC for positional.
        idempotency_key: stable fingerprint; also set on BracketOrder for OMS dedupe.
        """
        product_type = self._product_type(strategy_type)
        idempotency_key = self._idempotency_key(strategy_id, symbol, bar_datetime)

        entry_id = str(uuid.uuid4())
        sl_id = str(uuid.uuid4())
        tp_id = str(uuid.uuid4())

        entry_order = OrderLeg(
            order_id=entry_id,
            parent_order_id=None,
            order_role=BracketOrderLegRole.entry,
            symbol=symbol,
            side="BUY",
            quantity=quantity,
            price=round(entry_price, 4),
            order_type=ExchangeOrderType.limit,
            product_type=product_type,
            validity=OrderValidity.day,
        )

        # Stop-loss leg — sell at or below sl_price (SL-M for guaranteed fill)
        sl_order = OrderLeg(
            order_id=sl_id,
            parent_order_id=entry_id,
            order_role=BracketOrderLegRole.stop_loss_exit,
            symbol=symbol,
            side="SELL",
            quantity=quantity,
            trigger_price=round(stop_loss_price, 4),
            price=round(stop_loss_price, 4),
            order_type=ExchangeOrderType.sl_m,
            product_type=product_type,
            validity=OrderValidity.day,
        )

        # Take-profit leg — sell at tp_price (LIMIT)
        tp_order = OrderLeg(
            order_id=tp_id,
            parent_order_id=entry_id,
            order_role=BracketOrderLegRole.take_profit_exit,
            symbol=symbol,
            side="SELL",
            quantity=quantity,
            price=round(take_profit_price, 4),
            order_type=ExchangeOrderType.limit,
            product_type=product_type,
            validity=OrderValidity.day,
        )

        metadata = {
            "strategy_id": strategy_id,
            "mode": mode,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "product_type": product_type.value,
        }

        logger.info(
            "Bracket order built | symbol=%s qty=%d entry=%.4f sl=%.4f tp=%.4f key=%s",
            symbol, quantity, entry_price, stop_loss_price, take_profit_price, idempotency_key,
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
    ) -> ExitInstruction:
        """Return an exit instruction for an existing open position."""
        return ExitInstruction(
            position_id=position.position_id,
            symbol=position.symbol,
            reason=reason,  # type: ignore[arg-type]
            exit_price=round(exit_price, 4),
            quantity=position.quantity,
            product_type=self._product_type(strategy_type),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _product_type(strategy_type: str) -> ProductType:
        return ProductType.mis if strategy_type.lower() == "intraday" else ProductType.cnc

    @staticmethod
    def _idempotency_key(
        strategy_id: Optional[str],
        symbol: str,
        bar_datetime: Optional[str],
    ) -> str:
        raw = f"{strategy_id or 'none'}::{symbol}::{bar_datetime or datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]
