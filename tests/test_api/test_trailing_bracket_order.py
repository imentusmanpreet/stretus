"""Phase 3 — the bracket order carries the trailing spec for the OMS.

A trailing take-profit leg becomes a TRIGGER order (it fires on a pull-back, not a
resting LIMIT) and carries the trailing block; a trailing stop rides on the SL leg.
Without trailing the bracket is byte-for-byte the legacy shape.
"""
from __future__ import annotations

from decimal import Decimal

from app.schemas.execution import AssetClass, ExchangeOrderType, TrailingOrderSpec
from app.services.execution.trade_manager import TradeManager


def _tm() -> TradeManager:
    return TradeManager()


def _ttp() -> TrailingOrderSpec:
    return TrailingOrderSpec(distance_pct=0.5, activate_after_pct=1.0)


def test_crypto_trailing_take_profit_leg_is_trigger_with_block():
    b = _tm().build_bracket_order(
        "BTC_USDT", Decimal("0.01"), 100.0, 98.0, 101.0,
        "intraday", "sid", "paper", asset_class=AssetClass.crypto_spot,
        trailing_take_profit=_ttp(),
    )
    tp = b.take_profit_order
    # Trigger type (not a resting LIMIT), trailing block present, trigger_price set.
    assert tp.order_type == ExchangeOrderType.take_profit
    assert tp.trailing is not None and tp.trailing.distance_pct == 0.5
    assert tp.trailing.activate_after_pct == 1.0
    assert tp.trigger_price == 101.0
    # The stop leg is untouched when only the take-profit trails.
    assert b.stop_loss_order.trailing is None


def test_equity_trailing_take_profit_uses_sl_m_trigger():
    b = _tm().build_bracket_order(
        "RELIANCE.NS", 5, 100.0, 98.0, 105.0,
        "intraday", "sid", "paper", asset_class=AssetClass.equity_cash,
        trailing_take_profit=_ttp(),
    )
    assert b.take_profit_order.order_type == ExchangeOrderType.sl_m
    assert b.take_profit_order.trailing is not None


def test_trailing_stop_rides_on_sl_leg_tp_unchanged():
    b = _tm().build_bracket_order(
        "RELIANCE.NS", 5, 100.0, 98.0, 102.0,
        "intraday", "sid", "paper", asset_class=AssetClass.equity_cash,
        trailing_stop=TrailingOrderSpec(distance_pct=1.0),
    )
    assert b.stop_loss_order.trailing is not None
    assert b.stop_loss_order.trailing.distance_pct == 1.0
    # The take-profit leg stays a resting LIMIT when only the stop trails.
    assert b.take_profit_order.order_type == ExchangeOrderType.limit
    assert b.take_profit_order.trailing is None


def test_no_trailing_is_byte_for_byte_legacy():
    b = _tm().build_bracket_order(
        "RELIANCE.NS", 5, 100.0, 98.0, 102.0,
        "intraday", "sid", "paper", asset_class=AssetClass.equity_cash,
    )
    assert b.take_profit_order.order_type == ExchangeOrderType.limit
    assert b.take_profit_order.trailing is None
    assert b.stop_loss_order.trailing is None
    assert b.take_profit_order.trigger_price is None
