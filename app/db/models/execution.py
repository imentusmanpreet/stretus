"""
app/db/models/execution.py
───────────────────────────
ORM models for the strategy execution service.

Tables (all in ai_strategy schema):
  - strategy_risk_config  : 1-to-1 with strategies; stores SL/TP pct + all
                            risk_and_execution fields as typed columns.
  - instrument_metadata   : per-symbol tick/lot size for normalisation.
  - execution_states      : per-strategy runtime capital & position state.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.session import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ══════════════════════════════════════════════════════════════════════════════
# ai_strategy.strategy_risk_config
# ══════════════════════════════════════════════════════════════════════════════
class StrategyRiskConfig(Base):
    """
    Persists the computed risk_and_execution snapshot for a confirmed strategy.

    Written once by confirm_strategy(); read by the execution service
    RiskManager so it never relies on hardcoded defaults at evaluation time.
    """

    __tablename__ = "strategy_risk_config"
    __table_args__ = (
        UniqueConstraint("strategy_id", name="uq_risk_config_strategy_id"),
        {"schema": "ai_strategy"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    strategy_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai_strategy.strategies.id", ondelete="CASCADE"),
        nullable=False,
    )

    # ── SL / TP (numeric pct) ────────────────────────────────────────────────
    stop_loss_pct: Mapped[float] = mapped_column(Numeric(6, 3), nullable=False, default=2.0)
    take_profit_pct: Mapped[float] = mapped_column(Numeric(6, 3), nullable=False, default=5.0)

    # ── Risk sizing ──────────────────────────────────────────────────────────
    daily_loss_cap_pct: Mapped[float] = mapped_column(Numeric(6, 3), nullable=False, default=3.0)
    per_trade_risk_pct: Mapped[float] = mapped_column(Numeric(6, 3), nullable=False, default=2.0)
    max_trades_per_day: Mapped[int] = mapped_column(Integer, nullable=False, default=2)

    # ── Derived display fields ───────────────────────────────────────────────
    risk_reward: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ── Execution policy fields ──────────────────────────────────────────────
    execution_mode: Mapped[str] = mapped_column(Text, nullable=False, default="Backtest")
    position_sizing: Mapped[str] = mapped_column(Text, nullable=False, default="Risk based")
    trading_window: Mapped[str] = mapped_column(
        Text, nullable=False, default="9:15 - 15:30"
    )
    risk_validation: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="system risk guardials",
    )

    # ── Timestamps ───────────────────────────────────────────────────────────
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        server_default=func.now(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# ai_strategy.instrument_metadata
# ══════════════════════════════════════════════════════════════════════════════
class InstrumentMetadata(Base):
    """
    Exchange instrument details required for tick/lot normalisation and live quotes.

    Multi-asset shape (post-migration 0006):
      ``asset_class``     — 'equity_cash' | 'crypto_spot'
      ``adapter_id``      — 'upstox_rest' | 'binance_rest'
      ``adapter_symbol``  — venue-native symbol (e.g. 'NSE_EQ|INE002A01018' for
                            Upstox, 'BTCUSDT' for Binance)
      ``qty_step_size``   — fractional crypto qty step (NULL for equity)
      ``min_notional``    — venue-enforced min trade value in quote asset
                            (USDT for crypto; NULL for equity, where the
                            risk-config min_trade_value is the equivalent)

    Legacy column ``upstox_instrument_key`` is preserved (mirrored into
    ``adapter_symbol`` by the migration backfill) so existing equity callers
    continue to work without code changes.
    """

    __tablename__ = "instrument_metadata"
    __table_args__ = {"schema": "ai_strategy"}

    symbol: Mapped[str] = mapped_column(Text, primary_key=True)
    tick_size: Mapped[float] = mapped_column(Numeric(10, 5), nullable=False, default=0.05)
    lot_size: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    upper_circuit: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)
    lower_circuit: Mapped[Optional[float]] = mapped_column(Numeric(12, 4), nullable=True)

    # Legacy Upstox key (kept for backwards compatibility).
    upstox_instrument_key: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="Upstox v2 instrument key, e.g. NSE_EQ|INE002A01018. Used for live LTP/circuit.",
    )

    # Multi-asset additions (nullable for back-compat).
    asset_class: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="Asset class (equity_cash, crypto_spot). Mirrors ref_data.",
    )
    adapter_id: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="Venue adapter id (upstox_rest, binance_rest).",
    )
    adapter_symbol: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="Venue-native symbol (NSE_EQ|INE... for Upstox, BTCUSDT for Binance).",
    )
    qty_step_size: Mapped[Optional[float]] = mapped_column(
        Numeric(28, 10), nullable=True,
        comment="Fractional qty step (crypto). NULL for equity.",
    )
    min_notional: Mapped[Optional[float]] = mapped_column(
        Numeric(28, 10), nullable=True,
        comment="Min trade value enforced by the venue (USDT for crypto).",
    )

    last_refreshed: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# ai_strategy.execution_states
# ══════════════════════════════════════════════════════════════════════════════
class ExecutionState(Base):
    """
    Per-strategy runtime state used by the execution service.

    Stores capital allocation, position limits, cooldown tracking,
    and an optional OMS-sourced open positions snapshot (state_json).
    Created with system defaults on first evaluate/execute call if absent.
    """

    __tablename__ = "execution_states"
    __table_args__ = (
        UniqueConstraint("strategy_id", name="uq_exec_state_strategy_id"),
        {"schema": "ai_strategy"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    strategy_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai_strategy.strategies.id", ondelete="CASCADE"),
        nullable=False,
    )

    # ── Capital / sizing ─────────────────────────────────────────────────────
    capital: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False, default=100_000.0)
    max_risk_per_trade_pct: Mapped[float] = mapped_column(
        Numeric(6, 3), nullable=False, default=2.0
    )
    max_open_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    cash_reserve_pct: Mapped[float] = mapped_column(Numeric(6, 3), nullable=False, default=0.10)
    cooldown_bars: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    min_trade_value: Mapped[float] = mapped_column(Numeric(12, 4), nullable=False, default=500.0)

    # ── Runtime counters ─────────────────────────────────────────────────────
    bars_since_last_trade: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # ── OMS snapshot ─────────────────────────────────────────────────────────
    state_json: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Latest open_positions snapshot from OMS (updated per evaluate call)",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        server_default=func.now(),
    )
