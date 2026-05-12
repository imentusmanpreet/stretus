"""
app/db/models/strategy.py
──────────────────────────
SQLAlchemy ORM models that mirror the strategy.* PostgreSQL tables.

CHANGES:
  - chat_id renamed to session_id in Strategy and ChatMessage FK columns.
  - ChatMessage: added status, updated_at, error_message, parent_message_id,
    is_final columns for async processing support.
  - Added MessageStatus enum: pending / processing / completed / failed.

Note: we map to the existing schema rather than letting SQLAlchemy create it,
so MetaData is given schema="ai_strategy" and we set __table_args__ accordingly.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.db.session import Base

# ── Python-level Enums (mirror the PostgreSQL ENUM types) ──────────────────────
import enum


class MessageStatus(str, enum.Enum):
    """Tracks async processing state of an assistant message."""
    pending    = "pending"
    processing = "processing"
    completed  = "completed"
    failed     = "failed"


class ChatRole(str, enum.Enum):
    user      = "user"
    assistant = "assistant"
    system    = "system"


class StrategyStatus(str, enum.Enum):
    draft              = "draft"
    confirmed          = "confirmed"
    backtesting        = "backtesting"
    backtest_complete  = "backtest_complete"
    live               = "live"
    paused             = "paused"
    archived           = "archived"


class BacktestStatus(str, enum.Enum):
    pending   = "pending"
    running   = "running"
    completed = "completed"
    failed    = "failed"


# ── Helper: UTC now ────────────────────────────────────────────────────────────
def _utcnow():
    return datetime.now(timezone.utc)


# ══════════════════════════════════════════════════════════════════════════════
# ai_strategy.chats
# ══════════════════════════════════════════════════════════════════════════════
class Chat(Base):
    __tablename__ = "chats"
    __table_args__ = {"schema": "ai_strategy"}

    id         : Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id    : Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    title      : Mapped[Optional[str]]       = mapped_column(Text, nullable=True)
    created_at : Mapped[datetime]         = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at : Mapped[datetime]         = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )

    # Relationships
    messages   : Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage", back_populates="chat", cascade="all, delete-orphan"
    )
    strategies : Mapped[list["Strategy"]] = relationship(
        "Strategy", back_populates="chat", cascade="all, delete-orphan"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ai_strategy.strategies
# ══════════════════════════════════════════════════════════════════════════════
class Strategy(Base):
    __tablename__ = "strategies"
    __table_args__ = {"schema": "ai_strategy"}

    id              : Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id      : Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai_strategy.chats.id", ondelete="CASCADE"),
        nullable=False,
    )
    name            : Mapped[str]           = mapped_column(Text, nullable=False)
    symbol          : Mapped[str]           = mapped_column(Text, nullable=False)
    market          : Mapped[str]           = mapped_column(Text, nullable=False)
    timeframe       : Mapped[str]           = mapped_column(Text, nullable=False, server_default='1d')
    status          : Mapped[StrategyStatus] = mapped_column(
        Enum(StrategyStatus, schema="ai_strategy", name="strategy_status", create_type=False),
        default=StrategyStatus.draft,
        nullable=False,
    )
    yaml_path       : Mapped[Optional[str]]    = mapped_column(Text, nullable=True)
    strategy_config : Mapped[Optional[dict]]   = mapped_column(JSONB, nullable=True)
    created_at      : Mapped[datetime]      = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at      : Mapped[datetime]      = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )

    # Relationships
    chat      : Mapped["Chat"]          = relationship("Chat", back_populates="strategies", foreign_keys=[session_id])
    backtests : Mapped[list["Backtest"]] = relationship(
        "Backtest", back_populates="strategy", cascade="all, delete-orphan"
    )
    messages  : Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage", back_populates="strategy", foreign_keys="ChatMessage.strategy_id"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ai_strategy.backtest
# ══════════════════════════════════════════════════════════════════════════════
class Backtest(Base):
    __tablename__ = "backtest"
    __table_args__ = {"schema": "ai_strategy"}

    id            : Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    strategy_id   : Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai_strategy.strategies.id", ondelete="CASCADE"),
        nullable=False,
    )
    status        : Mapped[BacktestStatus] = mapped_column(
        Enum(BacktestStatus, schema="ai_strategy", name="backtest_status", create_type=False),
        default=BacktestStatus.pending,
        nullable=False,
    )
    result_json   : Mapped[Optional[dict]]  = mapped_column(JSONB, nullable=True)
    error_message : Mapped[Optional[str]]   = mapped_column(Text, nullable=True)
    started_at    : Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at  : Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at    : Mapped[datetime]        = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    # Relationships
    strategy : Mapped["Strategy"]        = relationship("Strategy", back_populates="backtests")
    messages : Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage", back_populates="backtest", foreign_keys="ChatMessage.backtest_id"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ai_strategy.chat_messages
# ══════════════════════════════════════════════════════════════════════════════
class ChatMessage(Base):
    __tablename__ = "chat_messages"
    __table_args__ = (
        Index("idx_chat_messages_chat",           "session_id", "created_at"),
        Index("idx_chat_messages_role",           "role"),
        Index("idx_chat_messages_status",         "status"),
        Index(
            "idx_chat_messages_backtest_id",      "backtest_id",
            postgresql_where="backtest_id IS NOT NULL",
        ),
        Index(
            "idx_chat_messages_strategy_id",      "strategy_id",
            postgresql_where="strategy_id IS NOT NULL",
        ),
        Index(
            "idx_chat_messages_ref_backtest_id",  "ref_backtest_id",
            postgresql_where="ref_backtest_id IS NOT NULL",
        ),
        {"schema": "ai_strategy"},
    )

    id              : Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id      : Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai_strategy.chats.id", ondelete="CASCADE"),
        nullable=False,
    )
    role            : Mapped[ChatRole] = mapped_column(
        Enum(ChatRole, schema="ai_strategy", name="chat_role", create_type=False),
        nullable=False,
    )
    content         : Mapped[str]          = mapped_column(Text, nullable=False)
    # ── Async processing fields ─────────────────────────────────────────────
    status          : Mapped[MessageStatus] = mapped_column(
        Enum(MessageStatus, schema="ai_strategy", name="message_status", create_type=False),
        default=MessageStatus.completed,
        nullable=False,
        server_default="completed",
    )
    updated_at      : Mapped[datetime]     = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )
    error_message   : Mapped[Optional[str]]   = mapped_column(Text, nullable=True)
    parent_message_id : Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai_strategy.chat_messages.id", ondelete="SET NULL"),
        nullable=True,
    )
    is_final        : Mapped[bool]         = mapped_column(
        Boolean, default=True, nullable=False, server_default="true"
    )
    # ── Existing fields ─────────────────────────────────────────────────────
    token_count     : Mapped[Optional[int]]   = mapped_column(Integer, nullable=True)
    model           : Mapped[Optional[str]]   = mapped_column(Text, nullable=True)
    strategy_draft  : Mapped[Optional[dict]]  = mapped_column(JSONB, nullable=True)
    backtest_id     : Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai_strategy.backtest.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at      : Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    strategy_id     : Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai_strategy.strategies.id", ondelete="SET NULL"),
        nullable=True,
    )
    ref_backtest_id : Mapped[Optional[str]]   = mapped_column(Text, nullable=True)
    strategy_json   : Mapped[Optional[dict]]  = mapped_column(
        JSONB, nullable=True,
        comment="Parsed strategy_json from assemble_strategy AI response (entry/exit signals, asset, parameters)"
    )

    # Relationships
    chat     : Mapped["Chat"]           = relationship("Chat",     back_populates="messages")
    strategy : Mapped[Optional["Strategy"]] = relationship(
        "Strategy", back_populates="messages", foreign_keys=[strategy_id]
    )
    backtest : Mapped[Optional["Backtest"]] = relationship(
        "Backtest", back_populates="messages", foreign_keys=[backtest_id]
    )
    replies  : Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage",
        foreign_keys="ChatMessage.parent_message_id",
        back_populates="parent_message",
    )
    parent_message : Mapped[Optional["ChatMessage"]] = relationship(
        "ChatMessage",
        foreign_keys=[parent_message_id],
        back_populates="replies",
        remote_side="ChatMessage.id",
    )
