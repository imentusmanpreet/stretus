"""
app/schemas/chat.py
════════════════════════════════════════════════════════════════════
All Pydantic models for the chat API.

CHANGES:
  - chat_id renamed to session_id across all schemas.
  - New AsyncMessageAcceptedResponse: immediate 202 acknowledgement on POST.
  - ChatMessageItem & MessageResponse extended with async status metadata.
"""
from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


# ── Request models ─────────────────────────────────────────────────────────────

class AssetClassCapability(BaseModel):
    """One asset class entry in a tenant's capabilities block."""

    asset_class_id: Literal["equity_cash", "crypto_spot"]
    enabled: bool = True


class ChatCapabilities(BaseModel):
    """Tenant-supplied capabilities scoped to this chat session."""

    asset_classes: List[AssetClassCapability] = Field(default_factory=list)


class ChatCreateRequest(BaseModel):
    title: Optional[str] = Field(None, example="My NIFTY Strategy")
    capabilities: Optional[ChatCapabilities] = Field(
        None,
        description=(
            "Asset classes the tenant has enabled for this chat. When omitted, "
            "defaults to equity_cash (Indian equities)."
        ),
    )


class MessageRequest(BaseModel):
    content: str = Field(..., min_length=1, example="I want to trade NIFTY on 15m")
    # Multi-asset backtest: explicit list of symbols to run sequentially when the
    # message triggers a backtest. None/empty → single asset from the strategy YAML.
    symbols: Optional[list[str]] = Field(default=None, example=["SBIN", "TCS", "INFY"])


# ── Response models ────────────────────────────────────────────────────────────

class ChatCreateResponse(BaseModel):
    session_id: str
    title:      str
    status:     str = "completed"
    created_at: str


class AsyncMessageAcceptedResponse(BaseModel):
    """
    Returned immediately by POST /chats/{session_id}/messages.

    The AI processes the request asynchronously in the background.
    The user message is saved instantly with status='processing'.
    Poll GET /chats/{session_id}/messages to retrieve the AI response
    once the background processing completes.
    """
    message_id:  str          # user message id saved in DB
    session_id:  str
    message:     str = "Your message is being processed."
    current_mode: str = "collect_user_input"
    created_at:  str


class MessageResponse(BaseModel):
    """
    Full response shape — used in GET /chats/{session_id}/messages history.

    Frontend guide:
    - content          → render in the AI chat bubble (only when status=completed)
    - status           → processing | completed | failed
    - error_message    → populated when status=failed
    - strategy_draft   → show progress indicators (which fields are filled)
    - strategy_preview → render the Strategy Preview Card
    - missing_fields   → checklist of what the user still needs to provide
    """
    model_config = ConfigDict(protected_namespaces=())

    message_id:       str
    session_id:       str
    role:             str = "assistant"
    state:            Optional[str] = None
    content:          str
    status:           str = "completed"
    error_message:    Optional[str] = None
    error:            Optional[dict] = None

    strategy_draft:   Optional[dict] = None
    strategy_preview: Optional[dict] = None
    missing_fields:   list[str] = []

    model_used:  Optional[str] = None
    created_at:  str
    updated_at:  Optional[str] = None


class ChatMessageItem(BaseModel):
    """
    A single message row as returned in GET /messages history.
    """
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    id:               str
    role:             str
    state:            Optional[str] = None
    content:          str
    status:           Optional[str] = None
    error_message:    Optional[str] = None
    error:            Optional[dict] = None
    strategy_draft:   Optional[dict] = None
    strategy_json:    Optional[dict] = None
    backtest_result:  Optional[dict] = None
    created_at:       Optional[str] = None
    updated_at:       Optional[str] = None


class ChatHistoryResponse(BaseModel):
    session_id:    str
    title:         str
    current_mode:  str
    status:        str
    message_count: int
    messages:      list[ChatMessageItem]


class ChatListItem(BaseModel):
    session_id:    str
    title:         str
    status:        str
    message_count: int
    created_at:    str
    updated_at:    str


class ChatListResponse(BaseModel):
    total: int
    chats: list[ChatListItem]
