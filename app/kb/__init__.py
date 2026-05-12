"""
app/kb — Stretus Knowledge Base (v2)

Structured, file-backed knowledge base for the strategy planner.
Replaces the legacy free-text .docx + ChromaDB pipeline.

Public API:
    from app.kb import kb               # process-wide singleton
    from app.kb.schemas import (
        SignalCard, Stock, Intent, RiskTier, MarketConfig,
        TimeframeConfig, IntentTaxonomy, StrategyPlan, PickedSignal,
    )
"""
from app.kb.loader import kb

__all__ = ["kb"]
