"""
app/schemas/strategy.py
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class StrategyConfirmRequest(BaseModel):
    session_id: str


class StrategyConfirmResponse(BaseModel):
    strategy_id: str
    name:        str
    symbol:      str
    market:      str
    timeframe:   str
    yaml_path:   str
    status:      str
