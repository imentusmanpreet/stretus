"""
app/services/execution/account_manager.py
───────────────────────────────────────────
Light affordability and position-limit guard.

Checks (in order):
  1. trade_value <= available_margin
  2. cash_reserve is maintained after trade
  3. open_positions count < max_open_positions
  4. cooldown_bars elapsed since last trade

This service does NOT calculate P&L or update any balance.
It returns (ok: bool, messages: List[str]).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AccountCheckInput:
    trade_value: float          # entry_price * quantity
    available_margin: float
    current_open_positions: int
    bars_since_last_trade: int
    # From ExecutionState (or defaults)
    max_open_positions: int = 3
    cash_reserve_pct: float = 0.10
    cooldown_bars: int = 5
    capital: float = 100_000.0


@dataclass
class AccountCheckResult:
    ok: bool
    messages: List[str] = field(default_factory=list)


class AccountManager:
    """Stateless; instantiate once per request."""

    def validate(self, inp: AccountCheckInput) -> AccountCheckResult:
        messages: List[str] = []

        # ── 1. Margin affordability ────────────────────────────────────────────
        if inp.trade_value > inp.available_margin:
            msg = (
                f"  ❌ Insufficient margin | trade=₹{inp.trade_value:,.2f}"
                f" > available=₹{inp.available_margin:,.2f} — trade blocked."
            )
            messages.append(msg)
            logger.warning(msg)
            return AccountCheckResult(ok=False, messages=messages)

        # ── 2. Cash reserve check ─────────────────────────────────────────────
        reserve_required = inp.capital * inp.cash_reserve_pct
        margin_after     = inp.available_margin - inp.trade_value
        if margin_after < reserve_required:
            msg = (
                f"  ❌ Cash reserve breach | margin after trade=₹{margin_after:,.2f}"
                f" < required reserve=₹{reserve_required:,.2f}"
                f" ({inp.cash_reserve_pct*100:.0f}% of ₹{inp.capital:,.0f}) — trade blocked."
            )
            messages.append(msg)
            logger.warning(msg)
            return AccountCheckResult(ok=False, messages=messages)

        # ── 3. Max open positions ─────────────────────────────────────────────
        if inp.current_open_positions >= inp.max_open_positions:
            msg = (
                f"  ❌ Max positions reached | open={inp.current_open_positions}"
                f" / limit={inp.max_open_positions} — trade blocked."
            )
            messages.append(msg)
            logger.warning(msg)
            return AccountCheckResult(ok=False, messages=messages)

        # ── 4. Cooldown check ─────────────────────────────────────────────────
        if inp.cooldown_bars > 0 and inp.bars_since_last_trade < inp.cooldown_bars:
            msg = (
                f"  ❌ Cooldown active | {inp.bars_since_last_trade} bars since last trade,"
                f" need {inp.cooldown_bars} bars — trade blocked."
            )
            messages.append(msg)
            logger.info(msg)
            return AccountCheckResult(ok=False, messages=messages)

        messages.append(
            f"  ✅ Account check passed | margin=₹{inp.available_margin:,.2f}"
            f"  margin_after=₹{margin_after:,.2f}"
            f"  reserve=₹{reserve_required:,.2f} ✓"
            f"  positions={inp.current_open_positions}/{inp.max_open_positions} ✓"
            f"  cooldown={inp.bars_since_last_trade}/{inp.cooldown_bars} bars ✓"
        )
        return AccountCheckResult(ok=True, messages=messages)
