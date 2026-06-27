"""
app/services/execution/portfolio_manager.py — the LIVE portfolio-aware risk layer (§9, §11).

Where the per-strategy ``risk_manager.py`` sizes ONE symbol's trade, this Portfolio Manager
sits ABOVE it and is the single authority for a dynamic deployment's shared capital and the
``max_positions`` cap across ALL active members (Invariant 9). Every entry intent a member
produces is checked here BEFORE an order is placed: is a slot free? is capital available? does
it breach the sector cap? — else the entry is skipped with an explicit reason (§12.1).

It is the live analog of the backtest ``engine.portfolio.PortfolioManager`` and uses the SAME
admission logic, so backtest and live behave identically (parity, §11). It is **stateful**
(it tracks the currently-open positions and free cash) and **pure** (no DB/broker imports), so
it is trivially testable and can be rebuilt from persisted state on restart (reconciliation,
§9 step 7).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

SizingMode = Literal["equal_weight", "risk_based"]
SkipReason = Literal["position_cap", "insufficient_capital", "sector_cap", "already_open"]


@dataclass(frozen=True)
class LivePosition:
    """An open position the deployment currently holds."""

    symbol: str
    allocated_capital: float
    sector: str | None = None


@dataclass(frozen=True)
class AdmissionDecision:
    """The portfolio layer's verdict on one entry intent.

    ``admit`` ⇒ place the order with ``allocated_capital``. Otherwise ``reason`` says why it
    was skipped (logged + surfaced; never silently dropped).
    """

    admit: bool
    allocated_capital: float = 0.0
    reason: SkipReason | None = None


@dataclass
class LivePortfolioManager:
    """Shared-capital + position-cap authority for one live dynamic deployment.

    Construct with the deployment's ``total_capital`` and caps. Drive it per member:
    :meth:`evaluate_entry` → (if admitted) :meth:`admit`; on exit :meth:`release`. Rebuild
    from persisted open positions with :meth:`rebuild` after a restart.
    """

    total_capital: float
    max_positions: int
    sizing_mode: SizingMode = "equal_weight"
    risk_per_trade_pct: float = 1.0
    sector_cap_pct: float | None = None

    _positions: dict[str, LivePosition] = field(default_factory=dict)
    _cash: float = field(default=0.0)

    def __post_init__(self) -> None:
        if self.total_capital <= 0:
            raise ValueError("total_capital must be > 0")
        if self.max_positions < 1:
            raise ValueError("max_positions must be ≥ 1")
        self._cash = float(self.total_capital)

    # ── views ────────────────────────────────────────────────────────────────
    @property
    def open_count(self) -> int:
        return len(self._positions)

    @property
    def open_symbols(self) -> list[str]:
        return sorted(self._positions)

    @property
    def free_cash(self) -> float:
        return self._cash

    @property
    def equity(self) -> float:
        """Free cash + capital locked in open positions (marked at cost)."""
        return self._cash + sum(p.allocated_capital for p in self._positions.values())

    def _sector_allocated(self, sector: str) -> float:
        return sum(p.allocated_capital for p in self._positions.values() if p.sector == sector)

    def _target_allocation(self, stop_distance_frac: float | None) -> float:
        if self.sizing_mode == "risk_based" and stop_distance_frac and stop_distance_frac > 0:
            return (self.risk_per_trade_pct / 100.0) * self.equity / stop_distance_frac
        return self.equity / self.max_positions

    # ── admission ──────────────────────────────────────────────────────────────
    def evaluate_entry(
        self, symbol: str, *, stop_distance_frac: float | None = None, sector: str | None = None,
    ) -> AdmissionDecision:
        """Decide whether ``symbol`` may open a new position right now (no state change).

        Order of checks mirrors the backtest engine: already-open → cap → capital → sector.
        """
        if symbol in self._positions:
            return AdmissionDecision(False, reason="already_open")
        if self.open_count >= self.max_positions:
            logger.info("portfolio|skip|symbol=%s|reason=position_cap|open=%d/%d",
                        symbol, self.open_count, self.max_positions)
            return AdmissionDecision(False, reason="position_cap")
        target = self._target_allocation(stop_distance_frac)
        allocated = min(target, self._cash)
        if allocated <= 0:
            logger.info("portfolio|skip|symbol=%s|reason=insufficient_capital|cash=%.2f",
                        symbol, self._cash)
            return AdmissionDecision(False, reason="insufficient_capital")
        if self.sector_cap_pct is not None and sector:
            cap_amount = (self.sector_cap_pct / 100.0) * self.equity
            if self._sector_allocated(sector) + allocated > cap_amount + 1e-9:
                logger.info("portfolio|skip|symbol=%s|reason=sector_cap|sector=%s", symbol, sector)
                return AdmissionDecision(False, reason="sector_cap")
        return AdmissionDecision(True, allocated_capital=allocated)

    def admit(self, symbol: str, allocated_capital: float, *, sector: str | None = None) -> None:
        """Record an opened position and lock its capital out of the shared pool."""
        if symbol in self._positions:
            return
        self._positions[symbol] = LivePosition(symbol, float(allocated_capital), sector)
        self._cash -= float(allocated_capital)
        logger.info("portfolio|admit|symbol=%s|allocated=%.2f|open=%d/%d|cash=%.2f",
                    symbol, allocated_capital, self.open_count, self.max_positions, self._cash)

    def release(self, symbol: str, *, realized_pnl: float = 0.0) -> None:
        """Close ``symbol``'s position, returning its capital + realised P&L to the pool."""
        pos = self._positions.pop(symbol, None)
        if pos is None:
            return
        self._cash += pos.allocated_capital + float(realized_pnl)
        logger.info("portfolio|release|symbol=%s|pnl=%.2f|open=%d|cash=%.2f",
                    symbol, realized_pnl, self.open_count, self._cash)

    def rebuild(self, positions: list[LivePosition]) -> None:
        """Reconciliation (§9 step 7): restore open positions after a restart/failover.

        Sets the open positions and recomputes free cash as ``total_capital`` minus the
        capital locked in those positions. Call before resuming so the cap + capital math
        continues from the true live state, not a fresh pool.
        """
        self._positions = {p.symbol: p for p in positions}
        self._cash = float(self.total_capital) - sum(p.allocated_capital for p in positions)
        logger.info("portfolio|rebuild|open=%d|cash=%.2f", self.open_count, self._cash)
