"""
quant_engine/engine/portfolio.py — the backtest Portfolio Manager (§8 step 6–7, §11).

A dynamic-universe backtest runs the EXISTING per-symbol simulator
(:func:`engine.simulator.simulate_trades`) once per member, then this Portfolio Manager
**wraps** those per-symbol trades under one shared capital pool (Invariant 1: reuse, don't
rewrite; Invariant 9: shared capital + position cap live HERE, never in the per-symbol
simulator). It is the single authority for capital, the ``max_positions`` cap, sizing, and
the one portfolio equity curve — identical logic for backtest and (later) live (parity, §11).

Model
-----
Each member's simulation yields independent candidate trades (entry/exit timestamps + a
fractional return). The manager replays them on ONE unified clock:
  * at each candidate entry, it first releases any positions that have since exited
    (returning their capital to the shared pool);
  * it admits the entry only if a position slot is free (``max_positions``) AND enough
    capital is available — otherwise it records a SKIP with the reason
    (``position_cap`` / ``insufficient_capital`` / ``sector_cap``);
  * on exit it realises ``allocated_capital × pnl_frac`` back into the pool.
The equity curve is the realised shared-pool equity through time (open positions marked at
cost — trade granularity, no intrabar marks needed from a ``Trade``). This is a faithful,
transparent portfolio overlay that never forks the simulator.

Pure module: depends only on numpy/pandas — no engine/app imports, so it is trivially
unit-testable with synthetic candidate trades (Invariant: tests, no network).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SizingMode = Literal["equal_weight", "risk_based"]
SkipReason = Literal["position_cap", "insufficient_capital", "sector_cap"]


@dataclass(frozen=True)
class CandidateTrade:
    """One member's trade offered to the portfolio (built from a simulator ``Trade``).

    ``pnl_frac`` is the fractional return on the capital allocated to THIS trade
    (e.g. ``0.05`` = +5%). ``stop_distance_frac`` is the entry→stop distance as a fraction
    of entry price, used only by ``risk_based`` sizing (``None`` ⇒ fall back to equal weight
    for this trade). ``entry_ts``/``exit_ts`` drive the unified clock.
    """

    symbol: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    side: str
    pnl_frac: float
    stop_distance_frac: float | None = None
    sector: str | None = None


@dataclass(frozen=True)
class PortfolioFill:
    """A candidate trade the portfolio actually took, with its capital + realised P&L."""

    symbol: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    side: str
    allocated_capital: float
    pnl_frac: float
    pnl_abs: float                 # allocated_capital × pnl_frac (rupees)
    sector: str | None = None


@dataclass(frozen=True)
class SkippedEntry:
    """A candidate trade the portfolio declined, with the reason (transparency, §12.1)."""

    symbol: str
    entry_ts: pd.Timestamp
    reason: SkipReason


@dataclass
class PortfolioResult:
    """The single portfolio outcome: equity curve, metrics, attribution, skips (§8 step 7)."""

    starting_capital: float
    ending_capital: float
    equity_curve: pd.Series                      # realised shared-pool equity through time
    fills: list[PortfolioFill] = field(default_factory=list)
    skipped: list[SkippedEntry] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    per_symbol_pnl: dict[str, float] = field(default_factory=dict)
    skip_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "starting_capital": self.starting_capital,
            "ending_capital": self.ending_capital,
            "equity_curve": [
                {"timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                 "equity": float(v)}
                for ts, v in self.equity_curve.items()
            ],
            "metrics": dict(self.metrics),
            "per_symbol_pnl": dict(self.per_symbol_pnl),
            "skip_counts": dict(self.skip_counts),
            "fills": [
                {"symbol": f.symbol, "entry": str(f.entry_ts), "exit": str(f.exit_ts),
                 "side": f.side, "allocated_capital": f.allocated_capital,
                 "pnl_abs": f.pnl_abs, "pnl_frac": f.pnl_frac, "sector": f.sector}
                for f in self.fills
            ],
            "skipped": [
                {"symbol": s.symbol, "entry": str(s.entry_ts), "reason": s.reason}
                for s in self.skipped
            ],
        }


class PortfolioManager:
    """Arbitrates per-symbol candidate trades into one shared-capital portfolio.

    Args:
      starting_capital: the shared pool the whole universe trades against.
      max_positions: hard cap on concurrent open positions (Invariant 9). The resolver's
        platform cap (§7.1) should already be reflected here by the caller.
      sizing_mode: ``equal_weight`` (1/max_positions of current equity per slot) or
        ``risk_based`` (risk ``risk_per_trade_pct`` of equity ÷ stop distance).
      risk_per_trade_pct: risk budget per trade for ``risk_based`` sizing.
      sector_cap_pct: optional ceiling on capital concurrently allocated to one sector
        (needs ``CandidateTrade.sector``); ``None`` disables the sector cap.
      periods_per_year: for annualised Sharpe/Sortino (daily-resampled curve → 252).
    """

    def __init__(
        self,
        *,
        starting_capital: float = 100_000.0,
        max_positions: int = 5,
        sizing_mode: SizingMode = "equal_weight",
        risk_per_trade_pct: float = 1.0,
        sector_cap_pct: float | None = None,
        periods_per_year: int = 252,
    ) -> None:
        if starting_capital <= 0:
            raise ValueError("starting_capital must be > 0")
        if max_positions < 1:
            raise ValueError("max_positions must be ≥ 1")
        self.starting_capital = float(starting_capital)
        self.max_positions = int(max_positions)
        self.sizing_mode = sizing_mode
        self.risk_per_trade_pct = float(risk_per_trade_pct)
        self.sector_cap_pct = sector_cap_pct
        self.periods_per_year = int(periods_per_year)

    # ── sizing ────────────────────────────────────────────────────────────────
    def _target_allocation(self, equity: float, cand: CandidateTrade) -> float:
        """Capital to allocate to ``cand`` given current ``equity`` (before cash cap)."""
        if self.sizing_mode == "risk_based" and cand.stop_distance_frac and cand.stop_distance_frac > 0:
            risk_amount = (self.risk_per_trade_pct / 100.0) * equity
            return risk_amount / cand.stop_distance_frac
        # equal_weight (default, and the risk_based fallback when no stop distance)
        return equity / self.max_positions

    # ── main loop ───────────────────────────────────────────────────────────────
    def run(self, candidates: list[CandidateTrade]) -> PortfolioResult:
        """Replay ``candidates`` on one unified clock under the shared pool + caps."""
        # Chronological by entry; deterministic tie-break by symbol so results are stable.
        ordered = sorted(candidates, key=lambda c: (c.entry_ts, c.symbol))

        cash = self.starting_capital
        # Open positions: each holds (exit_ts, allocated_capital, pnl_frac, candidate).
        open_positions: list[tuple[pd.Timestamp, float, float, CandidateTrade]] = []
        sector_allocated: dict[str, float] = {}

        fills: list[PortfolioFill] = []
        skipped: list[SkippedEntry] = []
        # Equity events: (timestamp, equity_after_event). Built as positions open/close.
        events: list[tuple[pd.Timestamp, float]] = [(self._t0(ordered), self.starting_capital)]

        def _release_until(now: pd.Timestamp) -> None:
            nonlocal cash
            still_open: list[tuple[pd.Timestamp, float, float, CandidateTrade]] = []
            # Close every position whose exit is at/just before `now`, oldest exit first.
            for exit_ts, alloc, pnl_frac, cand in sorted(open_positions, key=lambda p: p[0]):
                if exit_ts <= now:
                    realized = alloc * pnl_frac
                    cash += alloc + realized
                    if cand.sector:
                        sector_allocated[cand.sector] = max(
                            0.0, sector_allocated.get(cand.sector, 0.0) - alloc
                        )
                    fills.append(PortfolioFill(
                        symbol=cand.symbol, entry_ts=cand.entry_ts, exit_ts=exit_ts,
                        side=cand.side, allocated_capital=alloc, pnl_frac=pnl_frac,
                        pnl_abs=realized, sector=cand.sector,
                    ))
                    events.append((exit_ts, self._equity(cash, still_open + [
                        p for p in open_positions if p[0] > now
                    ])))
                else:
                    still_open.append((exit_ts, alloc, pnl_frac, cand))
            open_positions[:] = still_open

        for cand in ordered:
            _release_until(cand.entry_ts)
            equity_now = self._equity(cash, open_positions)

            if len(open_positions) >= self.max_positions:
                skipped.append(SkippedEntry(cand.symbol, cand.entry_ts, "position_cap"))
                continue

            target = self._target_allocation(equity_now, cand)
            alloc = min(target, cash)
            if alloc <= 0 or cash <= 0:
                skipped.append(SkippedEntry(cand.symbol, cand.entry_ts, "insufficient_capital"))
                continue

            if self.sector_cap_pct is not None and cand.sector:
                cap_amount = (self.sector_cap_pct / 100.0) * equity_now
                if sector_allocated.get(cand.sector, 0.0) + alloc > cap_amount + 1e-9:
                    skipped.append(SkippedEntry(cand.symbol, cand.entry_ts, "sector_cap"))
                    continue

            # Admit the position: lock capital out of the pool until it exits.
            cash -= alloc
            open_positions.append((cand.exit_ts, alloc, cand.pnl_frac, cand))
            if cand.sector:
                sector_allocated[cand.sector] = sector_allocated.get(cand.sector, 0.0) + alloc
            events.append((cand.entry_ts, self._equity(cash, open_positions)))

        # Flush any positions still open at the end (mark to their realised exit).
        if open_positions:
            last_exit = max(p[0] for p in open_positions)
            _release_until(last_exit)

        return self._build_result(fills, skipped, events)

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _t0(ordered: list[CandidateTrade]) -> pd.Timestamp:
        return ordered[0].entry_ts if ordered else pd.Timestamp("1970-01-01")

    @staticmethod
    def _equity(cash: float, open_positions: list) -> float:
        """Shared-pool equity = free cash + capital locked in open positions (marked at cost)."""
        return cash + sum(alloc for _, alloc, _, _ in open_positions)

    def _build_result(
        self,
        fills: list[PortfolioFill],
        skipped: list[SkippedEntry],
        events: list[tuple[pd.Timestamp, float]],
    ) -> PortfolioResult:
        # Collapse events into a sorted, last-write-wins equity curve.
        events.sort(key=lambda e: e[0])
        idx: list[pd.Timestamp] = []
        vals: list[float] = []
        for ts, eq in events:
            if idx and idx[-1] == ts:
                vals[-1] = eq
            else:
                idx.append(ts)
                vals.append(eq)
        equity_curve = pd.Series(vals, index=pd.DatetimeIndex(idx), name="equity")

        ending = float(equity_curve.iloc[-1]) if len(equity_curve) else self.starting_capital
        per_symbol: dict[str, float] = {}
        for f in fills:
            per_symbol[f.symbol] = per_symbol.get(f.symbol, 0.0) + f.pnl_abs

        skip_counts: dict[str, int] = {}
        for s in skipped:
            skip_counts[s.reason] = skip_counts.get(s.reason, 0) + 1

        metrics = self._metrics(equity_curve, fills, ending)
        logger.info(
            "portfolio|done|start=%.2f|end=%.2f|return=%.2f%%|fills=%d|skipped=%d|maxDD=%.2f%%",
            self.starting_capital, ending, metrics.get("total_return_pct", 0.0),
            len(fills), len(skipped), metrics.get("max_drawdown_pct", 0.0),
        )
        return PortfolioResult(
            starting_capital=self.starting_capital,
            ending_capital=ending,
            equity_curve=equity_curve,
            fills=fills,
            skipped=skipped,
            metrics=metrics,
            per_symbol_pnl=per_symbol,
            skip_counts=skip_counts,
        )

    def _metrics(
        self, equity_curve: pd.Series, fills: list[PortfolioFill], ending: float,
    ) -> dict[str, float]:
        total_return_pct = (ending / self.starting_capital - 1.0) * 100.0
        wins = [f for f in fills if f.pnl_abs > 0]
        win_rate = (len(wins) / len(fills) * 100.0) if fills else 0.0

        # Max drawdown on the realised equity curve.
        max_dd_pct = 0.0
        if len(equity_curve):
            running_peak = equity_curve.cummax()
            drawdown = (equity_curve - running_peak) / running_peak
            max_dd_pct = float(drawdown.min() * 100.0) if len(drawdown) else 0.0

        # Daily-resampled returns drive annualised Sharpe/Sortino/Calmar.
        sharpe = sortino = calmar = 0.0
        if len(equity_curve) >= 2:
            daily = equity_curve.resample("1D").last().ffill()
            rets = daily.pct_change().dropna()
            if len(rets) >= 2 and rets.std(ddof=0) > 0:
                ann = math.sqrt(self.periods_per_year)
                sharpe = float(rets.mean() / rets.std(ddof=0) * ann)
                downside = rets[rets < 0]
                if len(downside) and downside.std(ddof=0) > 0:
                    sortino = float(rets.mean() / downside.std(ddof=0) * ann)
            if max_dd_pct < 0:
                calmar = float(total_return_pct / abs(max_dd_pct))

        return {
            "total_return_pct": round(total_return_pct, 4),
            "ending_balance": round(ending, 2),
            "total_trades": float(len(fills)),
            "win_rate_pct": round(win_rate, 2),
            "max_drawdown_pct": round(max_dd_pct, 4),
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "calmar": round(calmar, 4),
        }


def candidate_from_trade(
    trade,
    *,
    sector: str | None = None,
) -> CandidateTrade:
    """Adapt a simulator ``engine.simulator.Trade`` into a :class:`CandidateTrade`.

    Uses ``pnl_abs`` (the simulator's fractional return) as ``pnl_frac`` so the portfolio
    compounds correctly on allocated capital. ``stop_distance_frac`` is derived from the
    entry/stop when present in ``entry_reason`` (for ``risk_based`` sizing), else left None.
    Keeps the portfolio layer decoupled from the Trade's exact field set.
    """
    entry_ts = pd.to_datetime(trade.entry_date)
    exit_ts = pd.to_datetime(trade.exit_date)
    stop_distance_frac: float | None = None
    detail = getattr(trade, "entry_reason", None) or {}
    try:
        entry_price = float(getattr(trade, "entry_price", 0.0) or 0.0)
        stop_price = float(detail.get("initial_stop") or detail.get("stop_price") or 0.0)
        if entry_price > 0 and stop_price > 0:
            stop_distance_frac = abs(entry_price - stop_price) / entry_price
    except (TypeError, ValueError):
        stop_distance_frac = None
    return CandidateTrade(
        symbol=trade.symbol,
        entry_ts=entry_ts,
        exit_ts=exit_ts if exit_ts >= entry_ts else entry_ts,
        side=str(trade.side),
        pnl_frac=float(trade.pnl_abs),
        stop_distance_frac=stop_distance_frac,
        sector=sector,
    )
