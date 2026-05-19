"""
app/services/discovery/types.py
───────────────────────────────
Discovery domain types. Plain dataclasses + Pydantic config so they can
serialize cleanly through the chat layer (which sometimes round-trips state
via JSON dicts on chat-message rows).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


# ── KB schema (declared on a preset) ────────────────────────────────────────


class TieBreakOption(BaseModel):
    """One tie-break method offered to the user when the scanner returns
    more than one candidate. The chat layer presents the `label` to the user
    and stores the `method` (a stable identifier from TIE_BREAK_METHODS) when
    the user picks one."""

    method: str
    label: str
    description: str = ""


class DiscoveryConfig(BaseModel):
    """Declared on a strategy preset to mark it as 'dynamic' — the runner
    scans the universe at backtest time instead of taking a user-supplied
    symbol.

    Convention:
      • `conditions` is a list of formula strings evaluated against each
        candidate's recent OHLCV (same AST as entry/exit conditions). A
        candidate passes ALL conditions to qualify.
      • `tie_break_options` is the SHORT-LIST shown to the user when the
        scan returns more than one candidate. Order matters — first item
        is the suggested default.
      • `lookback_days` controls how much OHLCV history each candidate is
        evaluated on (defaults match `volume_spike` / 52w-style needs).
    """

    enabled: bool = True
    universe: str = "kb_default"        # "kb_default" = stocks listed in app/kb/stocks/
    conditions: list[str] = Field(default_factory=list)
    tie_break_options: list[TieBreakOption] = Field(default_factory=list)
    lookback_days: int = 280            # ≥ 252 for 52-week-style filters
    # Phase 9c — when set, the scanner evaluates discovery conditions on
    # THIS timeframe instead of the strategy's main timeframe. Critical for
    # presets whose conditions are inherently daily (e.g. "near 52-week
    # high") even when the strategy itself trades on 5m. Defaults to None
    # (use the main timeframe) so existing presets keep their behavior.
    scan_timeframe: str | None = None
    # Phase 9h — default values for `{placeholder}` tokens in `conditions`.
    # The chat layer parses user-typed thresholds (e.g. "volume spike
    # 1.5x", "within 5% of 52-week high") and overrides these defaults
    # via builder.discovery_parameter_overrides. The orchestrator
    # substitutes the merged values into the condition strings BEFORE
    # passing them to the scanner. Presets without placeholders pass
    # an empty dict and behave identically to the pre-9h behavior.
    parameters: dict[str, float] = Field(default_factory=dict)

    def has_anything_to_scan(self) -> bool:
        return self.enabled and bool(self.conditions)


# ── Runtime types ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Candidate:
    """One stock that matched all discovery conditions, plus the metric
    values that downstream tie-break methods need."""

    symbol: str            # e.g. "HDFCBANK.NS"
    display_name: str      # e.g. "HDFC Bank"
    sector: str            # e.g. "banking"
    metrics: dict[str, float] = field(default_factory=dict)
    # Common metric keys (computed by the scanner so tie-break can be O(N) over
    # candidates without re-fetching anything):
    #   relative_volume   — VOL[-1] / AVG(VOL, 20)
    #   close             — last close
    #   distance_to_52w_high_pct  — (max_high_252 - close) / close * 100
    #   distance_to_52w_low_pct   — (close - min_low_252) / close * 100
    #   rsi_14            — RSI(14) at last bar
    #   atr_14_pct        — ATR(14) / close * 100

    def to_dict(self) -> dict:
        return {
            "symbol":       self.symbol,
            "display_name": self.display_name,
            "sector":       self.sector,
            "metrics":      dict(self.metrics),
        }


DiscoveryStatus = Literal["single", "none", "multiple"]


@dataclass(frozen=True)
class ScanResult:
    """Outcome of one universe scan. The chat layer dispatches on `status`:

      single   → builder.symbol = candidates[0].symbol; proceed.
      none     → tell user; ask to relax criteria.
      multiple → save (candidates + tie_break_options) on builder; ask user
                 which tie-break method to use.
    """

    status: DiscoveryStatus
    candidates: list[Candidate]
    tie_break_options: list[TieBreakOption]
    # Asof-date used for the scan (for traceability — same end-of-window for
    # backtest, "today" for live).
    asof_iso: str | None = None
    # Counts for diagnostics / chat composition.
    scanned_count: int = 0       # how many universe symbols were checked
    failed_fetches: int = 0       # how many symbols failed OHLCV fetch (excluded)
    # Phase 9m — per-condition fail counts. Keyed by the AST condition
    # string the scanner evaluated. Value = number of UNIVERSE symbols
    # that passed every PRECEDING condition but failed THIS one. This
    # turns a useless "no stocks matched" reply into actionable signal:
    # the user can see exactly which constraint was the bottleneck and
    # what to relax.
    fail_counts: dict[str, int] = field(default_factory=dict)
    # Phase 9n — sample of fetch errors (symbol → first exception
    # message). When the scanner can't reach the market-data service
    # for ANY universe symbol, this lets the chat layer surface the
    # actual root cause (network, auth, bad symbol format, etc.)
    # instead of the misleading "loosen your conditions" hint. Limited
    # to one entry per unique error string to bound size.
    fetch_errors: dict[str, str] = field(default_factory=dict)
    # Per-symbol scan transparency. The aggregate counters above answer
    # "how many?"; these answer "which ones?" so the no-match reply can
    # show the user the actual list of stocks scanned, which passed,
    # which failed which condition, and which couldn't be fetched —
    # instead of a black box.
    #   scanned_symbols           — every symbol the scanner attempted
    #   passed_symbols            — symbols that passed every condition
    #                               (always ⊆ {c.symbol for c in candidates})
    #   failed_fetch_symbols      — symbols where OHLCV fetch failed or
    #                               returned insufficient data; ⊆ scanned_symbols
    #   failed_condition_by_symbol — symbol → AST condition string of
    #                               the first condition the symbol failed.
    #                               Only populated for symbols that actually
    #                               reached evaluation (not fetch failures).
    scanned_symbols: list[str] = field(default_factory=list)
    passed_symbols: list[str] = field(default_factory=list)
    failed_fetch_symbols: list[str] = field(default_factory=list)
    failed_condition_by_symbol: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status":                     self.status,
            "candidates":                 [c.to_dict() for c in self.candidates],
            "tie_break_options":          [o.model_dump() for o in self.tie_break_options],
            "asof_iso":                   self.asof_iso,
            "scanned_count":              self.scanned_count,
            "failed_fetches":             self.failed_fetches,
            "fail_counts":                dict(self.fail_counts),
            "fetch_errors":               dict(self.fetch_errors),
            "scanned_symbols":            list(self.scanned_symbols),
            "passed_symbols":             list(self.passed_symbols),
            "failed_fetch_symbols":       list(self.failed_fetch_symbols),
            "failed_condition_by_symbol": dict(self.failed_condition_by_symbol),
        }
