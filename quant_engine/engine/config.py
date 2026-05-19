"""
Centralised configuration for the Stretus Quant Engine.

ALL numeric thresholds, boundary constants, and computation parameters live here.
No magic numbers should appear in business logic — import from this module instead.

Sections
--------
PASS_FAIL            — objective-specific backtest pass/fail thresholds
GRADING              — scoring boundaries for the A/B/C/D grade scale
RETURN_POTENTIAL     — thresholds for classifying return potential label
RISK_PROFILE         — thresholds for classifying risk profile label
DRAWDOWN_TOLERANCE   — thresholds for classifying drawdown tolerance label
RECOMMENDED_FOR      — thresholds used in trader-type recommendation
MARKET_PHASE         — thresholds for Bull/Bear/Sideways and trend direction
ALIGNMENT            — win-rate boundaries for upgrading/downgrading alignment
ENTRY_CONDITION      — window sizes for per-trade entry condition classification
MARKET_DATA_WINDOW   — enforced market-data range for backtests
METRICS              — computation parameters (trading calendar, caps, confidence)
"""
from __future__ import annotations

from datetime import datetime, timezone

# ── Pass / Fail thresholds (objective-aware) ──────────────────────────────────
# These control whether a backtest result is marked pass=True or pass=False.
# min_trades=0 disables trade-count gating; total_trades are still reported fully.
PASS_FAIL_THRESHOLDS: dict[str, dict[str, float | int]] = {
    "intraday": {
        "min_trades":        0,
        "min_win_rate_pct":  40.0,
        "min_profit_factor":  0.0,   # not enforced for intraday
    },
    "positional": {
        "min_trades":        0,
        "min_win_rate_pct":  40.0,
        "min_profit_factor":  1.2,
    },
}

# ── Overall grade scoring boundaries ──────────────────────────────────────────
# compute_overall_grade() builds a score 0–100 from sub-scores;
# these constants map that score to a letter grade.
GRADE_A_MIN_SCORE      = 85
GRADE_B_PLUS_MIN_SCORE = 72
GRADE_B_MIN_SCORE      = 62
GRADE_C_PLUS_MIN_SCORE = 52
GRADE_C_MIN_SCORE      = 42

# Sub-score component weights (must be consistent with _*_score() functions).
# Total possible = return(30) + sharpe(25) + drawdown(25) + consistency(15) + sample(5) = 100
GRADE_RETURN_SCORE_EXCELLENT_MIN_PCT  = 20.0   # → 30 pts
GRADE_RETURN_SCORE_GOOD_MIN_PCT       = 12.0   # → 24 pts
GRADE_RETURN_SCORE_MODERATE_MIN_PCT   =  5.0   # → 16 pts
GRADE_RETURN_SCORE_POSITIVE_MIN_PCT   =  0.0   # →  8 pts

GRADE_SHARPE_SCORE_EXCELLENT_MIN      = 1.5    # → 25 pts
GRADE_SHARPE_SCORE_GOOD_MIN           = 1.0    # → 18 pts
GRADE_SHARPE_SCORE_MODERATE_MIN       = 0.5    # → 10 pts
GRADE_SHARPE_SCORE_POSITIVE_MIN       = 0.0    # →  5 pts

GRADE_DRAWDOWN_SCORE_EXCELLENT_MAX_PCT = 8.0   # → 25 pts
GRADE_DRAWDOWN_SCORE_GOOD_MAX_PCT      = 12.0  # → 18 pts
GRADE_DRAWDOWN_SCORE_MODERATE_MAX_PCT  = 15.0  # → 12 pts
GRADE_DRAWDOWN_SCORE_FAIR_MAX_PCT      = 20.0  # →  8 pts
GRADE_DRAWDOWN_SCORE_POOR_MAX_PCT      = 30.0  # →  4 pts

GRADE_CONSISTENCY_SCORE_EXCELLENT_MIN_PF   = 1.5   # → 15 pts (combined with win_rate)
GRADE_CONSISTENCY_SCORE_EXCELLENT_MIN_WR   = 60.0
GRADE_CONSISTENCY_SCORE_GOOD_MIN_PF        = 1.2   # → 12 pts
GRADE_CONSISTENCY_SCORE_GOOD_MIN_WR        = 50.0
GRADE_CONSISTENCY_SCORE_MODERATE_MIN_PF    = 1.0   # →  8 pts
GRADE_CONSISTENCY_SCORE_MODERATE_MIN_WR    = 45.0
GRADE_CONSISTENCY_SCORE_FAIR_MIN_PF        = 0.8   # →  4 pts

GRADE_SAMPLE_SCORE_LARGE_MIN_TRADES    = 100   # →  5 pts
GRADE_SAMPLE_SCORE_MEDIUM_MIN_TRADES   =  40   # →  4 pts
GRADE_SAMPLE_SCORE_SMALL_MIN_TRADES    =  20   # →  2 pts

# ── Return potential classification ───────────────────────────────────────────
RETURN_POTENTIAL_STRONG_MIN_PCT    = 15.0   # total_return_pct ≥ this → potential "Strong"
RETURN_POTENTIAL_MODERATE_MIN_PCT  =  5.0   # total_return_pct ≥ this → potential "Moderate"
RETURN_POTENTIAL_STRONG_MIN_SHARPE =  1.0   # Sharpe must also be ≥ this for "Strong"
RETURN_POTENTIAL_STRONG_MIN_TRADES =  20    # trade count must also be ≥ this for "Strong"
RETURN_POTENTIAL_MODERATE_MIN_TRADES = 10   # trade count must be ≥ this for "Moderate"

# ── Risk profile classification ───────────────────────────────────────────────
RISK_CONSERVATIVE_MAX_DRAWDOWN_PCT   =  8.0
RISK_CONSERVATIVE_MAX_VOLATILITY_PCT = 12.0
RISK_CONSERVATIVE_MIN_VAR_95_PCT     = -2.0   # VaR must not be worse than this
RISK_MODERATE_MAX_DRAWDOWN_PCT       = 15.0
RISK_MODERATE_MAX_VOLATILITY_PCT     = 20.0
RISK_MODERATE_MIN_VAR_95_PCT         = -4.0

# ── Drawdown tolerance classification ─────────────────────────────────────────
DRAWDOWN_TOLERANCE_LOW_MAX_PCT             =  8.0
DRAWDOWN_TOLERANCE_LOW_MAX_RECOVERY_DAYS   = 30
DRAWDOWN_TOLERANCE_MEDIUM_MAX_PCT          = 15.0
DRAWDOWN_TOLERANCE_MEDIUM_MAX_RECOVERY_DAYS = 60

# ── Recommended user type thresholds ──────────────────────────────────────────
RECOMMENDED_EXPERIENCED_MIN_TRADES_PER_MONTH = 20
RECOMMENDED_INTERMEDIATE_MIN_TRADES_PER_MONTH = 8

# ── Market phase classification ───────────────────────────────────────────────
# classify_market_type() uses total price return over the period.
BULL_MARKET_MIN_RETURN_PCT = 8.0    # ≥ +8% price return → "Bull"
BEAR_MARKET_MAX_RETURN_PCT = -8.0   # ≤ -8% price return → "Bear"

# classify_market_phase() uses linear regression slope of normalized close prices.
TREND_SLOPE_UPTREND_MIN   =  0.0003   # slope ≥ this → "Uptrend"
TREND_SLOPE_DOWNTREND_MAX = -0.0003   # slope ≤ this → "Downtrend"

# ── Alignment upgrade / downgrade boundaries ──────────────────────────────────
# A "Moderate" base alignment is upgraded to "Strong" if win rate in that phase
# exceeds this value, and downgraded to "Weak" if it falls below.
ALIGNMENT_STRONG_UPGRADE_WIN_RATE_PCT = 60.0
ALIGNMENT_WEAK_DOWNGRADE_WIN_RATE_PCT = 30.0

# ── Per-trade entry condition classification ───────────────────────────────────
# classify_entry_condition() uses two rolling averages of close price.
ENTRY_CONDITION_FAST_WINDOW = 10    # bars for the fast moving average
ENTRY_CONDITION_SLOW_WINDOW = 30    # bars for the slow moving average

# ── Backtest market-data window ────────────────────────────────────────────────
# All backtests are constrained to this inclusive UTC window.
# Start date is fixed, end date is dynamically set to current date/time.
BACKTEST_MARKET_DATA_FROM_UTC = "2024-01-01T00:00:00Z"

def _get_current_utc_timestamp() -> str:
    """Return current UTC timestamp in ISO format with 'Z' suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

# Dynamic end date - always uses current date and time
BACKTEST_MARKET_DATA_TO_UTC = _get_current_utc_timestamp()

# ── Metrics computation parameters ────────────────────────────────────────────
DEFAULT_STARTING_BALANCE: float = 10_000.0
TRADING_DAYS_PER_YEAR: int      = 252       # standard annualisation factor
MONTHS_PER_YEAR: float          = 12.0
CALENDAR_DAYS_PER_MONTH: float  = 30.4375   # 365.25 / 12
PROFIT_FACTOR_MAX_CAP: float    = 9_999.0   # prevents Infinity in JSON
VAR_CONFIDENCE_LEVEL: float     = 0.05      # 5% tail = 95% VaR / Expected Shortfall
