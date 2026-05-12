"""
Parameterized description templates for backtest assessment and market phase analysis.

Architecture
------------
Every label (grade, return_potential, risk_profile, market_type, alignment) is computed
from numeric metrics elsewhere.  This module only contains:

  1. Template strings — static text with {placeholder} slots.
  2. One build_*() function per template group that selects a template and fills
     all placeholders with computed numeric/string values passed in by the caller.

Rules
-----
- No hard-coded business values (numbers, decisions) live here.
- All {placeholders} must be filled from computed data — never from literal constants.
- To change *when* a template is selected, update thresholds in config.py.
- To change *what* the text says, update the template string in this file.
- Every (grade, return_potential) pair has its own template so descriptions
  always reflect the actual computed metrics for that specific backtest.
"""
from __future__ import annotations

# ── Assessment notes ──────────────────────────────────────────────────────────
# Key: (overall_grade, return_potential)
# Placeholders filled by build_assessment_notes():
#   {total_return}  — total_return_pct
#   {num_days}      — num_days
#   {sharpe}        — sharpe_ratio
#   {max_drawdown}  — max_drawdown
#   {win_rate}      — win_rate
#   {total_trades}  — total_trades
#   {risk_profile}  — risk_profile label (already a computed string)

_ASSESSMENT_NOTES: dict[tuple[str, str], str] = {
    ("A", "Strong"): (
        "Exceptional performance: {total_return:.1f}% total return over {num_days} days "
        "with a Sharpe ratio of {sharpe:.2f} and max drawdown of {max_drawdown:.1f}%. "
        "{win_rate:.1f}% of {total_trades} trades were profitable, demonstrating "
        "consistently high signal quality."
    ),
    ("A", "Moderate"): (
        "High-quality strategy returning {total_return:.1f}% over {num_days} days. "
        "Sharpe {sharpe:.2f} with max drawdown {max_drawdown:.1f}% indicates solid "
        "risk-adjusted performance. {win_rate:.1f}% win rate across {total_trades} trades."
    ),
    ("A", "Weak"): (
        "Grade A achieved but on a limited trade sample ({total_trades} trades over {num_days} days). "
        "Return of {total_return:.1f}% and Sharpe {sharpe:.2f} are promising. "
        "Extend the backtest window to confirm statistical robustness before deploying."
    ),
    ("B+", "Strong"): (
        "Very strong performance: {total_return:.1f}% return, {win_rate:.1f}% win rate, "
        "Sharpe {sharpe:.2f}. Max drawdown of {max_drawdown:.1f}% is well managed "
        "across {total_trades} trades over {num_days} days."
    ),
    ("B+", "Moderate"): (
        "Above-average result with {total_return:.1f}% return and {win_rate:.1f}% win rate "
        "({total_trades} trades). Sharpe {sharpe:.2f} and drawdown {max_drawdown:.1f}% reflect "
        "a {risk_profile}-risk, reliable strategy over {num_days} days."
    ),
    ("B+", "Weak"): (
        "Solid grade with limited trade frequency ({total_trades} trades over {num_days} days). "
        "{total_return:.1f}% return and {max_drawdown:.1f}% drawdown. "
        "Increase the sample size to validate consistency."
    ),
    ("B", "Strong"): (
        "Good strategy: {total_return:.1f}% return, {win_rate:.1f}% win rate, "
        "Sharpe {sharpe:.2f} over {total_trades} trades. Max drawdown {max_drawdown:.1f}% "
        "is manageable. Minor improvements to risk management could push this to B+."
    ),
    ("B", "Moderate"): (
        "Moderate-good result: {total_return:.1f}% return, {win_rate:.1f}% win rate, "
        "Sharpe {sharpe:.2f}. Drawdown of {max_drawdown:.1f}% is within expected bounds "
        "for a {risk_profile}-risk strategy across {total_trades} trades."
    ),
    ("B", "Weak"): (
        "Acceptable performance on limited data ({total_trades} trades, {num_days} days). "
        "{total_return:.1f}% return with {max_drawdown:.1f}% drawdown. "
        "A longer backtest window is recommended before live deployment."
    ),
    ("C+", "Strong"): (
        "{total_return:.1f}% return and {win_rate:.1f}% win rate show potential, but grading "
        "is suppressed by elevated drawdown ({max_drawdown:.1f}%) or low sample count "
        "({total_trades} trades). Tighten stop-loss rules and re-test."
    ),
    ("C+", "Moderate"): (
        "Below-average grade: {total_return:.1f}% return, {win_rate:.1f}% win rate, "
        "Sharpe {sharpe:.2f} over {num_days} days. Drawdown of {max_drawdown:.1f}% "
        "is elevated. Review signal quality and risk management."
    ),
    ("C+", "Weak"): (
        "Low return ({total_return:.1f}%) and modest trade count ({total_trades}) limit "
        "confidence. Drawdown of {max_drawdown:.1f}% over {num_days} days. "
        "Widen the test window and review entry conditions."
    ),
    ("C", "Strong"): (
        "Mixed result: {total_return:.1f}% return and {win_rate:.1f}% win rate indicate signal "
        "potential, but {max_drawdown:.1f}% max drawdown and Sharpe {sharpe:.2f} "
        "drag the grade down. Focus on reducing max adverse excursion."
    ),
    ("C", "Moderate"): (
        "Marginal performance over {num_days} days: {total_return:.1f}% return, "
        "{win_rate:.1f}% win rate, Sharpe {sharpe:.2f}. Drawdown {max_drawdown:.1f}% "
        "and inconsistency across {total_trades} trades require attention."
    ),
    ("C", "Weak"): (
        "Strategy barely qualifies: {total_trades} trades, {total_return:.1f}% return, "
        "{win_rate:.1f}% win rate, {max_drawdown:.1f}% drawdown. "
        "Substantial revision of entry/exit conditions is recommended."
    ),
    ("D", "Strong"): (
        "Despite {total_return:.1f}% return and {win_rate:.1f}% win rate, the strategy "
        "did not meet minimum pass thresholds (trade count or profit factor). "
        "Run over a longer period with more trade opportunities."
    ),
    ("D", "Moderate"): (
        "Strategy returned {total_return:.1f}% with {win_rate:.1f}% win rate across "
        "{total_trades} trades but fell below quality thresholds. "
        "Max drawdown {max_drawdown:.1f}% and Sharpe {sharpe:.2f} indicate elevated risk."
    ),
    ("D", "Weak"): (
        "Strategy significantly underperformed over {num_days} days: {total_return:.1f}% return, "
        "{win_rate:.1f}% win rate across {total_trades} trades, "
        "{max_drawdown:.1f}% max drawdown. Review market alignment, entry conditions, "
        "and risk parameters before re-testing."
    ),
}

_ASSESSMENT_NOTES_FALLBACK = (
    "Strategy returned {total_return:.1f}% over {num_days} days with {total_trades} trades, "
    "{win_rate:.1f}% win rate, Sharpe {sharpe:.2f}, max drawdown {max_drawdown:.1f}%."
)


def build_assessment_notes(
    *,
    grade: str,
    return_potential: str,
    risk_profile: str,
    metrics: dict,
) -> str:
    """
    Select a template based on (grade, return_potential) and fill all
    placeholders with computed metric values.

    Every number in the returned string comes from the metrics dict —
    the template defines structure, not content.
    """
    template = (
        _ASSESSMENT_NOTES.get((grade, return_potential))
        or _ASSESSMENT_NOTES.get(("D", return_potential))
        or _ASSESSMENT_NOTES_FALLBACK
    )
    return template.format(
        total_return=float(metrics.get("total_return_pct") or 0.0),
        num_days=int(metrics.get("num_days") or 0),
        sharpe=float(metrics.get("sharpe_ratio") or 0.0),
        max_drawdown=float(metrics.get("max_drawdown") or 0.0),
        win_rate=float(metrics.get("win_rate") or 0.0),
        total_trades=int(metrics.get("total_trades") or 0),
        risk_profile=str(risk_profile),
    )


# ── Market phase descriptions ─────────────────────────────────────────────────
# Key: (market_type, strategy_side_upper, alignment)
# Placeholders filled by build_phase_description():
#   {period}         — e.g. "2026 Q1"
#   {market_type}    — "Bull" / "Bear" / "Sideways"
#   {market_phase}   — "Uptrend" / "Downtrend" / "Range-bound"
#   {strategy_side}  — "LONG" / "SHORT"
#   {alignment}      — "Strong" / "Moderate" / "Weak"
#   {price_change}   — price return pct (use :+.1f for sign)
#   {trade_count}    — integer number of trades
#   {win_rate}       — win rate pct (:.1f)
#   {phase_return}   — cumulative strategy return pct for this phase (:+.1f)

_PHASE_DESCRIPTIONS: dict[tuple[str, str, str], str] = {
    # Long strategy — Bull market
    ("Bull", "LONG", "Strong"): (
        "During {period}, the market was in a {market_phase} phase with {price_change:+.1f}% "
        "price movement. The long strategy aligned strongly with the uptrend: "
        "{trade_count} trades at {win_rate:.1f}% win rate generated {phase_return:+.1f}% return."
    ),
    ("Bull", "LONG", "Moderate"): (
        "During {period}, the market trended upward ({market_phase}, {price_change:+.1f}%). "
        "The long strategy had moderate alignment: {trade_count} trades, "
        "{win_rate:.1f}% win rate, {phase_return:+.1f}% return."
    ),
    ("Bull", "LONG", "Weak"): (
        "During {period}, the market rose {price_change:+.1f}% ({market_phase}) but the long "
        "strategy underperformed expectations: {trade_count} trades at {win_rate:.1f}% win rate, "
        "{phase_return:+.1f}% return. Review signal quality — the trend was favourable "
        "but entries may have been poorly timed."
    ),
    # Long strategy — Bear market
    ("Bear", "LONG", "Weak"): (
        "During {period}, the market declined {price_change:+.1f}% ({market_phase}). "
        "The long-only strategy traded against the prevailing downtrend: {trade_count} trades "
        "with {win_rate:.1f}% win rate and {phase_return:+.1f}% return. "
        "A market direction filter would have significantly reduced losses in this phase."
    ),
    ("Bear", "LONG", "Moderate"): (
        "During {period}, the market fell {price_change:+.1f}% ({market_phase}). "
        "The long strategy showed partial resilience despite headwinds: "
        "{trade_count} trades, {win_rate:.1f}% win rate, {phase_return:+.1f}% return."
    ),
    ("Bear", "LONG", "Strong"): (
        "During {period}, the market declined {price_change:+.1f}% ({market_phase}) yet the "
        "long strategy outperformed: {trade_count} trades, {win_rate:.1f}% win rate, "
        "{phase_return:+.1f}% return. Strong counter-trend signal quality observed — "
        "entries captured recoveries effectively."
    ),
    # Long strategy — Sideways market
    ("Sideways", "LONG", "Strong"): (
        "During {period}, the market was range-bound ({market_phase}, {price_change:+.1f}%). "
        "The long strategy navigated choppy conditions effectively: {trade_count} trades, "
        "{win_rate:.1f}% win rate, {phase_return:+.1f}% return."
    ),
    ("Sideways", "LONG", "Moderate"): (
        "During {period}, the market was in a {market_phase} phase ({price_change:+.1f}%). "
        "The long strategy achieved moderate results in mixed conditions: "
        "{trade_count} trades, {win_rate:.1f}% win rate, {phase_return:+.1f}% return."
    ),
    ("Sideways", "LONG", "Weak"): (
        "During {period}, the market moved sideways ({market_phase}, {price_change:+.1f}%). "
        "The long strategy struggled in range-bound conditions: {trade_count} trades, "
        "{win_rate:.1f}% win rate, {phase_return:+.1f}% return. "
        "Consider adding a volatility or momentum filter for ranging markets."
    ),
    # Short strategy — Bull market
    ("Bull", "SHORT", "Weak"): (
        "During {period}, the market rose {price_change:+.1f}% ({market_phase}). "
        "The short strategy traded against the uptrend: {trade_count} trades, "
        "{win_rate:.1f}% win rate, {phase_return:+.1f}% return."
    ),
    ("Bull", "SHORT", "Moderate"): (
        "During {period}, the market trended upward ({market_phase}, {price_change:+.1f}%). "
        "The short strategy showed selective success: {trade_count} trades, "
        "{win_rate:.1f}% win rate, {phase_return:+.1f}% return."
    ),
    ("Bull", "SHORT", "Strong"): (
        "During {period}, the market rose {price_change:+.1f}% ({market_phase}) yet the short "
        "strategy outperformed: {trade_count} trades, {win_rate:.1f}% win rate, "
        "{phase_return:+.1f}% return. Strong mean-reversion or pullback signals detected."
    ),
    # Short strategy — Bear market
    ("Bear", "SHORT", "Strong"): (
        "During {period}, the market declined {price_change:+.1f}% ({market_phase}). "
        "The short strategy aligned strongly with the downtrend: {trade_count} trades, "
        "{win_rate:.1f}% win rate, {phase_return:+.1f}% return."
    ),
    ("Bear", "SHORT", "Moderate"): (
        "During {period}, the market fell {price_change:+.1f}% ({market_phase}). "
        "The short strategy had moderate alignment: {trade_count} trades, "
        "{win_rate:.1f}% win rate, {phase_return:+.1f}% return."
    ),
    ("Bear", "SHORT", "Weak"): (
        "During {period}, the market declined {price_change:+.1f}% ({market_phase}) but the "
        "short strategy underperformed: {trade_count} trades, {win_rate:.1f}% win rate, "
        "{phase_return:+.1f}% return. Review entry timing."
    ),
    # Short strategy — Sideways market
    ("Sideways", "SHORT", "Strong"): (
        "During {period}, the market was range-bound ({market_phase}, {price_change:+.1f}%). "
        "The short strategy performed well in choppy conditions: {trade_count} trades, "
        "{win_rate:.1f}% win rate, {phase_return:+.1f}% return."
    ),
    ("Sideways", "SHORT", "Moderate"): (
        "During {period}, the market was in a {market_phase} phase ({price_change:+.1f}%). "
        "The short strategy recorded {trade_count} trades at {win_rate:.1f}% win rate, "
        "{phase_return:+.1f}% return."
    ),
    ("Sideways", "SHORT", "Weak"): (
        "During {period}, the market moved sideways ({market_phase}, {price_change:+.1f}%). "
        "The short strategy underperformed: {trade_count} trades, {win_rate:.1f}% win rate, "
        "{phase_return:+.1f}% return."
    ),
}

_PHASE_DESCRIPTION_FALLBACK = (
    "During {period}, the market was in a {market_phase} phase with {price_change:+.1f}% "
    "price movement ({market_type}). The {strategy_side} strategy executed {trade_count} "
    "trades with {win_rate:.1f}% win rate and {phase_return:+.1f}% cumulative return. "
    "Observed alignment: {alignment}."
)


def build_phase_description(
    *,
    period: str,
    market_type: str,
    market_phase: str,
    strategy_side: str,
    alignment: str,
    price_change_pct: float,
    trade_count: int,
    win_rate_pct: float,
    phase_return_pct: float,
) -> str:
    """
    Select a template based on (market_type, strategy_side, alignment) and fill
    all placeholders with computed values.

    Every number in the returned string comes from a parameter passed by the caller —
    the template defines narrative structure, not the values themselves.
    """
    key = (market_type, strategy_side.upper(), alignment)
    template = _PHASE_DESCRIPTIONS.get(key, _PHASE_DESCRIPTION_FALLBACK)
    return template.format(
        period=period,
        market_type=market_type,
        market_phase=market_phase,
        strategy_side=strategy_side.upper(),
        alignment=alignment,
        price_change=price_change_pct,
        trade_count=trade_count,
        win_rate=win_rate_pct,
        phase_return=phase_return_pct,
    )
