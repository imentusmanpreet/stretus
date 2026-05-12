"""
app/planner/stock_recommender.py — Phase 4 stock-selection (rule #24).

Maps a strategy prompt to a ranked subset of the supported universe with a
plain-English justification per pick. Selection is deterministic and
explainable:

  1.  characterise_strategy()  — read the user's prompt + the
                                  SemanticInstructions + any signal-filter
                                  / market-context toggles to produce a
                                  `StrategyProfile` (trending-ness, volatility
                                  tolerance, sector preference, liquidity
                                  need, indicator family).
  2.  score_stocks()           — score each supported stock against the
                                  profile using its `StockProfile`.
  3.  recommend()              — pick the top-N, attach per-stock
                                  justification lines explaining why.

The per-stock profiles are coarse-grained tier labels (high/medium/low),
chosen so the recommendations are credible without claiming false
precision. Tier values reflect typical behaviour of the named stock in
the Indian market over the last few years and can be tuned by editing
this single file without touching call sites.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ── Stock profiles ────────────────────────────────────────────────────────────


Tier = str  # "high" | "medium" | "low"


@dataclass(frozen=True)
class StockProfile:
    symbol:               str
    display_name:         str
    sector:               str
    market_cap_tier:      Tier
    liquidity_tier:       Tier
    volatility_tier:      Tier
    trending_tendency:    Tier        # how cleanly it respects trend indicators
    mean_reversion_score: Tier        # how reliably it reverts to MA
    intraday_friendliness: Tier
    notes:                str = ""


SUPPORTED_STOCKS: list[StockProfile] = [
    StockProfile(
        symbol="TCS.NS",
        display_name="TCS",
        sector="IT services",
        market_cap_tier="high",
        liquidity_tier="high",
        volatility_tier="low",
        trending_tendency="high",
        mean_reversion_score="medium",
        intraday_friendliness="medium",
        notes="Steady large-cap IT major. Respects EMAs cleanly; muted spikes.",
    ),
    StockProfile(
        symbol="INFY.NS",
        display_name="Infosys",
        sector="IT services",
        market_cap_tier="high",
        liquidity_tier="high",
        volatility_tier="low",
        trending_tendency="high",
        mean_reversion_score="medium",
        intraday_friendliness="medium",
        notes="Sister-stock to TCS. Strong trend behaviour on results calendar.",
    ),
    StockProfile(
        symbol="RELIANCE.NS",
        display_name="Reliance",
        sector="energy / diversified conglomerate",
        market_cap_tier="high",
        liquidity_tier="high",
        volatility_tier="medium",
        trending_tendency="high",
        mean_reversion_score="medium",
        intraday_friendliness="high",
        notes="Index-heavyweight; deep liquidity; suitable for momentum and pullback styles.",
    ),
    StockProfile(
        symbol="ADANIENT.NS",
        display_name="Adani Enterprises",
        sector="diversified / infrastructure",
        market_cap_tier="high",
        liquidity_tier="medium",
        volatility_tier="high",
        trending_tendency="medium",
        mean_reversion_score="low",
        intraday_friendliness="medium",
        notes="Headline-driven; large gaps. Best with explicit news-event filtering.",
    ),
    StockProfile(
        symbol="HDFCBANK.NS",
        display_name="HDFC Bank",
        sector="banking",
        market_cap_tier="high",
        liquidity_tier="high",
        volatility_tier="low",
        trending_tendency="high",
        mean_reversion_score="high",
        intraday_friendliness="high",
        notes="Banking heavyweight; deep order book; clean technical behaviour.",
    ),
    StockProfile(
        symbol="NHPC.NS",
        display_name="NHPC",
        sector="utilities / power",
        market_cap_tier="medium",
        liquidity_tier="medium",
        volatility_tier="low",
        trending_tendency="medium",
        mean_reversion_score="high",
        intraday_friendliness="low",
        notes="Utility — low intraday range; better suited to swing / positional.",
    ),
    StockProfile(
        symbol="SUZLON.NS",
        display_name="Suzlon",
        sector="renewable energy",
        market_cap_tier="medium",
        liquidity_tier="high",
        volatility_tier="high",
        trending_tendency="medium",
        mean_reversion_score="low",
        intraday_friendliness="high",
        notes="Volatile small/mid-cap; works for momentum breakouts; gappy.",
    ),
    StockProfile(
        symbol="GMRAIRPORTS.NS",
        display_name="GMR Airports",
        sector="infrastructure",
        market_cap_tier="medium",
        liquidity_tier="medium",
        volatility_tier="medium",
        trending_tendency="medium",
        mean_reversion_score="medium",
        intraday_friendliness="medium",
    ),
    StockProfile(
        symbol="IDEA.NS",
        display_name="Vodafone Idea",
        sector="telecom",
        market_cap_tier="low",
        liquidity_tier="medium",
        volatility_tier="high",
        trending_tendency="low",
        mean_reversion_score="low",
        intraday_friendliness="medium",
        notes="Penny-stock dynamics; news-driven; indicators less reliable.",
    ),
]


# ── Strategy profile ──────────────────────────────────────────────────────────


@dataclass
class StrategyProfile:
    """Coarse-grained characterisation of the user's strategy used for
    scoring. Values are normalised 0.0-1.0 weights."""
    trend_weight:        float = 0.0    # 0 = not trend-style; 1 = pure trend.
    mean_reversion_weight: float = 0.0
    volatility_tolerance:  float = 0.5  # 0 = wants stable, 1 = wants volatile.
    liquidity_need:        float = 0.7  # 0 = doesn't care, 1 = needs deep book.
    intraday_focus:        float = 0.5
    needs_news_resilience: float = 0.5  # 0 = tolerant, 1 = wants clean technicals.
    sector_preferences:    list[str] = field(default_factory=list)
    raw_signals:           list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trend_weight":         self.trend_weight,
            "mean_reversion_weight": self.mean_reversion_weight,
            "volatility_tolerance":  self.volatility_tolerance,
            "liquidity_need":        self.liquidity_need,
            "intraday_focus":        self.intraday_focus,
            "needs_news_resilience": self.needs_news_resilience,
            "sector_preferences":    list(self.sector_preferences),
            "raw_signals":           list(self.raw_signals),
        }


_TREND_KEYWORDS = (
    "trend", "trending", "pullback", "breakout", "ema cross", "ema pullback",
    "supertrend", "momentum", "ichimoku", "donchian", "vwap reclaim",
    "higher highs", "higher lows", "structural trend",
)
_MEAN_REVERSION_KEYWORDS = (
    "mean reversion", "reversion", "oversold bounce", "overbought reversal",
    "bollinger reversal", "rsi reversal", "fade",
)
_VOLATILE_KEYWORDS = (
    "scalp", "scalping", "breakout", "volatile", "spike", "explosive",
    "wide ranging", "big move",
)
_STABLE_KEYWORDS = (
    "stable", "low volatility", "smooth trend", "quiet", "controlled",
)
_INTRADAY_KEYWORDS = (
    "intraday", "1 min", "1min", "5 min", "5min", "15 min", "15min",
    "scalp", "day trade",
)
_POSITIONAL_KEYWORDS = (
    "positional", "swing", "weekly", "daily chart", "long term",
)


def characterise_strategy(
    prompt_text: str,
    instructions: Any | None,
    builder: Any,
) -> StrategyProfile:
    haystack = re.sub(r"\s+", " ", (prompt_text or "")).lower()
    profile = StrategyProfile()

    trend_hits        = sum(1 for kw in _TREND_KEYWORDS         if kw in haystack)
    mr_hits           = sum(1 for kw in _MEAN_REVERSION_KEYWORDS if kw in haystack)
    volatile_hits     = sum(1 for kw in _VOLATILE_KEYWORDS       if kw in haystack)
    stable_hits       = sum(1 for kw in _STABLE_KEYWORDS         if kw in haystack)
    intraday_hits     = sum(1 for kw in _INTRADAY_KEYWORDS       if kw in haystack)
    positional_hits   = sum(1 for kw in _POSITIONAL_KEYWORDS     if kw in haystack)

    # Trend / mean reversion: normalise relative to whichever wins.
    total_style = max(trend_hits + mr_hits, 1)
    profile.trend_weight         = min(1.0, trend_hits / total_style)
    profile.mean_reversion_weight = min(1.0, mr_hits / total_style)

    # Strategy family from semantic extraction overrides keyword counts.
    family = getattr(instructions, "strategy_family", None) if instructions else None
    if family in ("ORB", "BREAKOUT", "MOMENTUM"):
        profile.trend_weight = max(profile.trend_weight, 0.8)
    elif family in ("MEAN_REVERSION",):
        profile.mean_reversion_weight = max(profile.mean_reversion_weight, 0.8)
    elif family in ("EMA_PULLBACK", "VWAP_RECLAIM", "ICT_BOS_FVG"):
        profile.trend_weight = max(profile.trend_weight, 0.7)
    elif family in ("REVERSAL",):
        profile.mean_reversion_weight = max(profile.mean_reversion_weight, 0.7)

    # Volatility tolerance.
    if volatile_hits and not stable_hits:
        profile.volatility_tolerance = 0.85
    elif stable_hits and not volatile_hits:
        profile.volatility_tolerance = 0.2
    else:
        profile.volatility_tolerance = 0.5

    # Intraday vs positional focus.
    timeframe = (getattr(builder, "timeframe", "") or "").lower()
    objective = (getattr(builder, "objective", "") or "").lower()
    if intraday_hits or objective == "intraday" or timeframe in ("1m", "3m", "5m", "15m", "30m"):
        profile.intraday_focus = 0.85
    elif positional_hits or objective == "positional" or timeframe in ("1d", "1w"):
        profile.intraday_focus = 0.15
    else:
        profile.intraday_focus = 0.5

    # Liquidity need: scalping and tight-stop strategies need deep books.
    if profile.intraday_focus > 0.7 or "scalp" in haystack:
        profile.liquidity_need = 0.95
    elif profile.intraday_focus < 0.3:
        profile.liquidity_need = 0.5
    else:
        profile.liquidity_need = 0.75

    # News resilience: strategies that depend on clean technicals (EMA
    # pullback, multiple-confirmation entries) need stocks that don't gap
    # on headlines.
    if any(kw in haystack for kw in ("news filter", "earnings", "event blackout", "clean technicals", "pullback")):
        profile.needs_news_resilience = 0.85
    else:
        profile.needs_news_resilience = 0.5

    # Sector preferences — explicit mentions only.
    for sec_kw, label in (
        ("it ", "IT services"), ("technology", "IT services"),
        ("banking", "banking"), ("bank ", "banking"),
        ("energy", "energy / diversified conglomerate"),
        ("infrastructure", "infrastructure"), ("infra ", "infrastructure"),
        ("renewable", "renewable energy"), ("wind", "renewable energy"),
        ("utilities", "utilities / power"), ("power ", "utilities / power"),
        ("telecom", "telecom"),
    ):
        if sec_kw in haystack and label not in profile.sector_preferences:
            profile.sector_preferences.append(label)

    if family:
        profile.raw_signals.append(f"strategy_family={family}")
    return profile


# ── Scoring ──────────────────────────────────────────────────────────────────


_TIER_VALUE = {"low": 0.2, "medium": 0.55, "high": 0.85}


def _tier(value: Tier) -> float:
    return _TIER_VALUE.get(value, 0.5)


@dataclass
class StockRecommendation:
    stock:         StockProfile
    score:         float
    rationale:     list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol":       self.stock.symbol,
            "display_name": self.stock.display_name,
            "sector":       self.stock.sector,
            "score":        round(self.score, 3),
            "rationale":    list(self.rationale),
        }


def score_stocks(profile: StrategyProfile) -> list[StockRecommendation]:
    results: list[StockRecommendation] = []
    for stock in SUPPORTED_STOCKS:
        score = 0.0
        rationale: list[str] = []

        # Trend behaviour vs strategy's trend weight.
        if profile.trend_weight > 0:
            contrib = profile.trend_weight * _tier(stock.trending_tendency)
            score += contrib * 2.5
            if _tier(stock.trending_tendency) >= 0.7:
                rationale.append(
                    f"trends cleanly ({stock.trending_tendency} trending-tendency) — "
                    "good fit for a trend-following style"
                )

        # Mean-reversion vs strategy's mean-reversion weight.
        if profile.mean_reversion_weight > 0:
            contrib = profile.mean_reversion_weight * _tier(stock.mean_reversion_score)
            score += contrib * 2.5
            if _tier(stock.mean_reversion_score) >= 0.7:
                rationale.append(
                    "reverts predictably to its mean — works well for reversion entries"
                )

        # Volatility tolerance match. Penalty when mismatched.
        vol_match = 1.0 - abs(_tier(stock.volatility_tier) - profile.volatility_tolerance)
        score += vol_match * 1.5
        if profile.volatility_tolerance >= 0.7 and stock.volatility_tier == "high":
            rationale.append("high intraday volatility matches the strategy's appetite")
        elif profile.volatility_tolerance <= 0.3 and stock.volatility_tier == "low":
            rationale.append("low volatility matches the strategy's preference for stable conditions")
        elif vol_match < 0.4:
            rationale.append(
                f"volatility mismatch ({stock.volatility_tier} vs strategy preference) — caution"
            )

        # Liquidity.
        liq_match = 1.0 - abs(_tier(stock.liquidity_tier) - profile.liquidity_need)
        score += liq_match * 1.0
        if profile.liquidity_need >= 0.7 and stock.liquidity_tier == "high":
            rationale.append("deep liquidity supports tight-stop / scalp orders")
        elif profile.liquidity_need >= 0.7 and stock.liquidity_tier != "high":
            rationale.append(
                f"liquidity tier ({stock.liquidity_tier}) below what the strategy ideally needs"
            )

        # Intraday focus.
        intraday_match = 1.0 - abs(_tier(stock.intraday_friendliness) - profile.intraday_focus)
        score += intraday_match * 0.75
        if profile.intraday_focus >= 0.7 and stock.intraday_friendliness == "high":
            rationale.append("intraday-friendly behaviour (range and liquidity)")
        elif profile.intraday_focus >= 0.7 and stock.intraday_friendliness == "low":
            rationale.append("limited intraday range — less suited to fast strategies")

        # News resilience: penalise news-driven stocks when strategy wants
        # clean technicals.
        if profile.needs_news_resilience >= 0.7 and stock.volatility_tier == "high" and stock.notes:
            note = stock.notes.lower()
            if any(kw in note for kw in ("news", "headline", "gappy")):
                score -= 1.0
                rationale.append("often news-driven — may break clean-technical setups")

        # Sector preference override.
        if profile.sector_preferences:
            if stock.sector in profile.sector_preferences:
                score += 1.5
                rationale.append(f"sector match ({stock.sector})")
            else:
                score -= 0.25

        results.append(StockRecommendation(stock=stock, score=score, rationale=rationale))

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def recommend(
    prompt_text: str,
    instructions: Any | None,
    builder: Any,
    *,
    top_n: int = 3,
) -> dict[str, Any]:
    """Top-level entry. Returns a dict shaped for persistence on the
    StrategyBuilder draft and for direct rendering in the chat summary."""
    profile = characterise_strategy(prompt_text, instructions, builder)
    ranked = score_stocks(profile)
    top = ranked[:max(1, top_n)]
    rest = ranked[max(1, top_n):]
    return {
        "profile":      profile.to_dict(),
        "top_picks":    [r.to_dict() for r in top],
        "other_picks":  [r.to_dict() for r in rest],
        "supported_universe": [s.symbol for s in SUPPORTED_STOCKS],
    }


# ── Summary rendering ────────────────────────────────────────────────────────


def render_recommendations_for_summary(rec: dict[str, Any]) -> list[str]:
    lines: list[str] = ["── Recommended supported stocks ──"]
    if not rec or not rec.get("top_picks"):
        lines.append("No clear recommendation — provide more strategy detail.")
        return lines
    profile = rec.get("profile", {})
    style = []
    if profile.get("trend_weight", 0) >= 0.5:
        style.append("trend-following")
    if profile.get("mean_reversion_weight", 0) >= 0.5:
        style.append("mean-reversion")
    if profile.get("volatility_tolerance", 0.5) >= 0.7:
        style.append("volatility-tolerant")
    elif profile.get("volatility_tolerance", 0.5) <= 0.3:
        style.append("stable-conditions")
    if profile.get("intraday_focus", 0.5) >= 0.7:
        style.append("intraday")
    style_str = ", ".join(style) or "general-purpose"
    lines.append(f"I read your strategy as {style_str}. Based on that, the best supported stocks are:")
    for idx, pick in enumerate(rec["top_picks"], start=1):
        lines.append(f"  {idx}. {pick['display_name']} ({pick['symbol']})  — score {pick['score']}")
        for reason in pick.get("rationale", []):
            lines.append(f"      • {reason}")
    if rec.get("other_picks"):
        also = ", ".join(p["display_name"] for p in rec["other_picks"][:3])
        lines.append(f"  Other supported stocks (lower fit): {also}")
    lines.append(
        "Reply with the stock you want to use (or 'use my pick' to keep the one you already named)."
    )
    return lines


__all__ = [
    "SUPPORTED_STOCKS",
    "StockProfile",
    "StockRecommendation",
    "StrategyProfile",
    "characterise_strategy",
    "recommend",
    "render_recommendations_for_summary",
    "score_stocks",
]
