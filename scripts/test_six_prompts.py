"""
scripts/test_six_prompts.py — Validate fidelity of the chat planner extraction
on six representative user prompts. Run from repo root:

    python3 scripts/test_six_prompts.py

For each prompt we report what the system captures (with the fixes applied),
compare against the user's literal asks, and flag every gap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.planner.semantic_extractor import SemanticExtractor
from app.planner.unsupported_indicators import detect_unsupported
from app.services.strategy.builder import _extract_rms_from_text


@dataclass
class Expectation:
    """What the user actually asked for in the prompt."""
    symbol: str
    timeframe: str
    direction: str | None         # long_only / short_only / both / None
    objective: str | None         # swing / intraday / etc.
    experience: str | None
    indicators: dict[str, list[int]]   # {"SMA": [20], ...}
    indicator_phrases: list[str]       # raw mentions (for UNSUPPORTED check)
    entry_logic: list[str]             # human-readable list of entry rules
    exit_logic: list[str]
    stop_loss: str | None              # e.g. "candle_low (prev)", "atr"
    take_profit: str | None
    risk_reward: float | None
    trailing_stop: str | None
    htf_filter: str | None


PROMPTS: list[tuple[str, Expectation]] = [
    (
        "Create a beginner-friendly swing trading strategy for TCS using 15-minute candles. "
        "Buy when price crosses above the 20 SMA and sell when price closes below the 20 SMA. "
        "Add stop loss below previous candle low and target 2:1 reward risk ratio.",
        Expectation(
            symbol="TCS", timeframe="15m", direction="long_only",
            objective="swing", experience="beginner",
            indicators={"SMA": [20]},
            indicator_phrases=["20 SMA"],
            entry_logic=["price crosses above SMA(20)"],
            exit_logic=["price closes below SMA(20)"],
            stop_loss="candle_low (previous)",
            take_profit=None,
            risk_reward=2.0,    # "2:1 reward risk" → reward 2 / risk 1 = 2.0
            trailing_stop=None, htf_filter=None,
        ),
    ),
    (
        "Build a simple intraday strategy for RELIANCE on 5-minute timeframe using RSI. "
        "Enter long when RSI crosses above 30 from oversold zone and candle closes bullish. "
        "Exit when RSI reaches above 65 or stop loss is hit.",
        Expectation(
            symbol="RELIANCE", timeframe="5m", direction="long_only",
            objective="intraday", experience=None,
            indicators={"RSI": []},     # period not specified — should NOT be invented
            indicator_phrases=["RSI"],
            entry_logic=[
                "RSI crosses above 30 (from oversold)",
                "candle closes bullish",
            ],
            exit_logic=["RSI > 65", "SL hit"],
            stop_loss=None, take_profit=None, risk_reward=None,
            trailing_stop=None, htf_filter=None,
        ),
    ),
    (
        "Design a beginner intraday strategy for HDFCBANK using VWAP on 5-minute timeframe. "
        "Buy only when price stays above VWAP with bullish candles and volume confirmation. "
        "Use trailing stop loss after 1% profit.",
        Expectation(
            symbol="HDFCBANK", timeframe="5m", direction="long_only",
            objective="intraday", experience="beginner",
            indicators={"VWAP": []},
            indicator_phrases=["VWAP"],
            entry_logic=[
                "price > VWAP",
                "bullish candle",
                "volume confirmation",
            ],
            exit_logic=["trailing stop after 1% profit"],
            stop_loss=None, take_profit=None, risk_reward=None,
            trailing_stop="1% activation",
            htf_filter=None,
        ),
    ),
    (
        "Create a breakout trading strategy for ITC on 15-minute chart. "
        "Enter buy trade when price breaks previous day high with strong bullish candle and higher than average volume. "
        "Keep stop loss below breakout candle low.",
        Expectation(
            symbol="ITC", timeframe="15m", direction="long_only",
            objective=None, experience=None,
            indicators={},
            indicator_phrases=[],
            entry_logic=[
                "price breaks previous day high",
                "strong bullish candle",
                "volume > average",
            ],
            exit_logic=[],
            stop_loss="candle_low (breakout candle)",
            take_profit=None, risk_reward=None,
            trailing_stop=None, htf_filter=None,
        ),
    ),
    (
        "Generate a trend-following strategy for INFY using 9 EMA and 21 EMA on 5-minute timeframe. "
        "Buy when 9 EMA crosses above 21 EMA and avoid trades during sideways market conditions.",
        Expectation(
            symbol="INFY", timeframe="5m", direction="long_only",
            objective=None, experience=None,
            indicators={"EMA": [9, 21]},
            indicator_phrases=["9 EMA", "21 EMA"],
            entry_logic=["EMA(9) crosses above EMA(21)"],
            exit_logic=[],
            stop_loss=None, take_profit=None, risk_reward=None,
            trailing_stop=None,
            htf_filter="avoid sideways (trending-regime only)",
        ),
    ),
    (
        "Build an intraday momentum strategy for ADANIPORTS on 5-minute timeframe. "
        "Enter only when RSI is above 60, MACD histogram turns positive, and price closes above VWAP. "
        "Use ATR-based stop loss and partial profit booking at 1.5R.",
        Expectation(
            symbol="ADANIPORTS", timeframe="5m", direction="long_only",
            objective="intraday", experience=None,
            indicators={"RSI": [], "MACD": [], "VWAP": []},
            indicator_phrases=["RSI", "MACD", "VWAP"],
            entry_logic=[
                "RSI > 60",
                "MACD histogram > 0",
                "price > VWAP",
            ],
            exit_logic=["partial profit at 1.5R"],
            stop_loss="ATR-based",
            take_profit=None, risk_reward=1.5,    # 1.5R = reward/risk = 1.5
            trailing_stop=None, htf_filter=None,
        ),
    ),
]


# ───────────────────────────────────────────────────────────────────────────────
# Lightweight extractors for the items the chat layer normally handles via the
# LLM agent + agent_tool_parameters. We mimic them with regex so this script can
# run without the LLM. These mirror the same logic chat_service.py would apply.
# ───────────────────────────────────────────────────────────────────────────────

def _extract_symbol(text: str) -> str | None:
    KNOWN = ("TCS", "RELIANCE", "HDFCBANK", "ITC", "INFY", "ADANIPORTS")
    for s in KNOWN:
        if re.search(rf"\b{s}\b", text, re.IGNORECASE):
            return s
    return None


def _extract_timeframe(text: str) -> str | None:
    m = re.search(r"\b(\d+)[\s\-]*(?:minute|min|m)\b", text, re.IGNORECASE)
    if m:
        return f"{m.group(1)}m"
    if re.search(r"\b(?:daily|day)\b", text, re.IGNORECASE):
        return "1d"
    if re.search(r"\bhourly|1h\b", text, re.IGNORECASE):
        return "1h"
    return None


def _extract_objective(text: str) -> str | None:
    if re.search(r"\bswing\b", text, re.IGNORECASE):
        return "swing"
    if re.search(r"\bintraday\b|\bday\s+trade\b|\bday\s+trading\b", text, re.IGNORECASE):
        return "intraday"
    return None


def _extract_experience(text: str) -> str | None:
    if re.search(r"\bbeginner\b", text, re.IGNORECASE):
        return "beginner"
    if re.search(r"\bintermediate\b", text, re.IGNORECASE):
        return "intermediate"
    if re.search(r"\b(?:expert|advanced|pro)\b", text, re.IGNORECASE):
        return "expert"
    return None


# ───────────────────────────────────────────────────────────────────────────────
# Per-prompt validator
# ───────────────────────────────────────────────────────────────────────────────

def validate_prompt(idx: int, prompt: str, expected: Expectation) -> dict:
    """Run extraction and compare to expectation. Returns a report dict."""
    inst = SemanticExtractor().extract(prompt)
    rms = _extract_rms_from_text(prompt)
    unsupported = detect_unsupported(prompt)

    sym = _extract_symbol(prompt)
    tf = _extract_timeframe(prompt)
    obj = _extract_objective(prompt)
    exp = _extract_experience(prompt)

    # Build the captured snapshot
    captured = {
        "symbol": sym,
        "timeframe": tf,
        "objective": obj,
        "experience": exp,
        "direction": inst.direction,
        "indicators": dict(inst.indicators or {}),
        "stop_loss": (
            {"type": inst.stop_loss.type, "anchor": inst.stop_loss.anchor}
            if inst.stop_loss else None
        ),
        "risk_reward": (
            inst.risk_reward.ratio if inst.risk_reward and inst.risk_reward.ratio
            else (rms.get("risk_reward") or {}).get("value")
        ),
        "trailing_stop": (
            {
                "type": inst.trailing_stop.type if inst.trailing_stop else None,
                "activate_after_pct": inst.trailing_stop.activate_after_pct if inst.trailing_stop else None,
            }
            if inst.trailing_stop and inst.trailing_stop.enabled else None
        ),
        "htf_rules": [r.dict() for r in (inst.htf_rules or [])],
        "rsi_band": {
            "min": inst.rsi_entry_band_min,
            "max": inst.rsi_entry_band_max,
        } if inst.rsi_entry_band_min or inst.rsi_entry_band_max else None,
        "rsi_thresholds": inst.rsi_thresholds,
        "macd_states": inst.macd_states,
        "vwap_relations": inst.vwap_relations,
        "regime_preference": inst.regime_preference,
        "sl_type_hint": inst.sl_type_hint,
        "partial_exits": inst.partial_exits,
        "volume_ratio": inst.volume_ratio_threshold,
        "extraction_quality": round(inst.extraction_quality_score, 2),
        "rms_extracted": {k: v for k, v in rms.items()},
        "unsupported_flagged": [m.matched_phrase for m in unsupported],
    }

    # Build a gap report
    gaps: list[str] = []

    if sym != expected.symbol:
        gaps.append(f"symbol: got {sym!r}, want {expected.symbol!r}")
    if tf != expected.timeframe:
        gaps.append(f"timeframe: got {tf!r}, want {expected.timeframe!r}")
    if obj != expected.objective:
        gaps.append(f"objective: got {obj!r}, want {expected.objective!r}")
    if exp != expected.experience:
        gaps.append(f"experience: got {exp!r}, want {expected.experience!r}")

    # Direction may be inferred long_only from "buy"
    if expected.direction and inst.direction != expected.direction:
        gaps.append(f"direction: got {inst.direction!r}, want {expected.direction!r}")

    # Indicator families — every expected family should appear
    for family, periods in expected.indicators.items():
        got_periods = inst.indicators.get(family, [])
        if family not in inst.indicators:
            gaps.append(f"indicator: {family} not captured")
        elif periods and sorted(periods) != sorted(got_periods):
            gaps.append(
                f"indicator.{family}: periods got {got_periods}, want {periods}"
            )

    # SL anchor — accept either StructuralStopLoss object (candle anchors) OR
    # sl_type_hint=atr|percent for non-structural stops.
    if expected.stop_loss:
        want = expected.stop_loss.lower()
        if "atr" in want:
            if inst.sl_type_hint != "atr" and "atr" not in (
                (inst.stop_loss.type if inst.stop_loss else "") or ""
            ):
                gaps.append(f"stop_loss: ATR-based not captured (got sl_type_hint={inst.sl_type_hint!r})")
        elif "candle" in want:
            if not inst.stop_loss or "candle" not in (
                (inst.stop_loss.anchor or inst.stop_loss.type or "").lower()
            ):
                got = (inst.stop_loss.anchor if inst.stop_loss else None)
                gaps.append(f"stop_loss anchor: got {got!r}, want {expected.stop_loss!r}")
        elif not inst.stop_loss and inst.sl_type_hint is None:
            gaps.append(f"stop_loss: not captured (wanted {expected.stop_loss!r})")

    # RR
    if expected.risk_reward is not None:
        got_rr = (
            inst.risk_reward.ratio if inst.risk_reward and inst.risk_reward.ratio
            else (rms.get("risk_reward") or {}).get("value")
        )
        if got_rr is None:
            gaps.append(f"risk_reward: not captured (wanted {expected.risk_reward})")
        elif abs(float(got_rr) - expected.risk_reward) > 0.01:
            gaps.append(f"risk_reward: got {got_rr}, want {expected.risk_reward}")

    # Trailing
    if expected.trailing_stop and not (inst.trailing_stop and inst.trailing_stop.enabled):
        gaps.append(f"trailing_stop: not captured (wanted {expected.trailing_stop!r})")

    # HTF filter — accept either explicit htf_rules OR a regime_preference
    # capture (e.g. "avoid sideways" → regime_preference="trending").
    if expected.htf_filter and not inst.htf_rules and not inst.regime_preference:
        gaps.append(f"htf_filter: no htf_rules or regime_preference captured (wanted {expected.htf_filter!r})")

    # Entry logic items the system should at least see
    entry_seen: list[str] = []
    entry_missed: list[str] = []
    for rule in expected.entry_logic:
        rule_l = rule.lower()
        # Heuristic: check if any indicator/keyword from the rule landed in the
        # captured structures
        landed = False
        if "rsi" in rule_l:
            if "above 60" in rule_l:
                landed = any(t["op"] == "above" and t["value"] >= 60 for t in inst.rsi_thresholds)
            elif "above 30" in rule_l:
                landed = any(t["op"] == "above" and 25 <= t["value"] <= 35 for t in inst.rsi_thresholds)
            elif "RSI" in inst.indicators:
                landed = True
        elif "vwap" in rule_l:
            landed = bool(inst.vwap_relations) or "VWAP" in inst.indicators
        elif "macd" in rule_l:
            landed = bool(inst.macd_states) or "MACD" in inst.indicators
        elif "ema" in rule_l and "EMA" in inst.indicators:
            landed = True
        elif "sma" in rule_l and "SMA" in inst.indicators:
            landed = True
        elif "volume" in rule_l and inst.volume_ratio_threshold:
            landed = True
        elif "bullish candle" in rule_l and inst.candle_confirmation:
            landed = True
        elif "breaks" in rule_l or "breakout" in rule_l:
            landed = False
        (entry_seen if landed else entry_missed).append(rule)

    return {
        "idx": idx,
        "prompt": prompt,
        "captured": captured,
        "gaps": gaps,
        "entry_seen": entry_seen,
        "entry_missed": entry_missed,
        "expected": expected,
    }


# ───────────────────────────────────────────────────────────────────────────────
# Report
# ───────────────────────────────────────────────────────────────────────────────

def _yn(v) -> str:
    if v is None: return "—"
    if v is False: return "no"
    if v is True: return "yes"
    return str(v)


def render_report(reports: list[dict]) -> str:
    out: list[str] = []
    total_gaps = 0
    for r in reports:
        c = r["captured"]
        e = r["expected"]
        out.append("=" * 78)
        out.append(f"PROMPT #{r['idx']} — {e.symbol} / {e.timeframe}")
        out.append("=" * 78)
        out.append(r["prompt"])
        out.append("")
        out.append("[ CAPTURED ]")
        out.append(f"  symbol            : {c['symbol']}")
        out.append(f"  timeframe         : {c['timeframe']}")
        out.append(f"  objective         : {c['objective']}")
        out.append(f"  experience        : {c['experience']}")
        out.append(f"  direction         : {c['direction']}")
        out.append(f"  indicators        : {c['indicators']}")
        out.append(f"  stop_loss         : {c['stop_loss']}")
        out.append(f"  risk_reward       : {c['risk_reward']}")
        out.append(f"  trailing_stop     : {c['trailing_stop']}")
        out.append(f"  htf_rules         : {len(c['htf_rules'])} rule(s)")
        out.append(f"  rsi_band          : {c['rsi_band']}")
        out.append(f"  rsi_thresholds    : {c['rsi_thresholds']}")
        out.append(f"  macd_states       : {c['macd_states']}")
        out.append(f"  vwap_relations    : {c['vwap_relations']}")
        out.append(f"  regime_preference : {c['regime_preference']}")
        out.append(f"  sl_type_hint      : {c['sl_type_hint']}")
        out.append(f"  partial_exits     : {c['partial_exits']}")
        out.append(f"  volume_ratio      : {c['volume_ratio']}")
        out.append(f"  extraction_quality: {c['extraction_quality']}")
        out.append(f"  rms_extracted     : {c['rms_extracted']}")
        if c["unsupported_flagged"]:
            out.append(f"  unsupported       : {c['unsupported_flagged']}")
        out.append("")
        out.append("[ ENTRY-LOGIC COVERAGE ]")
        for rule in e.entry_logic:
            mark = "✓" if rule in r["entry_seen"] else "✗"
            out.append(f"  {mark} {rule}")
        if e.exit_logic:
            out.append("[ EXIT-LOGIC USER REQUESTS — captured?]")
            for rule in e.exit_logic:
                # We don't formally check exit-logic here; just enumerate the user asks
                out.append(f"  • {rule}")
        out.append("")
        if r["gaps"]:
            out.append("[ GAPS — must fix ]")
            for g in r["gaps"]:
                out.append(f"  ✗ {g}")
            total_gaps += len(r["gaps"])
        else:
            out.append("[ GAPS ] none")
        out.append("")
    out.append("=" * 78)
    out.append(f"SUMMARY: {len(reports)} prompts, {total_gaps} gaps total")
    out.append("=" * 78)
    return "\n".join(out)


if __name__ == "__main__":
    reports = [validate_prompt(i + 1, p, e) for i, (p, e) in enumerate(PROMPTS)]
    print(render_report(reports))
