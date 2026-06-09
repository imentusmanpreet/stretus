"""
Quick check: does the chat extraction pipeline capture EVERY param in a
dense 13-param user prompt?
"""
from __future__ import annotations

from app.kb.loader import KB
from app.planner.catalog_signal_picker import (
    auto_fill_missing_families,
    pick_plan_from_catalog,
)
from app.planner.semantic_extractor import SemanticExtractor
from app.services.strategy.builder import StrategyBuilder, extract_strategy_details


PROMPT = (
    "Create a clean intraday breakout strategy for HDFCBANK.NS on 5-minute "
    "candles. I only want trades when volume is strong and price is above "
    "VWAP. Use 1.5% stop loss with minimum 1:2.5 risk-reward, risk only 2% "
    "capital per trade, maximum 3 trades a day, and stop trading after 3 "
    "consecutive losses. Avoid trading on big gap-up or gap-down openings "
    "above 1%, and square off all positions before market close. Also add "
    "a trailing stop once trade reaches 2% profit."
)


EXPECTATIONS = {
    # symbol is set by the agent + stock matcher path, not the regex extractor
    "timeframe":  "5m",
    "objective":  "intraday",
    "stop_loss":  1.5,
    "risk_execution_config.stop_loss_pct":   1.5,
    "risk_execution_config.risk_reward":     2.5,
    "risk_execution_config.per_trade_risk":  2.0,
    "risk_execution_config.max_trades":      3,
    "max_consecutive_losses":  3,
    "gap_filter":  "ignore_both",
    "gap_threshold_pct":  1.0,
    "trailing_stop_spec.type":  "percent",
    "trailing_stop_spec.activate_after_pct":  2.0,
    # Square-off / time-exit
    "time_exit.exit_time":  "15:15",
}


def _get_path(obj, path: str):
    if "." not in path:
        return getattr(obj, path, None)
    head, rest = path.split(".", 1)
    cur = getattr(obj, head, None)
    if isinstance(cur, dict):
        for part in rest.split("."):
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                return None
        return cur
    return _get_path(cur, rest) if cur is not None else None


def run() -> None:
    print("PROMPT:")
    print(PROMPT)
    print()

    builder = StrategyBuilder()
    extract_strategy_details(PROMPT, builder)

    # Catalog picker
    kb = KB()
    sem = SemanticExtractor().extract(PROMPT)
    pick = pick_plan_from_catalog(
        PROMPT, kb=kb, timeframe="5m", sentiment="bullish",
        exit_on_opposite=getattr(sem, "exit_on_opposite", False),
    )
    plan = pick.signal_plan
    auto_fill_missing_families(plan, PROMPT, kb=kb, timeframe="5m", sentiment="bullish")

    print("===== EXTRACTED VALUES =====")
    pass_count = 0
    fail_lines: list[str] = []
    for path, expected in EXPECTATIONS.items():
        got = _get_path(builder, path)
        ok = False
        try:
            if isinstance(expected, (int, float)) and got is not None:
                ok = abs(float(got) - float(expected)) < 0.01
            else:
                ok = got == expected
        except (TypeError, ValueError):
            ok = got == expected
        mark = "✓" if ok else "✗"
        print(f"  {mark}  {path:50s} = {got!r:25s}  expected={expected!r}")
        if ok:
            pass_count += 1
        else:
            fail_lines.append(path)

    print()
    print("===== SIGNAL PLAN =====")
    for s in plan.get("entry", []):
        print(f"  entry: {s['name']:35s} ({s['signal_type']:6s})  params={s.get('params')}")
    for s in plan.get("exit", []):
        print(f"  exit : {s['name']:35s} ({s['signal_type']:6s})  params={s.get('params')}")
    if not plan.get("exit"):
        print("  exit : (none — only SL/TP / trailing stop)")

    print()
    print("===== RISK CONFIG SOURCES =====")
    sources = (builder.risk_execution_config or {}).get("rms_sources", {})
    for k, v in sources.items():
        print(f"  {k:30s} = {v}")

    print()
    print("===== HTF RULES =====")
    print(f"  {builder.htf_rules or '(none)'}")

    print()
    print(f"===== SUMMARY: {pass_count}/{len(EXPECTATIONS)} params captured =====")
    if fail_lines:
        print("Still missing / wrong:")
        for f in fail_lines:
            print(f"  - {f}")


if __name__ == "__main__":
    run()
