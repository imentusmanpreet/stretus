"""
scripts/test_four_user_prompts.py — End-to-end test of the 4 user-reported
chat prompts against the full extraction pipeline (regex + semantic + catalog
picker + auto-fill).

For each prompt we list the user's stated params and check whether each is
present (and correctly tagged source=user) in the resulting builder state.
"""
from __future__ import annotations

from app.kb.loader import KB
from app.planner.catalog_signal_picker import (
    auto_fill_missing_families,
    pick_plan_from_catalog,
)
from app.planner.semantic_extractor import SemanticExtractor
from app.services.strategy.builder import StrategyBuilder, extract_strategy_details


PROMPTS = [
    {
        "name": "Test 1 — SBIN / Supertrend + MACD",
        "text": (
            "Design an intermediate swing strategy for SBIN.NS using Supertrend "
            "and MACD confirmation. Use 1.5% stop loss, 3% take profit, 1:2 RR, "
            "risk 2% capital per trade, maximum 3 open trades, and 20% maximum "
            "capital allocation per trade."
        ),
        "expect_rms": {
            "stop_loss_pct":             {"value": 1.5,  "source": "user"},
            "take_profit_pct":           {"value": 3.0,  "source": "user"},
            "risk_reward":               {"value": 2.0,  "source": "user"},
            "per_trade_risk":            {"value": 2.0,  "source": "user"},
            "max_trades":                {"value": 3,    "source": "user"},
            "max_capital_allocation_pct": {"value": 20.0, "source": "user"},
        },
        "expect_builder_attr": {
            "experience": "intermediate",
            "objective":  "positional",   # "swing" → positional
            "stop_loss":  1.5,
            "take_profit": 3.0,
        },
    },
    {
        "name": "Test 2 — HDFCBANK / Basic RSI",
        "text": (
            "Build a basic RSI strategy for HDFCBANK.NS that only trades between "
            "9:30 AM to 11:30 AM, with strict 1:2.5 risk-reward and daily loss "
            "limit of 3%"
        ),
        "expect_rms": {
            "risk_reward":  {"value": 2.5,  "source": "user"},
            "daily_loss_cap": {"value": 3.0, "source": "user"},
        },
        "expect_builder_attr": {
            "experience":          "beginner",      # "basic" → beginner now
            "entry_window_start":  "09:30",         # set by semantic_extractor / agent
            "entry_window_end":    "11:30",
        },
    },
    {
        "name": "Test 3 — SBIN / Breakout + HTF",
        "text": (
            "Design an easy breakout strategy for SBIN.NS using higher "
            "timeframe confirmation, with cooldown of 3 bars after a loss "
            "and volume at least 1.5x average."
        ),
        "expect_rms": {
            "cooldown_bars_after_loss": {"value": 3,    "source": "user"},
        },
        "expect_builder_attr": {
            "experience": "beginner",                # "easy" → beginner
            "cooldown_bars_after_loss": 3,
            "volume_ratio_threshold":   1.5,
        },
        "expect_htf_rules": True,                    # "higher timeframe confirmation"
    },
    {
        "name": "Test 4 — HDFCBANK / MACD+RSI + ATR stops",
        "text": (
            "Develop a MACD + RSI combined strategy for HDFCBANK.NS with "
            "ATR-based stops, 2.5 risk-reward, max 5 trades, and 3-bar "
            "cooldown after losses."
        ),
        "expect_rms": {
            "risk_reward":              {"value": 2.5, "source": "user"},
            "max_trades":               {"value": 5,   "source": "user"},
            "cooldown_bars_after_loss": {"value": 3,   "source": "user"},
        },
        "expect_builder_attr": {
            "cooldown_bars_after_loss": 3,
        },
        "expect_atr_stop": True,
    },
]


def _check_rms(builder: StrategyBuilder, expect: dict) -> list[str]:
    msgs: list[str] = []
    rms = builder.risk_execution_config or {}
    sources = rms.get("rms_sources") or {}
    for fld, spec in expect.items():
        got = rms.get(fld)
        got_src = sources.get(fld)
        expected_val = spec["value"]
        expected_src = spec["source"]
        if got is None:
            msgs.append(f"  ✗ {fld}: MISSING (expected {expected_val} from {expected_src})")
            continue
        try:
            if abs(float(got) - float(expected_val)) > 0.01:
                msgs.append(f"  ✗ {fld}: got {got}, want {expected_val}")
                continue
        except (TypeError, ValueError):
            if got != expected_val:
                msgs.append(f"  ✗ {fld}: got {got!r}, want {expected_val!r}")
                continue
        if got_src != expected_src:
            msgs.append(f"  ✗ {fld} source: got {got_src!r}, want {expected_src!r}")
            continue
        msgs.append(f"  ✓ {fld} = {got} (source={got_src})")
    return msgs


def _check_attrs(builder: StrategyBuilder, expect: dict) -> list[str]:
    msgs: list[str] = []
    for attr, want in expect.items():
        got = getattr(builder, attr, None)
        if got != want:
            msgs.append(f"  ✗ builder.{attr}: got {got!r}, want {want!r}")
        else:
            msgs.append(f"  ✓ builder.{attr} = {got!r}")
    return msgs


def run() -> None:
    pass_count = 0
    total_count = 0

    for case in PROMPTS:
        print("=" * 78)
        print(case["name"])
        print("=" * 78)
        print(case["text"])
        print()
        builder = StrategyBuilder()
        extract_strategy_details(case["text"], builder)

        # Capture results
        all_ok = True
        for line in _check_rms(builder, case["expect_rms"]):
            print(line)
            total_count += 1
            if line.lstrip().startswith("✓"):
                pass_count += 1
            else:
                all_ok = False

        for line in _check_attrs(builder, case["expect_builder_attr"]):
            print(line)
            total_count += 1
            if line.lstrip().startswith("✓"):
                pass_count += 1
            else:
                all_ok = False

        if case.get("expect_htf_rules"):
            total_count += 1
            if builder.htf_rules:
                print(f"  ✓ htf_rules captured: {builder.htf_rules}")
                pass_count += 1
            else:
                print("  ✗ htf_rules: MISSING")
                all_ok = False

        if case.get("expect_atr_stop"):
            total_count += 1
            spec = builder.stop_loss_spec or {}
            if spec.get("type") == "atr":
                print(f"  ✓ stop_loss_spec = ATR-based: {spec}")
                pass_count += 1
            else:
                print(f"  ✗ stop_loss_spec: expected ATR-based, got {spec}")
                all_ok = False

        print()

    print("=" * 78)
    print(f"SUMMARY: {pass_count}/{total_count} checks pass")
    print("=" * 78)


if __name__ == "__main__":
    run()
