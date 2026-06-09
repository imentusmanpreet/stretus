"""
Add display `formula:` fields to signal YAMLs that currently lack one.

The engine evaluates these signals via the registry (Python), not by parsing
the formula. But the `entry_condition` / `exit_condition` strings persisted
in the strategy YAML need a readable representation, and the constraint
compiler's `_rebuild_conditions` skips signals whose render_formula returns
empty — which is why Supertrend etc. were dropping out of conditions.

Run once:

    PYTHONPATH=. python3 scripts/add_missing_formulas.py
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SIGNALS_DIR = REPO / "app" / "kb" / "signals"

DISPLAY_FORMULAS: dict[str, str] = {
    # ADX
    "adx_strong_trend":            "ADX({window}) > {threshold}",
    "adx_bullish_di":              "PLUS_DI({window}) > MINUS_DI({window})",
    "adx_bearish_di":              "MINUS_DI({window}) > PLUS_DI({window})",

    # ATR
    "atr_high_volatility":         "ATR({window}) > AVG(ATR({window}), 50)",
    "atr_low_volatility":          "ATR({window}) < AVG(ATR({window}), 50)",

    # Bollinger
    "bb_squeeze":                  "BB_WIDTH({window}, {window_dev}) < AVG(BB_WIDTH({window}, {window_dev}), 50)",

    # CCI
    "cci_overbought":              "CCI({window}) > {threshold}",
    "cci_oversold":                "CCI({window}) < -{threshold}",

    # CMF
    "chaikin_money_flow_positive": "CMF({window}) > 0",
    "chaikin_money_flow_negative": "CMF({window}) < 0",

    # Keltner
    "keltner_breakout_up":         "CLOSE > KC_UPPER({window})",
    "keltner_breakout_down":       "CLOSE < KC_LOWER({window})",

    # MFI
    "mfi_overbought":              "MFI({window}) > {threshold}",
    "mfi_oversold":                "MFI({window}) < {threshold}",

    # OBV
    "obv_rising":                  "OBV > PREV(OBV, 1)",
    "obv_falling":                 "OBV < PREV(OBV, 1)",

    # Stochastic
    "stoch_overbought":            "STOCH_K({window}) > {threshold}",
    "stoch_oversold":              "STOCH_K({window}) < {threshold}",

    # Supertrend
    "supertrend_bullish":          "CLOSE > SUPERTREND({window}, {multiplier})",
    "supertrend_bearish":          "CLOSE < SUPERTREND({window}, {multiplier})",
}


def main() -> None:
    added = 0
    skipped = 0
    for name, formula in DISPLAY_FORMULAS.items():
        path = SIGNALS_DIR / f"{name}.yaml"
        if not path.exists():
            print(f"  missing file: {name}")
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if data.get("formula"):
            skipped += 1
            continue
        # Insert formula early in the dict so it appears near top of the YAML
        ordered: dict = {}
        for k, v in data.items():
            ordered[k] = v
            if k == "kind":
                ordered["formula"] = formula
        # If "kind" wasn't present, just append
        if "formula" not in ordered:
            ordered["formula"] = formula
        path.write_text(
            yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True, width=1000),
            encoding="utf-8",
        )
        added += 1
    print(f"\nDone — {added} added, {skipped} already had formula")


if __name__ == "__main__":
    main()
