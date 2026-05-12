"""
engine/entry_sequence.py
─────────────────────────
Multi-step entry state machine.

A strategy's natural form is sometimes a sequence of conditions that must fire
*in order*, not a single boolean formula. For example, the 20-EMA pullback
play:

    Step 1  trend          CLOSE > EMA(20)          # confirm uptrend
    Step 2  pullback       LOW <= EMA(20)           # price returns to MA
    Step 3  reversal       IS_BULLISH_ENGULFING     # confirmation candle
    Step 4  volume         VOL_SPIKE(5, 1.5)        # participation
    => entry

A single formula combining all four with AND would only fire when every check
holds *on the same bar*, which never happens in real markets. The sequence
relaxes that: each step has a window of bars in which the next step can fire,
and only the FINAL step's bar produces the entry signal.

Design
──────
- Each step compiles to a CompiledCondition.
- `evaluate_entry_sequence(steps, df)` returns a pd.Series of bools — True at
  each bar where the entire sequence completed AT that bar (last step fired).
- Linear, single-track machine: at any time the runner is "waiting for step N
  to fire after step N-1 fired at bar K". If step N times out (more than
  `within_bars` bars elapsed since K) the machine resets and looks for step 0
  again.
- Strict no-look-ahead: every step's evaluation at bar i depends only on
  data 0..i.

Schema (YAML)
─────────────
    entry_sequence:
      - id: trend
        condition: "CLOSE > EMA(20)"
        within_bars: 50           # this step must fire within 50 bars of the
                                  # previous one (ignored on the first step)
      - id: pullback
        condition: "LOW <= EMA(20)"
        within_bars: 20
      - id: reversal
        condition: "IS_BULLISH_ENGULFING"
        within_bars: 3

When `entry_sequence` is present, the simulator uses the sequence's final
True bar as the entry trigger. The legacy `entry.condition`, if also set,
is AND-combined with the sequence: both must be true at the final bar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from engine.conditions import CompiledCondition, compile_condition


@dataclass(frozen=True)
class SequenceStep:
    id: str
    compiled: CompiledCondition
    within_bars: int          # 0 = no timeout


def parse_entry_sequence(raw: Any) -> tuple[SequenceStep, ...]:
    """Validate and compile each step. Returns () when the block is absent."""
    if raw in (None, "", [], {}):
        return ()
    if not isinstance(raw, list):
        raise ValueError("entry_sequence must be a list of step mappings.")

    steps: list[SequenceStep] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"entry_sequence[{idx}] must be a mapping.")
        step_id = str(item.get("id") or "").strip()
        if not step_id:
            raise ValueError(f"entry_sequence[{idx}] missing 'id'.")
        if step_id in seen_ids:
            raise ValueError(f"entry_sequence[{idx}] duplicate id={step_id!r}.")
        seen_ids.add(step_id)

        cond = str(item.get("condition") or "").strip()
        if not cond:
            raise ValueError(f"entry_sequence[{idx}] missing 'condition'.")
        compiled = compile_condition(cond)
        if compiled is None:
            raise ValueError(f"entry_sequence[{idx}] condition compiled to None.")

        within_bars = int(item.get("within_bars", 0) or 0)
        if within_bars < 0:
            raise ValueError(
                f"entry_sequence[{idx}] within_bars must be >= 0 (0 = no timeout)."
            )
        steps.append(SequenceStep(id=step_id, compiled=compiled, within_bars=within_bars))

    return tuple(steps)


def evaluate_entry_sequence(
    steps: tuple[SequenceStep, ...],
    df: pd.DataFrame,
) -> pd.Series:
    """Return a Series of bools aligned to df.index. True at bar i iff the
    entry sequence COMPLETED at bar i (i.e. the final step fired at bar i,
    every prior step fired in order within its `within_bars` window).

    Single-track: only one in-progress chain at a time. If a step times out,
    the machine resets and starts looking for step 0 again from the next bar.
    """
    n = len(df)
    result = pd.Series([False] * n, index=df.index, dtype=bool)
    if not steps or n == 0:
        return result

    # Track which step we're currently waiting on and when the most recent
    # step fired. step_index == 0 means "looking for the first step".
    step_index = 0
    last_fire_bar = -1            # bar where step_index-1 fired (or -1 for none)

    for i in range(n):
        # Timeout check: if a step is in progress and too many bars have
        # elapsed, reset before evaluating.
        if step_index > 0:
            current_step = steps[step_index]
            if current_step.within_bars > 0:
                elapsed = i - last_fire_bar
                if elapsed > current_step.within_bars:
                    step_index = 0
                    last_fire_bar = -1

        candidate = steps[step_index]
        fired = bool(candidate.compiled.evaluate(df, i))
        if not fired:
            continue

        # This step fired. Advance the machine.
        if step_index == len(steps) - 1:
            # Final step → sequence complete at this bar.
            result.iloc[i] = True
            # Reset so the next sequence can start from bar i+1.
            step_index = 0
            last_fire_bar = -1
        else:
            last_fire_bar = i
            step_index += 1

    return result
