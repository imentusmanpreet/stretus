"""
app/services/universe/breadth.py — the market-context / breadth gate (fail-closed).

Computed ONCE per refresh over the Tier-A pool (cheap daily frames the resolver already
fetched for screening). When the gate passes, the refresh may admit members; when it
fails — or its input cannot be computed — it admits NONE (fail-closed, Invariant 5).

Phase A computes the two breadth measures derivable from the pool's OHLCV alone:
  * ``advance_decline`` — advancers / decliners over the last bar; pass when ≥ ``min_ratio``;
  * ``new_high_new_low`` — symbols at a 52-bar-window new high vs new low; pass when the
    high/low ratio ≥ ``min_ratio``.
``sector_strength`` and ``market_trend`` need reference/sector feeds not wired in Phase A,
so they fail closed with a clear reason rather than admitting on data we don't have.

Pure function: no I/O, fully testable with synthetic frames.
"""
from __future__ import annotations

import logging
import math

import pandas as pd

from app.strategy.spec import UniverseBreadthGate

from .metrics import safe_distance_52w, safe_pct_change

logger = logging.getLogger(__name__)

_NEW_HIGH_WINDOW = 252


def evaluate_breadth(
    gate: UniverseBreadthGate | None,
    frames: dict[str, pd.DataFrame],
) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for the breadth ``gate`` over the pool ``frames``.

    ``gate is None`` ⇒ ``(True, "no_gate")`` (no gate configured, admit normally).
    A gate whose measure cannot be computed from the available data ⇒ ``(False, ...)``.
    """
    if gate is None:
        return True, "no_gate"
    if not frames:
        return False, "breadth|no_pool_data"

    min_ratio = gate.min_ratio if gate.min_ratio is not None else 1.0

    if gate.metric == "advance_decline":
        advancers = decliners = 0
        for df in frames.values():
            chg = safe_pct_change(df, window=1)
            if math.isnan(chg):
                continue
            if chg > 0:
                advancers += 1
            elif chg < 0:
                decliners += 1
        if advancers + decliners == 0:
            return False, "breadth|advance_decline|no_valid_changes"
        ratio = advancers / max(decliners, 1)
        ok = ratio >= min_ratio
        reason = f"breadth|advance_decline|adv={advancers}|dec={decliners}|ratio={ratio:.2f}|min={min_ratio:.2f}"
        return ok, reason

    if gate.metric == "new_high_new_low":
        new_highs = new_lows = 0
        for df in frames.values():
            dist_high = safe_distance_52w(df, side="high", window=_NEW_HIGH_WINDOW)
            dist_low = safe_distance_52w(df, side="low", window=_NEW_HIGH_WINDOW)
            if not math.isnan(dist_high) and dist_high <= 0.0:
                new_highs += 1
            if not math.isnan(dist_low) and dist_low <= 0.0:
                new_lows += 1
        if new_highs + new_lows == 0:
            return False, "breadth|new_high_new_low|none_at_extreme"
        ratio = new_highs / max(new_lows, 1)
        ok = ratio >= min_ratio
        reason = f"breadth|new_high_new_low|nh={new_highs}|nl={new_lows}|ratio={ratio:.2f}|min={min_ratio:.2f}"
        return ok, reason

    # sector_strength / market_trend need feeds not wired in Phase A → fail-closed.
    return False, f"breadth|{gate.metric}|input_unavailable_phase_a"
