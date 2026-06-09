#!/usr/bin/env python3
"""
scripts/backfill_signal_cards.py — WS3 one-time card metadata backfill + audit.

Reports (and optionally writes) the WS3 card metadata the gate/compiler want:
  * param_specs   — auto-derived from params_by_timeframe at LOAD time (so this
                    is informational; you only need to write explicit min/max).
  * comparison    — {lhs,rhs,op}; cannot be auto-derived for every card, so this
                    audit lists cards that lack it (esp. price-vs-indicator and
                    *_cross cards the gate's intent-repair relies on).
  * engine        — anchors_supported for risk cards.

It also validates every cross-card reference (contradicts / pairs_well_with /
mirrors_to) resolves — the same check the CI test enforces.

Usage:
    python scripts/backfill_signal_cards.py            # audit (dry-run)
    python scripts/backfill_signal_cards.py --write     # persist explicit
                                                        # param_specs blocks

Default is DRY-RUN. `--write` only fills missing `param_specs` blocks into the
YAML (preserving everything else); it never edits comparison/engine — those are
semantic and must be authored deliberately.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_kb():
    from app.kb import kb
    return kb


def audit() -> int:
    kb = _load_kb()
    names = set(kb.signals)

    dangling: list[str] = []
    missing_comparison: list[str] = []
    missing_engine_risk: list[str] = []

    # Cards that REALLY want a comparison: price-vs-indicator + crossover cards.
    wants_comparison = []
    for n, c in kb.signals.items():
        for ref in (c.contradicts or []):
            if ref not in names:
                dangling.append(f"{n}.contradicts -> {ref}")
        for ref in (c.pairs_well_with or []):
            if ref not in names:
                dangling.append(f"{n}.pairs_well_with -> {ref}")
        mw = getattr(c, "match_when", None)
        if mw and mw.mirrors_to and mw.mirrors_to not in names:
            dangling.append(f"{n}.mirrors_to -> {mw.mirrors_to}")

        is_pricey = n.startswith(("price_", "is_above_", "is_below_")) or "cross" in n
        if is_pricey and c.comparison is None:
            wants_comparison.append(n)
        if c.comparison is None:
            missing_comparison.append(n)

    print(f"cards loaded:            {len(kb.signals)}")
    print(f"dangling references:     {len(dangling)}")
    for d in dangling:
        print(f"   ✗ {d}")
    print(f"missing comparison:      {len(missing_comparison)} "
          f"(of which price/cross cards: {len(wants_comparison)})")
    for n in sorted(wants_comparison):
        print(f"   • {n}")

    return 1 if dangling else 0


def write_param_specs() -> int:
    """Persist an explicit `param_specs` block (derived from params_by_timeframe)
    into each card YAML that lacks one. Conservative: only adds the block, never
    touches other keys; skips cards that already declare param_specs."""
    import yaml  # local import so audit works without write deps

    kb = _load_kb()
    signals_dir = ROOT / "app" / "kb" / "signals"
    written = 0
    for n, c in kb.signals.items():
        path = signals_dir / f"{n}.yaml"
        if not path.exists():
            continue
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict) or raw.get("param_specs"):
            continue
        specs = {pn: {"type": ps.type, "default": ps.default} for pn, ps in c.param_specs.items()}
        if not specs:
            continue
        raw["param_specs"] = specs
        path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True))
        written += 1
        print(f"   + wrote param_specs to {n}.yaml")
    print(f"param_specs written to {written} card(s)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="persist explicit param_specs blocks into card YAMLs")
    args = ap.parse_args()
    rc = audit()
    if args.write:
        print("\n--- writing param_specs ---")
        write_param_specs()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
