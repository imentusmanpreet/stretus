#!/usr/bin/env python3
"""
Test semantic extraction against ICT (Break of Structure / Fair Value Gap) prompt.
Verify session filters, structural stop-loss, and ATR padding.

Run: python3 test_ict_prompt.py
"""
import json
from app.planner.semantic_extractor import SemanticExtractor
from app.planner.strategy_family_preserver import StrategyFamilyPreserver


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def test_ict_prompt():
    """Test semantic extraction against the ICT strategy prompt."""

    # User's ICT prompt
    prompt = """please help me creating following strategy : Smart Money ICT Strategy
    —Best for Institutional Market Structure Trading Works extremely well on: Adani.
    Logic -Large market participants often create break of structure followed by
    fair value gap retracements.You enter after smart money confirmation.
    Setup Timeframe 15-minute chart Market Timing Trade during high liquidity market hours
    Entry Rules BUY Bullish break of structure should happen Price should retrace
    into bullish fair value gap Higher highs should continue forming Volume should remain strong
    SELL Exit when bearish structure appears Stop Loss Below recent swing low with ATR padding
    Target Risk:Reward = 1:3 minimum."""

    print_section("ICT BREAK OF STRUCTURE / FAIR VALUE GAP STRATEGY TEST")
    print("User Prompt:")
    print("-" * 80)
    print(prompt)
    print("-" * 80)

    # Initialize extractors
    extractor = SemanticExtractor()
    family_preserver = StrategyFamilyPreserver()

    # Extract semantic instructions
    print("\n🔄 Extracting semantic instructions from ICT prompt...")
    instructions = extractor.extract(prompt)

    # ── VERIFICATION 1: Strategy Family ────────────────────────────────────────
    print_section("VERIFICATION 1: STRATEGY FAMILY DETECTION")
    print(f"✓ Detected Family: {instructions.strategy_family}")

    expected_family = "ICT_BOS_FVG"
    if instructions.strategy_family == expected_family:
        print(f"  ✅ PASS: Correctly identified as {expected_family}")
    else:
        print(f"  ⚠️  FAMILY: Got {instructions.strategy_family} (expected {expected_family})")

    # Get family spec
    family_spec = family_preserver.get_family_spec(instructions.strategy_family)
    if family_spec:
        print(f"\n  Family Specification:")
        print(f"  - Canonical signals: {[s.signal_name for s in family_spec.canonical_signals]}")
        print(f"  - SL anchor type: {family_spec.sl_anchor_type}")
        print(f"  - Preserved invariants: {len(family_spec.preserved_invariants)}")

    # ── VERIFICATION 2: Structural Stop-Loss with ATR Padding ──────────────────
    print_section("VERIFICATION 2: STRUCTURAL STOP-LOSS WITH ATR PADDING")
    print(f"✓ Stop-Loss Extracted: {instructions.stop_loss is not None}")

    if instructions.stop_loss:
        sl = instructions.stop_loss
        print(f"\n  Stop-Loss Specification:")
        print(f"    - Type: {sl.type}")
        print(f"    - Anchor: {sl.anchor}")
        print(f"    - Padding Method: {sl.padding.method if sl.padding else 'None'}")
        print(f"    - ATR Multiple: {sl.padding.atr_multiple if sl.padding and sl.padding.atr_multiple else 'None'}")
        print(f"    - Description: {sl.description}")

        # Verify structural SL extraction
        if sl.type == "swing_low":
            print(f"\n  ✅ PASS: Structural SL type = 'swing_low' (NOT percentage)")
        else:
            print(f"\n  ⚠️  SL TYPE: Got {sl.type} (expected swing_low)")

        # Verify ATR padding extraction
        if sl.padding and sl.padding.method == "atr":
            print(f"  ✅ PASS: ATR padding detected (method={sl.padding.method})")
            if sl.padding.atr_multiple:
                print(f"           ATR Multiple: {sl.padding.atr_multiple}x")
        else:
            print(f"  ⚠️  ATR PADDING: Got {sl.padding.method if sl.padding else 'None'} (expected 'atr')")
    else:
        print("  ❌ FAIL: No stop-loss extracted")

    # ── VERIFICATION 3: Session Filters (High Liquidity) ───────────────────────
    print_section("VERIFICATION 3: SESSION & TIME-WINDOW FILTERS")
    print(f"✓ Session Filters: {instructions.session_filters is not None}")

    if instructions.session_filters:
        sf = instructions.session_filters
        print(f"\n  Session Filter Specification:")
        print(f"    - Enabled: {sf.enabled}")
        print(f"    - Session Type: {sf.session or 'Custom windows'}")
        print(f"    - Valid Windows: {len(sf.valid_windows)}")

        if sf.valid_windows:
            for i, window in enumerate(sf.valid_windows, 1):
                print(f"\n    Window {i}:")
                if window.start_time:
                    print(f"      - Start: {window.start_time}")
                if window.end_time:
                    print(f"      - End: {window.end_time}")
                if window.duration_minutes:
                    print(f"      - Duration: {window.duration_minutes} min")
                if window.from_open:
                    print(f"      - From Market Open")

        # Verify high liquidity extraction
        if sf.valid_windows or sf.session:
            print(f"\n  ✅ PASS: Session filtering detected")
            print(f"           (High liquidity market hours)")
        else:
            print(f"\n  ⚠️  SESSION FILTER: No windows found")
    else:
        print("  ⚠️  SESSION FILTER: No session filters extracted")

    # ── VERIFICATION 4: Risk:Reward ────────────────────────────────────────────
    print_section("VERIFICATION 4: RISK:REWARD ENFORCEMENT")
    print(f"✓ Risk:Reward Extracted: {instructions.risk_reward is not None}")

    if instructions.risk_reward:
        rr = instructions.risk_reward
        print(f"\n  Risk:Reward Specification:")
        print(f"    - Type: {rr.type}")
        print(f"    - Ratio: 1:{rr.ratio if rr.ratio else 'N/A'}")
        print(f"    - Description: {rr.description}")

        # Verify 1:3 ratio extraction
        if rr.ratio == 3.0:
            print(f"\n  ✅ PASS: Correctly extracted '1:3 minimum' as ratio=3.0")
        else:
            print(f"\n  ⚠️  RR RATIO: Got {rr.ratio} (expected 3.0 for 1:3)")
    else:
        print("  ⚠️  RISK:REWARD: Not extracted")

    # ── VERIFICATION 5: Multi-Timeframe Rules ──────────────────────────────────
    print_section("VERIFICATION 5: MULTI-TIMEFRAME CONDITIONS")
    print(f"✓ HTF Rules Detected: {len(instructions.htf_rules)}")

    for i, rule in enumerate(instructions.htf_rules, 1):
        print(f"\n  Rule {i}:")
        print(f"    - Timeframe: {rule.timeframe}")
        print(f"    - Condition: {rule.condition or 'Not parsed'}")
        print(f"    - Role: {rule.role}")
        print(f"    - Description: {rule.description[:60]}...")

    # ── VERIFICATION 6: Volume Confirmation ────────────────────────────────────
    print_section("VERIFICATION 6: VOLUME & MOMENTUM CONFIRMATION")
    print(f"✓ Volume/Momentum Filters: {instructions.volume_momentum is not None}")

    if instructions.volume_momentum:
        vm = instructions.volume_momentum
        if vm.volume:
            print(f"\n  Volume Filter:")
            print(f"    - Type: {vm.volume.filter_type}")
            print(f"    ✅ Volume confirmation detected")

        if vm.momentum:
            print(f"\n  Momentum Filter:")
            print(f"    - Type: {vm.momentum.filter_type}")
            print(f"    ✅ Momentum confirmation detected")
    else:
        print("  ⚠️  No volume/momentum filters extracted")

    # ── FINAL SUMMARY ──────────────────────────────────────────────────────────
    print_section("EXTRACTION SUMMARY - ICT STRATEGY")

    summary = {
        "strategy_family": instructions.strategy_family,
        "htf_rules_count": len(instructions.htf_rules),
        "structural_sl_present": instructions.stop_loss is not None,
        "sl_type": instructions.stop_loss.type if instructions.stop_loss else None,
        "sl_anchor": instructions.stop_loss.anchor if instructions.stop_loss else None,
        "atr_padding_method": (instructions.stop_loss.padding.method
                               if instructions.stop_loss and instructions.stop_loss.padding
                               else None),
        "atr_multiple": (instructions.stop_loss.padding.atr_multiple
                         if instructions.stop_loss and instructions.stop_loss.padding
                         else None),
        "risk_reward_ratio": instructions.risk_reward.ratio if instructions.risk_reward else None,
        "session_filters_enabled": (instructions.session_filters and instructions.session_filters.enabled),
        "session_windows_count": (len(instructions.session_filters.valid_windows)
                                  if instructions.session_filters else 0),
        "volume_filter_present": (instructions.volume_momentum and instructions.volume_momentum.volume is not None),
        "extraction_quality_score": instructions.extraction_quality_score,
    }

    print("JSON Summary:")
    print(json.dumps(summary, indent=2))

    # ── DETAILED VERIFICATION CHECKLIST ────────────────────────────────────────
    print_section("VERIFICATION CHECKLIST")

    checks = {
        "✅ Family: ICT_BOS_FVG": instructions.strategy_family == "ICT_BOS_FVG",
        "✅ Structural SL (swing_low)": instructions.stop_loss and instructions.stop_loss.type == "swing_low",
        "✅ ATR Padding detected": (instructions.stop_loss and instructions.stop_loss.padding
                                    and instructions.stop_loss.padding.method == "atr"),
        "✅ RR 1:3 extracted": instructions.risk_reward and instructions.risk_reward.ratio == 3.0,
        "✅ Session filter present": instructions.session_filters is not None,
        "✅ Volume confirmation": (instructions.volume_momentum and
                                   instructions.volume_momentum.volume is not None),
        "✅ Quality score >= 0.60": instructions.extraction_quality_score >= 0.60,
    }

    all_pass = True
    for check, result in checks.items():
        if result:
            print(f"  {check}")
        else:
            print(f"  ❌ {check.replace('✅', 'FAILED:')}")
            all_pass = False

    print("\n" + "=" * 80)
    if all_pass:
        print("🎉 ALL VERIFICATIONS PASSED - ICT Semantic extraction working perfectly!")
    else:
        print("⚠️  Some verifications failed - see above for details")
    print("=" * 80 + "\n")

    return all_pass


if __name__ == "__main__":
    success = test_ict_prompt()
    exit(0 if success else 1)
