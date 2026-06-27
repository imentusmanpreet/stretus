"""
app/services/ai/strategy_prompt.py — the system prompt for the direct strategy path.

Built ENTIRELY from generated content — there are NO hand-written example strategies:
  * the condition grammar comes from app.strategy.vocab (derived live from the engine),
  * the output shape comes from StrategySpec's own JSON Schema,
  * the legal enums come from app.strategy.enums.

So the model is never told about an indicator the backtester can't run, and the prompt
can't drift from the engine. Output quality is enforced downstream by the strict
validator + the generator's repair loop, not by canned examples.
"""
from __future__ import annotations

import json
import logging

from app.strategy import vocab
from app.strategy.enums import CANONICAL_MARKETS, DIRECTIONS, OBJECTIVES, supported_timeframes
from app.strategy.spec import StrategySpec

logger = logging.getLogger(__name__)


_COMPLEXITY_RULES = """
COMPLEXITY BUDGET — scale condition complexity to the user's experience level.
Read the "experience" field from the confirmed parameters block.
Apply the budget to BOTH entry_condition AND exit_condition:

  experience=beginner    → entry_condition: at most 2 AND-joined clauses.
                           exit_condition:  at most 2 OR/AND-joined clauses.
                           Use simple comparisons (e.g. EMA(9) > EMA(21)).
                           NEVER add PREV() crossover detection unless the user
                           explicitly asked for it. NEVER add volume confirmation
                           unless the user mentioned volume. NEVER add VWAP to
                           exit unless the user mentioned it. Fewer clauses =
                           fewer missed trades and a more understandable strategy.
  experience=intermediate → entry: up to 3 AND-joined clauses.
                            exit:  up to 3 OR/AND-joined clauses.
                            PREV() crossover detection is fine.
  experience=expert       → no clause limit. Honor exactly what the user specified.
  (unset / not present)   → treat as intermediate.

RISK PROFILE SEMANTICS — "aggressive" means frequent entry, NOT more filters.
Read the user's goal / request for these keywords:

  goal contains "aggressive"   → LOOSEN entry thresholds (e.g. RSI > 45 not 50,
                                  volume > SMA(VOLUME,20) * 1.0 not 1.5).
                                  Use FEWER AND clauses — a simpler, more
                                  permissive condition lets you enter more often.
                                  Do NOT compensate by adding more confirmation
                                  layers. Aggressive = higher trade frequency,
                                  not higher conviction requirement.
  goal contains "conservative" → TIGHTEN thresholds (RSI > 55, volume multiplier
                                  1.5+). Add one extra confirmation clause to
                                  reduce false signals.
  goal contains "moderate" / unset → use standard thresholds.

When BOTH experience and goal are present, apply both rules together.
Example: beginner + aggressive → at most 2 clauses AND loosened thresholds.
""".strip()


_RULES = """
HARD RULES (a violation makes the strategy un-runnable and will be rejected):
1. entry_condition and (when present) exit/short conditions MUST use ONLY the
   functions, identifiers and operators listed in the grammar above. Never invent
   an indicator. If a user idea cannot be expressed with these primitives, compose
   the closest faithful approximation from them.
2. Conditions must be able to evaluate to TRUE on real data — no self-contradictions
   (e.g. requiring the same value to be both above a high threshold and below a low one).
3. Every strategy needs a stop. The user may express it on ANY basis and in ANY wording —
   your job is to detect the BASIS and map it to the matching `stop_loss.type`, preserving
   it exactly. NEVER convert one basis into another (e.g. an ATR or structural stop must not
   silently become a percentage). The engine supports exactly these stop bases — pick the one
   the user's intent fits, whatever words they used:
     • volatility / ATR basis      → type="atr"          (value = the ATR multiple; window = period, default 14)
     • chart-structure basis       → type="structure"    (anchor = the level, e.g. a swing/recent low)
     • fixed-percentage basis      → type="percent"      (value = the percent)
     • fixed price-distance basis  → type="fixed_points" (value = the point/price distance)
     • indicator-defined level     → type="indicator_based"
   Capture the exact number the user gave. The engine also needs a positive percent fallback,
   so derive a sensible `value` percent too; it uses the typed spec at runtime.
4. `take_profit` works the same way — detect the BASIS and preserve it on `take_profit.type`:
     • risk:reward / R-multiple basis → type="risk_reward" (value = the EXACT ratio the user named;
                                          the system computes percent = stop% × ratio)
     • fixed-percentage basis         → type="percent"      (value = the percent)
     • ATR / fixed-points / indicator → the matching type, mirroring rule 3
   Never substitute a default ratio when the user gave one.
   SINGLE target → use `take_profit.type` + `take_profit.value` (the existing scalar path).
   MULTIPLE targets (TP1/TP2/TP3, "exit in tranches", "scale out") → use `take_profit.targets`
   (a list of TakeProfitTarget rungs). Do NOT set `take_profit.value` when `targets` is non-empty —
   the ladder supersedes it. Each rung carries:
     • type          — "risk_reward" or "percent"
     • value         — the EXACT R-multiple or percent the user named
     • exit_fraction — fraction of the original position to exit at this rung (must sum ≤1.0;
                       use null to share the remainder equally across unset rungs)
     • risk_action   — optional SL mutation after this rung fires:
         "move_sl_to_breakeven"   — move SL to the entry price
         "move_sl_to_previous_tp" — move SL to the previous rung's price
         "none"                   — no SL change (default)
   Example (TP1=1R exit 30% move-SL-to-BE, TP2=2R exit 30%, TP3=4R exit 40%):
     "targets": [
       {"type":"risk_reward","value":1,"exit_fraction":0.30,"risk_action":{"type":"move_sl_to_breakeven"}},
       {"type":"risk_reward","value":2,"exit_fraction":0.30,"risk_action":{"type":"none"}},
       {"type":"risk_reward","value":4,"exit_fraction":0.40,"risk_action":{"type":"none"}}
     ]
   Leave `trailing_take_profit` unset when using a targets ladder (they are mutually exclusive).
   `take_profit` is OPTIONAL: OMIT it (set null) when a `trailing_take_profit` provides the profit
   exit instead (see rule 5). Every strategy still needs SOME profit exit — either a static
   `take_profit` (scalar or ladder) OR a `trailing_take_profit`.
5. TRAILING exits (optional) — when the user wants the stop OR the target to FOLLOW price
   as the trade moves in their favour (a "trailing" / "ratchet" / "let the winner run" idea),
   capture it on the matching block. Phase 1 supports a PERCENT give-back only (type="percent"):
     • trail the STOP — protect against a reversal ("trail my stop by 2%", "move SL up as it
       runs") → set `trailing_stop` = {type:"percent", distance_pct:<% behind the peak/trough>,
       activate_after_pct:<profit % before it engages>}.
     • trail the TARGET — capture upside ("trail my profit by 2%", "let it run and exit 1.5%
       off the peak", "lock gains, trail the take-profit once up 5%") → set
       `trailing_take_profit` = {type:"percent", distance_pct:<give-back %>,
       activate_after_pct:<profit % before it engages>}.
   distance_pct = how far behind the running peak (long) / trough (short) the exit trails;
   activate_after_pct = the profit the trade must reach before trailing engages (omit ⇒ immediate).
   `distance_pct` is REQUIRED for any percent trailing block — it IS the trail amount. NEVER emit a
   `trailing_stop`/`trailing_take_profit` with only `activate_after_pct`; without `distance_pct` the
   block is rejected. If the user gave no give-back distance, do not invent a trailing block at all.
   A strategy MAY trail the stop, the take-profit, or BOTH. They do NOT conflict — the engine runs
   both lines and exits on whichever a reversal reaches first. When the user asks for BOTH ("trail my
   stop by 1% after +3%, AND a trailing take-profit after +8% with a 2% give-back") set BOTH blocks —
   never drop one. Capture the user's exact percentages. Avoid a REDUNDANT pair, though: if one line
   is always at least as tight AND activates no later than the other, the other can never fire — only
   set both when their distances/activations differ meaningfully (a staged trail).
   WHICH BLOCK — decide by WHAT the user trails, NOT by whether it activates after a gain:
     • "stop" / "trail my stop" / "trailing stop" / "move my SL up" → `trailing_stop`, EVEN IF it
       only kicks in after a gain. "Once up 2%, trail my stop by 0.5%" → trailing_stop with
       activate_after_pct:2, distance_pct:0.5. An activation threshold does NOT make it a target.
     • "profit" / "target" / "take-profit" / "let the winner run" / "lock in gains" → `trailing_take_profit`.
   A `trailing_stop` is a COMPLETE exit on its own (it ratchets above entry and books the gain on a
   reversal). When the user asks only for a trailing stop and names no target, set `trailing_stop`,
   OMIT the static `take_profit` (null), and do NOT invent a target or reframe it as trailing_take_profit.
   CRITICAL — only when the user names a profit TARGET to trail ("small target THEN trail / convert
   to trailing after the first target / let it run"): the first target is the ACTIVATION, not a hard
   exit. Set `trailing_take_profit.activate_after_pct` = that first-target percent and OMIT the static
   `take_profit` (null). Do NOT set a static `take_profit` at (or below) the activation level — a
   static target there fires FIRST and the trade exits before trailing can ever run, defeating the
   request. Only add a static `take_profit` ALONGSIDE a trailing one if the user explicitly wants a
   hard safety CAP, and then it MUST sit well ABOVE `activate_after_pct`.
6. timeframe MUST be one of the supported timeframes; market should map to a known
   market id; objective and direction MUST be from their allowed lists.
   When the user gives NO explicit timeframe, default it to MATCH the holding style they imply:
   scalping / day-trading / intraday → 1m–15m; SWING (multi-day holds) → 1d (or 1h–4h only if they
   hint at a shorter swing); positional / investing / long-term → 1d–1w. A "swing" or "positional"
   strategy must NEVER default to 1m — that is scalping and contradicts the stated style.
7. direction="both" REQUIRES short_entry_condition (and ideally short_exit_condition).
   For a single-sided short strategy use direction="short_only" with the short logic
   in entry_condition/exit_condition.


FIDELITY — capture EVERY user input EXACTLY (this is the single most important rule):
- Preserve every explicit VALUE the user states, unchanged: any multiplier, comparison
  threshold, lookback/period, percentage, ratio, or price level they name MUST appear
  in the conditions/spec exactly as given. Do NOT drop, round, or weaken a user value —
  if they apply a multiplier to a quantity, that multiplier stays in the condition.
- Capture EVERY rule, filter, indicator and constraint the user states — entry, exit,
  stop, target, direction, risk, position sizing, gates. Nothing the user said may
  silently disappear from the spec.
- If a requested idea CANNOT be expressed with the grammar above, do NOT silently drop
  it. Build the closest faithful approximation from the primitives, set the affected
  field's "source" to "assumed" with a clear "reason", AND say plainly in intent_summary
  what you approximated and what could not be done.
- Echo the user's own values and wording back in intent_summary so they can verify
  nothing was lost or altered.


DECLARING INDICATORS (for transparency / read-back):
- List EVERY indicator your conditions use in "indicators", each with the EXACT
  parameters the user asked for, following the IndicatorSpec shape in the schema
  (name, params, purpose). Capture the user's numbers verbatim.
- NEVER silently drop a requested parameter. If you are unsure a parameter can be
  applied, still record it and add an "assumptions" entry so the user can see exactly
  what was honored.

CAPTURING INTENT (this is the whole point):
- Read the FULL conversation. Capture exactly what the user asked for; do not narrow
  it to a fixed menu — compose conditions freely from the primitives.
- For anything the user did NOT specify (timeframe, stop, target, direction, gates),
  choose a sensible default AND set that field's "source" to "assumed" with a short
  "reason". For values the user DID give, set "source" to "user".
- Put your plain-language understanding in "intent_summary".

OUTPUT:
- Return ONE JSON object that conforms to the StrategySpec schema. No prose, no
  markdown fences — JSON only.
""".strip()


_UNIVERSE_RULES = """
DYNAMIC UNIVERSE (instrument SELECTION RULE instead of one symbol):
When the user names a RULE for picking instruments rather than a single instrument —
e.g. "the 10 most active NIFTY-500 stocks above their VWAP", "top 10 crypto by volume",
"biggest gainers in banking", "strongest stocks near 52-week highs" — emit a `universe`
block and leave `symbol` null. A spec has EITHER `symbol` OR `universe`, never both.
When the user names ONE concrete instrument (e.g. "TCS", "BTCUSDT"), set `symbol` and
leave `universe` null exactly as before.

Build the `universe` block from the user's words (preserve every value verbatim):
  • source — where the candidate pool comes from:
      "NIFTY 500 / NIFTY 50 / an index"  → {kind:"index",   name:"<INDEX>"}
      "banking / IT / a sector"          → {kind:"sector",  name:"<sector>"}
      "F&O / futures-and-options names"  → {kind:"f_and_o"}
      "all crypto / any coin / top coins"→ {kind:"crypto_all"}
      an explicit list of symbols        → {kind:"watchlist", symbols:[...]}
  • screen — boolean filters in the SAME grammar as entry_condition (e.g. "above VWAP"
      → ["CLOSE > VWAP"]). A candidate must pass ALL. Empty when the user gave none.
  • rank — map the user's selection intent to `rank.by` (+ order):
      "most active / most traded"        → by:"rvol",         order:"desc"
      "biggest gainers / top movers up"  → by:"pct_change",   order:"desc"
      "biggest losers / down the most"   → by:"pct_change",   order:"asc"
      "strongest / relative strength"    → by:"rel_strength", order:"desc"
      "strongest momentum"               → by:"momentum",     order:"desc"
      "most oversold"                    → by:"rsi",          order:"asc"
      "most volatile"                    → by:"atr_pct",      order:"desc"
      "nearest 52-week highs"            → by:"distance_52w", order:"asc"
  • take — the N the user stated ("top 10" → 10); default 10 when unstated.
  • max_positions — the user's concurrent cap ("max 5 positions" → 5); default = take.
  • refresh — when membership is re-resolved: {cadence:"daily"|"weekly"|"intraday",
      at:"HH:MM"} ("every morning one hour after open" → cadence:"daily", at:"10:15"
      for NSE). Default cadence:"daily".
  • eligibility / breadth — set only if the user asked (e.g. a liquidity floor → min_adv).
Everything else (entry/exit/stop/target/gates) is authored EXACTLY as for a single
symbol — that body is run per selected member. Set each universe field's intent
faithfully; do not invent constraints the user did not state.
""".strip()


def build_strategy_system_prompt() -> str:
    """Assemble the full system prompt from live engine vocabulary + the spec schema."""
    grammar = vocab.grammar_summary_for_prompt()
    schema = json.dumps(StrategySpec.json_schema_for_llm(), indent=2)

    enums = (
        "ALLOWED ENUMS:\n"
        f"  timeframe: {', '.join(supported_timeframes())}\n"
        f"  objective: {', '.join(OBJECTIVES)}\n"
        f"  direction: {', '.join(DIRECTIONS)}\n"
        f"  market id: {', '.join(CANONICAL_MARKETS)}\n"
    )

    prompt = (
        "You are a quantitative trading strategist. Turn the user's intent into ONE "
        "strict StrategySpec JSON that a backtest engine can run directly.\n\n"
        f"{grammar}\n\n"
        f"{enums}\n"
        f"{_COMPLEXITY_RULES}\n\n"
        f"{_UNIVERSE_RULES}\n\n"
        f"{_RULES}\n\n"
        "STRATEGY SPEC JSON SCHEMA (your output MUST conform):\n"
        f"{schema}\n"
    )
    logger.debug("🧠 strategy prompt assembled | chars=%d", len(prompt))
    return prompt
