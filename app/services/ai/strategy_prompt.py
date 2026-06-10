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
   Never substitute a default ratio when the user gave one. The spec carries a SINGLE target, so
   if the user names several scaled targets (e.g. multiple R-multiples or partial exits), use the
   furthest as `take_profit` and record the full ladder in intent_summary so nothing is dropped.
5. timeframe MUST be one of the supported timeframes; market should map to a known
   market id; objective and direction MUST be from their allowed lists.
6. direction="both" REQUIRES short_entry_condition (and ideally short_exit_condition).
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
        f"{_RULES}\n\n"
        "STRATEGY SPEC JSON SCHEMA (your output MUST conform):\n"
        f"{schema}\n"
    )
    logger.debug("🧠 strategy prompt assembled | chars=%d", len(prompt))
    return prompt
