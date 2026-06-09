"""
Trader persona responder — generates LLM-powered, markdown-formatted milestone
responses that sound like a seasoned algo trader, not a scripted chatbot.

Usage:
    from app.services.chat.persona_responder import compose_milestone_response

    assistant_text = await compose_milestone_response(
        event="signal_plan_ready",
        context={...},
        llm=llm_service,          # optional — falls back to template if None
    )

Events:
    signal_plan_ready     — signals selected, awaiting assembly confirmation
    strategy_assembled    — strategy built, awaiting backtest confirmation
    backtest_complete     — backtest finished (passed or failed)
    backtest_failed       — hard backtest error (infra / data)
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trader persona system prompt
# ---------------------------------------------------------------------------

TRADER_PERSONA_SYSTEM_PROMPT = """You are Stretus — an AI algo trading strategist with deep experience in Indian
equity markets. You communicate exactly like a seasoned professional trader who
has been coding and running strategies for 15+ years.

VOICE & TONE
- Direct. Confident. Zero filler. Cut "Sure!", "Absolutely!", "Of course!" entirely.
- Reference the actual data — stock, timeframe, signal names, numbers.
- Vary your openings every response. Never start the same way twice.
- Short is powerful. Long is only justified when unpacking results or flags.
- Use trader vocabulary naturally: "go long", "go short", "trail the stop",
  "cut losses", "R:R", "ATR", "breakout", "confluence". This is the job.

FORMAT — ALWAYS MARKDOWN
Every response MUST be valid markdown. The UI renders markdown so plain text
looks broken.

Response types and their format:

signal_plan_ready:
  ## Signal Plan — {Symbol} {Timeframe}
  **Bias:** {Bullish/Bearish} | **Type:** {Intraday/Positional}
  
  Brief (1-2 sentence) trader-level rationale for WHY these signals fit this
  setup. Reference the strategy goal. Be specific — don't say "I picked signals
  that match your goal", say what they actually do and why they fit.
  
  *Confirm to assemble — I'll wire in entry, exit, and risk.*

strategy_assembled:
  ## Strategy Built — {Symbol} {Timeframe}
  **Bias:** {Bullish/Bearish} | **Preset:** {name or "Custom build"}
  
  ### Entry
  - **{signal}** — {what it detects and why it matches the goal in one clause}
  - **{filter}** — {pairing logic}
  
  ### Exit
  - **{signal}** — {what it does}
  
  ### Risk
  | Parameter | Value | Source |
  |-----------|-------|--------|
  | SL | {val}% | {user/default} |
  | TP | {val}% | {user/default} |
  | R:R | 1:{val} | — |
  
  {Optional: ⚠️ **Flag:** only if there's a real concern — wide SL for this
  timeframe, low historical win-rate class, etc. Skip entirely if clean.}
  
  *Ready to backtest. Confirm or tell me what to adjust.*

backtest_complete (passed):
  ## Backtest: {Symbol} {Timeframe} — ✅ Passed
  
  | Metric | Value |
  |--------|-------|
  | Return | {pct}% |
  | Win Rate | {pct}% |
  | Trades | {n} |
  | Grade | {grade} |
  
  1-2 sentences of trader-level interpretation: what the win rate / return
  combination means in practice, one concrete observation (e.g. "Win rate at
  62% is solid for a trend-following setup — expect equity curve drawdowns
  between consecutive losers, that's normal"). Suggest one actionable next
  step if relevant.

backtest_complete (failed):
  ## Backtest: {Symbol} {Timeframe} — ❌ Did Not Pass
  
  | Metric | Value |
  |--------|-------|
  | Return | {pct}% |
  | Win Rate | {pct}% |
  | Trades | {n} |
  | Grade | {grade} |
  
  Brief diagnosis: what likely caused the failure (too few trades, low win
  rate, poor RR). One specific suggestion to improve.

backtest_failed (infra error):
  ## Backtest Failed
  
  > ⚠️ {brief reason}
  
  One sentence on what to check or try next. Don't pad.

RULES
- Never start with "I" — it sounds weak. Lead with the action or the data.
- Never say "I have reviewed the knowledge base" — that's internal plumbing.
- Never overpromise: "this strategy will profit", "guaranteed returns" are
  never in your vocabulary.
- Beginner users: replace signal names with plain-English labels in brackets,
  e.g. "RSI(14) [momentum oversold filter]". Keep the structure the same.
- Expert users: include signal scores/ranks if available, skip the plain-
  English labels, add any regime or confluence notes.
"""

# ---------------------------------------------------------------------------
# Fallback templates (used when LLM is unavailable)
# ---------------------------------------------------------------------------

def _fallback_signal_plan(ctx: dict[str, Any]) -> str:
    asset = ctx.get("asset", "this asset")
    timeframe = ctx.get("timeframe", "")
    sentiment = ctx.get("sentiment", "")
    header = f"## Signal Plan — {asset} {timeframe}".strip()
    bias_line = f"**Bias:** {sentiment.capitalize()}" if sentiment else ""
    return (
        f"{header}\n\n"
        + (f"{bias_line}\n\n" if bias_line else "")
        + "Signals selected from the knowledge base, ranked by fit for your setup.\n\n"
        "*Confirm to assemble the strategy — entry, exit, and risk parameters will be wired in.*"
    )


def _fallback_strategy_assembled(ctx: dict[str, Any]) -> str:
    asset = ctx.get("asset", "this asset")
    timeframe = ctx.get("timeframe", "")
    entry_names = ctx.get("entry_names", "")
    exit_names = ctx.get("exit_names", "")
    header = f"## Strategy Built — {asset} {timeframe}".strip()
    entry_row = f"| Entry | `{entry_names}` |" if entry_names else "| Entry | — |"
    exit_row = f"| Exit | `{exit_names}` |" if exit_names else "| Exit | — |"
    return (
        f"{header}\n\n"
        f"| Role | Signals |\n"
        f"|------|---------|\n"
        f"{entry_row}\n"
        f"{exit_row}\n\n"
        "Strategy is assembled and ready.\n\n"
        "*Confirm to run the backtest.*"
    )


def _fallback_backtest_complete(ctx: dict[str, Any]) -> str:
    asset = ctx.get("asset", "this asset")
    timeframe = ctx.get("timeframe", "")
    passed = bool(ctx.get("passed"))
    verdict = "✅ **Passed**" if passed else "❌ **Did not pass**"
    header = f"## Backtest: {asset} {timeframe} — {'✅ Passed' if passed else '❌ Did Not Pass'}".strip()
    rows = [
        "| Metric | Value |",
        "|--------|-------|",
    ]
    if ctx.get("total_return_pct") is not None:
        rows.append(f"| Return | {ctx['total_return_pct']:.2f}% |")
    if ctx.get("win_rate") is not None:
        rows.append(f"| Win Rate | {ctx['win_rate']:.2f}% |")
    if ctx.get("total_trades") is not None:
        rows.append(f"| Trades | {ctx['total_trades']} |")
    if ctx.get("overall_grade"):
        rows.append(f"| Grade | **{ctx['overall_grade']}** |")
    table = "\n".join(rows)
    suffix = ""
    if not passed and ctx.get("failure_reason"):
        suffix = f"\n\n> ⚠️ **Reason:** {ctx['failure_reason']}"
    return (
        f"{header}\n\n"
        f"**Verdict:** {verdict}\n\n"
        f"{table}{suffix}\n\n"
        "Review the results. Want to adjust parameters and re-run?"
    )


def _fallback_backtest_failed(ctx: dict[str, Any]) -> str:
    reason = ctx.get("reason", "An unexpected error occurred during backtesting.")
    return f"## Backtest Failed\n\n> ⚠️ {reason}\n\nReview the configuration and try again."


_FALLBACKS = {
    "signal_plan_ready": _fallback_signal_plan,
    "strategy_assembled": _fallback_strategy_assembled,
    "backtest_complete": _fallback_backtest_complete,
    "backtest_failed": _fallback_backtest_failed,
}

# ---------------------------------------------------------------------------
# Context → user prompt builders
# ---------------------------------------------------------------------------

def _build_user_prompt(event: str, ctx: dict[str, Any]) -> str:
    asset = ctx.get("asset", "")
    timeframe = ctx.get("timeframe", "")
    sentiment = ctx.get("sentiment", "")
    objective = ctx.get("objective", "")
    experience = ctx.get("experience", "intermediate")
    goal = ctx.get("goal", "")
    preset = ctx.get("preset", "")
    entry_names = ctx.get("entry_names", "")
    exit_names = ctx.get("exit_names", "")
    sl = ctx.get("sl_pct")
    tp = ctx.get("tp_pct")
    sl_source = ctx.get("sl_source", "default")
    tp_source = ctx.get("tp_source", "default")
    regime = ctx.get("regime", "")
    signals_used = ctx.get("signals_used", [])
    plan_summary = ctx.get("plan_summary", "")

    if event == "signal_plan_ready":
        signals_str = ", ".join(str(s) for s in signals_used) if signals_used else entry_names
        return (
            f"EVENT: signal_plan_ready\n"
            f"Stock: {asset} | Timeframe: {timeframe} | Bias: {sentiment} | Type: {objective}\n"
            f"Experience: {experience} | Goal: {goal}\n"
            f"Preset: {preset or 'custom'}\n"
            f"Signals selected: {signals_str}\n"
            f"Market regime: {regime}\n"
            f"Plan summary: {plan_summary}\n\n"
            "Generate the signal plan ready message. Reference the actual signals and goal. "
            "Keep it concise — 3-5 lines max. End with confirmation prompt."
        )

    if event == "strategy_assembled":
        rr = round(tp / sl, 2) if sl and tp and sl > 0 else "—"
        return (
            f"EVENT: strategy_assembled\n"
            f"Stock: {asset} | Timeframe: {timeframe} | Bias: {sentiment} | Type: {objective}\n"
            f"Experience: {experience} | Preset: {preset or 'custom'}\n"
            f"Entry signals: {entry_names}\n"
            f"Exit signals: {exit_names}\n"
            f"SL: {sl}% ({sl_source}) | TP: {tp}% ({tp_source}) | R:R: 1:{rr}\n"
            f"Market regime: {regime}\n"
            + (f"Validation flags: {ctx.get('validation_flags', '')}\n" if ctx.get("validation_flags") else "")
            + "\nGenerate the strategy assembled message. Use the exact signal names. "
            "Include the risk table. Flag only real concerns. End with backtest prompt."
        )

    if event == "backtest_complete":
        passed = bool(ctx.get("passed"))
        tr = ctx.get("total_return_pct")
        wr = ctx.get("win_rate")
        trades = ctx.get("total_trades")
        grade = ctx.get("overall_grade", "")
        reason = ctx.get("failure_reason", "")
        return (
            f"EVENT: backtest_complete\n"
            f"Stock: {asset} | Timeframe: {timeframe} | Bias: {sentiment}\n"
            f"Entry: {entry_names} | Exit: {exit_names}\n"
            f"Result: {'PASSED' if passed else 'FAILED'}\n"
            f"Return: {tr}% | Win Rate: {wr}% | Trades: {trades} | Grade: {grade}\n"
            + (f"Failure reason: {reason}\n" if reason else "")
            + f"Experience: {experience}\n\n"
            "Generate the backtest complete message. Interpret the numbers like a real trader — "
            "what do the win rate and return combination tell you? Give one concrete, actionable "
            "observation. Suggest one next step. Keep it to 2-3 sentences after the table."
        )

    if event == "backtest_failed":
        return (
            f"EVENT: backtest_failed\n"
            f"Stock: {asset} | Timeframe: {timeframe}\n"
            f"Error: {ctx.get('reason', 'Unknown error')}\n\n"
            "Generate a brief backtest failed message. Diagnose the likely cause in one sentence. "
            "Suggest one specific fix. No padding."
        )

    return f"EVENT: {event}\nContext: {ctx}\n\nGenerate an appropriate trader persona response."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def compose_milestone_response(
    event: str,
    context: dict[str, Any],
    llm: Any | None = None,
) -> str:
    """
    Generate a markdown-formatted, trader-persona response for a workflow milestone.

    Falls back to a structured markdown template if the LLM is unavailable or
    raises an exception. The fallback is always production-quality markdown —
    it just lacks the LLM's contextual variation.

    Args:
        event:   One of signal_plan_ready | strategy_assembled | backtest_complete
                 | backtest_failed
        context: Dict with strategy context (asset, timeframe, signals, metrics…)
        llm:     LLMService instance. If None, falls back to template.

    Returns:
        Markdown string ready to send to the UI.
    """
    fallback_fn = _FALLBACKS.get(event)
    if fallback_fn is None:
        logger.warning("⚠️ persona_responder|unknown_event=%s|using_empty_fallback", event)
        return context.get("fallback_text", "")

    if llm is None:
        return fallback_fn(context)

    try:
        user_prompt = _build_user_prompt(event, context)
        messages = [
            {"role": "system", "content": TRADER_PERSONA_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        response = await llm.chat(messages)
        if response and response.strip():
            return response.strip()
    except Exception as exc:
        logger.warning(
            "⚠️ persona_responder|llm_failed|event=%s|err=%s|using_fallback",
            event,
            str(exc)[:120],
        )

    return fallback_fn(context)
