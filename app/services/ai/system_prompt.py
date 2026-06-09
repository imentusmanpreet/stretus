"""
app/services/ai/system_prompt.py
──────────────────────────────────
Builds the system instruction stored on each chat session.

The scope (equity vs crypto vs both) is chosen from the chat's capabilities
when the session is created. See build_system_instruction().
"""

from app.services.agent.prompt import build_agentic_system_prompt


_FOOTER_TEMPLATE = """

USER-FACING OPENING RULE
- For a brand-new session, the welcome message should briefly introduce Stretus and ask which supported {asset_phrase} the user wants to start with.
- If the user simply greets during an active workflow, pause and ask what they want to focus on instead of forcing the workflow forward.

SUPPORTED USER INPUTS
- Collect only: {asset_word} name or symbol, timeframe, objective, sentiment, experience, and goal.
- Do not ask for daily loss cap, max trade duration, max trades, entry conditions, or exit conditions during input collection.
- Supported timeframes are exactly: 1m, 5m, 10m, 15m, 30m, 1h, 1d.

OUTPUT CONTRACT
- Backend actions must be represented as structured tool calls.
- Plain text may only be used for conversational or educational responses where no backend action is needed.
"""


def _footer_for(asset_classes: list[str] | None) -> str:
    enabled = set(asset_classes or [])
    has_equity = "equity_cash" in enabled
    has_crypto = "crypto_spot" in enabled

    # Empty / unconfigured → default to the multi-asset footer.
    if not enabled or (has_equity and has_crypto):
        return _FOOTER_TEMPLATE.format(
            asset_phrase="stock or crypto pair",
            asset_word="asset",
        )
    if has_crypto and not has_equity:
        return _FOOTER_TEMPLATE.format(
            asset_phrase="crypto pair",
            asset_word="crypto pair",
        )
    # equity_cash only
    return _FOOTER_TEMPLATE.format(
        asset_phrase="Indian stock",
        asset_word="stock",
    )


def build_system_instruction(asset_classes: list[str] | None = None) -> str:
    """Return the full system instruction for a chat with these enabled asset classes."""
    return build_agentic_system_prompt(asset_classes) + _footer_for(asset_classes)


# Back-compat: callers that still import SYSTEM_INSTRUCTION get the equity
# variant, matching pre-Step-4 behavior.
SYSTEM_INSTRUCTION = build_system_instruction(None)
