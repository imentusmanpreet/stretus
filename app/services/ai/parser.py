"""
app/services/ai/parser.py
──────────────────────────
Helpers for parsing structured blocks out of raw LLM responses.
"""
import json
import re
from typing import Optional

# ── Confirmation keywords — user must say one of these to set is_confirm=True ──
# Covers natural English and Hinglish phrases
_CONFIRM_PATTERNS = [
    # ── Explicit affirmations ─────────────────────────────────────────────────
    r'\byes\b',
    r'\byep\b',
    r'\byeah\b',
    r'\bconfirm\b',
    r'\bconfirmed\b',
    r'\bgo ahead\b',
    r'\bgo for it\b',
    r'\bproceed\b',
    r'\bapprove\b',
    r'\bapproved\b',
    r'\blooks good\b',
    r'\bsounds good\b',
    r'\bsounds right\b',
    r'\ball good\b',
    r'\bthis works\b',
    r'\bthat works\b',
    r'\bworks for me\b',
    r'\bfine by me\b',
    r'\bthat.?s (correct|right|good|fine|perfect)\b',
    r'\bcontinue\b',
    r'\bcarry on\b',
    r'\bmove ahead\b',
    r'\bok(ay)?\b',
    r'\bfine\b',
    r'\bperfect\b',
    r'\bsure\b',
    r'\babsolutely\b',
    r'\bdo it\b',
    r'\blet.?s (go|do it|create|build|start|proceed)\b',
    r'\bsounds (good|right|great|correct|perfect)\b',
    r'\bthat.?s (it|right|correct|good|great|fine)\b',
    r'\byes.{0,20}proceed\b',
    r'\byes.{0,20}(go|start|plan|build|create|confirm)\b',
    r'\bi\'?m\s+(?:happy|good|fine|ready|done|ok|okay|set)\b',
    r'\bi\s+am\s+(?:happy|good|fine|ready|done|ok|okay|set)\b',
    r'\b(?:all|everything)\s+(?:is\s+)?(?:fine|good|set|done|ok|okay|correct|right|perfect)\b',
    r'\bnothing\s+to\s+(?:change|update|modify|edit)\b',
    r'\bno\s+changes?\s+(?:required|needed|necessary)?\b',
    r'\bmove\s+(?:to\s+)?(?:next|ahead|forward|on)\b',
    r'\b(?:next|move\s+on)\s+(?:step|please)?\b',
    r'\bnext\s+step\b',
    r'\bcorrect\s*(?:,|\.|!)?\s*(?:proceed|continue|next|go)\b',
    r'\b(?:keep|leave)\s+(?:them|it|all|everything|current|same)\b',

    # ── Action-intent commands — user tells the system what to do next ────────
    r'\bplan (signal|signals|the signal|the signals)\b',
    r'\b(plan|create|generate|build|make|start).{0,10}signal\b',
    r'\b(plan|create|generate|build|make).{0,10}strateg(y|ies)\b',
    r'\bcreate (the )?strategy\b',
    r'\bgenerate (the )?strategy\b',
    r'\bmake (the )?strategy\b',
    r'\bassemble (the )?strategy\b',
    r'\bstart (the )?planning\b',
    r'\bstart (the )?backtest\b',
    r'\brun (it|backtest|the strategy|the backtest)\b',
    r'\brun backtest\b',
    r'\bbacktest (it|now|this|the strategy)\b',
    r'\btest (it|this|the strategy)\b',
    r'\bgenerate (the )?signals?\b',
    r'\bpick (the )?signals?\b',
    r'\bselect (the )?signals?\b',
    r'\bstart (the )?signal\b',
    r'\bgo with (this|these|it)\b',
    r'\buse (this|these|it)\b',

    # ── Hinglish confirmations ────────────────────────────────────────────────
    r'\b(haan|ha|hna|han)\b',
    r'\btheek hai\b',
    r'\baage badho\b',
    r'\baage badhte hain\b',
    r'\bkaro\b',
    r'\bchalo\b',
    r'\bkar do\b',
    r'\bbanao\b',
    r'\bbana do\b',
    r'\bshuru karo\b',
    r'\bshuru karte hain\b',
]

_CONFIRM_RE = re.compile(
    '|'.join(_CONFIRM_PATTERNS),
    re.IGNORECASE,
)

_NEGATIVE_ACTION_RE = re.compile(
    # Negation followed by an action verb (e.g. "don't run", "stop the backtest")
    r"\b(?:don'?t|do\s+not|stop|cancel|abort|not\s+now|wait|hold\s+on)\b"
    r".{0,80}\b(?:run|execute|proceed|continue|backtest|test|assemble|build|"
    r"create|plan|strategy|signals?)\b|"
    # Bare standalone refusals ("no", "nope", "nah") — must be the whole message,
    # not "no changes required" or "no thanks proceed".
    r"^\s*(?:no+|nope|nah|stop|cancel|abort|wait|hold\s+on|not\s+now)\s*[\.!,]?\s*$",
    re.IGNORECASE,
)

_STOCK_ADVICE_PATTERNS = [
    r"\b(?:recommend|suggest|advice|advise|tip|tips)\b.*\b(?:stock|stocks|share|shares|buy|invest|trade)\b",
    r"\b(?:which|what|best|koi|kaunsa|konsa)\b.*\b(?:stock|stocks|share|shares)\b.*\b(?:buy|invest|trade|pick|recommend|suggest)\b",
    r"\bshould\s+i\s+buy\b",
    r"\b(?:recommend|suggest)\s+(?:me\s+)?(?:a|some|any)?\s*(?:stock|stocks|share|shares)\b",
    r"\b(?:stock|stocks|share|shares)\s+(?:advice|suggestion|recommendation|tip|tips)\b",
]

_STOCK_ADVICE_RE = re.compile(
    "|".join(_STOCK_ADVICE_PATTERNS),
    re.IGNORECASE,
)

# Asking about the universe — informational, NOT an advice request.
# Examples: "what stocks do you support", "list the stocks", "which stocks are available",
# "show me the stock list", "give me the list of stocks".
_SUPPORTED_STOCKS_QUESTION_PATTERNS = [
    r"\b(?:what|which)\s+(?:stocks?|shares?|companies|tickers?)\s+(?:do\s+you\s+|are\s+|can\s+i\s+|"
    r"that\s+you\s+)?(?:support|cover|offer|allow|available|use)\b",
    r"\b(?:list|show|tell)\s+(?:me\s+)?(?:the\s+|all\s+|some\s+)?(?:stocks?|shares?|companies|tickers?|symbols?)\b",
    r"\b(?:supported|available|allowed)\s+(?:stocks?|shares?|companies|tickers?|symbols?|list)\b",
    r"\b(?:give|share|provide)\s+(?:me\s+)?(?:the\s+|a\s+)?(?:list\s+of\s+)?(?:stocks?|shares?|companies|tickers?|symbols?)\b",
    r"\b(?:stock|share|company|ticker|symbol)\s+(?:list|universe|options)\b",
    r"\bwhat\s+(?:stocks?|shares?|companies)\b",
    r"\b(?:list|universe)\s+of\s+(?:stocks?|shares?|companies|tickers?|symbols?)\b",
]

_SUPPORTED_STOCKS_QUESTION_RE = re.compile(
    "|".join(_SUPPORTED_STOCKS_QUESTION_PATTERNS),
    re.IGNORECASE,
)


def detect_supported_stocks_question(user_message: str) -> bool:
    """Return True when the user asks what stocks the assistant supports / lists / covers.

    This is NOT a buy/sell advice request and must override `detect_stock_advice_request`.
    """
    return bool(_SUPPORTED_STOCKS_QUESTION_RE.search((user_message or "").strip()))


def detect_user_confirmation(user_message: str) -> bool:
    """
    Returns True if the user's message contains an explicit confirmation.

    This is used to set is_confirm=True on the StrategyBuilder.
    Confirmation is REQUIRED (along with is_complete) before STRATEGY_PREVIEW
    is emitted and the YAML is generated.

    Examples that return True:
        "yes", "yes go ahead", "looks good confirm",
        "haan bhai karo", "theek hai", "ok create the strategy"

    Examples that return False:
        "no", "change the stop loss", "what is EMA?", "add RSI"
    """
    cleaned = (user_message or "").strip()
    if _NEGATIVE_ACTION_RE.search(cleaned):
        return False
    return bool(_CONFIRM_RE.search(cleaned))


def detect_stock_advice_request(user_message: str) -> bool:
    """Return True when the user asks for stock advice or recommendations.

    A question about the supported universe ("what stocks do you support",
    "list the stocks") is informational, not advice, and must NOT trigger
    the SEBI safety boundary message.
    """
    cleaned = (user_message or "").strip()
    if detect_supported_stocks_question(cleaned):
        return False
    return bool(_STOCK_ADVICE_RE.search(cleaned))


def parse_strategy_preview(ai_response: str) -> Optional[dict]:
    """Extract the <STRATEGY_PREVIEW>{...}</STRATEGY_PREVIEW> JSON block."""
    m = re.search(
        r'<STRATEGY_PREVIEW>\s*(\{.*?\})\s*</STRATEGY_PREVIEW>',
        ai_response,
        re.DOTALL,
    )
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None
