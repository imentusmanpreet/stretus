"""
End-to-end validation of all 15 strategy prompts against the live API.

Flow per test:
  1. Create session
  2. Send strategy prompt
  3. Wait for AI input-capture message
  4. Send "yes" to confirm inputs
  5. Wait for AI signal plan message
  6. Send "yes" to assemble
  7. Wait for assembled strategy (ready_for_backtest OR strategy assembled in content)
  8. Validate the strategy_draft against expected criteria
"""
from __future__ import annotations

import json, time, sys
from dataclasses import dataclass, field

import requests

BASE = "http://localhost:8000/api/v1/strategy"
POLL  = 3      # seconds between polls
WAIT  = 120    # max seconds per turn

GREEN, RED, YELLOW, CYAN, BOLD, RESET = (
    "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[1m", "\033[0m"
)

def ok(m):   print(f"  {GREEN}✓ {m}{RESET}", flush=True)
def fail(m): print(f"  {RED}✗ {m}{RESET}", flush=True)
def info(m): print(f"  {CYAN}→ {m}{RESET}", flush=True)


# ── API ───────────────────────────────────────────────────────────────────────

def new_session() -> str:
    r = requests.post(f"{BASE}/chats", json={})
    return r.json()["session_id"]

def send(sid: str, text: str):
    requests.post(f"{BASE}/chats/{sid}/messages", json={"content": text})

def messages(sid: str) -> list[dict]:
    r = requests.get(f"{BASE}/chats/{sid}/messages")
    d = r.json()
    return d if isinstance(d, list) else d.get("messages", [])

def assistant_msgs(sid: str) -> list[dict]:
    return [m for m in messages(sid) if m.get("role") == "assistant"]

def wait_for_response(sid: str, current_count: int) -> dict | None:
    """Block until there are more assistant messages than current_count."""
    deadline = time.time() + WAIT
    while time.time() < deadline:
        ams = assistant_msgs(sid)
        if len(ams) > current_count:
            return ams[-1]
        time.sleep(POLL)
    return None

def draft(msg: dict) -> dict:
    return msg.get("strategy_draft") or {} if msg else {}

def is_assembled(msg: dict) -> bool:
    d = draft(msg)
    status = (d.get("processing_status") or "").lower()
    content = (msg.get("content") or "").lower()
    return (
        "ready_for_backtest" in status
        or "assembled" in status
        or ("assembled" in content and "ready" in content)
        or ("assembled" in content and "backtest" in content)
        or ("strategy" in content and "ready for" in content)
    )

def is_awaiting_confirm(msg: dict) -> bool:
    d = draft(msg)
    status = (d.get("processing_status") or "").lower()
    content = (msg.get("content") or "").lower()
    return (
        "awaiting_confirmation" in status
        or "confirm" in content
    )

def run_prompt(prompt: str) -> tuple[dict, str]:
    """Run the full chat flow for a single prompt. Returns (final_draft, log)."""
    log_lines = []
    sid = new_session()
    log_lines.append(f"session={sid}")

    # Step 1: send prompt
    send(sid, prompt)
    count = len(assistant_msgs(sid))
    resp = wait_for_response(sid, count)
    if resp is None:
        return {}, "TIMEOUT on initial response"

    content = (resp.get("content") or "")[:200]
    log_lines.append(f"AI1: {content}")

    # Possibly AI asked a clarifying question → confirm
    if is_awaiting_confirm(resp):
        count = len(assistant_msgs(sid))
        send(sid, "yes, proceed")
        resp = wait_for_response(sid, count)
        if resp is None:
            return draft(resp or {}), "TIMEOUT after first confirm"
        content = (resp.get("content") or "")[:200]
        log_lines.append(f"AI2: {content}")

    # If already assembled, done
    if is_assembled(resp):
        return draft(resp), "\n".join(log_lines)

    # Step 2: if we see the signal plan / another confirmation → confirm again to assemble
    if is_awaiting_confirm(resp):
        count = len(assistant_msgs(sid))
        send(sid, "yes, assemble it")
        resp = wait_for_response(sid, count)
        if resp is None:
            return draft(resp or {}), "TIMEOUT after assemble confirm"
        content = (resp.get("content") or "")[:200]
        log_lines.append(f"AI3: {content}")

    # If still not assembled, try one more confirm
    if not is_assembled(resp):
        count = len(assistant_msgs(sid))
        send(sid, "yes")
        resp2 = wait_for_response(sid, count)
        if resp2:
            resp = resp2
        log_lines.append(f"AI4: {(resp.get('content') or '')[:200]}")

    return draft(resp), "\n".join(log_lines)


# ── Signal/draft helpers ──────────────────────────────────────────────────────

def sig_names(d: dict) -> list[str]:
    sp = d.get("signal_plan") or {}
    return list({
        s["name"] for leg in ("entry", "exit")
        for s in (sp.get(leg) or []) if s.get("name")
    })

def has_sig(d: dict, frag: str) -> bool:
    return any(frag.lower() in n.lower() for n in sig_names(d))

def sl(d: dict) -> dict:
    return d.get("stop_loss_spec") or {}

def ts(d: dict) -> dict:
    return d.get("trailing_stop_spec") or {}

def rr(d: dict) -> float | None:
    intent = d.get("semantic_intent") or {}
    rm = intent.get("risk_model") or {}
    r = (rm.get("risk_reward") or {}).get("ratio")
    if r is not None:
        return float(r)
    cfg = d.get("risk_execution_config") or {}
    v = cfg.get("risk_reward_ratio")
    return float(v) if v is not None else None

def htf(d: dict) -> list:
    return ((d.get("semantic_intent") or {}).get("htf_confluence") or [])

def ref_sym(d: dict) -> str:
    return (d.get("reference_symbol") or "").upper()

def chk(label: str, ok_: bool, detail: str = "") -> bool:
    suf = f"  [{detail}]" if detail else ""
    if ok_:
        print(f"  {GREEN}✓{RESET} {label}{suf}", flush=True)
    else:
        print(f"  {RED}✗{RESET} {label}{suf}", flush=True)
    return ok_


# ── Test cases ────────────────────────────────────────────────────────────────

@dataclass
class TC:
    name: str
    prompt: str
    sym: str
    tf: str
    sentiment: str = "bullish"
    objective: str = "intraday"
    experience: str = "beginner"
    signals: list[str] = field(default_factory=list)
    preset: str = ""
    sl_structural: bool = False
    sl_anchor: str = ""
    trailing: bool = False
    rr: float | None = None
    htf: bool = False
    ref: str = ""

TESTS: list[TC] = [
    TC("B1 SMA Crossover TCS 15m",
       "Create a beginner-friendly swing trading strategy for TCS using 15-minute candles. "
       "Buy when price crosses above the 20 SMA and sell when price closes below the 20 SMA. "
       "Add stop loss below previous candle low and target 2:1 reward risk ratio.",
       sym="TCS", tf="15m", sentiment="bullish", objective="positional", experience="beginner",
       signals=["ema", "sma"], rr=2.0,
       ),

    TC("B2 RSI Oversold RELIANCE 5m",
       "Build a simple intraday strategy for RELIANCE on 5-minute timeframe using RSI. "
       "Enter long when RSI crosses above 30 from oversold zone and candle closes bullish. "
       "Exit when RSI reaches above 65 or stop loss is hit.",
       sym="RELIANCE", tf="5m", sentiment="bullish", objective="intraday", experience="beginner",
       signals=["rsi"],
       ),

    TC("B3 VWAP Trailing HDFCBANK 5m",
       "Design a beginner intraday strategy for HDFCBANK using VWAP on 5-minute timeframe. "
       "Buy only when price stays above VWAP with bullish candles and volume confirmation. "
       "Use trailing stop loss after 1% profit.",
       sym="HDFC", tf="5m", sentiment="bullish", objective="intraday", experience="beginner",
       signals=["vwap", "volume"], trailing=True,
       ),

    TC("B4 PDH Breakout ITC 15m",
       "Create a breakout trading strategy for ITC on 15-minute chart. "
       "Enter buy trade when price breaks previous day high with strong bullish candle "
       "and higher than average volume. Keep stop loss below breakout candle low.",
       sym="ITC", tf="15m", sentiment="bullish", objective="intraday", experience="beginner",
       signals=["volume", "breakout"], sl_structural=True,
       ),

    TC("B5 EMA 9/21 Crossover INFY 5m",
       "Generate a trend-following strategy for INFY using 9 EMA and 21 EMA on 5-minute timeframe. "
       "Buy when 9 EMA crosses above 21 EMA and avoid trades during sideways market conditions.",
       sym="INFY", tf="5m", sentiment="bullish", objective="intraday", experience="beginner",
       signals=["ema"],
       ),

    TC("I1 RSI+MACD+VWAP ADANIPORTS 5m",
       "Build an intraday momentum strategy for ADANIPORTS on 5-minute timeframe. "
       "Enter only when RSI is above 60, MACD histogram turns positive, and price closes above VWAP. "
       "Use ATR-based stop loss and partial profit booking at 1.5R.",
       sym="ADANI", tf="5m", sentiment="bullish", objective="intraday", experience="intermediate",
       signals=["rsi", "macd", "vwap"],
       ),

    TC("I2 Breakout Retest BHARTIARTL 15m",
       "Create a breakout retest strategy for BHARTIARTL using 15-minute candles. "
       "Detect strong resistance breakout with volume confirmation, wait for retest of breakout zone, "
       "then enter on bullish engulfing confirmation candle. Use structure-based stop loss.",
       sym="BHARTI", tf="15m", sentiment="bullish", objective="intraday", experience="intermediate",
       signals=["volume", "breakout"], sl_structural=True,
       ),

    TC("I3 ORB SBIN 5m",
       "Design an Opening Range Breakout strategy for SBIN using first 15 minutes high and low "
       "on 5-minute timeframe. Enter only if breakout candle volume is 1.5x average volume "
       "and EMA slope confirms trend direction.",
       sym="SBIN", tf="5m", sentiment="bullish", objective="intraday", experience="intermediate",
       signals=["opening_range", "volume"], preset="orb",
       ),

    TC("I4 1h HTF EMA Pullback LT 15m",
       "Generate a pullback continuation strategy for LT on 15-minute chart. "
       "Identify higher timeframe uptrend using 1-hour EMA trend, then enter on pullback "
       "to 20 EMA with bullish rejection wick confirmation.",
       sym="LT", tf="15m", sentiment="bullish", objective="intraday", experience="intermediate",
       signals=["ema"], htf=True,
       ),

    TC("I5 BankNifty Gating ICICIBANK 5m",
       "Build a sector momentum strategy for ICICIBANK where trades are allowed only if "
       "Bank Nifty sector sentiment is bullish. Use VWAP, RSI, and volume breakout "
       "confirmation for entries on 5-minute timeframe.",
       sym="ICICI", tf="5m", sentiment="bullish", objective="intraday", experience="intermediate",
       signals=["vwap", "rsi", "volume"], ref="BANKNIFTY",
       ),

    TC("E1 Opening Drive GMRAIRPORT 5m",
       "Design an institutional-grade opening drive strategy for GMRAIRPORT on 5-minute timeframe. "
       "Detect aggressive opening auction imbalance and trade only when first expansion candle closes "
       "above opening range high with stacked bullish order flow characteristics, strong relative volume "
       "expansion, and rising EMA slope. Use opening range low as structural stop loss and activate "
       "percentage trailing stop after 1R profit.",
       sym="GMRAIR", tf="5m", sentiment="bullish", objective="intraday", experience="expert",
       signals=["opening_range", "volume"], preset="orb",
       sl_structural=True, sl_anchor="opening_range_low", trailing=True,
       ),

    TC("E2 Liquidity Sweep SUZLON 5m",
       "Build a smart-money liquidity sweep reversal strategy for SUZLON on 5-minute timeframe. "
       "Detect sell-side liquidity grab below previous swing low followed by immediate bullish "
       "displacement candle with volume spike. Enter on reclaim of liquidity zone and use "
       "dynamic ATR trailing stop.",
       sym="SUZLON", tf="5m", sentiment="bullish", objective="intraday", experience="expert",
       signals=["volume"], trailing=True,
       ),

    TC("E3 MTF RELIANCE 5m",
       "Create a professional multi-timeframe trend alignment strategy for RELIANCE combining "
       "1-hour market structure, 15-minute momentum confirmation, and 5-minute execution entries. "
       "Only take longs during higher timeframe bullish structure and enter on pullback absorption "
       "near VWAP or 20 EMA.",
       sym="RELIANCE", tf="5m", sentiment="bullish", objective="intraday", experience="expert",
       signals=["vwap", "ema"], htf=True,
       ),

    TC("E4 Volatility Squeeze TATASTEEL 15m",
       "Generate a volatility compression breakout strategy for TATASTEEL using Bollinger Band squeeze "
       "and ATR contraction on 15-minute timeframe. Trigger entry only when compression resolves "
       "with strong directional candle, expanding volume, and ADX rising above 25.",
       sym="TATA", tf="15m", sentiment="bullish", objective="intraday", experience="expert",
       signals=["volume", "adx"],
       ),

    TC("E5 RS Rotation HDFCBANK 5m",
       "Design an advanced relative strength rotation strategy for HDFCBANK comparing its intraday "
       "relative performance against Bank Nifty and ICICIBANK. Enter only when relative strength ratio "
       "improves alongside breakout structure, institutional volume participation, and bullish VWAP "
       "positioning. Use staged exits and trailing structure lows for risk management.",
       sym="HDFC", tf="5m", sentiment="bullish", objective="intraday", experience="expert",
       signals=["vwap", "volume", "rs"], trailing=True, ref="BANKNIFTY",
       ),
]


def validate(tc: TC, d: dict) -> int:
    fails = 0
    if not d:
        fail("No strategy draft returned")
        return 1

    symbol = (d.get("symbol") or "").upper()
    timeframe = (d.get("timeframe") or "").lower()
    sentiment_val = (d.get("sentiment") or "").lower()
    objective_val = (d.get("objective") or "").lower()
    experience_val = (d.get("experience") or "").lower()

    if not chk("Symbol", tc.sym.upper() in symbol, f"expected '{tc.sym}', got '{symbol}'"): fails+=1
    if not chk("Timeframe", timeframe == tc.tf, f"expected '{tc.tf}', got '{timeframe}'"): fails+=1
    if not chk("Sentiment", sentiment_val == tc.sentiment, f"got '{sentiment_val}'"): fails+=1

    obj_ok = (
        (tc.objective == "intraday" and "intraday" in objective_val) or
        (tc.objective == "positional" and ("positional" in objective_val or "swing" in objective_val)) or
        (tc.objective == "swing" and ("positional" in objective_val or "swing" in objective_val))
    )
    if not chk("Objective", obj_ok, f"expected '{tc.objective}', got '{objective_val}'"): fails+=1

    exp_ok = experience_val == tc.experience or experience_val == "default"
    if not chk("Experience", exp_ok, f"expected '{tc.experience}', got '{experience_val}'"): fails+=1

    sigs = sig_names(d)
    for frag in tc.signals:
        if not chk(f"Signal '{frag}'", has_sig(d, frag), f"signals={sigs}"): fails+=1

    if tc.preset:
        p = (d.get("strategy_preset") or "").lower()
        fw = ((d.get("semantic_intent") or {}).get("base_framework") or "").lower()
        if not chk(f"Preset/framework '{tc.preset}'",
                   tc.preset in p or tc.preset in fw, f"preset='{p}' fw='{fw}'"): fails+=1

    if tc.sl_structural:
        sltype = (sl(d).get("type") or "").lower()
        if not chk("SL structural", sltype == "structural", f"sl_type='{sltype}'"): fails+=1

    if tc.sl_anchor:
        anchor = (sl(d).get("anchor") or "").lower()
        if not chk(f"SL anchor '{tc.sl_anchor}'",
                   tc.sl_anchor in anchor, f"anchor='{anchor}'"): fails+=1

    if tc.trailing:
        ts_val = ts(d)
        if not chk("Trailing stop present", bool(ts_val and ts_val.get("type")),
                   f"trailing={ts_val}"): fails+=1

    if tc.rr is not None:
        rr_val = rr(d)
        if not chk(f"RR {tc.rr}", rr_val is not None and abs(rr_val - tc.rr) < 0.25,
                   f"got rr={rr_val}"): fails+=1

    if tc.htf:
        rules = htf(d)
        if not chk("HTF rules present", len(rules) > 0, f"htf={rules}"): fails+=1

    if tc.ref:
        # Check in reference_symbol, filters.benchmark, or htf condition text
        ref_v = ref_sym(d)
        filters = ((d.get("semantic_intent") or {}).get("filters") or [])
        benchmarks = [str(f.get("benchmark", "")).upper() for f in filters]
        htf_text = json.dumps(htf(d)).upper()
        found = (
            tc.ref.upper() in ref_v or
            any(tc.ref.upper() in b for b in benchmarks) or
            tc.ref.upper() in htf_text
        )
        if not chk(f"Reference '{tc.ref}' captured", found,
                   f"ref='{ref_v}' benchmarks={benchmarks}"): fails+=1

    return fails


def main():
    results: list[tuple[str, bool, dict]] = []

    for tc in TESTS:
        print(f"\n{BOLD}{CYAN}{'─'*65}{RESET}", flush=True)
        print(f"{BOLD}▶ {tc.name}{RESET}", flush=True)
        print(f"  Prompt: {tc.prompt[:110]}...", flush=True)

        d, log = run_prompt(tc.prompt)

        print(f"  {YELLOW}Flow:{RESET}", flush=True)
        for line in log.split("\n"):
            print(f"    {line}", flush=True)

        print(f"  {YELLOW}Draft snapshot:{RESET}", flush=True)
        if d:
            print(f"    symbol={d.get('symbol')} tf={d.get('timeframe')} "
                  f"preset={d.get('strategy_preset')} "
                  f"fw={((d.get('semantic_intent') or {}).get('base_framework'))}", flush=True)
            print(f"    signals={sig_names(d)}", flush=True)
            print(f"    sl={sl(d)} trailing={ts(d)}", flush=True)
            print(f"    rr={rr(d)} htf_rules={len(htf(d))} ref={ref_sym(d)}", flush=True)
        else:
            print("    (empty)", flush=True)

        print(f"  {YELLOW}Checks:{RESET}", flush=True)
        fails = validate(tc, d)
        passed = fails == 0
        results.append((tc.name, passed, d))

        verdict = f"{GREEN}{BOLD}PASS{RESET}" if passed else f"{RED}{BOLD}FAIL ({fails}){RESET}"
        print(f"  {verdict}", flush=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*65}{RESET}", flush=True)
    print(f"{BOLD}FINAL RESULTS{RESET}", flush=True)
    print(f"{'='*65}", flush=True)
    ok_count = sum(1 for _, p, _ in results if p)
    for name, p, _ in results:
        c = GREEN if p else RED
        print(f"  {c}{'✓' if p else '✗'}{RESET} {name}", flush=True)
    print(f"\n{BOLD}{ok_count}/{len(results)} passed{RESET}", flush=True)

    with open("/tmp/strategy_test_results.json", "w") as f:
        json.dump({name: d for name, _, d in results}, f, indent=2, default=str)
    print("\nFull drafts → /tmp/strategy_test_results.json", flush=True)

    return 0 if ok_count == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())
