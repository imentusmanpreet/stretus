#!/bin/bash
# ════════════════════════════════════════════════════════════════
#  Stretus Chat System — Quick API Test Script
#  Run:  bash scripts/chat_system_test.sh
#  Make sure the FastAPI server is running on port 8000 first.
# ════════════════════════════════════════════════════════════════

BASE="http://localhost:8000/api/v1/strategy"
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Stretus Chat System API Test${NC}"
echo -e "${BLUE}══════════════════════════════════════════════════════${NC}\n"

# ── Step 1: Create a chat session ────────────────────────────────
echo -e "${YELLOW}Step 1: Creating chat session...${NC}"
CHAT_RESP=$(curl -s -X POST "$BASE/chats" \
  -H "Content-Type: application/json" \
  -d '{"title": "Test NIFTY Strategy"}')
echo "$CHAT_RESP" | python3 -m json.tool
CHAT_ID=$(echo "$CHAT_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['chat_id'])")
echo -e "${GREEN}✅ Chat ID: $CHAT_ID${NC}\n"

# ── Step 2: List all chats ───────────────────────────────────────
echo -e "${YELLOW}Step 2: Listing all chats...${NC}"
curl -s "$BASE/chats" | python3 -m json.tool
echo ""

# ── Step 3: Send message 1 – market ─────────────────────────────
echo -e "${YELLOW}Step 3: Sending message 1 (market)...${NC}"
curl -s -X POST "$BASE/chats/$CHAT_ID/messages" \
  -H "Content-Type: application/json" \
  -d '{"content": "I want to trade NIFTY on Indian indices market"}' \
  | python3 -m json.tool
echo ""

# ── Step 4: Send message 2 – timeframe ──────────────────────────
echo -e "${YELLOW}Step 4: Sending message 2 (timeframe)...${NC}"
curl -s -X POST "$BASE/chats/$CHAT_ID/messages" \
  -H "Content-Type: application/json" \
  -d '{"content": "Use 15m timeframe"}' \
  | python3 -m json.tool
echo ""

# ── Step 5: Send message 3 – entry condition ─────────────────────
echo -e "${YELLOW}Step 5: Sending message 3 (entry)...${NC}"
curl -s -X POST "$BASE/chats/$CHAT_ID/messages" \
  -H "Content-Type: application/json" \
  -d '{"content": "Entry condition: RSI(14) < 30 AND CLOSE > SMA(200)"}' \
  | python3 -m json.tool
echo ""

# ── Step 6: Send message 4 – exit condition ──────────────────────
echo -e "${YELLOW}Step 6: Sending message 4 (exit)...${NC}"
curl -s -X POST "$BASE/chats/$CHAT_ID/messages" \
  -H "Content-Type: application/json" \
  -d '{"content": "Exit when RSI(14) > 70"}' \
  | python3 -m json.tool
echo ""

# ── Step 7: Send message 5 – risk ───────────────────────────────
echo -e "${YELLOW}Step 7: Sending message 5 (risk)...${NC}"
MSG_RESP=$(curl -s -X POST "$BASE/chats/$CHAT_ID/messages" \
  -H "Content-Type: application/json" \
  -d '{"content": "Stop loss 1.5% and take profit 4%"}')
echo "$MSG_RESP" | python3 -m json.tool
IS_COMPLETE=$(echo "$MSG_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('is_complete', False))")
echo -e "${GREEN}Is Complete: $IS_COMPLETE${NC}\n"

# ── Step 8: Get chat history ─────────────────────────────────────
echo -e "${YELLOW}Step 8: Fetching chat history...${NC}"
curl -s "$BASE/chats/$CHAT_ID/messages" | python3 -m json.tool
echo ""

# ── Step 9: Confirm strategy (if complete) ────────────────────────
if [ "$IS_COMPLETE" = "True" ]; then
  echo -e "${YELLOW}Step 9: Confirming strategy...${NC}"
  curl -s -X POST "$BASE/strategies" \
    -H "Content-Type: application/json" \
    -d "{\"chat_id\": \"$CHAT_ID\"}" \
    | python3 -m json.tool
  echo ""
fi

echo -e "${BLUE}══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Test complete! Check http://localhost:8000/docs${NC}"
echo -e "${BLUE}══════════════════════════════════════════════════════${NC}"
