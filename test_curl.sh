#!/bin/bash
# ============================================================
# Research-OS API - Quick cURL Smoke Tests
# ============================================================

API_URL="${API_URL:-http://localhost:8000}"
API_KEY="${RESEARCH_OS_API_KEY:-your-secret-key-here}"

echo "=========================================="
echo "  Research-OS cURL Tests"
echo "  Target: $API_URL"
echo "=========================================="

# Test 1: Root
echo ""
echo "[1] GET /"
curl -s "$API_URL/" | python3 -m json.tool

# Test 2: Health
echo ""
echo "[2] GET /health"
curl -s "$API_URL/health" | python3 -m json.tool

# Test 3: No API key
echo ""
echo "[3] POST /v1/chat (no key - expect 401/422)"
curl -s -X POST "$API_URL/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}' | python3 -m json.tool

# Test 4: Wrong API key
echo ""
echo "[4] POST /v1/chat (wrong key - expect 403)"
curl -s -X POST "$API_URL/v1/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: wrong-key" \
  -d '{"query": "test"}' | python3 -m json.tool

# Test 5: Streaming chat
echo ""
echo "[5] POST /v1/chat (streaming)"
echo "    Streaming tokens below:"
echo "    ========================"
curl -s -N -X POST "$API_URL/v1/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"query": "What is the softmax function? Keep it brief."}'

# Test 6: Chat with history
echo ""
echo ""
echo "[6] POST /v1/chat (with history)"
curl -s -N -X POST "$API_URL/v1/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"query":"Explain that more simply","history":[{"role":"user","content":"What is softmax?"},{"role":"assistant","content":"Softmax converts logits to probabilities."}]}'

# Test 7: Ingest URL
echo ""
echo ""
echo "[7] POST /v1/ingest/url"
curl -s -X POST "$API_URL/v1/ingest/url" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"url":"https://arxiv.org/pdf/1706.03762v5","filename":"attention.pdf"}' \
  | python3 -m json.tool

# Test 8: Ingest file
echo ""
echo "[8] POST /v1/ingest/file"
echo "This is a test document with enough content for chunking." > /tmp/test_research.md
curl -s -X POST "$API_URL/v1/ingest/file" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@/tmp/test_research.md" \
  | python3 -m json.tool
rm -f /tmp/test_research.md

echo ""
echo "=========================================="
echo "  Done"
echo "=========================================="
