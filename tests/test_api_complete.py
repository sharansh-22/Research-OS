#!/usr/bin/env python3
"""
Research-OS API - Complete Test Suite
Tests every endpoint: health, chat (SSE), ingest/file, ingest/url.

Usage:
    python run_api.py          # Terminal 1
    python test_api_complete.py # Terminal 2
"""

import os
import sys
import json
import time
import requests
import tempfile
from pathlib import Path

BASE_URL = os.environ.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("RESEARCH_OS_API_KEY", "")

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

passed = 0
failed = 0
skipped = 0


def log_pass(name, detail=""):
    global passed
    passed += 1
    print(f"  {GREEN}PASS{RESET}  {name}  {detail}")


def log_fail(name, detail=""):
    global failed
    failed += 1
    print(f"  {RED}FAIL{RESET}  {name}  {detail}")


def log_skip(name, detail=""):
    global skipped
    skipped += 1
    print(f"  {YELLOW}SKIP{RESET}  {name}  {detail}")


def log_section(title):
    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}{RESET}\n")


def test_server_reachable():
    log_section("TEST 0: Server Reachability")
    try:
        r = requests.get(f"{BASE_URL}/", timeout=5)
        if r.status_code == 200:
            data = r.json()
            log_pass("Root endpoint", f"service={data.get('service', '?')}")
            return True
        else:
            log_fail("Root endpoint", f"HTTP {r.status_code}")
            return False
    except requests.ConnectionError:
        log_fail("Server connection", f"Cannot reach {BASE_URL}")
        print(f"\n  {RED}Server is not running!{RESET}")
        print(f"  Start it with: python run_api.py\n")
        return False
    except Exception as e:
        log_fail("Server connection", str(e))
        return False


def test_health():
    log_section("TEST 1: Health Endpoint (no auth)")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        if r.status_code == 200:
            data = r.json()
            log_pass(
                "GET /health",
                f"status={data['status']} chunks={data['index_chunks']} files={data['indexed_files']}"
            )
            required = {"status", "version", "index_chunks", "indexed_files", "backends", "smart_routing"}
            missing = required - set(data.keys())
            if not missing:
                log_pass("Response schema", "All required fields present")
            else:
                log_fail("Response schema", f"Missing: {missing}")
            backends = data.get("backends", {})
            if backends.get("groq", False):
                log_pass("Groq backend", "Connected")
            else:
                log_fail("Groq backend", "Not connected (check GROQ_API_KEY)")
        elif r.status_code == 503:
            log_fail("GET /health", f"503: {r.text[:200]}")
        else:
            log_fail("GET /health", f"HTTP {r.status_code}")
    except Exception as e:
        log_fail("GET /health", str(e))


def test_authentication():
    log_section("TEST 2: Authentication")
    chat_url = f"{BASE_URL}/v1/chat"
    payload = {"query": "test"}

    try:
        r = requests.post(chat_url, json=payload, timeout=10)
        if r.status_code in (401, 422):
            log_pass("Missing API key rejected", f"HTTP {r.status_code}")
        else:
            log_fail("Missing API key", f"Expected 401/422, got {r.status_code}")
    except Exception as e:
        log_fail("Missing API key test", str(e))

    try:
        bad = {"X-API-Key": "wrong-key-12345", "Content-Type": "application/json"}
        r = requests.post(chat_url, json=payload, headers=bad, timeout=10)
        if r.status_code == 403:
            log_pass("Wrong API key rejected", "HTTP 403")
        else:
            log_fail("Wrong API key", f"Expected 403, got {r.status_code}")
    except Exception as e:
        log_fail("Wrong API key test", str(e))

    if not API_KEY:
        log_skip("Valid API key", "RESEARCH_OS_API_KEY not set")
        return
    try:
        r = requests.post(chat_url, json=payload, headers=HEADERS, timeout=30, stream=True)
        if r.status_code not in (401, 403):
            log_pass("Valid API key accepted", f"HTTP {r.status_code}")
        else:
            log_fail("Valid API key rejected", f"HTTP {r.status_code}")
        r.close()
    except Exception as e:
        log_fail("Valid API key test", str(e))


def test_chat():
    log_section("TEST 3: Chat Endpoint (SSE Streaming)")

    if not API_KEY:
        log_skip("All chat tests", "RESEARCH_OS_API_KEY not set")
        return

    chat_url = f"{BASE_URL}/v1/chat"

    try:
        payload = {"query": "What is self-attention in transformers?"}
        r = requests.post(chat_url, json=payload, headers=HEADERS, stream=True, timeout=60)

        if r.status_code != 200:
            log_fail("Basic chat", f"HTTP {r.status_code}: {r.text[:200]}")
            return

        events = []
        token_count = 0
        full_response = ""

        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str:
                continue
            try:
                event = json.loads(data_str)
                events.append(event)
                if event.get("event") == "chunk" and event.get("data"):
                    token_count += 1
                    full_response += event["data"]
            except json.JSONDecodeError:
                pass

        r.close()

        if token_count > 0:
            log_pass("Basic chat streaming", f"{token_count} tokens received")
        else:
            log_fail("Basic chat streaming", "0 tokens received")

        event_types = [e.get("event") for e in events]

        if "start" in event_types:
            log_pass("start event received")
        else:
            log_fail("start event missing")

        if "context" in event_types:
            ctx = next(e for e in events if e.get("event") == "context")
            log_pass("context event received", f"code={ctx.get('code', 0)} theory={ctx.get('theory', 0)}")
        else:
            log_fail("context event missing")

        if "sources" in event_types:
            src_event = next(e for e in events if e.get("event") == "sources")
            src_count = len(src_event.get("sources", []))
            log_pass("sources event received", f"{src_count} sources")
        else:
            log_fail("sources event missing")

        if "done" in event_types:
            log_pass("done event received")
        else:
            log_fail("done event missing")

        if len(full_response) > 50:
            preview = full_response[:80].replace("\n", " ")
            log_pass("Response content", f'"{preview}..."')
        else:
            log_fail("Response too short", f"{len(full_response)} chars")

    except requests.Timeout:
        log_fail("Basic chat", "Timeout (60s)")
    except Exception as e:
        log_fail("Basic chat", str(e))

    try:
        payload_hist = {
            "query": "Can you explain that in simpler terms?",
            "history": [
                {"role": "user", "content": "What is self-attention?"},
                {"role": "assistant", "content": "Self-attention is a mechanism..."},
            ],
        }
        r = requests.post(chat_url, json=payload_hist, headers=HEADERS, stream=True, timeout=60)
        token_count = 0
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data:"):
                data_str = line[5:].strip()
                try:
                    event = json.loads(data_str)
                    if event.get("event") == "chunk":
                        token_count += 1
                except json.JSONDecodeError:
                    pass
        r.close()
        if token_count > 0:
            log_pass("Chat with history", f"{token_count} tokens")
        else:
            log_fail("Chat with history", "0 tokens")
    except Exception as e:
        log_fail("Chat with history", str(e))

    try:
        payload_filt = {"query": "Show me how attention works", "filter_type": "code"}
        r = requests.post(chat_url, json=payload_filt, headers=HEADERS, stream=True, timeout=60)
        found_start = False
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith("data:"):
                data_str = line[5:].strip()
                try:
                    event = json.loads(data_str)
                    if event.get("event") == "start":
                        found_start = True
                        intent = event.get("intent", "unknown")
                        if intent == "code":
                            log_pass("Filter type override", f"intent={intent}")
                        else:
                            log_fail("Filter type override", f"Expected code, got {intent}")
                except json.JSONDecodeError:
                    pass
        r.close()
        if not found_start:
            log_fail("Filter type override", "No start event found")
    except Exception as e:
        log_fail("Filter type override", str(e))

    try:
        r = requests.post(chat_url, json={"query": ""}, headers=HEADERS, timeout=10)
        if r.status_code == 422:
            log_pass("Empty query rejected", "HTTP 422")
        else:
            log_fail("Empty query validation", f"Expected 422, got {r.status_code}")
    except Exception as e:
        log_fail("Empty query test", str(e))

    try:
        bad_payload = {"query": "test", "history": [{"bad_key": "no role"}]}
        r = requests.post(chat_url, json=bad_payload, headers=HEADERS, timeout=10, stream=True)
        if r.status_code == 422:
            log_pass("Bad history rejected", "HTTP 422")
        else:
            log_fail("Bad history validation", f"Expected 422, got {r.status_code}")
        r.close()
    except Exception as e:
        log_fail("Bad history test", str(e))


def test_ingest_file():
    log_section("TEST 4: File Ingest Endpoint")

    if not API_KEY:
        log_skip("All ingest tests", "RESEARCH_OS_API_KEY not set")
        return

    ingest_url = f"{BASE_URL}/v1/ingest/file"
    auth_headers = {"X-API-Key": API_KEY}

    try:
        test_content = (
            "# Test Document for Research-OS\n\n"
            "## Introduction\n"
            "This is a test document to verify the ingest pipeline. "
            "It contains enough text to pass the minimum chunk size threshold.\n\n"
            "## Mathematical Background\n"
            "The softmax function maps a vector of real numbers to a probability distribution. "
            "It is widely used in classification tasks and attention mechanisms.\n\n"
            "## Implementation Notes\n"
            "The implementation uses PyTorch for tensor operations and "
            "supports both CPU and GPU execution backends.\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".md", prefix="test_research_", delete=False, mode="w") as f:
            f.write(test_content)
            temp_path = f.name

        with open(temp_path, "rb") as f:
            files = {"file": (Path(temp_path).name, f, "text/markdown")}
            r = requests.post(ingest_url, headers=auth_headers, files=files, timeout=30)

        os.unlink(temp_path)

        if r.status_code == 200:
            data = r.json()
            log_pass("Upload .md file", f"status={data['status']} file={data.get('filename', 'N/A')}")
        else:
            log_fail("Upload .md file", f"HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        log_fail("Upload .md file", str(e))

    try:
        with tempfile.NamedTemporaryFile(suffix=".xyz", prefix="test_bad_", delete=False, mode="w") as f:
            f.write("bad file")
            temp_path = f.name

        with open(temp_path, "rb") as f:
            files = {"file": (Path(temp_path).name, f, "application/octet-stream")}
            r = requests.post(ingest_url, headers=auth_headers, files=files, timeout=10)

        os.unlink(temp_path)

        if r.status_code == 415:
            log_pass("Unsupported type rejected", "HTTP 415")
        else:
            log_fail("Unsupported type", f"Expected 415, got {r.status_code}")
    except Exception as e:
        log_fail("Unsupported type test", str(e))

    try:
        with tempfile.NamedTemporaryFile(suffix=".txt", prefix="test_empty_", delete=False, mode="w") as f:
            temp_path = f.name

        with open(temp_path, "rb") as f:
            files = {"file": (Path(temp_path).name, f, "text/plain")}
            r = requests.post(ingest_url, headers=auth_headers, files=files, timeout=10)

        os.unlink(temp_path)

        if r.status_code == 400:
            log_pass("Empty file rejected", "HTTP 400")
        else:
            log_fail("Empty file", f"Expected 400, got {r.status_code}")
    except Exception as e:
        log_fail("Empty file test", str(e))


def test_ingest_url():
    log_section("TEST 5: URL Ingest Endpoint")

    if not API_KEY:
        log_skip("URL ingest tests", "RESEARCH_OS_API_KEY not set")
        return

    url_endpoint = f"{BASE_URL}/v1/ingest/url"

    try:
        payload = {"url": "https://arxiv.org/pdf/1706.03762v5", "filename": "attention_test.pdf"}
        r = requests.post(url_endpoint, json=payload, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            log_pass("URL ingest queued", f"status={data['status']}")
        else:
            log_fail("URL ingest", f"HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        log_fail("URL ingest", str(e))

    try:
        payload = {"url": "https://example.com/paper.pdf"}
        r = requests.post(url_endpoint, json=payload, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            log_pass("URL without filename", "Accepted")
        else:
            log_fail("URL without filename", f"HTTP {r.status_code}")
    except Exception as e:
        log_fail("URL without filename", str(e))


def test_docs():
    log_section("TEST 6: Documentation Endpoints")
    endpoints = [("/docs", "Swagger UI"), ("/redoc", "ReDoc"), ("/openapi.json", "OpenAPI JSON")]
    for path, name in endpoints:
        try:
            r = requests.get(f"{BASE_URL}{path}", timeout=10)
            if r.status_code == 200:
                log_pass(name, f"{BASE_URL}{path}")
            else:
                log_fail(name, f"HTTP {r.status_code}")
        except Exception as e:
            log_fail(name, str(e))


def main():
    key_display = "SET (" + API_KEY[:8] + "...)" if API_KEY else "NOT SET"
    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  RESEARCH-OS API TEST SUITE")
    print(f"{'=' * 60}{RESET}\n")
    print(f"  Target:   {BASE_URL}")
    print(f"  API Key:  {key_display}\n")

    if not test_server_reachable():
        print(f"\n{RED}{BOLD}Aborting: Server not reachable.{RESET}\n")
        sys.exit(1)

    test_health()
    test_authentication()
    test_chat()
    test_ingest_file()
    test_ingest_url()
    test_docs()

    total = passed + failed + skipped
    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}{RESET}\n")
    print(f"  {GREEN}Passed:  {passed}{RESET}")
    print(f"  {RED}Failed:  {failed}{RESET}")
    print(f"  {YELLOW}Skipped: {skipped}{RESET}")
    print(f"  Total:   {total}\n")

    if failed > 0:
        print(f"  {RED}{BOLD}Some tests failed. Check output above.{RESET}\n")
        sys.exit(1)
    else:
        print(f"  {GREEN}{BOLD}All tests passed!{RESET}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
