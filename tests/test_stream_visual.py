#!/usr/bin/env python3
"""Visual SSE stream test - watch tokens arrive in real time."""

import os
import json
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_KEY = os.environ.get("RESEARCH_OS_API_KEY", "")

print("Sending query: Explain backpropagation briefly\n")
print("-" * 50)

r = requests.post(
    f"{API_URL}/v1/chat",
    json={"query": "Explain backpropagation briefly"},
    headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
    stream=True,
    timeout=60,
)

if r.status_code != 200:
    print(f"ERROR: HTTP {r.status_code}")
    print(r.text)
    exit(1)

for line in r.iter_lines(decode_unicode=True):
    if not line or not line.startswith("data:"):
        continue

    data_str = line[5:].strip()
    if not data_str:
        continue

    try:
        event = json.loads(data_str)
        etype = event.get("event", "")

        if etype == "start":
            print(f"\nIntent: {event.get('intent', '?')}")
        elif etype == "context":
            print(f"Context: {event.get('code', 0)} code, {event.get('theory', 0)} theory\n")
        elif etype == "chunk":
            print(event.get("data", ""), end="", flush=True)
        elif etype == "done":
            print("\n\nStream complete")
        elif etype == "error":
            print(f"\nError: {event.get('error', '?')}")
    except json.JSONDecodeError:
        pass

r.close()
