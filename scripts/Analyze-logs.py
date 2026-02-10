import json
import os
from collections import defaultdict
import statistics

LOG_FILE = "logs/queries.jsonl"

def load_logs():
    if not os.path.exists(LOG_FILE):
        print("No logs found.")
        return []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def analyze_logs():
    logs = load_logs()
    if not logs:
        return

    # Group by query
    grouped = defaultdict(list)
    for entry in logs:
        query = entry["query"]
        for result in entry["results"]:
            grouped[query].append(result["distance"])

    # Print summary
    print("\n=== Drift Analysis ===")
    for query, distances in grouped.items():
        avg = statistics.mean(distances)
        min_d = min(distances)
        max_d = max(distances)
        print(f"Query: {query}")
        print(f"  Runs: {len(distances)}")
        print(f"  Avg distance: {avg:.4f}")
        print(f"  Min: {min_d:.4f}, Max: {max_d:.4f}\n")

if __name__ == "__main__":
    analyze_logs()
