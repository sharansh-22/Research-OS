#!/usr/bin/env python3
"""Quick check: what keys exist in chunk metadata after retrieval."""
import os
import logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING)

from src.rag import create_pipeline

pipeline = create_pipeline(index_dir="data/index", load_existing=True)
results = pipeline.retriever.search("What is attention?", top_k=5)

print("\n=== RETRIEVAL RESULTS ===")
print(f"Total results: {len(results)}\n")

if len(results) == 0:
    print("NO RESULTS RETURNED - index may be empty or query failed")
else:
    for i, r in enumerate(results):
        print(f"--- Chunk {i+1} ---")
        print(f"  chunk_type value: {r.chunk.chunk_type.value}")
        print(f"  chunk_id: {r.chunk.chunk_id}")
        print(f"  score: {r.score}")
        print(f"  source attr: {r.source}")
        print(f"  content preview: {r.chunk.content[:100]}")
        print(f"  metadata type: {type(r.chunk.metadata)}")
        print(f"  metadata keys: {sorted(r.chunk.metadata.keys())}")
        for k, v in r.chunk.metadata.items():
            val_str = str(v)[:80]
            print(f"    {k} = {val_str}")
        print()

    print("=== to_dict() OUTPUT FOR CHUNK 1 ===")
    print()
    import json
    d = results[0].to_dict()
    print(json.dumps(d, indent=2, default=str))
