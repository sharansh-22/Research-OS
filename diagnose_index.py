#!/usr/bin/env python3
"""Diagnose why search returns 0 results."""
import os
import logging
import numpy as np
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.WARNING)

from src.rag import create_pipeline

pipeline = create_pipeline(index_dir="data/index", load_existing=True)

r = pipeline.retriever

print("=== INDEX STATE ===")
print(f"  chunks count: {len(r.chunks)}")
print(f"  chunk_texts count: {len(r.chunk_texts)}")
print(f"  faiss_index: {r.faiss_index}")
print(f"  faiss_index.ntotal: {r.faiss_index.ntotal if r.faiss_index else 'None'}")
print(f"  bm25_index: {r.bm25_index}")
print(f"  min_similarity: {r.min_similarity}")
print(f"  enable_reranking: {r.enable_reranking}")
print()

if len(r.chunks) > 0:
    print("=== SAMPLE CHUNK ===")
    c = r.chunks[0]
    print(f"  chunk_type: {c.chunk_type.value}")
    print(f"  chunk_id: {c.chunk_id}")
    print(f"  content[:100]: {c.content[:100]}")
    print(f"  metadata keys: {sorted(c.metadata.keys())}")
    for k, v in c.metadata.items():
        print(f"    {k} = {str(v)[:80]}")
    print()

print("=== RAW FAISS SEARCH (bypass all filtering) ===")
query = "What is attention?"
q_vec = r.embedder.embed_query(query).reshape(1, -1).astype(np.float32)
scores, indices = r.faiss_index.search(q_vec, 5)
print(f"  Top 5 scores: {scores[0].tolist()}")
print(f"  Top 5 indices: {indices[0].tolist()}")
print()

for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
    if idx >= 0 and idx < len(r.chunks):
        c = r.chunks[idx]
        print(f"  Result {rank+1}: score={score:.4f} type={c.chunk_type.value}")
        print(f"    source={c.metadata.get('source', 'MISSING')}")
        print(f"    metadata keys: {sorted(c.metadata.keys())}")
        for k, v in c.metadata.items():
            print(f"      {k} = {str(v)[:80]}")
        print(f"    content[:120]: {c.content[:120]}")
        print()

print("=== BM25 STATUS ===")
if r.bm25_index is None:
    print("  BM25 index is None - this is the problem!")
    print("  BM25 gets rebuilt on add_chunks() but not on load().")
else:
    print("  BM25 index exists")
    test_tokens = ["attention", "transformer", "self"]
    bm25_scores = r.bm25_index.get_scores(test_tokens)
    nonzero = np.count_nonzero(bm25_scores)
    print(f"  Non-zero BM25 scores: {nonzero}/{len(bm25_scores)}")
