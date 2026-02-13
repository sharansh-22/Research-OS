import sys
import os
from src.rag.pipeline import create_pipeline

def test_audit():
    pipeline = create_pipeline()
    query = "What is the Transformer architecture?"
    print(f"Testing query: {query}")
    
    result = pipeline.query_with_audit(query)
    print("\nResult:")
    print(f"Answer: {result['answer'][:100]}...")
    print(f"Audit: {result['audit']}")

if __name__ == "__main__":
    test_audit()
