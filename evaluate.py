#!/usr/bin/env python3
"""
Automated Evaluation Framework for Research-OS RAG
==================================================

Evaluates:
1. Faithfulness: Is the answer supported by the context? (LLM Judge)
2. Relevancy: Does the answer address the user's query? (LLM Judge)
3. Groundedness: Are citations verified against the ledger? (Programmatic)

Usage:
    python evaluate.py --test-set data/test_set.json
"""

import sys
import os
import json
import argparse
import logging
from typing import List, Dict, Any
from pathlib import Path
import ollama

# src is reachable from root
from src.rag.pipeline import ResearchPipeline, PipelineConfig, create_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EVAL_METHOD = "Cross-Encoder (Deberta/MiniLM)"

def load_test_set(path: str) -> List[Dict]:
    """Load test questions."""
    # If path doesn't exist, return a dummy test set for verification
    if not os.path.exists(path):
        logger.warning(f"Test set not found at {path}. Using dummy test set.")
        return [
            {
                "id": "test_001",
                "question": "What is the Transformer architecture?",
                "ground_truth": "The Transformer is a deep learning architecture introduced in 'Attention Is All You Need' that relies entirely on an attention mechanism to draw global dependencies between input and output.",
            }
        ]
    
    with open(path, 'r') as f:
        return json.load(f)

def evaluate_groundedness(sources: List[Dict]) -> float:
    """Programmatically verify: Do the citations match the MD5-verified ledger?"""
    if not sources:
        return 0.0
    
    verified_count = sum(1 for s in sources if s.get("verified", False))
    return verified_count / len(sources)

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG Pipeline")
    parser.add_argument("--test-set", type=str, default="data/test_set.json", help="Path to test set JSON")
    parser.add_argument("--output", type=str, default="evaluation_report.json", help="Output report path")
    args = parser.parse_args()

    # Initialize Pipeline properly using factory to load index
    pipeline = create_pipeline()
    
    if pipeline.index_size == 0:
        logger.warning("Pipeline index is empty! Evaluation might not produce useful results.")
    
    # Load Test Set
    test_set = load_test_set(args.test_set)
    
    results = []
    
    print(f"Starting evaluation of {len(test_set)} items using: {EVAL_METHOD}")
    
    for item in test_set:
        qid = item.get("id", "unknown")
        question = item["question"]
        
        print(f"\nEvaluating: {question}...")
        
        # Query Pipeline
        try:
            result = pipeline.query(question, verify=False)
        except Exception as e:
            logger.error(f"Pipeline query failed for {qid}: {e}")
            results.append({
                "id": qid,
                "error": str(e)
            })
            continue

        # Prepare Context String (Optional, for logging or manual check)
        context_parts = []
        for c in result.code_context:
            context_parts.append(f"[Code] {c.get('text', '')}")
        for c in result.theory_context:
            context_parts.append(f"[Theory] {c.get('text', '')}")
        
        # Metrics from Pipeline
        faithfulness = result.evaluation.get("faithfulness", 0.0)
        relevancy = result.evaluation.get("relevancy", 0.0)
        
        # 3. Groundedness
        groundedness_score = evaluate_groundedness(result.sources)
        
        print(f"  -> Faithfulness: {faithfulness:.2f}")
        print(f"  -> Relevancy:    {relevancy:.2f}")
        print(f"  -> Groundedness: {groundedness_score:.2f}")
        
        results.append({
            "id": qid,
            "question": question,
            "response": result.response,
            "metrics": {
                "faithfulness": faithfulness,
                "relevancy": relevancy,
                "groundedness": groundedness_score,
                "num_sources": len(result.sources)
            },
            "intent": result.intent
        })

    # Calculate Aggregates
    total = len(results)
    if total > 0:
        avg_faithfulness = sum(r.get("metrics", {}).get("faithfulness", 0) for r in results) / total
        avg_relevancy = sum(r.get("metrics", {}).get("relevancy", 0) for r in results) / total
        avg_groundedness = sum(r.get("metrics", {}).get("groundedness", 0) for r in results) / total
        
        summary = {
            "total_samples": total,
            "faithfulness_avg": round(avg_faithfulness, 2),
            "relevancy_avg": round(avg_relevancy, 2),
            "groundedness_avg": round(avg_groundedness, 2)
        }
    else:
        summary = {"error": "No results", "faithfulness_avg": 0, "relevancy_avg": 0, "groundedness_avg": 0}

    # Generate Report Card
    from datetime import datetime
    report_card = f"""# Research-OS V1.0: Evaluation Report Card
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Method**: {EVAL_METHOD}

## Summary Metrics
- **Faithfulness**: {summary['faithfulness_avg']}
- **Relevancy**: {summary['relevancy_avg']}
- **Groundedness**: {summary['groundedness_avg']}
- **Total Tested**: {summary['total_samples']}

## Detailed Breakdown
"""
    for r in results:
        m = r.get('metrics', {})
        report_card += f"- **Q**: {r['question']}\n"
        report_card += f"  - Intent: `{r.get('intent', 'unknown')}`\n"
        report_card += f"  - Faithfulness: {m.get('faithfulness', 0):.2f}\n"
        report_card += f"  - Relevancy: {m.get('relevancy', 0):.2f}\n"
        report_card += f"  - Groundedness: {m.get('groundedness', 0):.2f}\n\n"

    report_card += f"""
## Certified by
Automated Evaluation Framework ({EVAL_METHOD})
"""
    
    with open("EVAL_REPORT.md", "w") as f:
        f.write(report_card)

    report = {
        "summary": summary,
        "details": results
    }
    
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
        
    print("\nEvaluation Complete!")
    print(json.dumps(summary, indent=2))
    print(f"Report saved to {args.output}")
    print(f"Report Card saved to EVAL_REPORT.md")

if __name__ == "__main__":
    main()

