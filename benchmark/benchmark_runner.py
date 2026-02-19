"""
Benchmark Runner
=================
Orchestrates head-to-head comparison between Research-OS and Vanilla RAG.

For each query in the test set:
  1. Runs through Vanilla RAG (Retrieve → Generate)
  2. Runs through Research-OS (Retrieve → Generate → Audit)
  3. Evaluates both using RAGEvaluator (DeBERTa NLI)
  4. Records metrics: response, latency, hallucination flag, source accuracy
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.rag.pipeline import ResearchPipeline, PipelineConfig, create_pipeline
from src.rag.evaluator import RAGEvaluator
from benchmark.vanilla_pipeline import VanillaRAGPipeline

logger = logging.getLogger(__name__)

# Thresholds — configurable via env vars
HALLUCINATION_THRESHOLD = float(os.getenv("BENCHMARK_HALLUC_THRESHOLD", "0.5"))
AUDITOR_PASS_THRESHOLD = float(os.getenv("BENCHMARK_AUDITOR_PASS", "0.7"))


class BenchmarkRunner:
    """
    Runs the full benchmark suite: Vanilla RAG vs Research-OS.
    
    Shares the same retriever and embedder between both pipelines
    to ensure a fair comparison.
    """

    def __init__(
        self,
        test_set_path: str = "benchmark/test_set.json",
        output_dir: str = "benchmark",
    ):
        """
        Initialize the benchmark runner.

        Args:
            test_set_path: Path to the test set JSON file.
            output_dir: Directory for output files (results.json, report).
        """
        self.test_set_path = test_set_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("BENCHMARK SUITE: Research-OS vs Vanilla RAG")
        logger.info("=" * 60)

        # --- Initialize Research-OS Pipeline (full) ---
        logger.info("Initializing Research-OS pipeline...")
        self.research_pipeline = create_pipeline()
        logger.info(f"  Index size: {self.research_pipeline.index_size} chunks")

        # --- Initialize Vanilla RAG Pipeline (shared components) ---
        logger.info("Initializing Vanilla RAG pipeline (shared retriever + generator)...")
        self.vanilla_pipeline = VanillaRAGPipeline(
            retriever=self.research_pipeline.retriever,
            generator=self.research_pipeline.generator,
            embedder=self.research_pipeline.embedder,
            config=self.research_pipeline.config,
            processed_files=self.research_pipeline._processed_files,
        )

        # --- Shared Evaluator (DeBERTa NLI — independent scorer) ---
        logger.info("Initializing shared RAGEvaluator (DeBERTa NLI)...")
        self.evaluator = RAGEvaluator()

        logger.info("All components initialized.")

    def load_test_set(self) -> List[Dict]:
        """Load the test set from JSON."""
        path = Path(self.test_set_path)
        if not path.exists():
            raise FileNotFoundError(f"Test set not found: {path}")
        with open(path, "r") as f:
            return json.load(f)

    def _evaluate_result(
        self, response: str, context_str: str, query: str, sources: List[Dict]
    ) -> Dict[str, Any]:
        """
        Evaluate a single result using the shared RAGEvaluator.

        Returns dict with faithfulness, relevancy, hallucination_flag,
        and source_citation_accuracy.
        """
        faithfulness = self.evaluator.evaluate_faithfulness(
            context=context_str, answer=response
        )
        relevancy = self.evaluator.evaluate_relevancy(
            query=query, answer=response
        )

        # Hallucination flag: faithfulness below threshold
        hallucination_flag = faithfulness < HALLUCINATION_THRESHOLD

        # Source citation accuracy: % of verified sources
        if sources:
            verified_count = sum(1 for s in sources if s.get("verified", False))
            source_accuracy = verified_count / len(sources)
        else:
            source_accuracy = 0.0

        return {
            "faithfulness": round(faithfulness, 4),
            "relevancy": round(relevancy, 4),
            "hallucination_flag": hallucination_flag,
            "source_citation_accuracy": round(source_accuracy, 4),
        }

    def run(self) -> Dict[str, Any]:
        """
        Execute the full benchmark.

        Returns:
            Complete results dict with per-query comparisons and summary.
        """
        test_set = self.load_test_set()
        total = len(test_set)

        logger.info(f"\nRunning {total} queries through both pipelines...\n")

        results = []
        hallucination_catches = 0

        for i, item in enumerate(test_set, 1):
            qid = item["id"]
            question = item["question"]
            category = item.get("category", "unknown")
            trick = item.get("trick", "")

            logger.info(f"[{i}/{total}] {qid}: {question[:80]}...")

            # Initialize context for use by both pipelines
            vanilla_context = ""

            # ----- Vanilla RAG -----
            try:
                vanilla_result = self.vanilla_pipeline.query(question)
                vanilla_context = vanilla_result.context_str

                vanilla_eval = self._evaluate_result(
                    response=vanilla_result.response,
                    context_str=vanilla_context,
                    query=question,
                    sources=vanilla_result.sources,
                )
                vanilla_entry = {
                    "response": vanilla_result.response,
                    "latency": round(vanilla_result.latency, 3),
                    **vanilla_eval,
                    "num_sources": len(vanilla_result.sources),
                }
            except Exception as e:
                logger.error(f"  Vanilla RAG failed: {e}")
                vanilla_entry = {
                    "response": f"ERROR: {e}",
                    "latency": 0.0,
                    "faithfulness": 0.0,
                    "relevancy": 0.0,
                    "hallucination_flag": True,
                    "source_citation_accuracy": 0.0,
                    "num_sources": 0,
                }

            # ----- Research-OS (with Audit) -----
            try:
                ros_start = time.time()
                ros_result = self.research_pipeline.query_with_audit(question)
                ros_latency = time.time() - ros_start

                ros_response = ros_result.get("answer", "")
                ros_audit = ros_result.get("audit", {})

                # Also evaluate with DeBERTa for apples-to-apples comparison
                # (the auditor uses Groq LLM, but we want the same scorer)
                # We need context — get it from the vanilla run (same retriever)
                ros_eval = self._evaluate_result(
                    response=ros_response,
                    context_str=vanilla_context,  # Same context (same retriever)
                    query=question,
                    sources=[],  # Audit result doesn't return full sources
                )

                ros_entry = {
                    "response": ros_response,
                    "latency": round(ros_latency, 3),
                    **ros_eval,
                    "auditor_faithfulness": round(
                        ros_audit.get("faithfulness", 0.0), 4
                    ),
                    "auditor_relevancy": round(
                        ros_audit.get("relevancy", 0.0), 4
                    ),
                    "auditor_reasoning": ros_audit.get("reasoning", ""),
                    "cached": ros_audit.get("cached", False),
                    "num_evidence": len(ros_audit.get("evidence", [])),
                }
            except Exception as e:
                logger.error(f"  Research-OS failed: {e}")
                ros_entry = {
                    "response": f"ERROR: {e}",
                    "latency": 0.0,
                    "faithfulness": 0.0,
                    "relevancy": 0.0,
                    "hallucination_flag": True,
                    "source_citation_accuracy": 0.0,
                    "auditor_faithfulness": 0.0,
                    "auditor_relevancy": 0.0,
                    "auditor_reasoning": f"Error: {e}",
                    "cached": False,
                    "num_evidence": 0,
                }

            # ----- Hallucination Catch Detection -----
            is_catch = (
                vanilla_entry["hallucination_flag"]
                and ros_entry.get("auditor_faithfulness", 0.0) >= AUDITOR_PASS_THRESHOLD
            )
            if is_catch:
                hallucination_catches += 1

            result_entry = {
                "id": qid,
                "question": question,
                "category": category,
                "trick": trick,
                "vanilla_rag": vanilla_entry,
                "research_os": ros_entry,
                "hallucination_catch": is_catch,
            }
            results.append(result_entry)

            # Print progress
            v_flag = "🚩 HALLUC" if vanilla_entry["hallucination_flag"] else "✅ OK"
            r_flag = f"🛡️ Auditor={ros_entry.get('auditor_faithfulness', 0.0):.2f}"
            catch = " ⚡ CATCH!" if is_catch else ""
            logger.info(f"  Vanilla: {v_flag} (f={vanilla_entry['faithfulness']:.2f})")
            logger.info(f"  Research-OS: {r_flag}")
            if catch:
                logger.info(f"  {catch}")

        # ----- Summary -----
        summary = self._compute_summary(results, hallucination_catches)

        full_results = {
            "benchmark_metadata": {
                "total_queries": total,
                "hallucination_threshold": HALLUCINATION_THRESHOLD,
                "auditor_pass_threshold": AUDITOR_PASS_THRESHOLD,
                "index_size": self.research_pipeline.index_size,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "summary": summary,
            "results": results,
        }

        # Save raw results
        results_path = self.output_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        logger.info(f"\nRaw results saved to {results_path}")

        return full_results

    def _compute_summary(
        self, results: List[Dict], hallucination_catches: int
    ) -> Dict[str, Any]:
        """Compute aggregate summary statistics."""
        n = len(results)
        if n == 0:
            return {"error": "No results"}

        def avg(key: str, system: str) -> float:
            vals = [r[system].get(key, 0.0) for r in results]
            return round(sum(vals) / n, 4)

        vanilla_halluc_count = sum(
            1 for r in results if r["vanilla_rag"]["hallucination_flag"]
        )
        ros_halluc_count = sum(
            1 for r in results if r["research_os"]["hallucination_flag"]
        )

        return {
            "vanilla_rag": {
                "avg_faithfulness": avg("faithfulness", "vanilla_rag"),
                "avg_relevancy": avg("relevancy", "vanilla_rag"),
                "avg_latency": avg("latency", "vanilla_rag"),
                "avg_source_accuracy": avg("source_citation_accuracy", "vanilla_rag"),
                "hallucination_count": vanilla_halluc_count,
            },
            "research_os": {
                "avg_faithfulness": avg("faithfulness", "research_os"),
                "avg_relevancy": avg("relevancy", "research_os"),
                "avg_latency": avg("latency", "research_os"),
                "avg_auditor_faithfulness": avg(
                    "auditor_faithfulness", "research_os"
                ),
                "avg_auditor_relevancy": avg("auditor_relevancy", "research_os"),
                "hallucination_count": ros_halluc_count,
            },
            "hallucination_catches": hallucination_catches,
            "catch_rate": (
                round(hallucination_catches / vanilla_halluc_count, 4)
                if vanilla_halluc_count > 0
                else 0.0
            ),
        }
