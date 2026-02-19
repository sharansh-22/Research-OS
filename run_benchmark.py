#!/usr/bin/env python3
"""
Research-OS Benchmark Runner
==============================
Entry point for the Benchmark Suite.

Compares Research-OS (Auditor + Cache) vs Vanilla RAG Baseline.

Usage:
    python run_benchmark.py
    python run_benchmark.py --test-set benchmark/test_set.json
    python run_benchmark.py --output benchmark/comparison_report.md
    python run_benchmark.py --results-only   # Skip report generation

Environment Variables:
    BENCHMARK_HALLUC_THRESHOLD  — Faithfulness threshold for hallucination flag (default: 0.5)
    BENCHMARK_AUDITOR_PASS      — Auditor faithfulness threshold for "pass" (default: 0.7)
"""

import sys
import argparse
import logging

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Suite: Research-OS vs Vanilla RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default="benchmark/test_set.json",
        help="Path to test set JSON (default: benchmark/test_set.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark/comparison_report.md",
        help="Path for the comparison report (default: benchmark/comparison_report.md)",
    )
    parser.add_argument(
        "--results-only",
        action="store_true",
        help="Only save raw results JSON, skip report generation",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  RESEARCH-OS BENCHMARK SUITE")
    logger.info("  Research-OS (Auditor + Cache) vs Vanilla RAG")
    logger.info("=" * 60)

    # Import here to avoid loading heavy models on --help
    from benchmark.benchmark_runner import BenchmarkRunner
    from benchmark.report_generator import generate_report

    # Run Benchmark
    runner = BenchmarkRunner(
        test_set_path=args.test_set,
        output_dir=str(args.output).rsplit("/", 1)[0] if "/" in args.output else "benchmark",
    )
    results = runner.run()

    # Generate Report
    if not args.results_only:
        logger.info(f"\nGenerating comparison report → {args.output}")
        report = generate_report(results, output_path=args.output)
        logger.info(f"Report saved to {args.output}")

    # Print Summary
    summary = results.get("summary", {})
    v = summary.get("vanilla_rag", {})
    r = summary.get("research_os", {})
    catches = summary.get("hallucination_catches", 0)

    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Vanilla RAG  | Faith: {v.get('avg_faithfulness', 0):.4f} | "
          f"Relev: {v.get('avg_relevancy', 0):.4f} | "
          f"Halluc: {v.get('hallucination_count', '?')} | "
          f"Latency: {v.get('avg_latency', 0):.2f}s")
    print(f"  Research-OS  | Faith: {r.get('avg_faithfulness', 0):.4f} | "
          f"Relev: {r.get('avg_relevancy', 0):.4f} | "
          f"Halluc: {r.get('hallucination_count', '?')} | "
          f"Latency: {r.get('avg_latency', 0):.2f}s")
    print(f"\n  ⚡ Hallucination Catches: {catches}")
    print(f"  📊 Catch Rate: {summary.get('catch_rate', 0) * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
