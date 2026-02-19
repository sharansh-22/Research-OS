"""
Comparison Report Generator
=============================
Generates a professional comparison_report.md from benchmark results.
"""

import json
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime


def generate_report(
    results: Dict[str, Any],
    output_path: str = "benchmark/comparison_report.md",
) -> str:
    """
    Generate a markdown comparison report.

    Args:
        results: Full results dict from BenchmarkRunner.run().
        output_path: Path to write the markdown report.

    Returns:
        The generated markdown string.
    """
    meta = results.get("benchmark_metadata", {})
    summary = results.get("summary", {})
    details = results.get("results", [])

    vanilla_summary = summary.get("vanilla_rag", {})
    ros_summary = summary.get("research_os", {})
    catches = summary.get("hallucination_catches", 0)
    catch_rate = summary.get("catch_rate", 0.0)

    md = []

    # --- Header ---
    md.append("# 📊 Benchmark Report: Research-OS vs Vanilla RAG")
    md.append("")
    md.append(f"**Generated**: {meta.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}")
    md.append(f"**Index Size**: {meta.get('index_size', 'N/A')} chunks")
    md.append(f"**Queries Tested**: {meta.get('total_queries', len(details))}")
    md.append(f"**Hallucination Threshold**: faithfulness < {meta.get('hallucination_threshold', 0.5)}")
    md.append(f"**Auditor Pass Threshold**: faithfulness ≥ {meta.get('auditor_pass_threshold', 0.7)}")
    md.append("")

    # --- Summary Table ---
    md.append("## Summary")
    md.append("")
    md.append("| Metric | Vanilla RAG | Research-OS | Winner |")
    md.append("|--------|------------|-------------|--------|")

    v_faith = vanilla_summary.get("avg_faithfulness", 0.0)
    r_faith = ros_summary.get("avg_faithfulness", 0.0)
    winner_faith = "🟢 Research-OS" if r_faith >= v_faith else "🔴 Vanilla RAG"
    md.append(f"| **Avg Faithfulness** | {v_faith:.4f} | {r_faith:.4f} | {winner_faith} |")

    v_rel = vanilla_summary.get("avg_relevancy", 0.0)
    r_rel = ros_summary.get("avg_relevancy", 0.0)
    winner_rel = "🟢 Research-OS" if r_rel >= v_rel else "🔴 Vanilla RAG"
    md.append(f"| **Avg Relevancy** | {v_rel:.4f} | {r_rel:.4f} | {winner_rel} |")

    v_lat = vanilla_summary.get("avg_latency", 0.0)
    r_lat = ros_summary.get("avg_latency", 0.0)
    winner_lat = "🟢 Research-OS" if r_lat < v_lat else "� Vanilla RAG"
    md.append(f"| **Avg Latency (s)** | {v_lat:.3f} | {r_lat:.3f} | {winner_lat} |")

    v_halluc = vanilla_summary.get("hallucination_count", 0)
    r_halluc = ros_summary.get("hallucination_count", 0)
    winner_halluc = "🟢 Research-OS" if r_halluc <= v_halluc else "🔴 Vanilla RAG"
    md.append(f"| **Hallucinations** | {v_halluc}/{meta.get('total_queries', '?')} | {r_halluc}/{meta.get('total_queries', '?')} | {winner_halluc} |")

    v_src = vanilla_summary.get("avg_source_accuracy", 0.0)
    md.append(f"| **Avg Source Accuracy** | {v_src:.4f} | — | — |")

    r_audit_f = ros_summary.get("avg_auditor_faithfulness", 0.0)
    md.append(f"| **Avg Auditor Faithfulness** | — | {r_audit_f:.4f} | — |")

    md.append("")

    # --- Hallucination Catch Highlight ---
    md.append("## ⚡ Hallucination Catch Events")
    md.append("")

    catch_events = [r for r in details if r.get("hallucination_catch")]

    if catch_events:
        md.append(f"**{catches} catch(es)** detected out of {v_halluc} vanilla hallucinations ")
        md.append(f"(**Catch Rate: {catch_rate * 100:.1f}%**).")
        md.append("")
        md.append("> A \"Hallucination Catch\" occurs when Vanilla RAG's faithfulness falls below")
        md.append(f"> the threshold ({meta.get('hallucination_threshold', 0.5)}), but the Research-OS")
        md.append(f"> Auditor scores the same query above the pass threshold ({meta.get('auditor_pass_threshold', 0.7)}).")
        md.append("")

        for event in catch_events:
            md.append(f"### 🛡️ {event['id']} — {event['category']}")
            md.append("")
            md.append(f"**Query**: {event['question']}")
            md.append("")
            md.append(f"**Trick**: {event.get('trick', 'N/A')}")
            md.append("")
            md.append("| Metric | Vanilla RAG | Research-OS |")
            md.append("|--------|------------|-------------|")
            v = event["vanilla_rag"]
            r = event["research_os"]
            md.append(f"| Faithfulness (DeBERTa) | {v['faithfulness']:.4f} 🚩 | {r['faithfulness']:.4f} |")
            md.append(f"| Auditor Faithfulness | — | {r.get('auditor_faithfulness', 0.0):.4f} ✅ |")
            md.append(f"| Relevancy | {v['relevancy']:.4f} | {r['relevancy']:.4f} |")
            md.append(f"| Latency | {v['latency']:.3f}s | {r['latency']:.3f}s |")
            md.append("")
            reasoning = r.get("auditor_reasoning", "N/A")
            md.append(f"> **Auditor Reasoning**: {reasoning}")
            md.append("")
    else:
        md.append("No hallucination catch events detected in this run.")
        md.append("")
        if v_halluc == 0:
            md.append("> Neither system produced hallucinated responses on this test set.")
        else:
            md.append("> Vanilla RAG produced hallucinations, but the Auditor did not")
            md.append("> score above the pass threshold for those queries.")
        md.append("")

    # --- Per-Query Breakdown ---
    md.append("## Detailed Per-Query Results")
    md.append("")
    md.append("| # | ID | Category | V-Faith | R-Faith | V-Halluc | R-Auditor | Catch | V-Latency | R-Latency |")
    md.append("|---|-----|----------|---------|---------|----------|-----------|-------|-----------|-----------|")

    for i, r in enumerate(details, 1):
        v = r["vanilla_rag"]
        ros = r["research_os"]
        v_flag = "🚩" if v["hallucination_flag"] else "✅"
        catch_flag = "⚡" if r.get("hallucination_catch") else ""
        md.append(
            f"| {i} | `{r['id']}` | {r['category']} | "
            f"{v['faithfulness']:.2f} | {ros['faithfulness']:.2f} | "
            f"{v_flag} | {ros.get('auditor_faithfulness', 0.0):.2f} | "
            f"{catch_flag} | {v['latency']:.2f}s | {ros['latency']:.2f}s |"
        )

    md.append("")

    # --- Response Samples ---
    md.append("## Response Samples (First 3)")
    md.append("")

    for r in details[:3]:
        md.append(f"### {r['id']}: {r['question'][:80]}...")
        md.append("")
        md.append("**Vanilla RAG Response** (truncated):")
        md.append("```")
        md.append(r["vanilla_rag"]["response"][:500])
        md.append("```")
        md.append("")
        md.append("**Research-OS Response** (truncated):")
        md.append("```")
        md.append(r["research_os"]["response"][:500])
        md.append("```")
        md.append("")

    # --- Footer ---
    md.append("---")
    md.append("")
    md.append("*Generated by Research-OS Benchmark Suite*")
    md.append(f"*Evaluation method: DeBERTa NLI (faithfulness) + MiniLM (relevancy) + Groq Auditor (chain-of-thought)*")

    report_text = "\n".join(md)

    # Write to file
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text
