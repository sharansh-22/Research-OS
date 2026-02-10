#!/usr/bin/env python3
"""
Research-OS: Main Entry Point
==============================
RAG system with Smart Query Routing, Memory, and Streaming.

Features:
- Smart Query Router (intent classification)
- Short-term memory (last 3 conversation turns)
- Real-time streaming output
- Code/theory context separation
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Dict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.rag import ResearchPipeline, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("research-os")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)


MAX_HISTORY_TURNS = 3


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                      RESEARCH-OS                             ║
║         Research Architect RAG System                        ║
║         Smart Routing | Memory | Streaming                   ║
╚══════════════════════════════════════════════════════════════╝
    """)


def add_to_history(
    history: List[Dict[str, str]],
    user_query: str,
    assistant_response: str,
    max_turns: int = MAX_HISTORY_TURNS,
) -> List[Dict[str, str]]:
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": assistant_response})
    max_messages = max_turns * 2
    if len(history) > max_messages:
        history = history[-max_messages:]
    return history


def clear_history() -> List[Dict[str, str]]:
    return []


def interactive_mode(pipeline: ResearchPipeline, use_streaming: bool = True):
    print_banner()
    print("Commands:")
    print("  • Type your research question")
    print("  • 'stats'    - Show index statistics")
    print("  • 'health'   - Check API connectivity")
    print("  • 'stream'   - Toggle streaming mode")
    print("  • 'history'  - Show conversation history")
    print("  • 'clear'    - Clear conversation history")
    print("  • 'intent'   - Test intent classification")
    print("  • 'quit'     - Exit")
    print(f"\n🔄 Streaming: {'ON' if use_streaming else 'OFF'}")
    print(f"🧠 Memory: ON (last {MAX_HISTORY_TURNS} turns)")
    print(f"🎯 Smart Routing: ON")
    print("=" * 60 + "\n")
    
    streaming_enabled = use_streaming
    chat_history: List[Dict[str, str]] = []
    
    while True:
        try:
            turns = len(chat_history) // 2
            memory_indicator = f"[{turns}/{MAX_HISTORY_TURNS}]" if turns > 0 else ""
            
            query = input(f"\n🔬 Query {memory_indicator}: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ('quit', 'exit', 'q'):
                print("\nGoodbye! 👋")
                break
            
            if query.lower() == 'stats':
                show_stats(pipeline)
                continue
            
            if query.lower() == 'health':
                show_health(pipeline)
                continue
            
            if query.lower() == 'stream':
                streaming_enabled = not streaming_enabled
                print(f"\n🔄 Streaming: {'ON' if streaming_enabled else 'OFF'}")
                continue
            
            if query.lower() == 'history':
                show_history(chat_history)
                continue
            
            if query.lower() == 'clear':
                chat_history = clear_history()
                print("\n🧠 Conversation history cleared.")
                continue
            
            if query.lower() == 'intent':
                test_intent(pipeline)
                continue
            
            # Classify intent first
            intent = pipeline.classify_intent(query)
            print(f"\n🎯 Intent: {intent.upper()}")
            
            if streaming_enabled:
                response = execute_streaming_query(pipeline, query, chat_history, intent)
            else:
                response = execute_standard_query(pipeline, query, chat_history, intent)
            
            if response and not response.startswith("⚠️"):
                chat_history = add_to_history(chat_history, query, response)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\n⚠️ Error: {e}")


def execute_streaming_query(
    pipeline: ResearchPipeline,
    query: str,
    history: List[Dict[str, str]],
    intent: str,
) -> str:
    print("⏳ Retrieving context...\n")
    
    results = pipeline.retriever.search_by_type_filtered(
        query, 
        pipeline.config.top_k,
        filter_type=intent,
    )
    code_results = results.get("code", [])
    theory_results = results.get("theory", [])
    
    print(f"📎 Found: {len(theory_results)} theory, {len(code_results)} code chunks")
    if history:
        print(f"🧠 Using {len(history) // 2} previous turns")
    print("\n" + "=" * 60)
    print("📝 Response:\n")
    
    full_response = ""
    try:
        for chunk in pipeline.generator.generate_stream(
            query, 
            code_results, 
            theory_results,
            history=history,
        ):
            print(chunk, end="", flush=True)
            full_response += chunk
        
        print("\n" + "=" * 60)
        
        if pipeline.config.verify_code and "```" in full_response:
            verify_code_blocks(pipeline, full_response)
        
        print(f"\n🤖 Backend: groq (streaming)")
        print(f"💬 Tip: Ask follow-ups! (e.g., 'Explain that' or 'Show me code')")
        
        return full_response
        
    except Exception as e:
        print(f"\n\n⚠️ Streaming error: {e}")
        return ""


def execute_standard_query(
    pipeline: ResearchPipeline,
    query: str,
    history: List[Dict[str, str]],
    intent: str,
) -> str:
    print("\n⏳ Processing...\n")
    
    result = pipeline.query(query, history=history, filter_type=intent)
    
    print("=" * 60)
    print(result.response)
    print("=" * 60)
    
    meta = result.generation_metadata
    backend = meta.get('backend', 'unknown')
    print(f"\n📎 Context: {len(result.theory_context)} theory, {len(result.code_context)} code")
    if history:
        print(f"🧠 Used {len(history) // 2} previous turns")
    print(f"🤖 Backend: {backend}")
    print(f"🎯 Intent: {result.intent}")
    
    if result.verification_results:
        passed = sum(1 for v in result.verification_results if v.get('success'))
        total = len(result.verification_results)
        print(f"✅ Code verified: {passed}/{total} blocks")
    
    print(f"💬 Tip: Ask follow-ups! (e.g., 'Explain that' or 'Show me code')")
    
    return result.response


def verify_code_blocks(pipeline: ResearchPipeline, response: str):
    verifications = pipeline.verifier.verify_generated_response(response)
    if verifications:
        passed = sum(1 for v in verifications if v.success)
        total = len(verifications)
        print(f"\n✅ Code verified: {passed}/{total} blocks")


def show_stats(pipeline: ResearchPipeline):
    stats = pipeline.get_stats()
    print(f"\n📊 Index Statistics:")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Chunk types: {stats['chunk_types']}")
    print(f"   Files indexed: {stats.get('processed_files', 'N/A')}")
    print(f"   Smart Routing: {'ON' if stats.get('smart_routing') else 'OFF'}")
    if 'backends' in stats:
        print(f"   Groq API: {'✓' if stats['backends'].get('groq') else '✗'}")
        print(f"   Fallback: {'✓' if stats['backends'].get('ollama_fallback') else '✗'}")


def show_health(pipeline: ResearchPipeline):
    print("\n🏥 Health Check...")
    status = pipeline.generator.health_check()
    print(f"   Groq API: {'✓ Connected' if status['groq'] else '✗ Failed'}")
    print(f"   Ollama: {'✓ Available' if status['ollama_fallback'] else '✗ Unavailable'}")


def show_history(history: List[Dict[str, str]]):
    if not history:
        print("\n📜 No conversation history yet.")
        return
    
    print(f"\n📜 Conversation History ({len(history) // 2} turns):")
    print("-" * 40)
    
    for i, msg in enumerate(history):
        role = msg['role'].upper()
        content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
        icon = "👤" if role == "USER" else "🤖"
        print(f"{icon} {role}: {content}")
        if msg['role'] == 'assistant' and i < len(history) - 1:
            print("-" * 40)


def test_intent(pipeline: ResearchPipeline):
    """Test intent classification on sample queries."""
    print("\n🎯 Intent Classification Test")
    print("-" * 40)
    
    test_queries = [
        "Show me Python code for attention",
        "Explain the math behind transformers",
        "What is the derivative of softmax?",
        "How to implement dropout in PyTorch?",
        "Why does batch normalization help?",
        "Give me a numpy example",
        "Prove the gradient descent convergence",
        "What is machine learning?",
    ]
    
    for q in test_queries:
        intent = pipeline.classify_intent(q)
        icon = "💻" if intent == "code" else "📐" if intent == "theory" else "🔀"
        print(f"  {icon} [{intent:6}] {q}")
    
    print("-" * 40)
    print("Enter your own query to test, or press Enter to return:")
    
    while True:
        custom = input("  Test: ").strip()
        if not custom:
            break
        intent = pipeline.classify_intent(custom)
        icon = "💻" if intent == "code" else "📐" if intent == "theory" else "🔀"
        print(f"  {icon} [{intent:6}] {custom}")


def single_query_mode(
    pipeline: ResearchPipeline,
    query: str,
    use_streaming: bool = True,
):
    intent = pipeline.classify_intent(query)
    
    if use_streaming:
        results = pipeline.retriever.search_by_type_filtered(
            query, 
            pipeline.config.top_k,
            filter_type=intent,
        )
        code_results = results.get("code", [])
        theory_results = results.get("theory", [])
        
        for chunk in pipeline.generator.generate_stream(query, code_results, theory_results):
            print(chunk, end="", flush=True)
        print(f"\n\n[Intent: {intent}]")
    else:
        result = pipeline.query(query, filter_type=intent)
        print(result.response)
        print(f"\n[Backend: {result.generation_metadata.get('backend', 'unknown')} | Intent: {result.intent}]")


def main():
    parser = argparse.ArgumentParser(description="Research-OS: Smart RAG System")
    parser.add_argument("--query", "-q", type=str, help="Single query")
    parser.add_argument("--index-dir", type=str, default="data/index")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    parser.add_argument("--no-fallback", action="store_true", help="Disable Ollama fallback")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not os.environ.get("GROQ_API_KEY"):
        print("\n⚠️  GROQ_API_KEY not set!")
        print("   Run: export GROQ_API_KEY='your-key-here'")
        print("   Get key: https://console.groq.com/keys\n")
        return
    
    config = PipelineConfig(
        index_dir=args.index_dir,
        enable_fallback=not args.no_fallback,
    )
    
    try:
        pipeline = ResearchPipeline(config)
    except ValueError as e:
        print(f"\n❌ Initialization Error:\n{e}")
        return
    
    if Path(args.index_dir).exists():
        try:
            pipeline.load_index()
            logger.info(f"Loaded {pipeline.index_size} chunks")
        except Exception as e:
            logger.warning(f"No existing index: {e}")
    
    use_streaming = not args.no_stream
    
    if args.query:
        single_query_mode(pipeline, args.query, use_streaming)
    else:
        interactive_mode(pipeline, use_streaming)


if __name__ == "__main__":
    main()
