#!/usr/bin/env python3
"""
Research-OS: Main Entry Point
==============================
RAG system with Groq API (llama-3.3-70b-versatile)
"""

import argparse
import logging
import os
from pathlib import Path

# Load environment variables
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


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                      RESEARCH-OS                             ║
║         Research Architect RAG System                        ║
║         Backend: Groq API (llama-3.3-70b-versatile)          ║
╚══════════════════════════════════════════════════════════════╝
    """)


def interactive_mode(pipeline: ResearchPipeline):
    print_banner()
    print("Commands:")
    print("  • Type your research question")
    print("  • 'stats' - Show index statistics")
    print("  • 'health' - Check API connectivity")
    print("  • 'quit' - Exit")
    print("=" * 60 + "\n")
    
    while True:
        try:
            q = input("\n🔬 Query: ").strip()
            
            if not q:
                continue
            
            if q.lower() in ('quit', 'exit', 'q'):
                print("\nGoodbye! 👋")
                break
            
            if q.lower() == 'stats':
                s = pipeline.get_stats()
                print(f"\n📊 Index Statistics:")
                print(f"   Chunks: {s['total_chunks']}")
                print(f"   Types: {s['chunk_types']}")
                if 'backends' in s:
                    print(f"   Groq API: {'✓' if s['backends'].get('groq') else '✗'}")
                    print(f"   Fallback: {'✓' if s['backends'].get('ollama_fallback') else '✗'}")
                continue
            
            if q.lower() == 'health':
                print("\n🏥 Health Check...")
                status = pipeline.generator.health_check()
                print(f"   Groq API: {'✓ Connected' if status['groq'] else '✗ Failed'}")
                print(f"   Ollama Fallback: {'✓ Available' if status['ollama_fallback'] else '✗ Unavailable'}")
                continue
            
            print("\n⏳ Processing...\n")
            result = pipeline.query(q)
            
            print("=" * 60)
            print(result.response)
            print("=" * 60)
            
            # Show metadata
            meta = result.generation_metadata
            backend = meta.get('backend', 'unknown')
            print(f"\n📎 Context: {len(result.theory_context)} theory, {len(result.code_context)} code chunks")
            print(f"🤖 Backend: {backend} ({meta.get('model', 'unknown')})")
            
            if result.verification_results:
                passed = sum(1 for v in result.verification_results if v.get('success'))
                total = len(result.verification_results)
                print(f"✅ Code verified: {passed}/{total} blocks passed")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Research-OS: RAG System with Groq API"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Single query to execute"
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/index",
        help="Index directory path"
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable Ollama fallback"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check API key
    if not os.environ.get("GROQ_API_KEY"):
        print("\n⚠️  GROQ_API_KEY not set!")
        print("   Run: export GROQ_API_KEY='your-key-here'")
        print("   Get key: https://console.groq.com/keys\n")
        return
    
    # Create config
    config = PipelineConfig(
        index_dir=args.index_dir,
        enable_fallback=not args.no_fallback,
    )
    
    # Create pipeline
    try:
        pipeline = ResearchPipeline(config)
    except ValueError as e:
        print(f"\n❌ Initialization Error:\n{e}")
        return
    
    # Load existing index
    if Path(args.index_dir).exists():
        try:
            pipeline.load_index()
            logger.info(f"Loaded {pipeline.index_size} chunks from index")
        except Exception as e:
            logger.warning(f"No existing index: {e}")
    
    # Execute
    if args.query:
        result = pipeline.query(args.query)
        print(result.response)
        print(f"\n[Backend: {result.generation_metadata.get('backend', 'unknown')}]")
    else:
        interactive_mode(pipeline)


if __name__ == "__main__":
    main()
