#!/usr/bin/env python3
"""Research-OS: Main Entry Point"""

import argparse
import logging
from pathlib import Path

from src.rag import ResearchPipeline, PipelineConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)7s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("research-os")


def interactive_mode(pipeline: ResearchPipeline):
    print("\n" + "=" * 60)
    print("  RESEARCH-OS: Interactive Mode")
    print("  Type 'quit' to exit, 'stats' for info")
    print("=" * 60 + "\n")
    
    while True:
        try:
            q = input("\n🔬 Query: ").strip()
            if not q:
                continue
            if q.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            if q.lower() == 'stats':
                s = pipeline.get_stats()
                print(f"\n📊 {s['total_chunks']} chunks | {s['chunk_types']}")
                continue
            
            print("\n⏳ Processing...\n")
            result = pipeline.query(q)
            print("=" * 60)
            print(result.response)
            print("-" * 60)
            print(f"📎 {len(result.theory_context)} theory, {len(result.code_context)} code chunks")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Research-OS")
    parser.add_argument("--query", "-q", help="Single query")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    args = parser.parse_args()
    
    config = PipelineConfig(index_dir=args.index_dir, model=args.model)
    pipeline = ResearchPipeline(config)
    
    if Path(args.index_dir).exists():
        try:
            pipeline.load_index()
            logger.info(f"Loaded {pipeline.index_size} chunks")
        except Exception as e:
            logger.warning(f"No index: {e}")
    
    if args.query:
        print(pipeline.query(args.query).response)
    else:
        interactive_mode(pipeline)


if __name__ == "__main__":
    main()
