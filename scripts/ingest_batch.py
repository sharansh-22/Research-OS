#!/usr/bin/env python3
"""
Batch Ingestion with Universal File Support
============================================
Supports: PDF, Python, Jupyter, Markdown, LaTeX, C++, CUDA

Usage:
    python scripts/ingest_batch.py              # Incremental
    python scripts/ingest_batch.py --force      # Reprocess all
    python scripts/ingest_batch.py --rebuild    # Full rebuild
    python scripts/ingest_batch.py --status     # Show status
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import ResearchPipeline, PipelineConfig, UniversalLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ingest")

logging.getLogger("faiss.loader").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


DATA_DIRS = [
    "data/01_fundamentals",
    "data/02_papers",
    "data/03_implementation",
    "data/04_misc",
]

INDEX_DIR = "data/index"

# All supported extensions
SUPPORTED_EXTENSIONS = UniversalLoader.supported_extensions()


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║         RESEARCH-OS: Universal Batch Ingestion               ║
║   Supports: PDF, Python, Jupyter, Markdown, LaTeX, C++       ║
╚══════════════════════════════════════════════════════════════╝
    """)


def find_all_files(data_dirs: list, extensions: list) -> list:
    """Find all supported files in data directories."""
    all_files = []
    
    for dir_str in data_dirs:
        dir_path = Path(dir_str)
        if not dir_path.exists():
            continue
        
        for ext in extensions:
            files = list(dir_path.glob(f"**/*{ext}"))
            all_files.extend(files)
    
    return sorted(set(all_files))


def print_summary(results: list, start_time: datetime):
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    processed = [r for r in results if r.status == "processed"]
    skipped = [r for r in results if r.status == "skipped"]
    failed = [r for r in results if r.status == "failed"]
    
    total_chunks = sum(r.chunks_added for r in results)
    
    # Group by extension
    by_ext = {}
    for r in processed:
        ext = Path(r.filename).suffix
        by_ext[ext] = by_ext.get(ext, 0) + 1
    
    print("\n" + "=" * 60)
    print("                    INGESTION SUMMARY")
    print("=" * 60)
    print(f"  📁 Total files found:    {len(results)}")
    print(f"  ✓  Processed (new):      {len(processed)}")
    print(f"  ⏭  Skipped (existing):   {len(skipped)}")
    print(f"  ✗  Failed:               {len(failed)}")
    print(f"  📄 Chunks added:         {total_chunks}")
    print(f"  ⏱  Duration:             {duration:.1f}s")
    
    if by_ext:
        print(f"\n  📊 By file type:")
        for ext, count in sorted(by_ext.items()):
            print(f"     {ext}: {count}")
    
    print("=" * 60)
    
    if failed:
        print("\n⚠️  FAILED FILES:")
        for r in failed:
            print(f"   • {r.filename}: {r.message}")
    
    print("")


def show_status(pipeline: ResearchPipeline):
    stats = pipeline.get_stats()
    processed = pipeline.get_processed_files()
    
    print("\n" + "=" * 60)
    print("                    INDEX STATUS")
    print("=" * 60)
    print(f"  📊 Total chunks:         {stats['total_chunks']}")
    print(f"  📁 Files tracked:        {stats['processed_files']}")
    print(f"  📂 Chunk types:          {stats['chunk_types']}")
    print("=" * 60)
    
    if processed:
        # Group by extension
        by_ext = {}
        for f in processed:
            ext = Path(f).suffix
            by_ext.setdefault(ext, []).append(f)
        
        print("\n📋 PROCESSED FILES:")
        for ext in sorted(by_ext.keys()):
            print(f"\n  {ext} ({len(by_ext[ext])}):")
            for f in sorted(by_ext[ext])[:10]:
                print(f"    ✓ {f}")
            if len(by_ext[ext]) > 10:
                print(f"    ... and {len(by_ext[ext]) - 10} more")
    else:
        print("\n📋 No files processed yet.")
    
    print("")


def run_ingestion(
    data_dirs: list,
    index_dir: str,
    force: bool = False,
    rebuild: bool = False,
):
    print_banner()
    start_time = datetime.now()
    
    if rebuild:
        print("🔄 Mode: REBUILD (clearing existing index)")
    elif force:
        print("🔄 Mode: FORCE (reprocessing all files)")
    else:
        print("🔄 Mode: INCREMENTAL (skipping processed files)")
    
    print(f"📂 Index: {index_dir}")
    print(f"📁 Data dirs: {data_dirs}")
    print(f"📄 Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
    print("-" * 60)
    
    # Find all files
    all_files = find_all_files(data_dirs, SUPPORTED_EXTENSIONS)
    
    if not all_files:
        print("\n❌ No supported files found!")
        return
    
    # Count by type
    by_ext = {}
    for f in all_files:
        ext = f.suffix
        by_ext[ext] = by_ext.get(ext, 0) + 1
    
    print("\n📊 Files found:")
    for ext, count in sorted(by_ext.items()):
        print(f"   {ext}: {count}")
    print(f"\n   Total: {len(all_files)}")
    print("-" * 60)
    
    # Initialize pipeline
    print("\n⏳ Initializing pipeline...")
    config = PipelineConfig(index_dir=index_dir)
    pipeline = ResearchPipeline(config)
    
    if not rebuild and Path(index_dir).exists():
        try:
            pipeline.load_index()
            print(f"✓ Loaded existing index ({pipeline.index_size} chunks)")
        except Exception as e:
            print(f"⚠️ Could not load index: {e}")
    
    if rebuild:
        print("🗑️  Clearing existing index...")
        pipeline.clear_index()
    
    print("-" * 60)
    print("\n📥 PROCESSING FILES:\n")
    
    results = []
    for i, file_path in enumerate(all_files, 1):
        filename = file_path.name
        prefix = f"[{i:3d}/{len(all_files)}]"
        
        result = pipeline.ingest_pdf(file_path, force=force or rebuild)
        results.append(result)
        
        if result.status == "processed":
            print(f"{prefix} ✓ PROCESSED: {filename}")
            print(f"        → {result.chunks_added} chunks ({result.processing_time:.1f}s)")
        elif result.status == "skipped":
            print(f"{prefix} ⏭ SKIPPED:   {filename}")
        else:
            print(f"{prefix} ✗ FAILED:    {filename}")
            print(f"        → {result.message}")
    
    print("\n" + "-" * 60)
    print("💾 Saving index...")
    pipeline.save_index()
    print(f"✓ Index saved to {index_dir}")
    
    print_summary(results, start_time)


def main():
    parser = argparse.ArgumentParser(description="Research-OS Universal Batch Ingestion")
    parser.add_argument("--force", "-f", action="store_true", help="Force reprocess all")
    parser.add_argument("--rebuild", "-r", action="store_true", help="Clear and rebuild")
    parser.add_argument("--status", "-s", action="store_true", help="Show status")
    parser.add_argument("--data-dir", "-d", action="append", dest="data_dirs")
    parser.add_argument("--index-dir", "-i", default=INDEX_DIR)
    
    args = parser.parse_args()
    data_dirs = args.data_dirs if args.data_dirs else DATA_DIRS
    
    if args.status:
        config = PipelineConfig(index_dir=args.index_dir)
        pipeline = ResearchPipeline(config)
        if Path(args.index_dir).exists():
            try:
                pipeline.load_index()
            except:
                pass
        show_status(pipeline)
        return
    
    run_ingestion(
        data_dirs=data_dirs,
        index_dir=args.index_dir,
        force=args.force,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
