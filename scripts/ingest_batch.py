#!/usr/bin/env python3
"""Batch PDF Ingestion"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import ResearchDocumentLoader, HybridRetriever, get_embedder

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("ingest")

DATA_DIRS = ["data/01_fundamentals", "data/02_papers", "data/03_implementation"]
INDEX_DIR = "data/index"


def main():
    logger.info("=" * 50)
    logger.info("Research-OS Batch Ingestion")
    logger.info("=" * 50)
    
    pdfs = []
    for d in DATA_DIRS:
        p = Path(d)
        if p.exists():
            pdfs.extend(p.glob("**/*.pdf"))
    
    if not pdfs:
        logger.error("No PDFs found!")
        return
    
    logger.info(f"Found {len(pdfs)} PDFs")
    
    embedder = get_embedder()
    loader = ResearchDocumentLoader()
    retriever = HybridRetriever(embedder=embedder)
    
    total = 0
    for i, pdf in enumerate(pdfs, 1):
        logger.info(f"[{i}/{len(pdfs)}] {pdf.name}")
        try:
            chunks = loader.load_pdf(pdf)
            retriever.add_chunks(chunks)
            total += len(chunks)
        except Exception as e:
            logger.error(f"  Failed: {e}")
    
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    retriever.save(INDEX_DIR)
    
    logger.info("=" * 50)
    logger.info(f"Done! {total} chunks → {INDEX_DIR}")


if __name__ == "__main__":
    main()
