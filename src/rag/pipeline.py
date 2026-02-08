"""Research Pipeline - Full RAG orchestration with Groq API"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import logging

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .data_loader import ResearchDocumentLoader, Chunk, ChunkType
from .embedder import FastEmbedder, get_embedder
from .retriever import HybridRetriever, RetrievalResult
from .generator import ResearchArchitect, GenerationResult
from .verifier import ArchitectureVerifier

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for Research Pipeline."""
    # Index settings
    index_dir: str = "data/index"
    top_k: int = 5
    min_similarity: float = 0.25
    faiss_weight: float = 0.7
    bm25_weight: float = 0.3
    
    # Generation settings
    temperature: float = 0.3
    max_tokens: int = 2048
    enable_fallback: bool = True
    
    # Verification settings
    verify_code: bool = True
    verification_timeout: int = 10
    
    def to_dict(self) -> Dict:
        return {
            "index_dir": self.index_dir,
            "top_k": self.top_k,
            "min_similarity": self.min_similarity,
            "faiss_weight": self.faiss_weight,
            "bm25_weight": self.bm25_weight,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "enable_fallback": self.enable_fallback,
            "verify_code": self.verify_code,
            "verification_timeout": self.verification_timeout,
        }


@dataclass
class QueryResult:
    """Result from a query."""
    query: str
    response: str
    code_context: List[Dict]
    theory_context: List[Dict]
    verification_results: List[Dict]
    generation_metadata: Dict
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "response": self.response,
            "context": {
                "code": self.code_context,
                "theory": self.theory_context,
            },
            "verification": self.verification_results,
            "metadata": self.generation_metadata,
        }


class ResearchPipeline:
    """Complete RAG pipeline with Groq API."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        logger.info("Initializing Research Pipeline...")
        
        # Initialize embedder
        self.embedder = get_embedder()
        
        # Initialize document loader
        self.loader = ResearchDocumentLoader()
        
        # Initialize retriever
        self.retriever = HybridRetriever(
            embedder=self.embedder,
            faiss_weight=self.config.faiss_weight,
            bm25_weight=self.config.bm25_weight,
            min_similarity=self.config.min_similarity,
        )
        
        # Initialize generator (Groq API)
        self.generator = ResearchArchitect(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            enable_fallback=self.config.enable_fallback,
        )
        
        # Initialize verifier
        self.verifier = ArchitectureVerifier(
            timeout_seconds=self.config.verification_timeout
        )
        
        logger.info("✓ Pipeline ready (Groq API)")
    
    def ingest_pdf(self, pdf_path: str | Path) -> int:
        """Ingest a single PDF."""
        chunks = self.loader.load_pdf(pdf_path)
        self.retriever.add_chunks(chunks)
        return len(chunks)
    
    def ingest_directory(self, dir_path: str | Path, recursive: bool = True) -> int:
        """Ingest all PDFs from directory."""
        chunks = self.loader.load_directory(dir_path, recursive)
        self.retriever.add_chunks(chunks)
        return len(chunks)
    
    def query(self, question: str, verify: Optional[bool] = None) -> QueryResult:
        """Execute a RAG query."""
        do_verify = verify if verify is not None else self.config.verify_code
        
        # Retrieve
        results = self.retriever.search_by_type(question, self.config.top_k)
        code_res = results.get("code", [])
        theory_res = results.get("theory", [])
        
        # Generate
        gen = self.generator.generate(question, code_res, theory_res)
        
        # Verify code blocks (if present)
        verifs = []
        if do_verify and "```" in gen.response:
            verifs = [v.to_dict() for v in self.verifier.verify_generated_response(gen.response)]
        
        return QueryResult(
            query=question,
            response=gen.response,
            code_context=[r.to_dict() for r in code_res],
            theory_context=[r.to_dict() for r in theory_res],
            verification_results=verifs,
            generation_metadata=gen.to_dict(),
        )
    
    def search_only(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Search without generation."""
        k = top_k or self.config.top_k
        return self.retriever.search(query, top_k=k)
    
    def save_index(self, path: Optional[str] = None) -> None:
        """Save index to disk."""
        self.retriever.save(path or self.config.index_dir)
        logger.info(f"Index saved to {path or self.config.index_dir}")
    
    def load_index(self, path: Optional[str] = None) -> None:
        """Load index from disk."""
        load_path = path or self.config.index_dir
        self.retriever = HybridRetriever.load(load_path, self.embedder)
        logger.info(f"Index loaded from {load_path}")
    
    @property
    def index_size(self) -> int:
        """Number of chunks in index."""
        return self.retriever.size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        types = {}
        for c in self.retriever.chunks:
            t = c.chunk_type.value
            types[t] = types.get(t, 0) + 1
        
        # Backend status
        backend_status = self.generator.health_check()
        
        return {
            "total_chunks": self.index_size,
            "chunk_types": types,
            "backends": backend_status,
            "config": self.config.to_dict(),
        }
    
    def clear_index(self) -> None:
        """Clear the index."""
        self.retriever.clear()
        logger.info("Index cleared")


def create_pipeline(
    index_dir: str = "data/index",
    load_existing: bool = True
) -> ResearchPipeline:
    """Factory function to create pipeline."""
    config = PipelineConfig(index_dir=index_dir)
    pipeline = ResearchPipeline(config)
    
    if load_existing and Path(index_dir).exists():
        try:
            pipeline.load_index()
            logger.info(f"Loaded {pipeline.index_size} chunks")
        except Exception as e:
            logger.warning(f"Could not load index: {e}")
    
    return pipeline
