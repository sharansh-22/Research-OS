"""
Research Pipeline - Full RAG orchestration with Groq API
=========================================================
Features:
- Incremental indexing with processed files ledger
- Skip already-processed PDFs
- Force reprocessing option
- Persistent JSON ledger
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Set
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


# =============================================================================
# CONFIGURATION
# =============================================================================

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
    
    # Ledger settings
    ledger_filename: str = "processed_files.json"
    use_file_hash: bool = True  # Detect file changes via MD5
    
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
    
    @property
    def ledger_path(self) -> Path:
        """Full path to the processed files ledger."""
        return Path(self.index_dir) / self.ledger_filename


# =============================================================================
# QUERY RESULT
# =============================================================================

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


# =============================================================================
# INGESTION RESULT
# =============================================================================

@dataclass
class IngestionResult:
    """Result from ingesting a file."""
    filename: str
    status: str  # 'processed', 'skipped', 'failed'
    chunks_added: int
    message: str
    processing_time: float = 0.0


# =============================================================================
# RESEARCH PIPELINE
# =============================================================================

class ResearchPipeline:
    """
    Complete RAG pipeline with Groq API and incremental indexing.
    
    Features:
    - Incremental indexing (skip already-processed files)
    - File hash detection (reprocess if file changed)
    - Persistent JSON ledger
    - Force reprocessing option
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        logger.info("Initializing Research Pipeline...")
        
        # Ensure index directory exists
        Path(self.config.index_dir).mkdir(parents=True, exist_ok=True)
        
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
        
        # Load processed files ledger
        self._processed_files: Dict[str, Dict] = {}
        self.load_processed_log()
        
        logger.info("✓ Pipeline ready (Groq API + Incremental Indexing)")
    
    # =========================================================================
    # LEDGER MANAGEMENT
    # =========================================================================
    
    def load_processed_log(self) -> Dict[str, Dict]:
        """
        Load the processed files ledger from disk.
        
        Returns:
            Dict mapping filename -> metadata
        """
        ledger_path = self.config.ledger_path
        
        if ledger_path.exists():
            try:
                with open(ledger_path, 'r') as f:
                    data = json.load(f)
                    self._processed_files = data.get("files", {})
                    logger.info(f"Loaded ledger: {len(self._processed_files)} files tracked")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load ledger: {e}. Starting fresh.")
                self._processed_files = {}
        else:
            logger.info("No existing ledger found. Starting fresh.")
            self._processed_files = {}
        
        return self._processed_files
    
    def save_processed_log(self) -> None:
        """Save the processed files ledger to disk."""
        ledger_path = self.config.ledger_path
        
        # Ensure directory exists
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "total_files": len(self._processed_files),
            "files": self._processed_files,
        }
        
        try:
            with open(ledger_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Ledger saved: {len(self._processed_files)} files")
        except IOError as e:
            logger.error(f"Failed to save ledger: {e}")
    
    def is_processed(self, pdf_path: str | Path) -> bool:
        """
        Check if a file has already been processed.
        
        Also checks file hash if use_file_hash is enabled.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if file was processed and unchanged
        """
        pdf_path = Path(pdf_path)
        filename = pdf_path.name
        
        if filename not in self._processed_files:
            return False
        
        # If hash checking is enabled, verify file hasn't changed
        if self.config.use_file_hash:
            stored_hash = self._processed_files[filename].get("file_hash")
            current_hash = self._compute_file_hash(pdf_path)
            
            if stored_hash != current_hash:
                logger.info(f"File changed (hash mismatch): {filename}")
                return False
        
        return True
    
    def mark_processed(
        self,
        pdf_path: str | Path,
        chunks_added: int,
    ) -> None:
        """
        Mark a file as processed in the ledger.
        
        Args:
            pdf_path: Path to processed PDF
            chunks_added: Number of chunks added to index
        """
        pdf_path = Path(pdf_path)
        
        self._processed_files[pdf_path.name] = {
            "path": str(pdf_path.absolute()),
            "processed_at": datetime.now().isoformat(),
            "chunks_added": chunks_added,
            "file_hash": self._compute_file_hash(pdf_path) if self.config.use_file_hash else None,
            "file_size": pdf_path.stat().st_size,
        }
        
        # Save immediately for durability
        self.save_processed_log()
    
    def clear_processed_log(self) -> None:
        """Clear the processed files ledger."""
        self._processed_files = {}
        self.save_processed_log()
        logger.info("Ledger cleared")
    
    def get_processed_files(self) -> List[str]:
        """Get list of processed filenames."""
        return list(self._processed_files.keys())
    
    def _compute_file_hash(self, pdf_path: Path) -> str:
        """Compute MD5 hash of file for change detection."""
        try:
            hasher = hashlib.md5()
            with open(pdf_path, 'rb') as f:
                # Read in chunks for large files
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except IOError:
            return ""
    
    # =========================================================================
    # INGESTION
    # =========================================================================
    
    def ingest_pdf(
        self,
        pdf_path: str | Path,
        force: bool = False,
    ) -> IngestionResult:
        """
        Ingest a single PDF with incremental indexing support.
        
        Args:
            pdf_path: Path to PDF file
            force: If True, reprocess even if already in ledger
            
        Returns:
            IngestionResult with status and details
        """
        import time
        start_time = time.time()
        
        pdf_path = Path(pdf_path)
        filename = pdf_path.name
        
        # Check if file exists
        if not pdf_path.exists():
            return IngestionResult(
                filename=filename,
                status="failed",
                chunks_added=0,
                message=f"File not found: {pdf_path}",
            )
        
        # Check if already processed (skip if not forced)
        if not force and self.is_processed(pdf_path):
            return IngestionResult(
                filename=filename,
                status="skipped",
                chunks_added=0,
                message="Already processed (in ledger)",
            )
        
        # Process the file
        try:
            chunks = self.loader.load_pdf(pdf_path)
            
            if not chunks:
                return IngestionResult(
                    filename=filename,
                    status="failed",
                    chunks_added=0,
                    message="No chunks extracted",
                    processing_time=time.time() - start_time,
                )
            
            # Add to retriever
            self.retriever.add_chunks(chunks)
            
            # Mark as processed
            self.mark_processed(pdf_path, len(chunks))
            
            processing_time = time.time() - start_time
            
            return IngestionResult(
                filename=filename,
                status="processed",
                chunks_added=len(chunks),
                message=f"Added {len(chunks)} chunks",
                processing_time=processing_time,
            )
            
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            return IngestionResult(
                filename=filename,
                status="failed",
                chunks_added=0,
                message=str(e),
                processing_time=time.time() - start_time,
            )
    
    def ingest_directory(
        self,
        dir_path: str | Path,
        recursive: bool = True,
        force: bool = False,
    ) -> List[IngestionResult]:
        """
        Ingest all PDFs from directory with incremental indexing.
        
        Args:
            dir_path: Directory path
            recursive: Search subdirectories
            force: Reprocess all files
            
        Returns:
            List of IngestionResult for each file
        """
        dir_path = Path(dir_path)
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = sorted(dir_path.glob(pattern))
        
        results = []
        for pdf_path in pdf_files:
            result = self.ingest_pdf(pdf_path, force=force)
            results.append(result)
        
        return results
    
    # =========================================================================
    # QUERYING
    # =========================================================================
    
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
    
    # =========================================================================
    # INDEX MANAGEMENT
    # =========================================================================
    
    def save_index(self, path: Optional[str] = None) -> None:
        """Save index to disk."""
        save_path = path or self.config.index_dir
        self.retriever.save(save_path)
        self.save_processed_log()  # Also save ledger
        logger.info(f"Index saved to {save_path}")
    
    def load_index(self, path: Optional[str] = None) -> None:
        """Load index from disk."""
        load_path = path or self.config.index_dir
        self.retriever = HybridRetriever.load(load_path, self.embedder)
        self.load_processed_log()  # Also load ledger
        logger.info(f"Index loaded from {load_path}")
    
    def rebuild_index(self, data_dirs: List[str | Path]) -> Dict[str, Any]:
        """
        Rebuild index from scratch (clears existing).
        
        Args:
            data_dirs: List of directories to process
            
        Returns:
            Summary statistics
        """
        # Clear everything
        self.retriever.clear()
        self.clear_processed_log()
        
        # Process all directories
        all_results = []
        for dir_path in data_dirs:
            results = self.ingest_directory(dir_path, force=True)
            all_results.extend(results)
        
        # Save
        self.save_index()
        
        # Summary
        processed = sum(1 for r in all_results if r.status == "processed")
        failed = sum(1 for r in all_results if r.status == "failed")
        total_chunks = sum(r.chunks_added for r in all_results)
        
        return {
            "total_files": len(all_results),
            "processed": processed,
            "failed": failed,
            "total_chunks": total_chunks,
        }
    
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
            "processed_files": len(self._processed_files),
            "backends": backend_status,
            "config": self.config.to_dict(),
        }
    
    def clear_index(self) -> None:
        """Clear the index and ledger."""
        self.retriever.clear()
        self.clear_processed_log()
        logger.info("Index and ledger cleared")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_pipeline(
    index_dir: str = "data/index",
    load_existing: bool = True,
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
