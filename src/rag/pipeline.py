"""
Research Pipeline - Full RAG Orchestration with Smart Routing
==============================================================
Features:
- Smart Query Router (Intent Classification)
- Pre-filtering based on query intent (code/theory/hybrid)
- Incremental indexing with processed files ledger
- Hybrid search with reranking
- Groq API generation with streaming
- Short-term conversation memory
- Code verification
"""

import os
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Generator, Tuple
from dataclasses import dataclass
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
# INTENT CLASSIFICATION KEYWORDS
# =============================================================================

# Keywords indicating CODE intent (case-insensitive)
CODE_KEYWORDS = [
    r'\bcode\b',
    r'\bpython\b',
    r'\bimplementation\b',
    r'\bimplement\b',
    r'\bimpl\b',
    r'\bclass\b',
    r'\bdef\b',
    r'\bfunction\b',
    r'\bfunctions\b',
    r'\bapi\b',
    r'\bscript\b',
    r'\bnotebook\b',
    r'\bpytorch\b',
    r'\btensorflow\b',
    r'\bnumpy\b',
    r'\btorch\b',
    r'\bkeras\b',
    r'\bjax\b',
    r'\bscipy\b',
    r'\bsklearn\b',
    r'\bpandas\b',
    r'\bmodule\b',
    r'\blibrary\b',
    r'\bpackage\b',
    r'\bsnippet\b',
    r'\bexample code\b',
    r'\bcode example\b',
    r'\bhow to code\b',
    r'\bwrite code\b',
    r'\bshow me code\b',
    r'\bgive me code\b',
    r'\bsyntax\b',
    r'\bmethod\b',
    r'\bcuda\b',
    r'\bgpu\b',
    r'\btensor\b',
    r'\barray\b',
    r'\bmatrix operations\b',
    r'\bdebug\b',
    r'\berror\b',
    r'\bbug\b',
    r'\bfix\b',
]

# Keywords indicating THEORY intent (case-insensitive)
THEORY_KEYWORDS = [
    r'\bmath\b',
    r'\bmathematics\b',
    r'\bmathematical\b',
    r'\btheory\b',
    r'\btheoretical\b',
    r'\bproof\b',
    r'\bproofs\b',
    r'\bprove\b',
    r'\bequation\b',
    r'\bequations\b',
    r'\bderive\b',
    r'\bderivation\b',
    r'\bderivations\b',
    r'\bformula\b',
    r'\bformulas\b',
    r'\bformulae\b',
    r'\btheorem\b',
    r'\btheorems\b',
    r'\blemma\b',
    r'\bdefinition\b',
    r'\bdefinitions\b',
    r'\bdefine\b',
    r'\bconcept\b',
    r'\bconcepts\b',
    r'\bconceptual\b',
    r'\bwhy does\b',
    r'\bwhy is\b',
    r'\bwhy do\b',
    r'\bhow does\b',
    r'\bhow is\b',
    r'\bexplain\b',
    r'\bexplanation\b',
    r'\bunderstand\b',
    r'\bunderstanding\b',
    r'\bintuition\b',
    r'\bintuitively\b',
    r'\bfundamental\b',
    r'\bfundamentals\b',
    r'\bprinciple\b',
    r'\bprinciples\b',
    r'\baxiom\b',
    r'\bcorollary\b',
    r'\bproposition\b',
    r'\bnotation\b',
    r'\bsymbols\b',
    r'\blatex\b',
    r'\balgebra\b',
    r'\bcalculus\b',
    r'\bprobability\b',
    r'\bstatistics\b',
    r'\blinear algebra\b',
    r'\bgradient\b',
    r'\bloss function\b',
    r'\bobjective function\b',
    r'\boptimization theory\b',
    r'\bconvergence\b',
]

# Compile patterns for efficiency
CODE_PATTERN = re.compile('|'.join(CODE_KEYWORDS), re.IGNORECASE)
THEORY_PATTERN = re.compile('|'.join(THEORY_KEYWORDS), re.IGNORECASE)


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
    
    # Memory settings
    max_history_turns: int = 3
    
    # Smart routing settings
    enable_smart_routing: bool = True  # Enable/disable intent classification
    
    # Verification settings
    verify_code: bool = True
    verification_timeout: int = 10
    
    # Ledger settings
    ledger_filename: str = "processed_files.json"
    use_file_hash: bool = True
    
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
            "max_history_turns": self.max_history_turns,
            "enable_smart_routing": self.enable_smart_routing,
            "verify_code": self.verify_code,
            "verification_timeout": self.verification_timeout,
        }
    
    @property
    def ledger_path(self) -> Path:
        return Path(self.index_dir) / self.ledger_filename


# =============================================================================
# RESULT CLASSES
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
    intent: str = "hybrid"  # Detected intent
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "response": self.response,
            "intent": self.intent,
            "context": {
                "code": self.code_context,
                "theory": self.theory_context,
            },
            "verification": self.verification_results,
            "metadata": self.generation_metadata,
        }


@dataclass
class IngestionResult:
    """Result from ingesting a file."""
    filename: str
    status: str
    chunks_added: int
    message: str
    processing_time: float = 0.0


# =============================================================================
# RESEARCH PIPELINE WITH SMART ROUTING
# =============================================================================

class ResearchPipeline:
    """
    Complete RAG pipeline with Smart Query Routing.
    
    Features:
    - Intent classification (code/theory/hybrid)
    - Pre-filtering based on query intent
    - Hybrid search with reranking
    - Streaming generation
    - Short-term conversation memory
    - Incremental indexing
    - Code verification
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
        
        # Initialize generator
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
        
        logger.info("✓ Pipeline ready (Smart Routing enabled)")
    
    # =========================================================================
    # INTENT CLASSIFICATION (SMART QUERY ROUTER)
    # =========================================================================
    
    def _classify_intent(self, query: str) -> Tuple[str, Dict[str, bool]]:
        """
        Classify query intent using keyword matching.
        
        Args:
            query: User's query string
            
        Returns:
            Tuple of (intent, debug_info)
            - intent: "code", "theory", or "hybrid"
            - debug_info: Dict with matched keywords info
        """
        # Check for code keywords
        code_matches = CODE_PATTERN.findall(query)
        has_code_intent = len(code_matches) > 0
        
        # Check for theory keywords
        theory_matches = THEORY_PATTERN.findall(query)
        has_theory_intent = len(theory_matches) > 0
        
        # Build debug info
        debug_info = {
            "code_matches": code_matches,
            "theory_matches": theory_matches,
            "code_count": len(code_matches),
            "theory_count": len(theory_matches),
        }
        
        # Classification logic
        if has_code_intent and has_theory_intent:
            # Ambiguous - both types of keywords present
            # Use the one with more matches, or default to hybrid
            if len(code_matches) > len(theory_matches) * 1.5:
                intent = "code"
            elif len(theory_matches) > len(code_matches) * 1.5:
                intent = "theory"
            else:
                intent = "hybrid"  # Safety first
        elif has_code_intent:
            intent = "code"
        elif has_theory_intent:
            intent = "theory"
        else:
            intent = "hybrid"  # Default for unclassified queries
        
        debug_info["intent"] = intent
        
        logger.debug(
            f"Intent classified: '{query[:50]}...' → {intent} "
            f"(code:{len(code_matches)}, theory:{len(theory_matches)})"
        )
        
        return intent, debug_info
    
    def classify_intent(self, query: str) -> str:
        """
        Public method to classify query intent.
        
        Args:
            query: User's query string
            
        Returns:
            Intent string: "code", "theory", or "hybrid"
        """
        intent, _ = self._classify_intent(query)
        return intent
    
    # =========================================================================
    # QUERY WITH SMART ROUTING
    # =========================================================================
    
    def query(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        verify: Optional[bool] = None,
        filter_type: Optional[str] = None,  # Override auto-classification
    ) -> QueryResult:
        """
        Execute a RAG query with smart routing.
        
        Args:
            question: User's question
            history: Conversation history
            verify: Override code verification setting
            filter_type: Override intent classification ("code", "theory", "hybrid")
            
        Returns:
            QueryResult with response and metadata
        """
        history = history or []
        do_verify = verify if verify is not None else self.config.verify_code
        
        # Classify intent (or use override)
        if filter_type:
            intent = filter_type
        elif self.config.enable_smart_routing:
            intent, _ = self._classify_intent(question)
        else:
            intent = "hybrid"
        
        logger.info(f"Query intent: {intent}")
        
        # Retrieve with intent-based filtering
        results = self.retriever.search_by_type_filtered(
            query=question,
            top_k=self.config.top_k,
            filter_type=intent,
        )
        code_results = results.get("code", [])
        theory_results = results.get("theory", [])
        
        # Generate with history
        gen_result = self.generator.generate(
            query=question,
            code_results=code_results,
            theory_results=theory_results,
            history=history,
        )
        
        # Verify code blocks (if present)
        verifications = []
        if do_verify and "```" in gen_result.response:
            verifications = [
                v.to_dict() 
                for v in self.verifier.verify_generated_response(gen_result.response)
            ]
        
        return QueryResult(
            query=question,
            response=gen_result.response,
            code_context=[r.to_dict() for r in code_results],
            theory_context=[r.to_dict() for r in theory_results],
            verification_results=verifications,
            generation_metadata=gen_result.to_dict(),
            intent=intent,
        )
    
    def query_stream(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        filter_type: Optional[str] = None,
    ) -> Generator[str, None, Dict]:
        """
        Execute a streaming RAG query with smart routing.
        
        Args:
            question: User's question
            history: Conversation history
            filter_type: Override intent classification
            
        Yields:
            str: Response chunks
        """
        history = history or []
        
        # Classify intent
        if filter_type:
            intent = filter_type
        elif self.config.enable_smart_routing:
            intent, _ = self._classify_intent(question)
        else:
            intent = "hybrid"
        
        # Retrieve with filtering
        results = self.retriever.search_by_type_filtered(
            query=question,
            top_k=self.config.top_k,
            filter_type=intent,
        )
        code_results = results.get("code", [])
        theory_results = results.get("theory", [])
        
        # Stream generation with history
        yield from self.generator.generate_stream(
            query=question,
            code_results=code_results,
            theory_results=theory_results,
            history=history,
        )
    
    # =========================================================================
    # LEDGER MANAGEMENT
    # =========================================================================
    
    def load_processed_log(self) -> Dict[str, Dict]:
        """Load the processed files ledger from disk."""
        ledger_path = self.config.ledger_path
        
        if ledger_path.exists():
            try:
                with open(ledger_path, 'r') as f:
                    data = json.load(f)
                    self._processed_files = data.get("files", {})
                    logger.info(f"Loaded ledger: {len(self._processed_files)} files")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load ledger: {e}")
                self._processed_files = {}
        else:
            self._processed_files = {}
        
        return self._processed_files
    
    def save_processed_log(self) -> None:
        """Save the processed files ledger to disk."""
        ledger_path = self.config.ledger_path
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
        except IOError as e:
            logger.error(f"Failed to save ledger: {e}")
    
    def is_processed(self, pdf_path: str | Path) -> bool:
        """Check if a file has already been processed."""
        pdf_path = Path(pdf_path)
        filename = pdf_path.name
        
        if filename not in self._processed_files:
            return False
        
        if self.config.use_file_hash:
            stored_hash = self._processed_files[filename].get("file_hash")
            current_hash = self._compute_file_hash(pdf_path)
            if stored_hash != current_hash:
                return False
        
        return True
    
    def mark_processed(self, pdf_path: str | Path, chunks_added: int) -> None:
        """Mark a file as processed in the ledger."""
        pdf_path = Path(pdf_path)
        
        self._processed_files[pdf_path.name] = {
            "path": str(pdf_path.absolute()),
            "processed_at": datetime.now().isoformat(),
            "chunks_added": chunks_added,
            "file_hash": self._compute_file_hash(pdf_path) if self.config.use_file_hash else None,
            "file_size": pdf_path.stat().st_size,
        }
        
        self.save_processed_log()
    
    def clear_processed_log(self) -> None:
        """Clear the processed files ledger."""
        self._processed_files = {}
        self.save_processed_log()
    
    def get_processed_files(self) -> List[str]:
        """Get list of processed filenames."""
        return list(self._processed_files.keys())
    
    def _compute_file_hash(self, pdf_path: Path) -> str:
        """Compute MD5 hash of file."""
        try:
            hasher = hashlib.md5()
            with open(pdf_path, 'rb') as f:
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
        """Ingest a single PDF with incremental indexing."""
        import time
        start_time = time.time()
        
        pdf_path = Path(pdf_path)
        filename = pdf_path.name
        
        if not pdf_path.exists():
            return IngestionResult(filename, "failed", 0, f"File not found: {pdf_path}")
        
        if not force and self.is_processed(pdf_path):
            return IngestionResult(filename, "skipped", 0, "Already processed")
        
        try:
            chunks = self.loader.load_pdf(pdf_path)
            
            if not chunks:
                return IngestionResult(
                    filename, "failed", 0, "No chunks extracted",
                    processing_time=time.time() - start_time
                )
            
            self.retriever.add_chunks(chunks)
            self.mark_processed(pdf_path, len(chunks))
            
            return IngestionResult(
                filename, "processed", len(chunks),
                f"Added {len(chunks)} chunks",
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            return IngestionResult(
                filename, "failed", 0, str(e),
                processing_time=time.time() - start_time
            )
    
    def ingest_directory(
        self,
        dir_path: str | Path,
        recursive: bool = True,
        force: bool = False,
    ) -> List[IngestionResult]:
        """Ingest all files from directory."""
        dir_path = Path(dir_path)
        pattern = "**/*" if recursive else "*"
        
        from .data_loader import UniversalLoader
        extensions = UniversalLoader.supported_extensions()
        
        all_files = []
        for ext in extensions:
            all_files.extend(dir_path.glob(f"{pattern}{ext}"))
        
        results = []
        for file_path in sorted(set(all_files)):
            result = self.ingest_pdf(file_path, force=force)
            results.append(result)
        
        return results
    
    # =========================================================================
    # INDEX MANAGEMENT
    # =========================================================================
    
    def save_index(self, path: Optional[str] = None) -> None:
        """Save index to disk."""
        save_path = path or self.config.index_dir
        self.retriever.save(save_path)
        self.save_processed_log()
        logger.info(f"Index saved to {save_path}")
    
    def load_index(self, path: Optional[str] = None) -> None:
        """Load index from disk."""
        load_path = path or self.config.index_dir
        self.retriever = HybridRetriever.load(load_path, self.embedder)
        self.load_processed_log()
        logger.info(f"Index loaded from {load_path}")
    
    def rebuild_index(self, data_dirs: List[str | Path]) -> Dict[str, Any]:
        """Rebuild index from scratch."""
        self.retriever.clear()
        self.clear_processed_log()
        
        all_results = []
        for dir_path in data_dirs:
            results = self.ingest_directory(dir_path, force=True)
            all_results.extend(results)
        
        self.save_index()
        
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
        
        backend_status = self.generator.health_check()
        
        return {
            "total_chunks": self.index_size,
            "chunk_types": types,
            "processed_files": len(self._processed_files),
            "backends": backend_status,
            "smart_routing": self.config.enable_smart_routing,
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
