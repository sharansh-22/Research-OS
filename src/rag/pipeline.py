"""
Research Pipeline - Full RAG Orchestration with API Support
============================================================
Features:
- Smart Query Router (Intent Classification)
- Pre-filtering based on query intent
- Streaming generation (yields JSON-serializable chunks)
- Short-term conversation memory
- Incremental indexing
- Code verification
- Structured source citations
"""

import os
import re
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any, Generator, Tuple, Union
from dataclasses import dataclass
import logging

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

CODE_KEYWORDS = [
    r'\bcode\b', r'\bpython\b', r'\bimplementation\b', r'\bimplement\b',
    r'\bimpl\b', r'\bclass\b', r'\bdef\b', r'\bfunction\b', r'\bfunctions\b',
    r'\bapi\b', r'\bscript\b', r'\bnotebook\b', r'\bpytorch\b', r'\btensorflow\b',
    r'\bnumpy\b', r'\btorch\b', r'\bkeras\b', r'\bjax\b', r'\bscipy\b',
    r'\bsklearn\b', r'\bpandas\b', r'\bmodule\b', r'\blibrary\b', r'\bpackage\b',
    r'\bsnippet\b', r'\bexample code\b', r'\bcode example\b', r'\bhow to code\b',
    r'\bwrite code\b', r'\bshow me code\b', r'\bgive me code\b', r'\bsyntax\b',
    r'\bmethod\b', r'\bcuda\b', r'\bgpu\b', r'\btensor\b', r'\barray\b',
    r'\bdebug\b', r'\berror\b', r'\bbug\b', r'\bfix\b',
]

THEORY_KEYWORDS = [
    r'\bmath\b', r'\bmathematics\b', r'\bmathematical\b', r'\btheory\b',
    r'\btheoretical\b', r'\bproof\b', r'\bproofs\b', r'\bprove\b',
    r'\bequation\b', r'\bequations\b', r'\bderive\b', r'\bderivation\b',
    r'\bformula\b', r'\bformulas\b', r'\btheorem\b', r'\btheorems\b',
    r'\blemma\b', r'\bdefinition\b', r'\bdefine\b', r'\bconcept\b',
    r'\bconcepts\b', r'\bwhy does\b', r'\bwhy is\b', r'\bhow does\b',
    r'\bexplain\b', r'\bexplanation\b', r'\bunderstand\b', r'\bintuition\b',
    r'\bfundamental\b', r'\bprinciple\b', r'\baxiom\b', r'\bcorollary\b',
    r'\bnotation\b', r'\blatex\b', r'\balgebra\b', r'\bcalculus\b',
    r'\bprobability\b', r'\bstatistics\b', r'\bgradient\b', r'\bconvergence\b',
]

CODE_PATTERN = re.compile('|'.join(CODE_KEYWORDS), re.IGNORECASE)
THEORY_PATTERN = re.compile('|'.join(THEORY_KEYWORDS), re.IGNORECASE)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for Research Pipeline."""
    index_dir: str = "data/index"
    top_k: int = 5
    min_similarity: float = 0.25
    faiss_weight: float = 0.7
    bm25_weight: float = 0.3
    temperature: float = 0.3
    max_tokens: int = 2048
    enable_fallback: bool = True
    max_history_turns: int = 3
    enable_smart_routing: bool = True
    verify_code: bool = True
    verification_timeout: int = 10
    ledger_filename: str = "processed_files.json"
    use_file_hash: bool = True

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

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
    intent: str = "hybrid"
    sources: List[Dict] = None

    def __post_init__(self):
        if self.sources is None:
            self.sources = []

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "response": self.response,
            "intent": self.intent,
            "context": {"code": self.code_context, "theory": self.theory_context},
            "verification": self.verification_results,
            "metadata": self.generation_metadata,
            "sources": self.sources,
        }


@dataclass
class IngestionResult:
    """Result from ingesting a file."""
    filename: str
    status: str
    chunks_added: int
    message: str
    processing_time: float = 0.0


@dataclass
class StreamChunk:
    """A single chunk from streaming response."""
    event: str  # 'start', 'chunk', 'context', 'sources', 'done', 'error'
    data: Optional[str] = None
    intent: Optional[str] = None
    code_count: Optional[int] = None
    theory_count: Optional[int] = None
    sources: Optional[List[Dict]] = None
    error: Optional[str] = None

    def to_json(self) -> str:
        d = {"event": self.event}
        if self.data is not None:
            d["data"] = self.data
        if self.intent is not None:
            d["intent"] = self.intent
        if self.code_count is not None:
            d["code"] = self.code_count
        if self.theory_count is not None:
            d["theory"] = self.theory_count
        if self.sources is not None:
            d["sources"] = self.sources
        if self.error is not None:
            d["error"] = self.error
        return json.dumps(d)


# =============================================================================
# SOURCE EXTRACTION HELPER
# =============================================================================

def _extract_sources(
    code_results: List[RetrievalResult],
    theory_results: List[RetrievalResult],
) -> List[Dict]:
    """
    Build a deduplicated, structured list of source citations
    from retrieval results.

    Each source dict contains:
        - source: filename
        - type: "code" or "theory"
        - section: section title (if available)
        - score: relevance score
        - chunk_id: unique chunk identifier

    Deduplication is by (source filename + section).
    """
    seen = set()
    sources = []

    all_results = []
    for r in theory_results:
        all_results.append((r, "theory"))
    for r in code_results:
        all_results.append((r, "code"))

    # Sort by score descending so highest-relevance entry wins dedup
    all_results.sort(key=lambda x: x[0].score, reverse=True)

    for result, result_type in all_results:
        meta = result.chunk.metadata
        filename = meta.get("source", "unknown")
        section = meta.get("section", "")
        source_path = meta.get("source_path", "")

        # Dedup key: filename + section
        dedup_key = f"{filename}|{section}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        source_entry = {
            "source": filename,
            "type": result_type,
            "section": section if section else None,
            "score": round(result.score, 4),
            "chunk_id": result.chunk.chunk_id,
        }

        # Include source_path if available
        if source_path:
            source_entry["source_path"] = source_path

        sources.append(source_entry)

    return sources


# =============================================================================
# HALLUCINATION GUARD
# =============================================================================

_HALLUCINATED_SOURCE_PATTERNS = re.compile(
    r'(?:^|\n)(?:#{1,4}\s*)?(?:\*{0,2})'
    r'(?:sources?|references?|citations?|bibliography)'
    r'(?:\*{0,2})\s*:?\s*\n',
    re.IGNORECASE,
)


def _strip_hallucinated_sources(text: str) -> str:
    """
    Remove any LLM-hallucinated 'Sources' section from the end of a response.

    The LLM sometimes appends a fake sources/references block.
    We strip it so our real metadata-based citations are authoritative.
    """
    match = _HALLUCINATED_SOURCE_PATTERNS.search(text)
    if match:
        # Only strip if it appears in the last 30% of the response
        position_ratio = match.start() / len(text) if len(text) > 0 else 0
        if position_ratio > 0.7:
            stripped = text[:match.start()].rstrip()
            logger.debug(
                f"Stripped hallucinated sources section at position "
                f"{match.start()}/{len(text)} ({position_ratio:.0%})"
            )
            return stripped
    return text


# =============================================================================
# RESEARCH PIPELINE
# =============================================================================

class ResearchPipeline:
    """Complete RAG pipeline with Smart Query Routing and API support."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        logger.info("Initializing Research Pipeline...")
        Path(self.config.index_dir).mkdir(parents=True, exist_ok=True)

        self.embedder = get_embedder()
        self.loader = ResearchDocumentLoader()

        self.retriever = HybridRetriever(
            embedder=self.embedder,
            faiss_weight=self.config.faiss_weight,
            bm25_weight=self.config.bm25_weight,
            min_similarity=self.config.min_similarity,
        )

        self.generator = ResearchArchitect(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            enable_fallback=self.config.enable_fallback,
        )

        self.verifier = ArchitectureVerifier(
            timeout_seconds=self.config.verification_timeout
        )

        self._processed_files: Dict[str, Dict] = {}
        self.load_processed_log()

        logger.info("Pipeline ready (Smart Routing + API Support)")

    # =========================================================================
    # INTENT CLASSIFICATION
    # =========================================================================

    def _classify_intent(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """Classify query intent using keyword matching."""
        code_matches = CODE_PATTERN.findall(query)
        theory_matches = THEORY_PATTERN.findall(query)

        debug_info = {
            "code_matches": code_matches,
            "theory_matches": theory_matches,
            "code_count": len(code_matches),
            "theory_count": len(theory_matches),
        }

        has_code = len(code_matches) > 0
        has_theory = len(theory_matches) > 0

        if has_code and has_theory:
            if len(code_matches) > len(theory_matches) * 1.5:
                intent = "code"
            elif len(theory_matches) > len(code_matches) * 1.5:
                intent = "theory"
            else:
                intent = "hybrid"
        elif has_code:
            intent = "code"
        elif has_theory:
            intent = "theory"
        else:
            intent = "hybrid"

        debug_info["intent"] = intent
        return intent, debug_info

    def classify_intent(self, query: str) -> str:
        """Public method to classify query intent."""
        intent, _ = self._classify_intent(query)
        return intent

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def query(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        verify: Optional[bool] = None,
        filter_type: Optional[str] = None,
    ) -> QueryResult:
        """Execute a non-streaming RAG query with structured sources."""
        history = history or []
        do_verify = verify if verify is not None else self.config.verify_code

        # Classify intent
        if filter_type:
            intent = filter_type
        elif self.config.enable_smart_routing:
            intent, _ = self._classify_intent(question)
        else:
            intent = "hybrid"

        # Retrieve
        results = self.retriever.search_by_type_filtered(
            query=question,
            top_k=self.config.top_k,
            filter_type=intent,
        )
        code_results = results.get("code", [])
        theory_results = results.get("theory", [])

        # Generate
        gen_result = self.generator.generate(
            query=question,
            code_results=code_results,
            theory_results=theory_results,
            history=history,
        )

        # Strip hallucinated sources from LLM output
        clean_response = _strip_hallucinated_sources(gen_result.response)

        # Build structured sources from retrieval metadata
        sources = _extract_sources(code_results, theory_results)

        # Verify
        verifications = []
        if do_verify and "```" in clean_response:
            verifications = [
                v.to_dict()
                for v in self.verifier.verify_generated_response(clean_response)
            ]

        return QueryResult(
            query=question,
            response=clean_response,
            code_context=[r.to_dict() for r in code_results],
            theory_context=[r.to_dict() for r in theory_results],
            verification_results=verifications,
            generation_metadata=gen_result.to_dict(),
            intent=intent,
            sources=sources,
        )

    def query_stream(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        filter_type: Optional[str] = None,
        yield_json: bool = False,
    ) -> Generator[Union[str, StreamChunk], None, None]:
        """
        Execute a streaming RAG query with structured sources at the end.

        Event sequence:
            1. start   — intent classification result
            2. context — retrieval counts
            3. chunk   — LLM tokens (many)
            4. sources — structured citation metadata
            5. done    — stream complete

        Args:
            question: User's question
            history: Conversation history
            filter_type: Override intent classification
            yield_json: If True, yields JSON strings; else yields StreamChunk objects

        Yields:
            Either JSON strings or StreamChunk objects
        """
        history = history or []

        try:
            # Classify intent
            if filter_type:
                intent = filter_type
            elif self.config.enable_smart_routing:
                intent, _ = self._classify_intent(question)
            else:
                intent = "hybrid"

            # Yield start event
            start_chunk = StreamChunk(event="start", intent=intent)
            yield start_chunk.to_json() if yield_json else start_chunk

            # Retrieve
            results = self.retriever.search_by_type_filtered(
                query=question,
                top_k=self.config.top_k,
                filter_type=intent,
            )
            code_results = results.get("code", [])
            theory_results = results.get("theory", [])

            # Yield context event
            context_chunk = StreamChunk(
                event="context",
                code_count=len(code_results),
                theory_count=len(theory_results),
            )
            yield context_chunk.to_json() if yield_json else context_chunk

            # Stream generation
            full_response = ""
            for token in self.generator.generate_stream(
                query=question,
                code_results=code_results,
                theory_results=theory_results,
                history=history,
            ):
                full_response += token
                token_chunk = StreamChunk(event="chunk", data=token)
                yield token_chunk.to_json() if yield_json else token_chunk

            # Build structured sources from retrieval metadata
            sources = _extract_sources(code_results, theory_results)

            # Yield sources event (always, even if empty)
            sources_chunk = StreamChunk(event="sources", sources=sources)
            yield sources_chunk.to_json() if yield_json else sources_chunk

            # Yield done event
            done_chunk = StreamChunk(event="done")
            yield done_chunk.to_json() if yield_json else done_chunk

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_chunk = StreamChunk(event="error", error=str(e))
            yield error_chunk.to_json() if yield_json else error_chunk

    # =========================================================================
    # LEDGER MANAGEMENT
    # =========================================================================

    def load_processed_log(self) -> Dict[str, Dict]:
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
        self._processed_files = {}
        self.save_processed_log()

    def get_processed_files(self) -> List[str]:
        return list(self._processed_files.keys())

    def _compute_file_hash(self, pdf_path: Path) -> str:
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

    def ingest_pdf(self, pdf_path: str | Path, force: bool = False) -> IngestionResult:
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
        save_path = path or self.config.index_dir
        self.retriever.save(save_path)
        self.save_processed_log()
        logger.info(f"Index saved to {save_path}")

    def load_index(self, path: Optional[str] = None) -> None:
        load_path = path or self.config.index_dir
        self.retriever = HybridRetriever.load(load_path, self.embedder)
        self.load_processed_log()
        logger.info(f"Index loaded from {load_path}")

    def rebuild_index(self, data_dirs: List[str | Path]) -> Dict[str, Any]:
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
        return self.retriever.size

    def get_stats(self) -> Dict[str, Any]:
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
        self.retriever.clear()
        self.clear_processed_log()
        logger.info("Index and ledger cleared")


# =============================================================================
# FACTORY
# =============================================================================

def create_pipeline(
    index_dir: str = "data/index",
    load_existing: bool = True,
) -> ResearchPipeline:
    config = PipelineConfig(index_dir=index_dir)
    pipeline = ResearchPipeline(config)

    if load_existing and Path(index_dir).exists():
        try:
            pipeline.load_index()
            logger.info(f"Loaded {pipeline.index_size} chunks")
        except Exception as e:
            logger.warning(f"Could not load index: {e}")

    return pipeline
