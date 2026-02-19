"""
Vanilla RAG Pipeline (Baseline)
================================
Standard Retrieve → Generate flow using the SAME retriever and generator
as Research-OS, but WITHOUT:
  - ResearchAuditor (CoT LLM Judge)
  - SemanticCache
  - _strip_hallucinated_sources (Hallucination Guard)

This ensures a fair comparison: same LLM, same Vector DB, same embeddings.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from src.rag.retriever import HybridRetriever, RetrievalResult
from src.rag.generator import ResearchArchitect
from src.rag.embedder import FastEmbedder, get_embedder
from src.rag.pipeline import PipelineConfig, _extract_sources

logger = logging.getLogger(__name__)


@dataclass
class VanillaResult:
    """Result container for a Vanilla RAG query."""
    query: str
    response: str
    latency: float
    sources: List[Dict] = field(default_factory=list)
    context_str: str = ""
    code_results: List[RetrievalResult] = field(default_factory=list)
    theory_results: List[RetrievalResult] = field(default_factory=list)
    generation_metadata: Dict = field(default_factory=dict)


class VanillaRAGPipeline:
    """
    Baseline RAG: Retrieve → Generate.
    
    Reuses existing Research-OS components by composition.
    No Auditor, no Cache, no Hallucination Guard.
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        generator: Optional[ResearchArchitect] = None,
        embedder: Optional[FastEmbedder] = None,
        config: Optional[PipelineConfig] = None,
        processed_files: Optional[Dict] = None,
    ):
        """
        Initialize Vanilla RAG Pipeline.

        Args:
            retriever: Pre-loaded HybridRetriever (shared with Research-OS for fairness).
            generator: Pre-loaded ResearchArchitect (shared with Research-OS for fairness).
            embedder: Pre-loaded FastEmbedder instance.
            config: PipelineConfig for retrieval settings.
            processed_files: Ledger dict for source verification.
        """
        self.config = config or PipelineConfig()
        self.embedder = embedder or get_embedder()
        self.processed_files = processed_files or {}

        # Reuse existing components — don't reinitialize
        self.retriever = retriever or HybridRetriever(
            embedder=self.embedder,
            faiss_weight=self.config.faiss_weight,
            bm25_weight=self.config.bm25_weight,
            min_similarity=self.config.min_similarity,
        )
        self.generator = generator or ResearchArchitect(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            enable_fallback=self.config.enable_fallback,
        )

        logger.info("Vanilla RAG Pipeline initialized (no Auditor, no Cache)")

    def query(self, question: str, history: Optional[List[Dict]] = None) -> VanillaResult:
        """
        Execute a standard RAG query: Retrieve → Generate.
        
        No hallucination guard, no auditor, no cache.
        
        Args:
            question: User's query.
            history: Optional conversation history.
            
        Returns:
            VanillaResult with response, latency, and source info.
        """
        history = history or []
        start_time = time.time()

        # --- Step 1: Retrieve ---
        results = self.retriever.search_by_type_filtered(
            query=question,
            top_k=self.config.top_k,
            filter_type="hybrid",  # No smart routing — always hybrid
        )
        code_results = results.get("code", [])
        theory_results = results.get("theory", [])

        # --- Step 2: Generate (raw, no post-processing) ---
        gen_result = self.generator.generate(
            query=question,
            code_results=code_results,
            theory_results=theory_results,
            history=history,
        )

        # NO _strip_hallucinated_sources — raw LLM output
        raw_response = gen_result.response

        latency = time.time() - start_time

        # Build source citations (same logic for fair comparison)
        sources = _extract_sources(
            code_results,
            theory_results,
            self.processed_files,
            self.config,
        )

        # Build context string for later evaluation
        context_str = self.generator._format_context(code_results, theory_results)

        return VanillaResult(
            query=question,
            response=raw_response,
            latency=latency,
            sources=sources,
            context_str=context_str,
            code_results=code_results,
            theory_results=theory_results,
            generation_metadata=gen_result.to_dict(),
        )
