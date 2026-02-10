"""
Hybrid Retriever with Smart Filtering
======================================
- Dense retrieval (FAISS) for semantic similarity
- Sparse retrieval (BM25) for keyword matching
- Cross-encoder reranking (FlashRank) for precision
- Intent-based pre-filtering (code/theory/hybrid)
- Reciprocal Rank Fusion for combining results
- Full persistence support (save/load)
"""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

import faiss
from rank_bm25 import BM25Okapi

from .embedder import FastEmbedder, get_embedder
from .data_loader import Chunk, ChunkType

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RetrievalResult:
    """A single retrieval result with score and metadata."""
    chunk: Chunk
    score: float
    rank: int
    source: str  # 'faiss', 'bm25', 'hybrid', 'reranked', 'filtered'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (maintains backward compatibility)."""
        return {
            "content": self.chunk.content,
            "text": self.chunk.content,
            "type": self.chunk.chunk_type.value,
            "score": self.score,
            "rank": self.rank,
            "source": self.source,
            "metadata": self.chunk.metadata,
            "chunk_id": self.chunk.chunk_id,
        }


# =============================================================================
# HYBRID RETRIEVER WITH SMART FILTERING
# =============================================================================

class HybridRetriever:
    """
    Hybrid search with intent-based filtering.
    
    Pipeline:
    1. Retrieve k*3 candidates from FAISS (dense)
    2. Retrieve k*3 candidates from BM25 (sparse)
    3. Filter by intent (code/theory/hybrid)
    4. Merge and deduplicate by chunk_id
    5. Rerank with FlashRank cross-encoder
    6. Return top-k results
    
    Fallback: If filtering results in 0 chunks, uses unfiltered results.
    """
    
    # FlashRank model configuration
    RERANK_MODEL = "ms-marco-TinyBERT-L-2-v2"
    RERANK_CACHE_DIR = ".cache/flashrank"
    
    # Chunk type mappings for filtering
    CODE_TYPES = {ChunkType.CODE}
    THEORY_TYPES = {
        ChunkType.THEORY, 
        ChunkType.MATH, 
        ChunkType.THEOREM,
        ChunkType.DEFINITION, 
        ChunkType.PROOF, 
        ChunkType.ALGORITHM,
        ChunkType.MARKDOWN,
    }
    
    def __init__(
        self,
        embedder: Optional[FastEmbedder] = None,
        faiss_weight: float = 0.7,
        bm25_weight: float = 0.3,
        min_similarity: float = 0.25,
        enable_reranking: bool = True,
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            embedder: FastEmbedder instance
            faiss_weight: Weight for FAISS scores in RRF
            bm25_weight: Weight for BM25 scores in RRF
            min_similarity: Minimum similarity threshold
            enable_reranking: Whether to use FlashRank reranking
        """
        self.embedder = embedder or get_embedder()
        self.faiss_weight = faiss_weight
        self.bm25_weight = bm25_weight
        self.min_similarity = min_similarity
        self.enable_reranking = enable_reranking
        
        # Index components
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.bm25_index: Optional[BM25Okapi] = None
        
        # Chunk storage
        self.chunks: List[Chunk] = []
        self.chunk_texts: List[str] = []
        
        # Dimension from embedder
        self._dimension = self.embedder.dimension
        
        # Initialize reranker (lazy load)
        self.ranker = None
        if self.enable_reranking:
            self._init_reranker()
        
        logger.info(
            f"HybridRetriever initialized. "
            f"FAISS: {faiss_weight}, BM25: {bm25_weight}, "
            f"Reranking: {'ON' if self.enable_reranking and self.ranker else 'OFF'}"
        )
    
    def _init_reranker(self) -> None:
        """Initialize FlashRank reranker."""
        try:
            from flashrank import Ranker
            
            cache_path = Path(self.RERANK_CACHE_DIR)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            self.ranker = Ranker(
                model_name=self.RERANK_MODEL,
                cache_dir=str(cache_path),
            )
            logger.info(f"✓ FlashRank reranker loaded ({self.RERANK_MODEL})")
            
        except ImportError:
            logger.warning("FlashRank not installed. Reranking disabled.")
            self.ranker = None
            self.enable_reranking = False
        except Exception as e:
            logger.warning(f"Failed to initialize FlashRank: {e}")
            self.ranker = None
            self.enable_reranking = False
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0
    
    @property
    def size(self) -> int:
        return len(self.chunks)
    
    # =========================================================================
    # INDEXING
    # =========================================================================
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to the index."""
        if not chunks:
            return
        
        logger.info(f"Adding {len(chunks)} chunks to index...")
        
        self.chunks.extend(chunks)
        texts = [c.content for c in chunks]
        self.chunk_texts.extend(texts)
        
        # Build FAISS index
        embeddings = self.embedder.embed_documents(texts)
        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatIP(self._dimension)
        self.faiss_index.add(embeddings.astype(np.float32))
        
        # Build BM25 index
        tokenized = [self._tokenize(t) for t in self.chunk_texts]
        self.bm25_index = BM25Okapi(tokenized)
        
        logger.info(f"Index now contains {len(self.chunks)} chunks")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        import re
        return re.findall(r'\b\w+\b', text.lower())
    
    # =========================================================================
    # FILTERING
    # =========================================================================
    
    def _filter_by_type(
        self,
        results: List[RetrievalResult],
        filter_type: str,
    ) -> List[RetrievalResult]:
        """
        Filter results by chunk type.
        
        Args:
            results: List of retrieval results
            filter_type: "code", "theory", or "hybrid"
            
        Returns:
            Filtered results (or original if filter yields empty)
        """
        if filter_type == "hybrid":
            return results
        
        if filter_type == "code":
            filtered = [r for r in results if r.chunk.chunk_type in self.CODE_TYPES]
        elif filter_type == "theory":
            filtered = [r for r in results if r.chunk.chunk_type in self.THEORY_TYPES]
        else:
            return results
        
        # Fallback: if filtering yields 0 results, return unfiltered
        if len(filtered) == 0:
            logger.warning(
                f"Filter '{filter_type}' yielded 0 results. "
                f"Falling back to unfiltered ({len(results)} results)."
            )
            return results
        
        logger.debug(f"Filtered: {len(results)} → {len(filtered)} ({filter_type})")
        
        # Update source to indicate filtering
        for r in filtered:
            r.source = f"{r.source}+filtered"
        
        return filtered
    
    # =========================================================================
    # SEARCH
    # =========================================================================
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        chunk_type: Optional[ChunkType] = None,
        method: str = "hybrid",
        use_reranking: Optional[bool] = None,
        filter_type: str = "hybrid",  # NEW: Intent-based filtering
    ) -> List[RetrievalResult]:
        """
        Search for relevant chunks with intent-based filtering.
        
        Args:
            query: Search query
            top_k: Number of results to return
            chunk_type: Filter by specific chunk type (legacy)
            method: Search method ('hybrid', 'faiss', 'bm25')
            use_reranking: Override reranking setting
            filter_type: Intent filter ("code", "theory", "hybrid")
            
        Returns:
            List of RetrievalResult objects
        """
        if self.is_empty:
            logger.warning("Cannot search: index is empty")
            return []
        
        # Determine if we should rerank
        should_rerank = use_reranking if use_reranking is not None else self.enable_reranking
        should_rerank = should_rerank and self.ranker is not None
        
        # Expand retrieval for filtering and reranking
        expansion_factor = 4 if (filter_type != "hybrid" or should_rerank) else 1
        n_candidates = min(top_k * expansion_factor, len(self.chunks))
        
        # Get initial results
        if method == "faiss":
            results = self._search_faiss(query, n_candidates)
        elif method == "bm25":
            results = self._search_bm25(query, n_candidates)
        else:
            results = self._search_hybrid(query, n_candidates)
        
        # Apply intent-based filtering BEFORE reranking
        results = self._filter_by_type(results, filter_type)
        
        # Legacy chunk_type filter (if specified)
        if chunk_type is not None:
            results = [r for r in results if r.chunk.chunk_type == chunk_type]
        
        # Apply reranking if enabled
        if should_rerank and len(results) > 0:
            results = self._rerank(query, results, top_k)
        else:
            results = [r for r in results if r.score >= self.min_similarity][:top_k]
        
        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def search_by_type(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Search and separate results by type (code vs theory).
        Legacy method for backward compatibility.
        """
        all_results = self.search(query, top_k=top_k * 2)
        
        code_results = [
            r for r in all_results 
            if r.chunk.chunk_type in self.CODE_TYPES
        ][:top_k]
        
        theory_results = [
            r for r in all_results 
            if r.chunk.chunk_type in self.THEORY_TYPES
        ][:top_k]
        
        return {
            "code": code_results,
            "theory": theory_results,
        }
    
    def search_by_type_filtered(
        self,
        query: str,
        top_k: int = 5,
        filter_type: str = "hybrid",
    ) -> Dict[str, List[RetrievalResult]]:
        """
        Search with intent-based pre-filtering, then separate by type.
        
        Args:
            query: Search query
            top_k: Number of results per type
            filter_type: Intent filter ("code", "theory", "hybrid")
            
        Returns:
            Dict with 'code' and 'theory' keys
        """
        # Get filtered results
        all_results = self.search(
            query, 
            top_k=top_k * 2, 
            filter_type=filter_type,
        )
        
        # Separate by type
        code_results = [
            r for r in all_results 
            if r.chunk.chunk_type in self.CODE_TYPES
        ][:top_k]
        
        theory_results = [
            r for r in all_results 
            if r.chunk.chunk_type in self.THEORY_TYPES
        ][:top_k]
        
        # Log filtering results
        logger.debug(
            f"search_by_type_filtered({filter_type}): "
            f"{len(all_results)} total → {len(code_results)} code, {len(theory_results)} theory"
        )
        
        return {
            "code": code_results,
            "theory": theory_results,
        }
    
    def _search_faiss(self, query: str, n: int) -> List[RetrievalResult]:
        """Pure FAISS (dense) search."""
        query_vec = self.embedder.embed_query(query).reshape(1, -1).astype(np.float32)
        scores, indices = self.faiss_index.search(query_vec, n)
        
        return [
            RetrievalResult(
                chunk=self.chunks[idx],
                score=float(score),
                rank=rank + 1,
                source="faiss",
            )
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0]))
            if idx >= 0
        ]
    
    def _search_bm25(self, query: str, n: int) -> List[RetrievalResult]:
        """Pure BM25 (sparse) search."""
        query_tokens = self._tokenize(query)
        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:n]
        
        return [
            RetrievalResult(
                chunk=self.chunks[idx],
                score=min(scores[idx] / 10.0, 1.0),
                rank=rank + 1,
                source="bm25",
            )
            for rank, idx in enumerate(top_indices)
        ]
    
    def _search_hybrid(self, query: str, n: int) -> List[RetrievalResult]:
        """Hybrid search using Reciprocal Rank Fusion."""
        k = 60
        
        faiss_results = self._search_faiss(query, n)
        bm25_results = self._search_bm25(query, n)
        
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Tuple[Chunk, float, float]] = {}
        
        for r in faiss_results:
            cid = r.chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0) + self.faiss_weight / (k + r.rank)
            if cid not in chunk_map:
                chunk_map[cid] = (r.chunk, r.score, 0.0)
            else:
                chunk_map[cid] = (chunk_map[cid][0], r.score, chunk_map[cid][2])
        
        for r in bm25_results:
            cid = r.chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0) + self.bm25_weight / (k + r.rank)
            if cid not in chunk_map:
                chunk_map[cid] = (r.chunk, 0.0, r.score)
            else:
                chunk_map[cid] = (chunk_map[cid][0], chunk_map[cid][1], r.score)
        
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            RetrievalResult(
                chunk=chunk_map[cid][0],
                score=self.faiss_weight * chunk_map[cid][1] + self.bm25_weight * chunk_map[cid][2],
                rank=rank + 1,
                source="hybrid",
            )
            for rank, (cid, _) in enumerate(sorted_items[:n])
        ]
    
    def _rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Rerank results using FlashRank cross-encoder."""
        if not results:
            return results
        
        try:
            from flashrank import RerankRequest
            
            # Deduplicate
            seen_ids = set()
            unique_results = []
            for r in results:
                if r.chunk.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk.chunk_id)
                    unique_results.append(r)
            
            if not unique_results:
                return results[:top_k]
            
            # Prepare passages
            passages = [
                {
                    "id": r.chunk.chunk_id,
                    "text": r.chunk.content,
                    "meta": {"original_score": r.score},
                }
                for r in unique_results
            ]
            
            # Rerank
            rerank_request = RerankRequest(query=query, passages=passages)
            reranked = self.ranker.rerank(rerank_request)
            
            # Build result mapping
            chunk_by_id = {r.chunk.chunk_id: r.chunk for r in unique_results}
            
            reranked_results = []
            for rank, item in enumerate(reranked[:top_k]):
                chunk_id = item["id"]
                if chunk_id in chunk_by_id:
                    reranked_results.append(RetrievalResult(
                        chunk=chunk_by_id[chunk_id],
                        score=float(item["score"]),
                        rank=rank + 1,
                        source="reranked",
                    ))
            
            reranked_results = [
                r for r in reranked_results 
                if r.score >= self.min_similarity
            ]
            
            return reranked_results
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}")
            return [r for r in results if r.score >= self.min_similarity][:top_k]
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save(self, index_dir: str | Path) -> None:
        """Save index to disk."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(index_dir / "faiss.index"))
        
        with open(index_dir / "chunks.pkl", 'wb') as f:
            pickle.dump([c.to_dict() for c in self.chunks], f)
        
        with open(index_dir / "chunk_texts.pkl", 'wb') as f:
            pickle.dump(self.chunk_texts, f)
        
        if self.bm25_index is not None:
            with open(index_dir / "bm25.pkl", 'wb') as f:
                pickle.dump(self.bm25_index, f)
        
        config = {
            "faiss_weight": self.faiss_weight,
            "bm25_weight": self.bm25_weight,
            "min_similarity": self.min_similarity,
            "dimension": self._dimension,
            "n_chunks": len(self.chunks),
            "enable_reranking": self.enable_reranking,
        }
        with open(index_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Index saved: {len(self.chunks)} chunks → {index_dir}")
    
    @classmethod
    def load(
        cls,
        index_dir: str | Path,
        embedder: Optional[FastEmbedder] = None,
    ) -> "HybridRetriever":
        """Load index from disk."""
        index_dir = Path(index_dir)
        
        if not index_dir.exists():
            raise FileNotFoundError(f"Index not found: {index_dir}")
        
        with open(index_dir / "config.json") as f:
            config = json.load(f)
        
        retriever = cls(
            embedder=embedder,
            faiss_weight=config["faiss_weight"],
            bm25_weight=config["bm25_weight"],
            min_similarity=config["min_similarity"],
            enable_reranking=config.get("enable_reranking", True),
        )
        
        retriever.faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
        
        with open(index_dir / "chunks.pkl", 'rb') as f:
            retriever.chunks = [Chunk.from_dict(d) for d in pickle.load(f)]
        
        with open(index_dir / "chunk_texts.pkl", 'rb') as f:
            retriever.chunk_texts = pickle.load(f)
        
        with open(index_dir / "bm25.pkl", 'rb') as f:
            retriever.bm25_index = pickle.load(f)
        
        logger.info(f"Index loaded: {len(retriever.chunks)} chunks ← {index_dir}")
        return retriever
    
    def clear(self) -> None:
        """Clear all index data."""
        self.faiss_index = None
        self.bm25_index = None
        self.chunks = []
        self.chunk_texts = []
        logger.info("Index cleared")
