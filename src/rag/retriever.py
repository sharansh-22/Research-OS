"""Hybrid Retriever: FAISS + BM25 with persistence"""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

import faiss
from rank_bm25 import BM25Okapi

from .embedder import FastEmbedder, get_embedder
from .data_loader import Chunk, ChunkType

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int
    source: str
    
    def to_dict(self) -> Dict:
        return {
            "content": self.chunk.content,
            "type": self.chunk.chunk_type.value,
            "score": self.score,
            "rank": self.rank,
            "source": self.source,
            "metadata": self.chunk.metadata,
        }


class HybridRetriever:
    """Hybrid search combining FAISS and BM25."""
    
    def __init__(
        self,
        embedder: Optional[FastEmbedder] = None,
        faiss_weight: float = 0.7,
        bm25_weight: float = 0.3,
        min_similarity: float = 0.25,
    ):
        self.embedder = embedder or get_embedder()
        self.faiss_weight = faiss_weight
        self.bm25_weight = bm25_weight
        self.min_similarity = min_similarity
        
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.bm25_index: Optional[BM25Okapi] = None
        self.chunks: List[Chunk] = []
        self.chunk_texts: List[str] = []
        self._dimension = self.embedder.dimension
        
        logger.info(f"HybridRetriever initialized. FAISS: {faiss_weight}, BM25: {bm25_weight}")
    
    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0
    
    @property
    def size(self) -> int:
        return len(self.chunks)
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to index."""
        if not chunks:
            return
        
        logger.info(f"Adding {len(chunks)} chunks to index...")
        
        self.chunks.extend(chunks)
        texts = [c.content for c in chunks]
        self.chunk_texts.extend(texts)
        
        # Build FAISS
        embeddings = self.embedder.embed_documents(texts)
        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatIP(self._dimension)
        self.faiss_index.add(embeddings.astype(np.float32))
        
        # Build BM25
        tokenized = [self._tokenize(t) for t in self.chunk_texts]
        self.bm25_index = BM25Okapi(tokenized)
        
        logger.info(f"Index now contains {len(self.chunks)} chunks")
    
    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r'\b\w+\b', text.lower())
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        chunk_type: Optional[ChunkType] = None,
        method: str = "hybrid",
    ) -> List[RetrievalResult]:
        """Search for relevant chunks."""
        if self.is_empty:
            return []
        
        n_candidates = min(top_k * 3, len(self.chunks))
        
        if method == "faiss":
            results = self._search_faiss(query, n_candidates)
        elif method == "bm25":
            results = self._search_bm25(query, n_candidates)
        else:
            results = self._search_hybrid(query, n_candidates)
        
        if chunk_type:
            results = [r for r in results if r.chunk.chunk_type == chunk_type]
        
        results = [r for r in results if r.score >= self.min_similarity][:top_k]
        
        for i, r in enumerate(results):
            r.rank = i + 1
        
        return results
    
    def _search_faiss(self, query: str, n: int) -> List[RetrievalResult]:
        query_vec = self.embedder.embed_query(query).reshape(1, -1).astype(np.float32)
        scores, indices = self.faiss_index.search(query_vec, n)
        
        return [
            RetrievalResult(self.chunks[idx], float(score), rank + 1, "faiss")
            for rank, (score, idx) in enumerate(zip(scores[0], indices[0]))
            if idx >= 0
        ]
    
    def _search_bm25(self, query: str, n: int) -> List[RetrievalResult]:
        scores = self.bm25_index.get_scores(self._tokenize(query))
        top_idx = np.argsort(scores)[::-1][:n]
        
        return [
            RetrievalResult(self.chunks[idx], min(scores[idx] / 10.0, 1.0), rank + 1, "bm25")
            for rank, idx in enumerate(top_idx)
        ]
    
    def _search_hybrid(self, query: str, n: int) -> List[RetrievalResult]:
        """Reciprocal Rank Fusion."""
        k = 60
        faiss_results = self._search_faiss(query, n)
        bm25_results = self._search_bm25(query, n)
        
        rrf_scores = {}
        chunk_map = {}
        
        for r in faiss_results:
            cid = r.chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0) + self.faiss_weight / (k + r.rank)
            chunk_map[cid] = (r.chunk, r.score, 0.0)
        
        for r in bm25_results:
            cid = r.chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0) + self.bm25_weight / (k + r.rank)
            if cid in chunk_map:
                chunk_map[cid] = (chunk_map[cid][0], chunk_map[cid][1], r.score)
            else:
                chunk_map[cid] = (r.chunk, 0.0, r.score)
        
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            RetrievalResult(
                chunk_map[cid][0],
                self.faiss_weight * chunk_map[cid][1] + self.bm25_weight * chunk_map[cid][2],
                rank + 1,
                "hybrid"
            )
            for rank, (cid, _) in enumerate(sorted_items[:n])
        ]
    
    def search_by_type(self, query: str, top_k: int = 5) -> Dict[str, List[RetrievalResult]]:
        """Search and separate by type."""
        all_results = self.search(query, top_k=top_k * 2)
        return {
            "code": [r for r in all_results if r.chunk.chunk_type == ChunkType.CODE][:top_k],
            "theory": [r for r in all_results if r.chunk.chunk_type != ChunkType.CODE][:top_k],
        }
    
    def save(self, index_dir: str | Path) -> None:
        """Save index to disk."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        
        if self.faiss_index:
            faiss.write_index(self.faiss_index, str(index_dir / "faiss.index"))
        
        with open(index_dir / "chunks.pkl", 'wb') as f:
            pickle.dump([c.to_dict() for c in self.chunks], f)
        
        with open(index_dir / "chunk_texts.pkl", 'wb') as f:
            pickle.dump(self.chunk_texts, f)
        
        if self.bm25_index:
            with open(index_dir / "bm25.pkl", 'wb') as f:
                pickle.dump(self.bm25_index, f)
        
        config = {
            "faiss_weight": self.faiss_weight,
            "bm25_weight": self.bm25_weight,
            "min_similarity": self.min_similarity,
            "dimension": self._dimension,
            "n_chunks": len(self.chunks),
        }
        with open(index_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Index saved: {len(self.chunks)} chunks")
    
    @classmethod
    def load(cls, index_dir: str | Path, embedder: Optional[FastEmbedder] = None) -> "HybridRetriever":
        """Load index from disk."""
        index_dir = Path(index_dir)
        
        with open(index_dir / "config.json") as f:
            config = json.load(f)
        
        retriever = cls(
            embedder=embedder,
            faiss_weight=config["faiss_weight"],
            bm25_weight=config["bm25_weight"],
            min_similarity=config["min_similarity"],
        )
        
        retriever.faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
        
        with open(index_dir / "chunks.pkl", 'rb') as f:
            retriever.chunks = [Chunk.from_dict(d) for d in pickle.load(f)]
        
        with open(index_dir / "chunk_texts.pkl", 'rb') as f:
            retriever.chunk_texts = pickle.load(f)
        
        with open(index_dir / "bm25.pkl", 'rb') as f:
            retriever.bm25_index = pickle.load(f)
        
        logger.info(f"Index loaded: {len(retriever.chunks)} chunks")
        return retriever
    
    def clear(self) -> None:
        """Clear index."""
        self.faiss_index = None
        self.bm25_index = None
        self.chunks = []
        self.chunk_texts = []