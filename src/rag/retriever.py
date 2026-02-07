"""Hybrid Retriever: FAISS + BM25"""

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
        
        logger.info(f"HybridRetriever initialized")
    
    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0
    
    @property
    def size(self) -> int:
        return len(self.chunks)
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        
        logger.info(f"Adding {len(chunks)} chunks...")
        
        self.chunks.extend(chunks)
        texts = [c.content for c in chunks]
        self.chunk_texts.extend(texts)
        
        embeddings = self.embedder.embed_documents(texts)
        if self.faiss_index is None:
            self.faiss_index = faiss.IndexFlatIP(self._dimension)
        self.faiss_index.add(embeddings.astype(np.float32))
        
        tokenized = [self._tokenize(t) for t in self.chunk_texts]
        self.bm25_index = BM25Okapi(tokenized)
        
        logger.info(f"Index: {len(self.chunks)} chunks")
    
    def _tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r'\b\w+\b', text.lower())
    
    def search(self, query: str, top_k: int = 5, chunk_type: Optional[ChunkType] = None, method: str = "hybrid") -> List[RetrievalResult]:
        if self.is_empty:
            return []
        
        n = min(top_k * 3, len(self.chunks))
        
        if method == "faiss":
            results = self._search_faiss(query, n)
        elif method == "bm25":
            results = self._search_bm25(query, n)
        else:
            results = self._search_hybrid(query, n)
        
        if chunk_type:
            results = [r for r in results if r.chunk.chunk_type == chunk_type]
        
        results = [r for r in results if r.score >= self.min_similarity][:top_k]
        for i, r in enumerate(results):
            r.rank = i + 1
        
        return results
    
    def _search_faiss(self, query: str, n: int) -> List[RetrievalResult]:
        vec = self.embedder.embed_query(query).reshape(1, -1).astype(np.float32)
        scores, indices = self.faiss_index.search(vec, n)
        return [RetrievalResult(self.chunks[idx], float(sc), rk+1, "faiss") for rk, (sc, idx) in enumerate(zip(scores[0], indices[0])) if idx >= 0]
    
    def _search_bm25(self, query: str, n: int) -> List[RetrievalResult]:
        scores = self.bm25_index.get_scores(self._tokenize(query))
        top = np.argsort(scores)[::-1][:n]
        return [RetrievalResult(self.chunks[i], min(scores[i]/10, 1.0), rk+1, "bm25") for rk, i in enumerate(top)]
    
    def _search_hybrid(self, query: str, n: int) -> List[RetrievalResult]:
        k = 60
        f_res = self._search_faiss(query, n)
        b_res = self._search_bm25(query, n)
        
        rrf, cmap = {}, {}
        for r in f_res:
            cid = r.chunk.chunk_id
            rrf[cid] = rrf.get(cid, 0) + self.faiss_weight / (k + r.rank)
            cmap[cid] = (r.chunk, r.score, 0.0)
        
        for r in b_res:
            cid = r.chunk.chunk_id
            rrf[cid] = rrf.get(cid, 0) + self.bm25_weight / (k + r.rank)
            if cid in cmap:
                cmap[cid] = (cmap[cid][0], cmap[cid][1], r.score)
            else:
                cmap[cid] = (r.chunk, 0.0, r.score)
        
        sorted_items = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        return [RetrievalResult(cmap[cid][0], self.faiss_weight*cmap[cid][1] + self.bm25_weight*cmap[cid][2], rk+1, "hybrid") for rk, (cid, _) in enumerate(sorted_items[:n])]
    
    def search_by_type(self, query: str, top_k: int = 5) -> Dict[str, List[RetrievalResult]]:
        results = self.search(query, top_k=top_k * 2)
        return {
            "code": [r for r in results if r.chunk.chunk_type == ChunkType.CODE][:top_k],
            "theory": [r for r in results if r.chunk.chunk_type != ChunkType.CODE][:top_k],
        }
    
    def save(self, index_dir: str | Path) -> None:
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
        
        with open(index_dir / "config.json", 'w') as f:
            json.dump({"faiss_weight": self.faiss_weight, "bm25_weight": self.bm25_weight, "min_similarity": self.min_similarity, "dimension": self._dimension, "n_chunks": len(self.chunks)}, f)
        
        logger.info(f"Saved {len(self.chunks)} chunks")
    
    @classmethod
    def load(cls, index_dir: str | Path, embedder: Optional[FastEmbedder] = None) -> "HybridRetriever":
        index_dir = Path(index_dir)
        
        with open(index_dir / "config.json") as f:
            cfg = json.load(f)
        
        ret = cls(embedder=embedder, faiss_weight=cfg["faiss_weight"], bm25_weight=cfg["bm25_weight"], min_similarity=cfg["min_similarity"])
        ret.faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
        
        with open(index_dir / "chunks.pkl", 'rb') as f:
            ret.chunks = [Chunk.from_dict(d) for d in pickle.load(f)]
        
        with open(index_dir / "chunk_texts.pkl", 'rb') as f:
            ret.chunk_texts = pickle.load(f)
        
        with open(index_dir / "bm25.pkl", 'rb') as f:
            ret.bm25_index = pickle.load(f)
        
        logger.info(f"Loaded {len(ret.chunks)} chunks")
        return ret
    
    def clear(self) -> None:
        self.faiss_index = None
        self.bm25_index = None
        self.chunks = []
        self.chunk_texts = []
