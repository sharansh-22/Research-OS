import faiss
import numpy as np
import os
import pickle
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

class Retriever:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        # Each document is expected to be a dict with at least:
        # {"text": str, "metadata": {...}} where metadata can include "type".
        self.documents: List[Dict[str, Any]] = []
        self.bm25 = None  # type: ignore[assignment]
        self.tokenized_corpus: List[List[str]] = []

    def add(self, embeddings: np.ndarray, docs: List[Dict[str, Any]]):
        """
        Add documents to both FAISS and BM25 indices.
        """
        # Store docs (preserve metadata intact)
        self.documents.extend(docs)

        # FAISS
        self.index.add(embeddings)
        
        # BM25
        # Tokenize all docs (simple whitespace tokenization).
        new_tokenized = [doc.get("text", "").lower().split() for doc in docs]
        self.tokenized_corpus.extend(new_tokenized)

        # Rebuild BM25 index with full corpus
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    def save(self, directory: str):
        """Save index and documents to directory"""
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        
        # Persist full document dicts (including metadata such as "type").
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

        # Crucial: persist BM25 object so keyword search survives restarts.
        with open(os.path.join(directory, "bm25.pkl"), "wb") as f:
            pickle.dump(self.bm25, f)

    def load(self, directory: str):
        """Load index and documents from directory"""
        index_path = os.path.join(directory, "index.faiss")
        docs_path = os.path.join(directory, "documents.pkl")
        bm25_path = os.path.join(directory, "bm25.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            print("Warning: index.faiss or documents.pkl not found; starting fresh.")
            return
            
        self.index = faiss.read_index(index_path)
        
        with open(docs_path, "rb") as f:
            # Loaded documents retain any metadata fields (including "type").
            self.documents = pickle.load(f)
            
        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
            # tokenized_corpus is only used for rebuilding; reconstruct if needed.
            if self.documents and not self.tokenized_corpus:
                self.tokenized_corpus = [d.get("text", "").lower().split() for d in self.documents]
        else:
            print("Warning: BM25 index not found. Hybrid search might fail if not rebuilt.")

    def search(self, query_embedding: np.ndarray, query_text: str, k: int = 5):
        """
        Hybrid search: dense FAISS + sparse BM25, union + deduplicate.
        
        Args:
            query_embedding: Vector for dense search
            query_text: Text for sparse search (BM25)
            k: Number of results to return
        """
        # 1. Dense Search (FAISS)
        D, I = self.index.search(np.array([query_embedding]), k)
        dense_results: List[Dict[str, Any]] = []
        for j, i in enumerate(I[0]):
            if i < len(self.documents):
                doc = self.documents[i]
                dense_results.append(
                    {"text": doc.get("text", ""), "score": float(D[0][j]), "metadata": doc.get("metadata", {})}
                )

        # If BM25 isn't available, return dense results directly.
        if self.bm25 is None or not query_text:
            return dense_results[:k]
            
        # 2. Sparse Search (BM25)
        tokenized_query = query_text.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(scores)[::-1][:k]
        
        sparse_results: List[Dict[str, Any]] = []
        for i in top_n_indices:
            if i < len(self.documents):
                doc = self.documents[i]
                sparse_results.append(
                    {"text": doc.get("text", ""), "score": float(scores[i]), "metadata": doc.get("metadata", {})}
                )
                
        # 3. Ensemble union + deduplicate.
        # Since dense scores are distances (lower better) and sparse scores are similarities
        # (higher better), we avoid mixing them numerically. Instead, we interleave by rank.
        def _key(r: Dict[str, Any]) -> str:
            meta = r.get("metadata", {}) or {}
            cid = meta.get("chunk_id")
            src = meta.get("source", "")
            if cid is not None:
                return f"{src}::{cid}"
            return r.get("text", "")[:2000]

        seen = set()
        combined: List[Dict[str, Any]] = []

        # Interleave by rank: dense[0], sparse[0], dense[1], sparse[1], ...
        for rank in range(max(len(dense_results), len(sparse_results))):
            if rank < len(dense_results):
                r = dense_results[rank]
                k0 = _key(r)
                if k0 not in seen:
                    seen.add(k0)
                    combined.append(r)
                    if len(combined) >= k:
                        break
            if rank < len(sparse_results):
                r = sparse_results[rank]
                k0 = _key(r)
                if k0 not in seen:
                    seen.add(k0)
                    combined.append(r)
                    if len(combined) >= k:
                        break

        return combined

    def filter_by_type(self, results: List[Dict[str, Any]], type_filter: str) -> List[Dict[str, Any]]:
        """
        Filter search results to only those whose metadata.type matches `type_filter`
        (e.g., "code" or "theory").
        """
        if not type_filter:
            return results

        return [
            r
            for r in results
            if r.get("metadata", {}).get("type") == type_filter
        ]
