import pytest
import numpy as np
import os
import shutil
from src.rag.retriever import HybridRetriever, RetrievalResult
from src.rag.data_loader import Chunk, ChunkType

@pytest.fixture
def temp_dir():
    dirname = "test_vector_store"
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    yield dirname
    if os.path.exists(dirname):
        shutil.rmtree(dirname)

@pytest.fixture
def mock_embedder():
    class MockEmbedder:
        def __init__(self):
            self.dimension = 4
        
        def embed_documents(self, texts):
            # Deterministic mock embeddings
            # Apple -> [1, 0, 0, 0]
            # Banana -> [0, 1, 0, 0]
            # Carrot -> [0, 0, 1, 0]
            embs = []
            for t in texts:
                if "Apple" in t:
                    embs.append([1.0, 0.0, 0.0, 0.0])
                elif "Banana" in t:
                    embs.append([0.0, 1.0, 0.0, 0.0])
                elif "Carrot" in t:
                    embs.append([0.0, 0.0, 1.0, 0.0])
                else:
                    embs.append([0.0, 0.0, 0.0, 1.0])
            return np.array(embs).astype('float32')
            
        def embed_query(self, text):
            if "Apple" in text:
                return np.array([1.0, 0.0, 0.0, 0.0]).astype('float32')
            elif "Banana" in text:
                return np.array([0.0, 1.0, 0.0, 0.0]).astype('float32')
            elif "Carrot" in text:
                return np.array([0.0, 0.0, 1.0, 0.0]).astype('float32')
            return np.array([0.0, 0.0, 0.0, 1.0]).astype('float32')
            
    return MockEmbedder()

def test_retriever_hybrid_integration(temp_dir, mock_embedder):
    # Setup
    retriever = HybridRetriever(
        embedder=mock_embedder,
        faiss_weight=0.5,
        bm25_weight=0.5,
        min_similarity=0.0, # Disable filter for this test
        enable_reranking=False
    )
    
    # Mock data
    chunks = [
        Chunk(
            chunk_id="c0",
            content="Apple is a fruit",
            chunk_type=ChunkType.THEORY,
            metadata={"source": "doc1"}
        ),
        Chunk(
            chunk_id="c1",
            content="Banana is yellow",
            chunk_type=ChunkType.THEORY,
            metadata={"source": "doc2"}
        ),
        Chunk(
            chunk_id="c2",
            content="Carrot is a vegetable",
            chunk_type=ChunkType.THEORY,
            metadata={"source": "doc3"}
        )
    ]
    
    # Add
    retriever.add_chunks(chunks)
    
    # Verify BM25 built
    assert retriever.bm25_index is not None
    assert retriever.size == 3
    
    # Search (Hybrid)
    # Query text "yellow" (matches Banana in sparse)
    # But let's say we want to test dense match for Carrot
    # We'll use a query that has dense similarity to Carrot but text match to Banana
    # "Carrot yellow"
    
    # In our mock embedder:
    # "Carrot" -> [0,0,1,0] (matches c2)
    # "Banana" -> [0,1,0,0] (matches c1)
    
    # Search for "Banana" -> Dense should match c1, Sparse should match c1
    results = retriever.search("Banana", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk.content == "Banana is yellow"
    
    # Test RRF score retrieval
    # RRF score should be populated
    assert results[0].score > 0
    assert results[0].source == "hybrid"

def test_save_load(temp_dir, mock_embedder):
    retriever = HybridRetriever(embedder=mock_embedder, min_similarity=0.0)
    chunks = [Chunk(chunk_id="c0", content="test doc", chunk_type=ChunkType.THEORY, metadata={})]
    
    retriever.add_chunks(chunks)
    retriever.save(temp_dir)
    
    # Load
    new_retriever = HybridRetriever.load(temp_dir, embedder=mock_embedder)
    
    assert new_retriever.bm25_index is not None
    assert new_retriever.size == 1
    assert new_retriever.chunks[0].content == "test doc"

