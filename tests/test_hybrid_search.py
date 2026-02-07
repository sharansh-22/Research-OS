import pytest
import numpy as np
import os
import shutil
from src.rag.retriever import Retriever

@pytest.fixture
def temp_dir():
    dirname = "test_vector_store"
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    yield dirname
    if os.path.exists(dirname):
        shutil.rmtree(dirname)

def test_retriever_hybrid_integration(temp_dir):
    # Setup
    dim = 4
    retriever = Retriever(dim=dim)
    
    # Mock data
    # Embeddings: 4D vectors
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0], # Doc 1
        [0.0, 1.0, 0.0, 0.0], # Doc 2
        [0.0, 0.0, 1.0, 0.0], # Doc 3
    ]).astype('float32')
    
    docs = [
        {"text": "Apple is a fruit", "metadata": {"chunk_id": 0}},
        {"text": "Banana is yellow", "metadata": {"chunk_id": 1}},
        {"text": "Carrot is a vegetable", "metadata": {"chunk_id": 2}}
    ]
    
    # Add
    retriever.add(embeddings, docs)
    
    # Verify BM25 built
    assert retriever.bm25 is not None
    assert len(retriever.tokenized_corpus) == 3
    
    # Search (Dense only - implied by interface if query_text is None)
    # Query close to Doc 1
    q_emb = np.array([1.0, 0.0, 0.0, 0.0]).astype('float32')
    results_dense = retriever.search(q_emb, k=1)
    assert len(results_dense) == 1
    assert results_dense[0][0]["text"] == "Apple is a fruit"
    
    # Search (Hybrid)
    # Query text "yellow" (matches Doc 2) but embedding close to Doc 3 (mismatch test)
    # q_emb close to Doc 3
    q_emb_3 = np.array([0.0, 0.0, 1.0, 0.0]).astype('float32')
    
    results_hybrid = retriever.search(q_emb_3, query_text="yellow", k=2)
    
    # Should contain Doc 3 (dense match) and Doc 2 (sparse match)
    texts = [r[0]["text"] for r in results_hybrid]
    assert "Carrot is a vegetable" in texts # Dense match
    assert "Banana is yellow" in texts # Sparse match

def test_save_load(temp_dir):
    dim = 4
    retriever = Retriever(dim=dim)
    embeddings = np.array([[1,0,0,0]]).astype('float32')
    docs = [{"text": "test doc", "metadata": {"chunk_id": 0}}]
    
    retriever.add(embeddings, docs)
    retriever.save(temp_dir)
    
    # Load
    new_retriever = Retriever(dim=dim)
    new_retriever.load(temp_dir)
    
    assert new_retriever.bm25 is not None
    assert len(new_retriever.documents) == 1
    assert new_retriever.documents[0]["text"] == "test doc"
