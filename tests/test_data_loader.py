
import pytest
from unittest.mock import MagicMock, patch
from src.rag.data_loader import split_text_semantic, split_markdown_logic, load_and_chunk_pdf, MathLoader

def test_semantic_splitting():
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    # Target 20 chars.
    # Chunk 1: "Sentence one. Sentence two." (27 chars)
    chunks = split_text_semantic(text, target_chars=20, overlap_sentences=0)
    assert len(chunks) >= 2
    assert "Sentence one. Sentence two." in chunks[0]
    
    # Test overlap
    chunks_overlap = split_text_semantic(text, target_chars=20, overlap_sentences=1)
    # Chunk 2 should start with "Sentence two." (overlap) or similar logic depending on implementation
    # Implementation: if overlap, start from last N sentences.
    # Chunk 1: S1, S2.
    # Chunk 2: S2, S3, S4.
    assert "Sentence two." in chunks_overlap[1]

def test_split_markdown_logic():
    markdown = """Here is some theory.
```python
def foo():
    return 'bar'
```
More theory here.
"""
    chunks = split_markdown_logic(markdown, chunk_size=20, overlap=0)
    
    # We expect:
    # 1. "Here is some theory." (Theory)
    # 2. ```...``` (Code)
    # 3. "More theory here." (Theory)
    
    # Depending on semantic split config, "Here is some theory." might be one chunk.
    assert len(chunks) >= 3
    
    # Check types
    types = [c["type"] for c in chunks]
    assert "code" in types
    assert "theory" in types
    
    # Check content
    code_chunks = [c for c in chunks if c["type"] == "code"]
    assert len(code_chunks) == 1
    assert "def foo():" in code_chunks[0]["text"]

@patch('src.rag.data_loader.MathLoader')
@patch('os.path.basename') # mock basename to avoid path issues
def test_load_and_chunk_pdf(mock_basename, MockLoader):
    mock_basename.return_value = "dummy.pdf"
    
    # Mock instance
    loader_instance = MockLoader.return_value
    # Return a simple mixed markdown
    loader_instance.pdf_to_markdown.return_value = "Theory text.\n```code```"
    
    chunks = load_and_chunk_pdf("path/to/dummy.pdf")
    
    assert len(chunks) == 2
    assert chunks[0]["metadata"]["source"] == "dummy.pdf"
    assert chunks[0]["metadata"]["page"] == 0 # Default we set
    
    # Verify we got both types
    types = set(c["metadata"]["type"] for c in chunks)
    assert "theory" in types
    assert "code" in types
