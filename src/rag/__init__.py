"""Research-OS: Universal RAG System with Multi-Format Support"""

from .data_loader import (
    # Main classes
    UniversalLoader,
    ResearchDocumentLoader,  # Backward compat
    Chunk,
    ChunkType,
    # Parsers (for advanced usage)
    BaseParser,
    PDFParser,
    PythonParser,
    JupyterParser,
    MarkdownParser,
    LaTeXParser,
    CppParser,
    TextParser,
    # Convenience functions
    load_file,
    load_research_pdf,
)

from .embedder import FastEmbedder, get_embedder
from .retriever import HybridRetriever, RetrievalResult
from .generator import ResearchArchitect, GenerationResult
from .verifier import ArchitectureVerifier, VerificationResult
from .pipeline import (
    ResearchPipeline,
    PipelineConfig,
    QueryResult,
    IngestionResult,
    StreamChunk,
    create_pipeline,
)

__all__ = [
    # Data Loading
    "UniversalLoader",
    "ResearchDocumentLoader",
    "Chunk",
    "ChunkType",
    "BaseParser",
    "PDFParser",
    "PythonParser",
    "JupyterParser",
    "MarkdownParser",
    "LaTeXParser",
    "CppParser",
    "TextParser",
    "load_file",
    "load_research_pdf",
    # Embeddings
    "FastEmbedder",
    "get_embedder",
    # Retrieval
    "HybridRetriever",
    "RetrievalResult",
    # Generation
    "ResearchArchitect",
    "GenerationResult",
    # Verification
    "ArchitectureVerifier",
    "VerificationResult",
    # Pipeline
    "ResearchPipeline",
    "PipelineConfig",
    "QueryResult",
    "IngestionResult",
    "StreamChunk",
    "create_pipeline",
]

__version__ = "2.1.0"
