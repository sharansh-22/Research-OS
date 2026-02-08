# Research-OS: AI Agent Instructions

## Project Overview
Research-OS is a **Retrieval-Augmented Generation (RAG) system** that answers complex research questions by combining semantic search (FAISS), keyword search (BM25), and local LLM inference (Ollama). It processes PDFs across theory/code/math domains and generates answers with cited sources.

## Architecture & Data Flow

### Core Components (src/rag/)
1. **data_loader.py** → Extracts PDF text using `pymupdf4llm`, splits into semantic chunks with ChunkType classification (CODE, THEORY, MATH, ALGORITHM, etc.). Chunks carry metadata: source file, page number, chunk_id (MD5 hash).
2. **embedder.py** → Wraps SentenceTransformers (default: all-MiniLM-L6-v2, 384-dim embeddings). Provides `encode()` for batch and `encode_single()` for single texts.
3. **retriever.py** → HybridRetriever: weighted combination of FAISS (semantic, 70% default) + BM25 (keyword, 30% default). Persists indexes in `data/index/`.
4. **generator.py** → ResearchArchitect calls Ollama (default: qwen2.5-coder:7b) with system prompt emphasizing theory→code bridge, LaTeX math, and source citation. Structured output: response + token counts.
5. **verifier.py** → ArchitectureVerifier sandboxes code execution (with signal-based timeout) to validate tensor shapes and return execution traces.
6. **pipeline.py** → ResearchPipeline orchestrates all: ingest PDFs → embed → index → retrieve → generate → verify. Config: PipelineConfig controls temperatures, weights, model, verify_code flag.

### Entry Points
- **main.py** → CLI with `--query` (single) or interactive REPL mode with "stats" command
- **test_gemini.py** → Validation suite for API calls
- **Analyze-logs.py** → Query log analysis utility (reads logs/queries.jsonl)

## Critical Patterns & Conventions

### Chunk Metadata
Every `Chunk` has: `content`, `chunk_type` (enum), `metadata` (dict with source/page), `chunk_id`. Chunks serialize/deserialize via `.to_dict()` / `.from_dict()` for persistence.

### Search Weights
FAISS weight + BM25 weight must be manually tuned (see PipelineConfig.faiss_weight, bm25_weight). Scores normalized to 0-1 range before blending. Min_similarity threshold filters low-scoring results.

### Ollama Integration
Expects Ollama running locally (default: http://localhost:11434). Model name passed to generator; prompt includes explicit [THEORY] and [CODE] markers for context chunks. No streaming; full responses collected.

### Index Persistence
FAISS index stored as `data/index/faiss.index`, BM25 corpus as `data/index/bm25.pkl`, metadata as `data/index/chunks.jsonl`. Load via `pipeline.load_index()` or ingest via `pipeline.ingest_pdf()` / `pipeline.ingest_directory()`.

## Developer Workflows

### Adding New ChunkType
1. Add enum variant to `ChunkType` in data_loader.py
2. Update classification logic in `ResearchDocumentLoader._classify_chunk()` (regex/heuristics)
3. Update system prompt in generator.py to handle new type

### Custom Retrieval Weights
Edit PipelineConfig.faiss_weight and bm25_weight (sum should ~1.0), pass to ResearchPipeline constructor. Lower min_similarity to cast wider net.

### Testing
`pytest tests/` runs semantic splitting, markdown parsing, and hybrid search validation. Mock Ollama calls in unit tests; integration tests require local Ollama instance.

### Debugging Queries
Use interactive mode: type "stats" to see chunk counts per type, monitor logs for embedding/retrieval scores. Check verification_results in QueryResult for code execution traces.

## Environment & Dependencies
- Python 3.10+
- **Core**: pymupdf4llm (PDF), faiss-cpu, rank-bm25, ollama (local inference), sentence-transformers
- **Dev**: pytest, jupyter, rich (formatting), structlog (structured logging)
- **Config**: .env for API keys, local.env for local overrides (git-ignored)

## Key Files to Review First
- [src/rag/pipeline.py](src/rag/pipeline.py) → ResearchPipeline class (orchestration entry point)
- [src/rag/data_loader.py](src/rag/data_loader.py) → Chunk definition and split logic
- [src/rag/retriever.py](src/rag/retriever.py) → Hybrid search scoring formula
- [main.py](main.py) → CLI argument parsing and interactive loop
