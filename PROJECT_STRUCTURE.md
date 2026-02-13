# Research-OS - Complete Project Structure & Architecture

## 📋 Executive Summary

**Research-OS** is an end-to-end Retrieval-Augmented Generation (RAG) system that enables intelligent querying over a multi-format knowledge base. It combines dense vector search (FAISS), sparse keyword search (BM25), cross-encoder reranking (FlashRank), and LLM inference (Groq API primary, Ollama fallback) to provide accurate, context-aware answers with source attribution.

### Key Features
- **Hybrid Search**: FAISS (semantic) + BM25 (keyword) with Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking**: FlashRank (ms-marco-TinyBERT-L-2-v2) for precision
- **Multi-Format Ingestion**: PDF, Python, Jupyter, Markdown, LaTeX, C++/CUDA
- **Smart Query Routing**: Intent classification (code / theory / hybrid)
- **Streaming Generation**: Server-Sent Events (SSE) for real-time token delivery
- **3-Turn Conversation Memory**: Sliding window context for multi-turn chat
- **React Frontend**: Three-pane UI with session management and source inspector
- **FastAPI Backend**: RESTful API with auth, background ingestion, and health checks
- **System Integrity**: MD5 file hashing, ingestion ledger, constant-time API key comparison

---

## 📁 Directory Structure

```
Research-OS/
│
├── 📄 ROOT CONFIGURATION FILES
│   ├── .gitattributes              # Git LFS config for PDFs/binary files
│   ├── .gitignore                  # Git ignore patterns
│   ├── requirements.txt            # Python package dependencies (pinned)
│   ├── local.env                   # Local environment overrides
│   └── .env                        # Environment variables (GROQ_API_KEY, RESEARCH_OS_API_KEY, etc.)
│
├── 📄 ENTRY POINTS
│   ├── main.py                     # CLI application (interactive REPL + --ingest + --query)
│   ├── run_api.py                  # FastAPI server launcher (uvicorn)
│   └── verify_setup.py             # 22-section system verification (pre-flight checker)
│
├── 📁 src/                         # MAIN SOURCE CODE PACKAGE
│   │
│   ├── 📁 rag/                     # RAG Core Module
│   │   │
│   │   ├── __init__.py             # Package exports (18 classes, v2.1.0)
│   │   │
│   │   ├── data_loader.py          # UNIVERSAL DOCUMENT LOADER
│   │   │   ├── UniversalLoader     # Factory: routes files to format-specific parsers
│   │   │   ├── PDFParser           # pymupdf4llm → markdown → section splitting
│   │   │   ├── PythonParser        # Split by functions/classes
│   │   │   ├── JupyterParser       # Code/markdown cells
│   │   │   ├── MarkdownParser      # Split by headers
│   │   │   ├── LaTeXParser         # Split by \section, \subsection
│   │   │   ├── CppParser           # Brace-matched function extraction
│   │   │   ├── TextParser          # Fallback plain text
│   │   │   ├── Chunk               # Dataclass: content + ChunkType + metadata
│   │   │   └── ChunkType           # Enum: code | theory | algorithm | theorem | proof | definition
│   │   │
│   │   ├── embedder.py             # TEXT EMBEDDING (FastEmbed / ONNX)
│   │   │   ├── FastEmbedder        # Wraps fastembed.TextEmbedding
│   │   │   │   ├── Model: sentence-transformers/all-MiniLM-L6-v2
│   │   │   │   ├── Dimension: 384
│   │   │   │   ├── embed()         # Batch embed with L2 normalization
│   │   │   │   ├── embed_query()   # Single query embedding
│   │   │   │   └── embed_documents()  # Batch document embedding
│   │   │   └── get_embedder()      # Singleton accessor
│   │   │
│   │   ├── retriever.py            # HYBRID SEARCH (FAISS + BM25 + FlashRank)
│   │   │   ├── HybridRetriever
│   │   │   │   ├── FAISS IndexFlatIP (inner product, 384-dim)
│   │   │   │   ├── BM25Okapi sparse index
│   │   │   │   ├── FlashRank cross-encoder reranker
│   │   │   │   ├── search()                     # Hybrid search with RRF merge
│   │   │   │   ├── search_by_type_filtered()    # Intent-based pre-filtering
│   │   │   │   ├── _reciprocal_rank_fusion()    # RRF (k=60, FAISS: 0.7, BM25: 0.3)
│   │   │   │   ├── save() / load()              # Full persistence (faiss.index, chunks.pkl, bm25.pkl)
│   │   │   │   └── add_documents()              # Add chunks to both indices
│   │   │   └── RetrievalResult     # Dataclass: chunk + score + rank + source
│   │   │
│   │   ├── generator.py            # LLM ANSWER GENERATION
│   │   │   ├── ResearchArchitect
│   │   │   │   ├── Primary: Groq API (llama-3.3-70b-versatile)
│   │   │   │   ├── Fallback: Ollama (phi3:mini)
│   │   │   │   ├── MAX_HISTORY_TURNS = 3 (sliding window)
│   │   │   │   ├── generate_stream()    # Streaming token generation
│   │   │   │   ├── generate()           # Non-streaming generation
│   │   │   │   └── health_check()       # Backend connectivity test
│   │   │   └── GenerationResult    # Dataclass: response + metadata
│   │   │
│   │   ├── pipeline.py             # RAG PIPELINE ORCHESTRATION
│   │   │   ├── ResearchPipeline
│   │   │   │   ├── classify_intent()    # Smart query routing (code/theory/hybrid)
│   │   │   │   ├── query()              # Full RAG: retrieve → generate → cite
│   │   │   │   ├── query_stream()       # Streaming RAG with JSON-serializable chunks
│   │   │   │   ├── ingest_pdf()         # Single file ingestion with MD5 dedup
│   │   │   │   ├── ingest_directory()   # Batch directory ingestion
│   │   │   │   ├── rebuild_index()      # Full reindex from scratch
│   │   │   │   ├── save_index() / load_index()  # Persistence
│   │   │   │   └── get_stats()          # Index statistics
│   │   │   ├── PipelineConfig      # Dataclass: index_dir, enable_fallback, etc.
│   │   │   ├── QueryResult         # Dataclass: response + intent + context + metadata
│   │   │   ├── IngestionResult     # Dataclass: filename + status + chunks_added + hash
│   │   │   ├── StreamChunk         # Dataclass: event + data (for SSE)
│   │   │   └── create_pipeline()   # Factory function
│   │   │
│   │   └── verifier.py             # CODE VERIFICATION SANDBOX
│   │       ├── ArchitectureVerifier
│   │       │   ├── verify_dimensions()          # Execute code, extract tensor shapes
│   │       │   ├── verify_generated_response()  # Verify all code blocks in LLM output
│   │       │   ├── extract_code_blocks()        # Parse ```python``` fences
│   │       │   ├── _is_safe()                   # Regex safety check (blocks os, subprocess, eval, exec)
│   │       │   └── Timeout: SIGALRM-based (10s default)
│   │       └── VerificationResult  # Dataclass: success + output + shapes + execution_time
│   │
│   ├── 📁 api/                     # FastAPI Backend Module
│   │   ├── __init__.py             # Exports create_app()
│   │   ├── main.py                 # App factory + lifespan (startup/shutdown)
│   │   │   ├── create_app()        # FastAPI instance with CORS + routes
│   │   │   └── CORS origins: localhost:5173, 5174, 3000
│   │   ├── routes.py               # HTTP Endpoints
│   │   │   ├── POST /v1/chat       # Streaming chat (SSE) with RAG
│   │   │   ├── POST /v1/ingest/file    # Multipart file upload → background ingest
│   │   │   ├── POST /v1/ingest/url     # URL download → background ingest
│   │   │   ├── GET  /v1/ingest/status  # Ingestion task progress
│   │   │   ├── GET  /v1/index/files    # List indexed documents
│   │   │   └── GET  /health            # System health + backend status
│   │   ├── dependencies.py         # Security & Singleton
│   │   │   ├── PipelineState       # Global RAG pipeline singleton (lifespan-managed)
│   │   │   └── verify_api_key()    # X-API-Key header → hmac.compare_digest (constant-time)
│   │   └── ingestion_tracker.py    # Background task progress tracking
│   │       ├── IngestionStage      # Enum: DOWNLOADING, PARSING, EMBEDDING, INDEXING, COMPLETE, FAILED
│   │       └── tracker             # Global tracker instance
│   │
│   └── api.py                      # Legacy monolithic API (superseded by src/api/)
│
├── 📁 scripts/                     # UTILITY SCRIPTS
│   ├── download_data.py            # Download knowledge base PDFs
│   ├── auto_download.py            # Auto-download + classify documents
│   ├── ingest_batch.py             # Batch PDF ingestion
│   ├── verify_setup.py             # Duplicate of root verify_setup.py
│   ├── check_metadata.py           # Index metadata inspector
│   ├── diagnose_index.py           # Index diagnostics
│   └── Analyze-logs.py             # Query log analysis
│
├── 📁 frontend/                    # REACT FRONTEND (Vite + Tailwind v3)
│   ├── package.json                # npm config (React 19, Tailwind 3)
│   ├── vite.config.js              # Vite build config
│   ├── tailwind.config.js          # Tailwind theme (custom dark palette)
│   ├── postcss.config.js           # PostCSS config
│   ├── index.html                  # HTML entry point
│   └── src/
│       ├── main.jsx                # React entry point
│       ├── App.jsx                 # Three-pane layout + session management
│       │   ├── MAX_HISTORY_TURNS = 3 (mirrored from backend)
│       │   ├── pushHistory()       # Sliding window history manager
│       │   └── Health polling (30s interval)
│       ├── api.js                  # API client
│       │   ├── streamChat()        # SSE streaming via fetch + ReadableStream
│       │   ├── uploadFile()        # Multipart file upload
│       │   ├── ingestUrl()         # URL ingestion
│       │   ├── fetchHealth()       # Health check
│       │   └── API key via localStorage (X-API-Key header)
│       ├── chatHistory.js          # Session persistence (localStorage, 50-session cap)
│       ├── index.css               # Tailwind styles + custom theme
│       └── components/
│           ├── ChatPane.jsx        # Chat interface with streaming
│           ├── MessageBubble.jsx   # Markdown renderer (react-markdown, KaTeX, highlight.js)
│           ├── LeftPane.jsx        # Sidebar: sessions + file upload + URL ingest
│           ├── SourcePane.jsx      # Source inspector (right pane)
│           ├── ApiKeyModal.jsx     # API key configuration modal
│           └── ChatHistoryPanel.jsx  # Chat history UI
│
├── 📁 backend/                     # BACKEND MODELS
│   └── models/                     # ML model storage
│
├── 📁 data/                        # DATA & KNOWLEDGE BASE
│   ├── 📁 01_fundamentals/         # Fundamental ML resources
│   ├── 📁 02_papers/               # Research papers
│   ├── 📁 03_implementation/       # Implementation guides
│   ├── 📁 04_misc/                 # Uploaded / miscellaneous files
│   └── 📁 index/                   # PERSISTED SEARCH INDICES
│       ├── faiss.index             # FAISS vector database (384-dim, IndexFlatIP)
│       ├── chunks.pkl              # Chunk objects (pickle)
│       ├── chunk_texts.pkl         # Raw text for BM25
│       ├── bm25.pkl                # BM25 sparse index (pickle)
│       ├── config.json             # Index config (n_chunks, dimension, model)
│       └── processed_files.json    # Ingestion ledger (filename → MD5 hash + chunks + timestamp)
│
├── 📁 tests/                       # UNIT & INTEGRATION TESTS
│   ├── test_data_loader.py         # Chunking, metadata, format detection
│   └── test_hybrid_search.py       # Index persistence, hybrid ranking
│
├── 📁 notebooks/                   # JUPYTER NOTEBOOKS
│   ├── 1-text-extraction.ipynb     # PDF extraction experiments
│   └── 2-embedding.ipynb           # Embedding experiments
│
├── 📁 logs/                        # QUERY LOGS & ANALYTICS
│
├── 📁 .cache/                      # MODEL CACHES
│   └── flashrank/                  # FlashRank reranker model
│       └── ms-marco-TinyBERT-L-2-v2/
│           ├── flashrank-TinyBERT-L-2-v2.onnx
│           ├── tokenizer.json
│           └── config.json
│
├── .vscode/                        # IDE SETTINGS
│   └── settings.json               # Python interpreter + conda auto-activation
│
└── .github/                        # GitHub workflows
```

---

## 🔗 Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          USER INTERACTION                                     │
│  CLI (main.py) │ React Frontend (port 5173) │ API Docs (/docs)               │
└────────────┬──────────────────┬───────────────────────────────┬──────────────┘
             │                  │                               │
             │           FastAPI Backend (port 8000)            │
             │           X-API-Key auth + SSE streaming         │
             │                  │                               │
       ┌─────┴──────┐    ┌─────┴──────┐              ┌────────┴────────┐
       │ INGESTION   │    │   QUERY    │              │  HEALTH/STATS   │
       └─────────────┘    └────────────┘              └─────────────────┘
             │                  │
             ▼                  ▼
  ┌────────────────────┐  ┌──────────────────────────┐
  │ UniversalLoader    │  │ classify_intent(query)   │
  │ (PDF/LaTeX/C++/    │  │ → code | theory | hybrid │
  │  Python/Jupyter/   │  └──────────┬───────────────┘
  │  Markdown)         │             │
  └────────┬───────────┘             ▼
           │              ┌──────────────────────────┐
           ▼              │ HybridRetriever          │
  ┌────────────────────┐  │  ├─ FAISS (semantic)     │
  │ FastEmbedder       │  │  ├─ BM25  (keyword)      │
  │ (all-MiniLM-L6-v2) │  │  ├─ RRF merge (k=60)    │
  │ → 384-dim vectors  │  │  └─ FlashRank rerank     │
  └────────┬───────────┘  └──────────┬───────────────┘
           │                         │
           ▼                         ▼
  ┌────────────────────┐  ┌──────────────────────────┐
  │ Add to FAISS +     │  │ ResearchArchitect        │
  │ BM25 indices       │  │  ├─ Groq (primary)       │
  └────────┬───────────┘  │  ├─ Ollama (fallback)    │
           │              │  └─ 3-turn memory window  │
           ▼              └──────────┬───────────────┘
  ┌────────────────────┐             │
  │ Save to data/index │             ▼
  │ + update ledger    │  ┌──────────────────────────┐
  │ (MD5 hash tracking)│  │ SSE Stream → Frontend    │
  └────────────────────┘  │ (start → context →       │
                          │  chunks → sources → done) │
                          └──────────────────────────┘
```

---

## 🧠 Component Interaction Details

### 1. **Data Ingestion Pipeline**
```
Document (PDF / .py / .tex / .cpp / .ipynb / .md)
  ↓ UniversalLoader.load_file()
  ├─ Route to format-specific parser (PDFParser, CppParser, etc.)
  ├─ Extract text → create Chunk objects with ChunkType + metadata
  └─ Return: List[Chunk]
    ↓ ResearchPipeline.ingest_pdf()
    ├─ Compute MD5 hash → check ledger for duplicates
    ├─ Embed chunks via FastEmbedder (384-dim, L2-normalized)
    └─ Add to HybridRetriever (FAISS + BM25)
      ↓ save_index()
      ├─ Persist: data/index/faiss.index, chunks.pkl, bm25.pkl
      ├─ Write: data/index/config.json
      └─ Update: data/index/processed_files.json (ledger)
```

### 2. **Query & Retrieval Pipeline**
```
User Query (string)
  ↓ ResearchPipeline.query(question, history, filter_type)
  ├─ classify_intent(query)  →  code | theory | hybrid
  ├─ HybridRetriever.search_by_type_filtered(query, top_k, intent)
  │  ├─ FAISS search: inner product → top-k semantic matches
  │  ├─ BM25 search: Okapi BM25 → top-k keyword matches
  │  ├─ Reciprocal Rank Fusion (k=60, FAISS: 0.7, BM25: 0.3)
  │  └─ FlashRank cross-encoder reranking
  └─ Return: List[RetrievalResult] (sorted by score)
```

### 3. **Answer Generation Pipeline**
```
(Query, Retrieved Chunks, History)
  ↓ ResearchArchitect.generate_stream()
  ├─ Build conversation: system prompt + history[-6:] + context + query
  ├─ Primary: Groq API (llama-3.3-70b-versatile)
  │  └─ Fallback: Ollama (phi3:mini) if Groq fails
  ├─ Stream tokens via SSE events
  └─ Post-process: strip hallucinated sources, inject metadata-based citations
    ↓ Deliver via EventSourceResponse to frontend
```

---

## 📦 Dependencies & Requirements

### Core RAG Packages
```
pymupdf4llm        # PDF → markdown extraction
pymupdf (fitz)     # PDF parsing engine
fastembed           # ONNX-based text embeddings (all-MiniLM-L6-v2)
faiss-cpu           # Vector similarity search (CPU-only)
rank-bm25           # BM25 keyword ranking
flashrank           # Cross-encoder reranking (TinyBERT)
numpy (<2.0.0)      # Numerical computations
groq                # Groq cloud LLM API client
ollama              # Local LLM fallback client
pydantic            # Data validation
python-dotenv       # Environment variable management
```

### API Packages
```
fastapi             # Web framework
uvicorn             # ASGI server
sse-starlette       # Server-Sent Events support
python-multipart    # File upload handling
```

### Frontend Packages (npm)
```
react (^19.2.0)     # UI framework
react-markdown      # Markdown rendering
remark-math         # Math notation parsing
rehype-katex        # KaTeX rendering
katex               # Math typesetting
highlight.js        # Code syntax highlighting
lucide-react        # Icons
tailwindcss (^3.4)  # Utility-first CSS (v3)
```

### Development Dependencies
```
pytest              # Unit testing
pytest-asyncio      # Async test support
jupyter             # Interactive notebooks
httpx               # HTTP testing client
```

### System Requirements
- **Python**: 3.10+ (tested on 3.10)
- **RAM**: 16 GB (optimized for FAISS + BM25 + embedder in-memory)
- **GPU**: Not required (ONNX CPU inference via FastEmbed)
- **Ollama**: Optional fallback — `ollama serve` on localhost:11434
- **Groq API Key**: Required for primary LLM generation
- **Node.js**: 18+ (for frontend)

---

## 🚀 Usage Workflows

### Setup & Installation
```bash
# Clone repository
git clone <repo-url>
cd Research-OS

# Create conda environment
conda create -n Research-OS python=3.10
conda activate Research-OS

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..

# Set environment variables
cp .env.example .env   # Edit with your GROQ_API_KEY

# Verify everything
python verify_setup.py

# Download knowledge base PDFs
python scripts/download_data.py

# Ingest PDFs into index
python scripts/ingest_batch.py
```

### Usage Modes

#### 1. Interactive CLI Mode
```bash
python main.py
```

#### 2. API Server + Frontend
```bash
# Terminal 1: Start API
python run_api.py

# Terminal 2: Start frontend
cd frontend && npm run dev
```

#### 3. Single Query (CLI)
```bash
python main.py --query "What is the attention mechanism?"
```

#### 4. Batch Ingestion
```bash
python scripts/ingest_batch.py
```

#### 5. System Verification
```bash
python verify_setup.py
```

#### 6. Testing
```bash
pytest tests/
```

---

## 🔧 Configuration & Extensibility

### Environment Variables (.env)
```bash
GROQ_API_KEY=gsk_...                  # Groq API key (required for primary LLM)
RESEARCH_OS_API_KEY=...               # API authentication key
API_URL=http://localhost:8000         # Backend URL
RESEARCH_OS_INDEX_DIR=data/index      # Index storage directory
RESEARCH_OS_CORS_ORIGINS=*            # CORS policy (overridden in src/api/main.py)
```

### Key Configuration Points

**Change LLM Model** (generator.py):
```python
MODEL = "llama-3.3-70b-versatile"     # Groq primary
FALLBACK_MODEL = "phi3:mini"          # Ollama fallback
```

**Adjust Hybrid Search Weights** (retriever.py):
```python
self.faiss_weight = 0.7               # Semantic search weight
self.bm25_weight = 0.3                # Keyword search weight
```

**Change Embedding Model** (embedder.py):
```python
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
```

**Adjust Memory Window** (generator.py + App.jsx):
```python
MAX_HISTORY_TURNS = 3                 # 3 user+assistant turn pairs = 6 messages
```

---

## 📊 Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| PDF Ingestion (10 pages) | 5-10s | pymupdf4llm + embedding |
| Embedding Single Text | ~10ms | FastEmbed ONNX (CPU) |
| FAISS Search (3845 chunks) | ~2ms | Inner product, flat index |
| BM25 Search | ~5ms | Okapi BM25 ranking |
| RRF Merge + FlashRank Rerank | ~20ms | Cross-encoder reranking |
| Full Hybrid Search | ~30ms | FAISS + BM25 + RRF + rerank |
| LLM Generation (Groq) | 1-5s | Cloud API, streaming |
| LLM Generation (Ollama) | 5-30s | Local fallback, CPU |
| Full RAG Query | 2-10s | Retrieve + Generate (Groq) |

---

## 🧪 Testing Strategy

### Unit Tests
- **test_data_loader.py**: Chunking logic, metadata preservation, format detection
- **test_hybrid_search.py**: Index persistence, hybrid ranking, RRF correctness

### System Verification
- **verify_setup.py**: 22-section pre-flight checker (Python, env vars, packages, source files, index, FlashRank, imports, embedder, generator, pipeline, syntax, frontend, network, CORS, security)

### Manual Testing
- Interactive CLI testing
- API endpoint testing via `/docs` (Swagger UI)
- Frontend SSE streaming validation

---

## 📝 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "No module named 'faiss'" | FAISS not installed | `pip install faiss-cpu` |
| "GROQ_API_KEY not set" | Missing env var | Set in `.env` or `export GROQ_API_KEY='gsk_...'` |
| "Pipeline not initialized" | API started without index | Run `python scripts/ingest_batch.py` first |
| "CORS error in browser" | Frontend origin not allowed | Check `src/api/main.py` CORS origins |
| "401 Missing API key" | No X-API-Key header | Set API key in frontend settings modal |
| "Index not found" | No ingestion completed | Run `python scripts/ingest_batch.py` |
| "Slow query response" | Using Ollama fallback | Check Groq API key and connectivity |
| "Import errors" | Wrong conda env | `conda activate Research-OS` |

---

## 📄 License & Attribution

- **LLM**: Meta LLaMA 3.3 (via Groq), Microsoft Phi-3 (via Ollama)
- **Embeddings**: FastEmbed (ONNX) / Sentence Transformers
- **Vector DB**: Meta FAISS
- **PDF Extraction**: pymupdf4llm
- **Reranking**: FlashRank (ms-marco-TinyBERT-L-2-v2)
- **BM25**: Rank-bm25 library
- **Frontend**: React 19, Tailwind CSS 3, Vite

---

**Last Updated**: February 12, 2026
**Version**: 2.1.0
**Maintainers**: Research-OS Team
