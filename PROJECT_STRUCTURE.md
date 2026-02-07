# AI Knowledge Assistant - Complete Project Structure & Architecture

## 📋 Executive Summary

The **AI Knowledge Assistant** is an end-to-end Retrieval-Augmented Generation (RAG) system that enables intelligent querying over a knowledge base of PDFs. It combines dense vector search (FAISS), sparse keyword search (BM25), and local LLM inference (Ollama) to provide accurate, context-aware answers with source attribution.

### Key Features
- **Hybrid Search**: FAISS (semantic) + BM25 (keyword) for comprehensive retrieval
- **Smart Text Processing**: Intelligent PDF chunking with metadata preservation
- **Local LLM Generation**: Ollama integration for offline inference
- **Interactive CLI**: Real-time query processing with source tracking
- **Modular Architecture**: Pluggable components for extensibility
- **Batch Ingestion**: Process multiple PDFs efficiently

---

## 📁 Directory Structure

```
ai-knowledge-assistant/
│
├── 📄 ROOT CONFIGURATION FILES
│   ├── .gitattributes              # Git LFS (Large File Storage) config for PDFs/binary files
│   ├── .gitignore                  # Git ignore patterns
│   │   └── Excludes: logs/, .env, __pycache__, .vscode, data/*, etc.
│   ├── requirements.txt            # Python package dependencies (see section below)
│   ├── local.env                   # Local environment variables (not in version control)
│   └── .env                        # Environment variables template (Git-tracked)
│
├── 📄 ENTRY POINT & CLI
│   ├── main.py                     # Primary CLI application
│   │   ├── print_banner()          # Display application header
│   │   ├── main()                  # Parse args, orchestrate pipeline
│   │   │   ├── --ingest <path>     # Ingest and build index from PDF
│   │   │   └── --query <text>      # Run single query
│   │   ├── answer_query()          # Execute query, display results with sources
│   │   └── INTERACTIVE MODE        # REPL loop for continuous querying
│   │
│   ├── test_gemini.py              # Test suite for Gemini API integration
│   │   ├── API connectivity tests
│   │   ├── Model inference tests
│   │   └── Response parsing tests
│   │
│   └── Analyze-logs.py             # Query log analysis utility
│       ├── Load logs/queries.jsonl (query history)
│       ├── Analyze query patterns & drift
│       └── Generate statistics & insights
│
├── 📁 src/                         # MAIN SOURCE CODE PACKAGE
│   │
│   ├── 📁 rag/                     # RAG (Retrieval-Augmented Generation) Module
│   │   │                           # Purpose: Core RAG functionality
│   │   │
│   │   ├── __init__.py             # Package initialization (exports key classes)
│   │   │
│   │   ├── data_loader.py          # PDF TEXT EXTRACTION & CHUNKING
│   │   │   │
│   │   │   ├── extract_text_with_metadata()
│   │   │   │   ├── Input: PDF file path
│   │   │   │   ├── Process: Page-by-page extraction using marker-pdf
│   │   │   │   └── Output: List of {text, metadata{page, source}} dicts
│   │   │   │
│   │   │   ├── split_text_smart(text, chunk_size, overlap_ratio)
│   │   │   │   ├── Input: Raw text, target chunk size, overlap ratio
│   │   │   │   ├── Process: Word-based splitting with overlap preservation
│   │   │   │   │  └── Respects paragraph boundaries when possible
│   │   │   │   └── Output: List of text chunks with start/end indices
│   │   │   │
│   │   │   └── load_and_chunk_pdf(pdf_path, chunk_size, overlap_ratio)
│   │   │       ├── Input: PDF path, chunking parameters
│   │   │       ├── Process: Extract → Split → Attach metadata to each chunk
│   │   │       └── Output: List of {text, metadata{page, source, chunk_idx}} dicts
│   │   │
│   │   ├── embedder.py             # TEXT EMBEDDING (SentenceTransformers)
│   │   │   │
│   │   │   ├── Embedder class
│   │   │   │   ├── __init__(model_name: str)
│   │   │   │   │   └── Loads SentenceTransformer model (default: all-MiniLM-L6-v2)
│   │   │   │   │
│   │   │   │   ├── encode(texts: List[str]) -> np.ndarray
│   │   │   │   │   ├── Input: List of text strings
│   │   │   │   │   ├── Process: Batch encode using SentenceTransformer
│   │   │   │   │   └── Output: (N, 384) numpy array of embeddings
│   │   │   │   │
│   │   │   │   └── encode_single(text: str) -> np.ndarray
│   │   │   │       ├── Input: Single text string
│   │   │   │       ├── Process: Encode single text
│   │   │   │       └── Output: (384,) numpy array embedding
│   │   │   │
│   │   │   └── Model Details:
│   │   │       ├── Embedding Dimension: 384
│   │   │       ├── Max Sequence Length: 512 tokens
│   │   │       └── Inference: CPU or CUDA (auto-detected)
│   │   │
│   │   ├── retriever.py            # HYBRID SEARCH (FAISS + BM25)
│   │   │   │
│   │   │   ├── Retriever class
│   │   │   │   ├── __init__(embedding_dim: int = 384)
│   │   │   │   │   ├── Initialize FAISS index (flat, no GPU)
│   │   │   │   │   ├── Initialize BM25 ranker
│   │   │   │   │   └── Initialize document storage
│   │   │   │   │
│   │   │   │   ├── add(documents: List[Dict])
│   │   │   │   │   ├── Input: {text, metadata{page, source, ...}}
│   │   │   │   │   ├── Process 1: Extract text & embed with SentenceTransformer
│   │   │   │   │   ├── Process 2: Add embeddings to FAISS index
│   │   │   │   │   ├── Process 3: Build BM25 corpus from texts
│   │   │   │   │   └── Store original documents for retrieval
│   │   │   │   │
│   │   │   │   ├── search(query: str, k: int = 5) -> List[Dict]
│   │   │   │   │   ├── Input: Query text, number of results
│   │   │   │   │   ├── Process 1: Embed query text
│   │   │   │   │   ├── Process 2: Search FAISS for top-k semantic matches
│   │   │   │   │   ├── Process 3: Search BM25 for top-k keyword matches
│   │   │   │   │   ├── Process 4: Merge & re-rank results (hybrid scoring)
│   │   │   │   │   └── Output: Sorted list of {text, metadata, score}
│   │   │   │   │
│   │   │   │   ├── save(index_dir: str)
│   │   │   │   │   ├── Save FAISS index to disk
│   │   │   │   │   ├── Save documents to JSON/pickle
│   │   │   │   │   └── Save BM25 corpus metadata
│   │   │   │   │
│   │   │   │   ├── load(index_dir: str) -> bool
│   │   │   │   │   ├── Load FAISS index from disk
│   │   │   │   │   ├── Load stored documents
│   │   │   │   │   └── Rebuild BM25 ranker
│   │   │   │   │
│   │   │   │   └── filter_by_type(results, metadata_key, value)
│   │   │   │       └── Filter search results by metadata field
│   │   │   │
│   │   │   └── Indexing Strategy:
│   │   │       ├── FAISS: Flat L2 distance (no compression)
│   │   │       ├── BM25: TF-IDF variant for sparse retrieval
│   │   │       └── Hybrid: Weighted combination of both scores
│   │   │
│   │   ├── generator.py            # LLM ANSWER GENERATION (Ollama)
│   │   │   │
│   │   │   ├── Generator class
│   │   │   │   ├── __init__(model_name: str = "llama3.2:3b", 
│   │   │   │   │             base_url: str = "http://localhost:11434")
│   │   │   │   │   └── Initialize Ollama client
│   │   │   │   │
│   │   │   │   └── generate(query: str, context_chunks: List[str], 
│   │   │   │                 num_ctx: int = 2048) -> str
│   │   │   │       ├── Input: User query, list of context chunks
│   │   │   │       ├── Process:
│   │   │   │       │  1. Concatenate context chunks
│   │   │   │       │  2. Build prompt with context + question
│   │   │   │       │  3. Call Ollama API for generation
│   │   │   │       │  4. Stream and assemble response
│   │   │   │       └── Output: Complete generated answer string
│   │   │   │
│   │   │   └── Model Configuration:
│   │   │       ├── Default Model: llama3.2:3b (3B parameters, 8GB RAM)
│   │   │       ├── Alternative Models: llama2:7b, mistral:7b, neural-chat:7b
│   │   │       ├── Context Window: 2048 tokens (configurable)
│   │   │       └── Temperature: 0.7 (configurable for creativity vs accuracy)
│   │   │
│   │   ├── pipeline.py             # RAG PIPELINE ORCHESTRATION
│   │   │   │
│   │   │   ├── RAGPipeline class
│   │   │   │   ├── __init__()
│   │   │   │   │   ├── Initialize Embedder
│   │   │   │   │   ├── Initialize Generator
│   │   │   │   │   ├── Initialize Retriever
│   │   │   │   │   ├── Attempt to load existing index
│   │   │   │   │   └── Setup logging (logs/queries.jsonl)
│   │   │   │   │
│   │   │   │   ├── build_index(pdf_path: str)
│   │   │   │   │   ├── Input: Path to single PDF or directory
│   │   │   │   │   ├── Process:
│   │   │   │   │   │  1. Load & chunk PDF (data_loader)
│   │   │   │   │   │  2. Embed chunks (embedder)
│   │   │   │   │   │  3. Add to index (retriever)
│   │   │   │   │   │  4. Persist index to disk
│   │   │   │   │   └── Logging: Record ingestion timestamp & file size
│   │   │   │   │
│   │   │   │   ├── query(query_text: str, k: int = 3) -> List[Dict]
│   │   │   │   │   ├── Input: Query string, number of results
│   │   │   │   │   ├── Process:
│   │   │   │   │   │  1. Retrieve top-k chunks (retriever.search)
│   │   │   │   │   │  2. Log query to queries.jsonl
│   │   │   │   │   │  3. Attach metadata (timestamp, chunk_count)
│   │   │   │   │   └── Output: List of {text, metadata, score}
│   │   │   │   │
│   │   │   │   ├── ask(query_text: str, k: int = 3) -> Dict
│   │   │   │   │   ├── Input: Query string, context chunk count
│   │   │   │   │   ├── Process:
│   │   │   │   │   │  1. Retrieve context: query(query_text, k)
│   │   │   │   │   │  2. Generate answer: generator.generate(query, context)
│   │   │   │   │   │  3. Assemble response with sources
│   │   │   │   │   └── Output: {answer: str, sources: List[Dict], 
│   │   │   │   │             metadata: {retrieval_time, generation_time}}
│   │   │   │   │
│   │   │   │   └── State Persistence:
│   │   │   │       ├── Index Directory: indices/ (FAISS + documents)
│   │   │       │       └── Query Log: logs/queries.jsonl
│   │   │   │
│   │   │   └── Workflow Diagram:
│   │   │       PDF Input → Load & Chunk → Embed → FAISS/BM25 Index
│   │   │       User Query → Retrieve → Generate → Return with Sources
│   │   │
│   │   ├── verifier.py             # CODE VERIFICATION UTILITY (Optional)
│   │   │   │
│   │   │   ├── ArchitectureVerifier class
│   │   │   │   ├── __init__(safe_mode: bool = True)
│   │   │   │   │   └── Enable/disable code execution sandbox
│   │   │   │   │
│   │   │   │   ├── extract_code_blocks(text: str) -> List[str]
│   │   │   │   │   ├── Input: Markdown text from LLM response
│   │   │   │   │   ├── Process: Parse ```python``` code fences
│   │   │   │   │   └── Output: List of code block strings
│   │   │   │   │
│   │   │   │   ├── verify_dimensions(code: str) -> Dict
│   │   │   │   │   ├── Input: Python code string
│   │   │   │   │   ├── Process: Execute code, capture tensor shapes
│   │   │   │   │   └── Output: {success: bool, shapes: Dict, errors: str}
│   │   │   │   │
│   │   │   │   └── Purpose: Validate ML code correctness
│   │   │
│   │   └── __pycache__/            # Python bytecode cache (auto-generated)
│   │
│   ├── 📁 agents/                  # AGENT FRAMEWORK (Future Extension)
│   │   └── [Empty - Planned for multi-step reasoning agents]
│   │
│   └── 📁 knowledge-graph/         # KNOWLEDGE GRAPH (Future Extension)
│       └── [Empty - Planned for semantic relationship extraction]
│
├── 📁 scripts/                     # UTILITY SCRIPTS
│   │
│   ├── download_data.py            # DOWNLOAD KNOWLEDGE BASE PDFS
│   │   ├── Purpose: Populate data/ directory with ML PDFs
│   │   ├── Functions:
│   │   │   ├── download_fundamentals()
│   │   │   │   └── Downloads to data/01_fundamentals/
│   │   │   │       └── Linear Algebra for Machine Learning (Part 1)
│   │   │   │
│   │   │   ├── download_papers()
│   │   │   │   └── Downloads to data/02_papers/
│   │   │   │       ├── "Attention Is All You Need" (Transformer)
│   │   │   │       ├── "Deep Residual Learning for Image Recognition" (ResNet)
│   │   │   │       ├── "Adam: A Method for Stochastic Optimization"
│   │   │   │       ├── "Denoising Diffusion Probabilistic Models" (DDPM)
│   │   │   │       └── "Dropout: A Simple Way to Prevent Neural Networks..."
│   │   │   │
│   │   │   └── download_implementations()
│   │   │       └── Downloads to data/03_implementation/
│   │   │           ├── Deep Learning with PyTorch (Book)
│   │   │           └── The Little Book of Deep Learning
│   │   │
│   │   └── Execution: `python scripts/download_data.py`
│   │
│   └── ingest_batch.py             # BATCH PDF INGESTION
│       ├── Purpose: Process all PDFs in data/ directory
│       ├── Process:
│       │   1. Scan data/01_fundamentals/, data/02_papers/, data/03_implementation/
│       │   2. For each PDF:
│       │   │  └── Load → Chunk → Embed → Add to index
│       │   3. Save consolidated FAISS index to indices/
│       │   4. Log ingestion results
│       │
│       └── Execution: `python scripts/ingest_batch.py`
│
├── 📁 tests/                       # UNIT & INTEGRATION TESTS
│   │
│   ├── test_data_loader.py         # DATA LOADING TESTS
│   │   ├── test_paragraph_splitting()
│   │   │   └── Verify text chunking respects paragraphs
│   │   │
│   │   ├── test_chunking_limit()
│   │   │   └── Verify chunk sizes don't exceed limit
│   │   │
│   │   ├── test_overlap()
│   │   │   └── Verify overlap ratio applied correctly
│   │   │
│   │   └── test_smart_behavior()
│   │       └── Verify metadata attached to chunks
│   │
│   ├── test_hybrid_search.py       # RETRIEVAL TESTS
│   │   ├── test_retriever_hybrid_integration()
│   │   │   └── Verify FAISS + BM25 hybrid search works
│   │   │
│   │   └── test_save_load()
│   │       └── Verify index persistence/loading
│   │
│   └── __pycache__/                # Python bytecode cache
│
├── 📁 notebooks/                   # JUPYTER NOTEBOOKS (Experimental/Dev)
│   │
│   ├── 1-text-extraction.ipynb     # PDF EXTRACTION EXPERIMENTS
│   │   ├── Cell 1: Import libraries
│   │   ├── Cell 2: Load PDF with marker-pdf
│   │   ├── Cell 3: Extract text page-by-page
│   │   └── Cell 4: Visualize text quality
│   │
│   ├── 2-embedding.ipynb           # EMBEDDING EXPERIMENTS
│   │   ├── Cell 1: Load SentenceTransformer
│   │   ├── Cell 2: Embed sample texts
│   │   ├── Cell 3: Compute cosine similarity
│   │   └── Cell 4: Visualize embeddings (t-SNE/UMAP)
│   │
│   └── [Root-level duplicates for quick access]
│       ├── 1-text-extraction.ipynb
│       └── 2-embedding.ipynb
│
├── 📁 data/                        # DATA & KNOWLEDGE BASE
│   │
│   ├── 📄 sample.pdf               # Sample test PDF
│   ├── 📄 terminal.pdf             # Terminal/shell concepts PDF
│   │
│   ├── 📁 chunks/                  # PROCESSED TEXT CHUNKS
│   │   └── terminal_chunks.jsonl   # Chunked text from terminal.pdf
│   │       └── Format: One JSON object per line
│   │           └── {text: str, metadata: {page: int, source: str, chunk_idx: int}}
│   │
│   ├── 📁 01_fundamentals/         # FUNDAMENTAL ML RESOURCES
│   │   └── linear_algebra_for_ml_part1.pdf
│   │       └── Topics: Vectors, matrices, decomposition
│   │
│   ├── 📁 02_papers/               # SEMINAL RESEARCH PAPERS
│   │   ├── attention_is_all_you_need.pdf
│   │   │   └── Transformers, multi-head attention
│   │   ├── resnet.pdf
│   │   │   └── Residual networks, skip connections
│   │   ├── adam_optimizer.pdf
│   │   │   └── Adaptive learning rates for optimization
│   │   ├── ddpm_diffusion.pdf
│   │   │   └── Denoising diffusion probabilistic models
│   │   └── dropout_srivastava14a.pdf
│   │       └── Regularization technique for neural networks
│   │
│   └── 📁 03_implementation/       # IMPLEMENTATION GUIDES
│       ├── deep_learning_with_pytorch.pdf
│       │   └── PyTorch fundamentals, training loops, models
│       └── the_little_book_of_deep_learning.pdf
│           └── Deep learning principles, architectures, best practices
│
├── 📁 indices/                     # PERSISTED SEARCH INDICES (Generated)
│   ├── faiss.index                 # FAISS vector database
│   │   └── Contains embeddings for all chunks
│   ├── documents.json              # Original chunk documents + metadata
│   │   └── Array of {text, metadata} objects
│   └── bm25_metadata.pkl           # BM25 corpus metadata (pickle)
│
├── 📁 logs/                        # QUERY LOGS & ANALYTICS
│   │
│   ├── queries.jsonl               # QUERY HISTORY LOG
│   │   ├── Format: One JSON object per line
│   │   ├── Fields: {
│   │   │     timestamp: str (ISO 8601),
│   │   │     query: str,
│   │   │     retrieval_time: float (seconds),
│   │   │     generation_time: float (seconds),
│   │   │     num_chunks_retrieved: int,
│   │   │     model_used: str
│   │   │   }
│   │   └── Purpose: Track query patterns, performance, user interactions
│   │
│   └── [Additional logs]: errors.log, warnings.log (optional)
│
├── 📁 backend/                     # BACKEND API (Future)
│   └── [Empty - Planned for FastAPI/Flask REST API]
│
├── 📁 frontend/                    # FRONTEND UI (Future)
│   └── [Empty - Planned for React/Vue web interface]
│
├── 📁 docs/                        # DOCUMENTATION (Future)
│   └── [Empty - Planned for API docs, guides, examples]
│
├── .vscode/                        # VS CODE SETTINGS
│   ├── launch.json                 # Debugger configuration
│   ├── settings.json               # Editor settings
│   └── extensions.json             # Recommended extensions
│
├── .ipynb_checkpoints/             # Jupyter checkpoints (auto-generated)
│
├── .pytest_cache/                  # Pytest cache (auto-generated)
│
├── .git/                           # Git repository metadata
│
└── System Volume Information/      # Windows system folder (ignore)

```

---

## 🔗 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERACTION                             │
│  CLI (main.py) → Interactive REPL / --ingest / --query              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          ▼                                  ▼
    ┌──────────────┐              ┌──────────────────┐
    │ INDEX BUILD  │              │  QUERY EXECUTION │
    └──────────────┘              └──────────────────┘
          │                              │
          ▼                              ▼
    ┌──────────────────────┐      ┌──────────────────────┐
    │ data_loader.py       │      │ retriever.search()   │
    │ (PDF → Chunks)       │      │ (FAISS + BM25)       │
    └──────────┬───────────┘      └──────────┬───────────┘
               │                             │
               ▼                             ▼
    ┌──────────────────────┐      ┌──────────────────────┐
    │ embedder.encode()    │      │ Retrieved Chunks     │
    │ (Text → Embeddings)  │      │ (Top-k Results)      │
    └──────────┬───────────┘      └──────────┬───────────┘
               │                             │
               ▼                             ▼
    ┌──────────────────────┐      ┌──────────────────────┐
    │ retriever.add()      │      │ generator.generate() │
    │ (Build Indices)      │      │ (LLM → Answer)       │
    └──────────┬───────────┘      └──────────┬───────────┘
               │                             │
               ▼                             ▼
    ┌──────────────────────┐      ┌──────────────────────┐
    │ retriever.save()     │      │ Format Response      │
    │ (Persist Indices)    │      │ (Answer + Sources)   │
    └──────────┬───────────┘      └──────────┬───────────┘
               │                             │
               └────────────────┬────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │ Output to User   │
                        │ (CLI Display)    │
                        └──────────────────┘
```

---

## 🧠 Component Interaction Details

### 1. **Data Ingestion Pipeline**
```
PDF File
  ↓ data_loader.load_and_chunk_pdf()
  ├─ Extract text + metadata (page, source)
  ├─ Split into chunks (smart word-based)
  └─ Return: List[{text, metadata}]
    ↓ pipeline.build_index()
    ├─ Pass chunks to embedder.encode()
    ├─ embedder calls SentenceTransformer
    └─ Returns: List[embedding_vector]
      ↓ retriever.add()
      ├─ Add to FAISS index (flat L2)
      ├─ Build BM25 corpus
      └─ Persist: indices/faiss.index, indices/documents.json
```

### 2. **Query & Retrieval Pipeline**
```
User Query (string)
  ↓ pipeline.query(query_text, k=3)
  ├─ embedder.encode_single(query_text)
  │  └─ Returns: 1D embedding vector
  ├─ retriever.search(query_text, k=3)
  │  ├─ FAISS search: L2 distance → top-3 semantic matches
  │  ├─ BM25 search: TF-IDF → top-3 keyword matches
  │  └─ Merge & rank: Combined hybrid score
  ├─ Log to logs/queries.jsonl
  └─ Return: List[{text, metadata, score}]
```

### 3. **Answer Generation Pipeline**
```
(Query, Retrieved Chunks)
  ↓ pipeline.ask(query, context_chunks)
  ├─ Concatenate chunk texts
  ├─ Build prompt: "Context: ...\n\nQuestion: {query}\n\nAnswer:"
  ├─ generator.generate(query, chunks)
  │  ├─ Call Ollama API
  │  ├─ Stream response tokens
  │  └─ Assemble full answer
  ├─ Format response dict:
  │  ├─ answer: str
  │  ├─ sources: List[{text, metadata}]
  │  └─ metadata: {retrieval_time, generation_time}
  └─ Return: Response dict
    ↓ Display to user with formatted output
```

---

## 📦 Dependencies & Requirements

### Core Dependencies
```
ollama              # Local LLM inference client
faiss-cpu          # Vector similarity search (CPU-only)
sentence-transformers  # Text embeddings
marker-pdf         # High-quality PDF text extraction
numpy              # Numerical computations
rank-bm25          # BM25 keyword ranking
python-dotenv      # Environment variable management
```

### Development Dependencies
```
pytest             # Unit testing framework
jupyter            # Interactive notebooks
```

### System Requirements
- **Python**: 3.8+ (tested on 3.10)
- **RAM**: 8+ GB (for Ollama model + FAISS index)
- **GPU**: Optional (CUDA for faster embeddings)
- **Ollama**: Installed and running on localhost:11434

---

## 🚀 Usage Workflows

### Setup & Installation
```bash
# Clone repository
git clone <repo-url>
cd ai-knowledge-assistant

# Create virtual environment (conda)
conda create -n rag python=3.10
conda activate rag

# Install dependencies
pip install -r requirements.txt

# Download Ollama model
ollama pull llama3.2:3b

# Download knowledge base PDFs
python scripts/download_data.py

# Ingest PDFs into index
python scripts/ingest_batch.py
```

### Usage Modes

#### 1. Interactive Mode
```bash
python main.py
# Type queries and get answers with source attribution
```

#### 2. Single Query Mode
```bash
python main.py --query "What is attention mechanism?"
```

#### 3. Ingest + Query
```bash
python main.py --ingest data/sample.pdf --query "What is transformer?"
```

#### 4. Batch Ingestion
```bash
python scripts/ingest_batch.py
```

#### 5. Testing
```bash
pytest tests/
```

#### 6. Notebook Exploration
```bash
jupyter notebook notebooks/
# Open 1-text-extraction.ipynb or 2-embedding.ipynb
```

---

## 🔧 Configuration & Extensibility

### Environment Variables (local.env)
```bash
OLLAMA_MODEL=llama3.2:3b           # LLM model name
OLLAMA_BASE_URL=http://localhost:11434  # Ollama server URL
EMBEDDING_MODEL=all-MiniLM-L6-v2   # SentenceTransformer model
CHUNK_SIZE=256                     # Text chunk size (words)
CHUNK_OVERLAP=0.1                  # Overlap ratio (0.1 = 10%)
CONTEXT_WINDOW=2048                # LLM context window (tokens)
LOG_LEVEL=INFO                     # Logging level
```

### Customization Points

**Change LLM Model**:
```python
# In src/rag/generator.py
generator = Generator(model_name="mistral:7b")
```

**Change Embedding Model**:
```python
# In src/rag/pipeline.py
embedder = Embedder(model_name="sentence-transformers/all-mpnet-base-v2")
```

**Adjust Chunking Parameters**:
```python
# In src/rag/pipeline.py
pipeline.build_index(pdf_path, chunk_size=512, overlap_ratio=0.2)
```

**Change Search Strategy**:
```python
# In src/rag/retriever.py
results = retriever.search(query, k=10)  # Return top-10 instead of 5
```

---

## 📊 Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| PDF Ingestion (10 pages) | 5-10s | Includes chunking + embedding |
| Embedding Single Text | ~20ms | CPU inference, varies by text length |
| FAISS Search (1M vectors) | ~5ms | L2 distance, flat index |
| BM25 Search | ~10ms | TF-IDF ranking |
| Hybrid Search | ~15ms | Combined FAISS + BM25 |
| LLM Generation | 5-30s | Depends on output length + model |
| Full RAG Query | 10-40s | Retrieve + Generate |

---

## 🧪 Testing Strategy

### Unit Tests
- **test_data_loader.py**: Chunking logic, metadata preservation
- **test_hybrid_search.py**: Index persistence, hybrid ranking

### Integration Tests
- End-to-end PDF ingestion + query + generation

### Manual Testing
- Interactive CLI testing
- Verify source attribution accuracy
- Check generation quality on test queries

---

## 🔮 Future Extensions

### Planned Features
1. **Backend API**: FastAPI REST endpoints
2. **Frontend UI**: React/Vue web interface
3. **Agents Framework**: Multi-step reasoning agents
4. **Knowledge Graph**: Automatic relationship extraction
5. **Advanced Retrieval**: MMR (Maximal Marginal Relevance), re-ranking
6. **Multi-Modal**: Support for images in PDFs
7. **Streaming UI**: Real-time token streaming to client
8. **Fine-tuning**: Custom embeddings/LLM fine-tuning
9. **Monitoring**: Query performance analytics, user feedback loops
10. **Caching**: Query result caching for common questions

---

## 📝 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "No module named 'faiss'" | FAISS not installed | `pip install faiss-cpu` |
| "Connection refused to Ollama" | Ollama not running | `ollama serve` in separate terminal |
| "CUDA out of memory" | GPU memory exceeded | Use CPU embeddings: remove CUDA |
| "PDF extraction failed" | Corrupted PDF or unsupported format | Try with sample.pdf |
| "Index not found" | No ingestion completed | Run `python scripts/ingest_batch.py` |
| "Slow query response" | Large context window or slow LLM | Reduce `k` parameter or use faster model |

---

## 📄 License & Attribution

- **Framework**: LLaMA 2 / Llama 3.2
- **Embeddings**: Hugging Face Sentence Transformers
- **Vector DB**: Meta FAISS
- **PDF Extraction**: Unified-IO Marker-PDF
- **BM25**: Rank-bm25 library

---

**Last Updated**: February 7, 2026
**Version**: 1.0 (Production Ready)
**Maintainers**: AI Knowledge Assistant Team
