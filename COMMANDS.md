# Research-OS — Command Reference
# ================================

# =============================================================================
# 0. ENVIRONMENT SETUP (Run once)
# =============================================================================

# Activate your conda/venv environment
conda activate Research-OS                    # or: source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt               # installs RAG + API + test deps

# Set required environment variables (add to .env or export manually)
export GROQ_API_KEY="gsk_..."                 # LLM backend — required for generation
export RESEARCH_OS_API_KEY="your-secret-key"  # API auth — required for API endpoints

# =============================================================================
# 1. CLI MODE (No server needed)
# =============================================================================

# Interactive mode — streaming ON, memory ON, smart routing ON
python main.py                                # launches the REPL chat loop

# Interactive mode — disable streaming (full response at once)
python main.py --no-stream                    # useful for copy-pasting long answers

# Single query — streaming
python main.py -q "What is self-attention?"   # prints answer + sources, then exits

# Single query — no streaming
python main.py -q "Explain backpropagation" --no-stream

# Verbose logging — see retrieval scores, intent classification, etc.
python main.py -v                             # sets log level to DEBUG

# Custom index directory
python main.py --index-dir data/custom_index  # use a different FAISS index

# Disable Ollama fallback (Groq only)
python main.py --no-fallback                  # skip local LLM if Groq fails

# Combine flags
python main.py -v --no-stream --no-fallback   # debug mode, no stream, no fallback

# =============================================================================
# 2. API SERVER
# =============================================================================

# Start the API server (default: 0.0.0.0:8000)
python run_api.py                             # production mode, single worker

# Start with auto-reload (development mode — restarts on code changes)
python run_api.py --reload                    # watches src/ for file changes

# Custom host and port
python run_api.py --port 8080                 # run on port 8080
python run_api.py --host 127.0.0.1            # localhost only (not exposed)
python run_api.py --host 0.0.0.0 --port 9000  # exposed on port 9000

# Multiple workers (production — cannot use with --reload)
python run_api.py --workers 4                 # 4 uvicorn workers

# Debug logging
python run_api.py --log-level debug           # verbose API + pipeline logs

# =============================================================================
# 3. API ENDPOINTS (while server is running)
# =============================================================================

# Health check — no auth required
curl http://localhost:8000/health              # returns index size, backend status

# Root info
curl http://localhost:8000/                    # returns service name, version, doc links

# Swagger docs — interactive API playground
# Open in browser: http://localhost:8000/docs

# ReDoc — clean API documentation
# Open in browser: http://localhost:8000/redoc

# Chat (streaming SSE) — requires API key
curl -N -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $RESEARCH_OS_API_KEY" \
  -d '{"query": "What is self-attention?"}'

# Chat with conversation history
curl -N -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $RESEARCH_OS_API_KEY" \
  -d '{"query": "Explain that more simply", "history": [{"role": "user", "content": "What is self-attention?"}, {"role": "assistant", "content": "Self-attention is..."}]}'

# Chat with intent filter override
curl -N -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $RESEARCH_OS_API_KEY" \
  -d '{"query": "Show me attention", "filter_type": "code"}'

# Upload and ingest a file — background task
curl -X POST http://localhost:8000/v1/ingest/file \
  -H "X-API-Key: $RESEARCH_OS_API_KEY" \
  -F "file=@path/to/paper.pdf"

# Ingest from URL — background task
curl -X POST http://localhost:8000/v1/ingest/url \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $RESEARCH_OS_API_KEY" \
  -d '{"url": "https://arxiv.org/pdf/1706.03762v5", "filename": "attention.pdf"}'

# =============================================================================
# 4. TESTING
# =============================================================================

# Full API test suite (24 tests — server must be running)
python test_api_complete.py                   # tests health, auth, chat, ingest, docs

# Quick curl smoke tests (server must be running)
bash test_curl.sh                             # 8 curl commands with formatted output

# Visual streaming test — watch tokens arrive live
python test_stream_visual.py                  # prints tokens in real-time

# Pytest (if you have unit tests in tests/)
pytest tests/ -v                              # run all unit tests
pytest tests/ -v --cov=src                    # with coverage report

# =============================================================================
# 5. DATA INGESTION (CLI — no server needed)
# =============================================================================

# Batch ingest — process all PDFs in data/ directories
python scripts/ingest_batch.py                # uses scripts/ingest_batch.py

# Auto download — fetch papers from configured sources
python scripts/auto_download.py               # uses scripts/auto_download.py

# Download specific data
python scripts/download_data.py               # uses scripts/download_data.py

# =============================================================================
# 6. NOTEBOOKS (Jupyter)
# =============================================================================

# Launch Jupyter for interactive exploration
jupyter notebook                              # opens browser with notebook list

# Key notebooks:
#   notebooks/1-text-extraction.ipynb         # test PDF parsing
#   notebooks/2-embedding.ipynb               # test embedding + retrieval

# =============================================================================
# 7. DIAGNOSTICS
# =============================================================================

# Check what metadata your chunks contain
python check_metadata.py                      # prints chunk metadata keys + values

# Diagnose index issues (FAISS scores, BM25 status)
python diagnose_index.py                      # raw FAISS search + BM25 check

# Analyze query logs
python Analyze-logs.py                        # reads logs/queries.jsonl

# =============================================================================
# 8. COMMON WORKFLOWS
# =============================================================================

# --- Workflow A: Fresh start (first time) ---
# 1. pip install -r requirements.txt
# 2. export GROQ_API_KEY="gsk_..."
# 3. python scripts/ingest_batch.py           # index your PDFs
# 4. python main.py                           # start chatting

# --- Workflow B: Start API for frontend ---
# 1. export GROQ_API_KEY="gsk_..."
# 2. export RESEARCH_OS_API_KEY="secret"
# 3. python run_api.py                        # server on :8000
# 4. python test_api_complete.py              # verify everything works

# --- Workflow C: Add new papers ---
# 1. Copy PDFs to data/02_papers/
# 2. python scripts/ingest_batch.py           # re-index
# 3. python main.py -q "What does the new paper say about X?"

# --- Workflow D: Development with hot reload ---
# 1. python run_api.py --reload --log-level debug
# 2. Edit code in src/ — server auto-restarts
# 3. python test_api_complete.py              # re-test after changes
