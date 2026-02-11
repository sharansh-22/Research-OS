# Research-OS — Command Reference
# ================================
# Version 2.1.0 | Single source of truth for all operations

# =============================================================================
# 0. QUICK START (One Command)
# =============================================================================

# Start everything — backend, frontend, opens browser
python webrun.py                              # Ctrl+C stops everything

# =============================================================================
# 1. ENVIRONMENT SETUP (First time only)
# =============================================================================

# Activate environment
conda activate Research-OS                    # or: source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt               # RAG + API packages

# Frontend dependencies
cd frontend && npm install && cd ..           # React + Tailwind packages

# Set environment variables (add to .env or export)
export GROQ_API_KEY="gsk_..."                 # required — LLM generation
export RESEARCH_OS_API_KEY="your-secret-key"  # required — API authentication

# =============================================================================
# 2. SYSTEM VERIFICATION
# =============================================================================

# Full 22-section system check (run before anything)
python verify_setup.py                        # checks env, packages, files, index, imports, network

# =============================================================================
# 3. WEB MODE (Frontend + Backend)
# =============================================================================

# One command — starts backend, frontend, opens browser
python webrun.py                              # everything on autopilot

# Or start separately in two terminals:

# Terminal 1: Backend
python run_api.py                             # FastAPI on :8000

# Terminal 2: Frontend
cd frontend && npm run dev                    # Vite on :5173

# Then open: http://localhost:5173

# Backend options:
python run_api.py --reload                    # auto-restart on code changes
python run_api.py --port 8080                 # custom port
python run_api.py --host 127.0.0.1            # localhost only
python run_api.py --workers 4                 # multiple workers (no --reload)
python run_api.py --log-level debug           # verbose logging

# =============================================================================
# 4. CLI MODE (No server needed)
# =============================================================================

# Interactive mode — streaming, memory, smart routing all ON
python main.py                                # launches REPL chat loop

# Single query
python main.py -q "What is self-attention?"   # prints answer + sources, exits

# Flags
python main.py --no-stream                    # full response at once
python main.py --no-fallback                  # Groq only, skip Ollama
python main.py -v                             # verbose/debug logging
python main.py --index-dir data/custom_index  # custom index path

# Combine
python main.py -v --no-stream --no-fallback

# Interactive commands (inside REPL):
#   stats    — index statistics
#   health   — backend connectivity
#   stream   — toggle streaming
#   history  — show conversation memory
#   clear    — clear conversation
#   intent   — test query classification
#   quit     — exit

# =============================================================================
# 5. API ENDPOINTS (curl examples)
# =============================================================================

# Health — no auth required
curl http://localhost:8000/health

# Swagger docs — open in browser
# http://localhost:8000/docs

# Chat (SSE streaming)
curl -N -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $RESEARCH_OS_API_KEY" \
  -d '{"query": "What is self-attention?"}'

# Chat with history
curl -N -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $RESEARCH_OS_API_KEY" \
  -d '{"query": "Explain that simply", "history": [{"role": "user", "content": "What is attention?"}, {"role": "assistant", "content": "Attention is..."}]}'

# Chat with intent override
curl -N -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $RESEARCH_OS_API_KEY" \
  -d '{"query": "Show me attention", "filter_type": "code"}'

# Upload file
curl -X POST http://localhost:8000/v1/ingest/file \
  -H "X-API-Key: $RESEARCH_OS_API_KEY" \
  -F "file=@path/to/paper.pdf"

# Ingest from URL
curl -X POST http://localhost:8000/v1/ingest/url \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $RESEARCH_OS_API_KEY" \
  -d '{"url": "https://arxiv.org/pdf/1706.03762v5", "filename": "attention.pdf"}'

# Check ingestion progress
curl http://localhost:8000/v1/ingest/status \
  -H "X-API-Key: $RESEARCH_OS_API_KEY"

# List indexed files
curl http://localhost:8000/v1/index/files \
  -H "X-API-Key: $RESEARCH_OS_API_KEY"

# =============================================================================
# 6. DATA INGESTION (CLI — no server needed)
# =============================================================================

# Batch ingest all files in data/ directories
python scripts/ingest_batch.py                # incremental (skips processed)
python scripts/ingest_batch.py --force         # reprocess everything
python scripts/ingest_batch.py --rebuild       # clear index and rebuild
python scripts/ingest_batch.py --status        # check what's indexed
python scripts/ingest_batch.py --data-dir data/02_papers  # specific directory

# Download papers
python scripts/auto_download.py --url "https://arxiv.org/pdf/..."
python scripts/auto_download.py --url "..." --folder 02_papers
python scripts/auto_download.py --dry-run      # preview without saving

# View ingestion ledger
cat data/index/processed_files.json

# =============================================================================
# 7. OLLAMA (Local LLM Fallback)
# =============================================================================

# Start server
ollama serve

# Manage models
ollama list                                   # show installed
ollama pull phi3:mini                         # download fallback model
ollama pull llama3.2:3b                       # alternative model
ollama rm <model_name>                        # remove model

# Test
ollama run phi3:mini "Hello"

# =============================================================================
# 8. GIT
# =============================================================================

git status
git add .
git commit -m "your message"
git push origin main
git pull origin main

# =============================================================================
# 9. COMMON WORKFLOWS
# =============================================================================

# --- Workflow A: Daily use ---
# 1. python webrun.py                         # starts everything, opens browser
# 2. Ask questions in the UI
# 3. Ctrl+C when done

# --- Workflow B: First time setup ---
# 1. conda activate Research-OS
# 2. pip install -r requirements.txt
# 3. cd frontend && npm install && cd ..
# 4. cp .env.example .env                     # edit with your keys
# 5. python scripts/ingest_batch.py           # index your PDFs
# 6. python verify_setup.py                   # confirm everything works
# 7. python webrun.py                         # launch

# --- Workflow C: Add new papers ---
# 1. Copy PDFs to data/02_papers/
# 2. python scripts/ingest_batch.py           # re-index (incremental)
# 3. python webrun.py                         # query the new content

# --- Workflow D: CLI only (no web) ---
# 1. conda activate Research-OS
# 2. python main.py                           # interactive chat
# 3. Or: python main.py -q "your question"    # single query

# --- Workflow E: Development ---
# 1. python run_api.py --reload --log-level debug
# 2. cd frontend && npm run dev               # in second terminal
# 3. Edit code — both servers auto-restart
# 4. python verify_setup.py                   # re-verify after changes
