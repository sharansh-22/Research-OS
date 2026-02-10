"""
Research-OS FastAPI Backend
============================
RESTful API with Server-Sent Events (SSE) for streaming responses.

Endpoints:
- POST /v1/chat     - Streaming chat with RAG
- POST /v1/ingest   - Ingest documents (background task)
- GET  /v1/search   - Search without generation
- GET  /health      - Health check
- GET  /stats       - Index statistics

Run:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import ResearchPipeline, PipelineConfig

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)7s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("research-os-api")

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Settings:
    """API Configuration."""
    INDEX_DIR: str = os.getenv("INDEX_DIR", "data/index")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    CORS_ORIGINS: List[str] = ["*"]  # Allow all for development
    MAX_HISTORY_TURNS: int = 3


settings = Settings()


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ChatMessage(BaseModel):
    """Single chat message."""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    """Request body for /v1/chat."""
    query: str = Field(..., min_length=1, max_length=10000)
    history: List[ChatMessage] = Field(default_factory=list)
    stream: bool = Field(default=True)
    top_k: int = Field(default=5, ge=1, le=20)
    filter_type: Optional[str] = Field(default=None, pattern="^(code|theory|hybrid)$")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Explain the attention mechanism in transformers",
                "history": [
                    {"role": "user", "content": "What is deep learning?"},
                    {"role": "assistant", "content": "Deep learning is..."}
                ],
                "stream": True,
                "top_k": 5
            }
        }


class ChatResponse(BaseModel):
    """Response body for non-streaming /v1/chat."""
    response: str
    intent: str
    context: Dict[str, int]
    model: str
    backend: str


class IngestRequest(BaseModel):
    """Request body for /v1/ingest."""
    url: Optional[str] = Field(default=None)
    path: Optional[str] = Field(default=None)
    category: Optional[str] = Field(
        default=None, 
        pattern="^(01_fundamentals|02_papers|03_implementation|04_misc)$"
    )
    force: bool = Field(default=False)
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://arxiv.org/pdf/1706.03762.pdf",
                "category": "02_papers"
            }
        }


class IngestResponse(BaseModel):
    """Response for /v1/ingest."""
    status: str
    message: str
    task_id: Optional[str] = None


class SearchRequest(BaseModel):
    """Request body for /v1/search."""
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    filter_type: Optional[str] = Field(default="hybrid")


class SearchResult(BaseModel):
    """Single search result."""
    content: str
    chunk_type: str
    score: float
    source: str
    metadata: Dict


class SearchResponse(BaseModel):
    """Response for /v1/search."""
    query: str
    intent: str
    results: List[SearchResult]
    total: int


class HealthResponse(BaseModel):
    """Response for /health."""
    status: str
    timestamp: str
    index_loaded: bool
    docs_count: int
    chunk_types: Dict[str, int]
    backends: Dict[str, bool]


class StatsResponse(BaseModel):
    """Response for /stats."""
    total_chunks: int
    chunk_types: Dict[str, int]
    processed_files: int
    backends: Dict[str, bool]
    smart_routing: bool
    config: Dict


# =============================================================================
# APPLICATION STATE
# =============================================================================

class AppState:
    """Application state container."""
    pipeline: Optional[ResearchPipeline] = None
    ready: bool = False
    startup_time: Optional[datetime] = None


state = AppState()


# =============================================================================
# LIFESPAN (STARTUP/SHUTDOWN)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    - Loads the RAG pipeline on startup
    - Cleans up on shutdown
    """
    # Startup
    logger.info("=" * 60)
    logger.info("Starting Research-OS API...")
    logger.info("=" * 60)
    
    # Check API key
    if not settings.GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set! Chat endpoints will fail.")
    
    try:
        # Initialize pipeline
        config = PipelineConfig(
            index_dir=settings.INDEX_DIR,
            enable_fallback=True,
        )
        
        state.pipeline = ResearchPipeline(config)
        
        # Load existing index if available
        index_path = Path(settings.INDEX_DIR)
        if index_path.exists() and (index_path / "faiss.index").exists():
            state.pipeline.load_index()
            logger.info(f"✓ Loaded index: {state.pipeline.index_size} chunks")
        else:
            logger.warning(f"No index found at {settings.INDEX_DIR}")
        
        state.ready = True
        state.startup_time = datetime.now()
        
        logger.info("✓ Pipeline ready")
        logger.info(f"✓ API running at http://{settings.HOST}:{settings.PORT}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Research-OS API...")
    state.ready = False
    state.pipeline = None
    logger.info("✓ Shutdown complete")


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Research-OS API",
    description="RAG System with Smart Query Routing, Memory, and Streaming",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# DEPENDENCY
# =============================================================================

def get_pipeline() -> ResearchPipeline:
    """Dependency to get the pipeline instance."""
    if not state.ready or state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Pipeline not initialized."
        )
    return state.pipeline


# =============================================================================
# STREAMING HELPER
# =============================================================================

async def stream_chat_response(
    pipeline: ResearchPipeline,
    query: str,
    history: List[Dict[str, str]],
    filter_type: Optional[str],
    top_k: int,
) -> AsyncGenerator[str, None]:
    """
    Async generator for SSE streaming.
    
    Yields JSON-formatted events:
    - {"event": "start", "intent": "..."}
    - {"event": "chunk", "data": "..."}
    - {"event": "context", "code": N, "theory": M}
    - {"event": "done"}
    - {"event": "error", "message": "..."}
    """
    try:
        # Classify intent
        if filter_type:
            intent = filter_type
        else:
            intent = pipeline.classify_intent(query)
        
        # Send start event
        yield json.dumps({
            "event": "start",
            "intent": intent,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Retrieve context
        results = pipeline.retriever.search_by_type_filtered(
            query=query,
            top_k=top_k,
            filter_type=intent,
        )
        code_results = results.get("code", [])
        theory_results = results.get("theory", [])
        
        # Send context info
        yield json.dumps({
            "event": "context",
            "code": len(code_results),
            "theory": len(theory_results),
        })
        
        # Stream generation
        for chunk in pipeline.generator.generate_stream(
            query=query,
            code_results=code_results,
            theory_results=theory_results,
            history=history,
        ):
            yield json.dumps({
                "event": "chunk",
                "data": chunk,
            })
            # Small delay to prevent overwhelming client
            await asyncio.sleep(0)
        
        # Send completion event
        yield json.dumps({
            "event": "done",
            "timestamp": datetime.now().isoformat(),
        })
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield json.dumps({
            "event": "error",
            "message": str(e),
        })


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Research-OS API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns API status and index statistics.
    """
    if not state.ready or state.pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    try:
        stats = state.pipeline.get_stats()
        backend_status = state.pipeline.generator.health_check()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            index_loaded=state.pipeline.index_size > 0,
            docs_count=state.pipeline.index_size,
            chunk_types=stats.get("chunk_types", {}),
            backends=backend_status,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse, tags=["Health"])
async def get_stats():
    """Get detailed index statistics."""
    pipeline = get_pipeline()
    stats = pipeline.get_stats()
    
    return StatsResponse(
        total_chunks=stats["total_chunks"],
        chunk_types=stats["chunk_types"],
        processed_files=stats["processed_files"],
        backends=stats["backends"],
        smart_routing=stats["smart_routing"],
        config=stats["config"],
    )


@app.post("/v1/chat", tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Chat endpoint with RAG.
    
    Supports both streaming (SSE) and non-streaming responses.
    
    **Streaming (default):**
    Returns Server-Sent Events with JSON payloads:
    - `{"event": "start", "intent": "code"}`
    - `{"event": "chunk", "data": "token"}`
    - `{"event": "context", "code": 3, "theory": 5}`
    - `{"event": "done"}`
    
    **Non-streaming:**
    Returns complete JSON response.
    """
    pipeline = get_pipeline()
    
    # Convert history to dict format
    history = [{"role": m.role, "content": m.content} for m in request.history]
    
    # Limit history
    max_messages = settings.MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        history = history[-max_messages:]
    
    if request.stream:
        # Streaming response (SSE)
        return EventSourceResponse(
            stream_chat_response(
                pipeline=pipeline,
                query=request.query,
                history=history,
                filter_type=request.filter_type,
                top_k=request.top_k,
            ),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming response
        try:
            result = pipeline.query(
                question=request.query,
                history=history,
                filter_type=request.filter_type,
            )
            
            return ChatResponse(
                response=result.response,
                intent=result.intent,
                context={
                    "code": len(result.code_context),
                    "theory": len(result.theory_context),
                },
                model=result.generation_metadata.get("model", "unknown"),
                backend=result.generation_metadata.get("backend", "unknown"),
            )
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    Search endpoint (retrieval only, no generation).
    
    Returns matching chunks from the index.
    """
    pipeline = get_pipeline()
    
    try:
        # Classify intent
        intent = pipeline.classify_intent(request.query)
        filter_type = request.filter_type or intent
        
        # Search
        results = pipeline.retriever.search(
            query=request.query,
            top_k=request.top_k,
            filter_type=filter_type,
        )
        
        # Format results
        search_results = [
            SearchResult(
                content=r.chunk.content[:500],  # Truncate for response
                chunk_type=r.chunk.chunk_type.value,
                score=r.score,
                source=r.source,
                metadata=r.chunk.metadata,
            )
            for r in results
        ]
        
        return SearchResponse(
            query=request.query,
            intent=intent,
            results=search_results,
            total=len(search_results),
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
):
    """
    Ingest documents endpoint.
    
    Runs ingestion as a background task to avoid blocking.
    
    **URL ingestion:** Downloads and classifies the document.
    **Path ingestion:** Ingests a local file.
    """
    pipeline = get_pipeline()
    
    if not request.url and not request.path:
        raise HTTPException(
            status_code=400,
            detail="Either 'url' or 'path' must be provided"
        )
    
    # Generate task ID
    task_id = f"ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Add background task
    if request.url:
        background_tasks.add_task(
            ingest_from_url,
            pipeline=pipeline,
            url=request.url,
            category=request.category,
            task_id=task_id,
        )
        message = f"Ingestion started for URL: {request.url}"
    else:
        background_tasks.add_task(
            ingest_from_path,
            pipeline=pipeline,
            path=request.path,
            force=request.force,
            task_id=task_id,
        )
        message = f"Ingestion started for path: {request.path}"
    
    return IngestResponse(
        status="accepted",
        message=message,
        task_id=task_id,
    )


@app.post("/v1/ingest/batch", tags=["Ingestion"])
async def ingest_batch(
    background_tasks: BackgroundTasks,
    force: bool = Query(default=False),
    rebuild: bool = Query(default=False),
):
    """
    Batch ingestion endpoint.
    
    Ingests all files from default data directories.
    """
    pipeline = get_pipeline()
    
    task_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    background_tasks.add_task(
        run_batch_ingestion,
        pipeline=pipeline,
        force=force,
        rebuild=rebuild,
        task_id=task_id,
    )
    
    return IngestResponse(
        status="accepted",
        message="Batch ingestion started",
        task_id=task_id,
    )


@app.get("/v1/intent", tags=["Utilities"])
async def classify_intent(query: str = Query(..., min_length=1)):
    """
    Test intent classification for a query.
    
    Returns the detected intent without performing search.
    """
    pipeline = get_pipeline()
    
    intent = pipeline.classify_intent(query)
    
    return {
        "query": query,
        "intent": intent,
    }


# =============================================================================
# BACKGROUND TASKS
# =============================================================================

async def ingest_from_url(
    pipeline: ResearchPipeline,
    url: str,
    category: Optional[str],
    task_id: str,
):
    """Background task: Ingest from URL."""
    logger.info(f"[{task_id}] Starting URL ingestion: {url}")
    
    try:
        # Import auto_download function
        from scripts.auto_download import download_and_classify
        
        success, filename, folder = download_and_classify(
            url=url,
            override_folder=category,
            dry_run=False,
        )
        
        if success:
            # Ingest the downloaded file
            file_path = Path("data") / folder / filename
            result = pipeline.ingest_pdf(file_path, force=True)
            
            if result.status == "processed":
                pipeline.save_index()
                logger.info(f"[{task_id}] ✓ Ingested: {filename} ({result.chunks_added} chunks)")
            else:
                logger.warning(f"[{task_id}] Ingestion failed: {result.message}")
        else:
            logger.error(f"[{task_id}] Download failed: {url}")
            
    except Exception as e:
        logger.error(f"[{task_id}] Ingestion error: {e}")


async def ingest_from_path(
    pipeline: ResearchPipeline,
    path: str,
    force: bool,
    task_id: str,
):
    """Background task: Ingest from local path."""
    logger.info(f"[{task_id}] Starting path ingestion: {path}")
    
    try:
        file_path = Path(path)
        
        if file_path.is_dir():
            results = pipeline.ingest_directory(file_path, force=force)
            processed = sum(1 for r in results if r.status == "processed")
            logger.info(f"[{task_id}] ✓ Ingested directory: {processed} files")
        else:
            result = pipeline.ingest_pdf(file_path, force=force)
            logger.info(f"[{task_id}] ✓ Ingested: {result.filename} ({result.status})")
        
        pipeline.save_index()
        
    except Exception as e:
        logger.error(f"[{task_id}] Ingestion error: {e}")


async def run_batch_ingestion(
    pipeline: ResearchPipeline,
    force: bool,
    rebuild: bool,
    task_id: str,
):
    """Background task: Run batch ingestion."""
    logger.info(f"[{task_id}] Starting batch ingestion (force={force}, rebuild={rebuild})")
    
    try:
        data_dirs = [
            "data/01_fundamentals",
            "data/02_papers",
            "data/03_implementation",
            "data/04_misc",
        ]
        
        if rebuild:
            summary = pipeline.rebuild_index(data_dirs)
            logger.info(f"[{task_id}] ✓ Rebuild complete: {summary}")
        else:
            total = 0
            for dir_path in data_dirs:
                if Path(dir_path).exists():
                    results = pipeline.ingest_directory(dir_path, force=force)
                    processed = sum(1 for r in results if r.status == "processed")
                    total += processed
            
            pipeline.save_index()
            logger.info(f"[{task_id}] ✓ Batch ingestion complete: {total} files")
            
    except Exception as e:
        logger.error(f"[{task_id}] Batch ingestion error: {e}")


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info",
    )
