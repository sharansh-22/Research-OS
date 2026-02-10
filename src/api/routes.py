"""
Research-OS API Routes
========================
All HTTP endpoints for the Research-OS backend.

Endpoints:
    POST /v1/chat          — Streaming chat (SSE)
    POST /v1/ingest/file   — Upload and ingest a file
    POST /v1/ingest/url    — Download and ingest from URL
    GET  /health           — System health and stats
"""

import os
import logging
import time
import shutil
from pathlib import Path
from typing import List, Dict, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File,
    BackgroundTasks,
    status,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
from sse_starlette.sse import EventSourceResponse

from .dependencies import verify_api_key, PipelineState

logger = logging.getLogger(__name__)


# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================

class ChatRequest(BaseModel):
    """Request body for /v1/chat."""
    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The research question to ask",
        examples=["Explain the math behind self-attention"],
    )
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Conversation history as [{role, content}, ...]",
        examples=[[
            {"role": "user", "content": "What is dropout?"},
            {"role": "assistant", "content": "Dropout is a regularization technique..."},
        ]],
    )
    filter_type: Optional[str] = Field(
        default=None,
        description="Override intent classification: 'code', 'theory', or 'hybrid'",
        pattern="^(code|theory|hybrid)$",
    )


class IngestURLRequest(BaseModel):
    """Request body for /v1/ingest/url."""
    url: str = Field(
        ...,
        description="URL of the file to download and ingest",
        examples=["https://arxiv.org/pdf/1706.03762v5"],
    )
    filename: Optional[str] = Field(
        default=None,
        description="Override filename (auto-detected from URL if omitted)",
    )


class IngestResponse(BaseModel):
    """Response for ingest endpoints."""
    status: str
    message: str
    filename: Optional[str] = None
    task_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Response for /health."""
    status: str
    version: str
    index_chunks: int
    indexed_files: int
    backends: Dict[str, bool]
    smart_routing: bool


# =============================================================================
# ROUTER
# =============================================================================

router = APIRouter()


# =============================================================================
# POST /v1/chat — Streaming Chat (SSE)
# =============================================================================

@router.post(
    "/v1/chat",
    summary="Streaming Research Chat",
    description=(
        "Send a research question and receive a Server-Sent Events stream. "
        "Events: `start`, `context`, `chunk` (tokens), `done`, `error`."
    ),
    response_description="SSE stream of JSON events",
    tags=["Chat"],
)
async def chat(
    request: ChatRequest,
    _api_key: str = Depends(verify_api_key),
):
    """
    Streaming RAG endpoint.
    
    Consumes `pipeline.query_stream(yield_json=True)` which yields
    JSON strings with event types: start, context, chunk, done, error.
    
    Each SSE `data:` field is a JSON object:
        {"event": "start", "intent": "hybrid"}
        {"event": "context", "code": 3, "theory": 5}
        {"event": "chunk", "data": "The attention"}
        {"event": "chunk", "data": " mechanism works"}
        {"event": "done"}
    """
    pipeline = PipelineState.get()
    
    # Validate history format
    for i, msg in enumerate(request.history):
        if "role" not in msg or "content" not in msg:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"history[{i}] must have 'role' and 'content' keys",
            )
        if msg["role"] not in ("user", "assistant"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"history[{i}].role must be 'user' or 'assistant'",
            )
    
    def event_generator():
        """
        Sync generator that yields SSE-formatted strings.
        
        pipeline.query_stream(yield_json=True) returns JSON strings
        from StreamChunk.to_json(). We yield them directly as SSE data.
        """
        try:
            for json_str in pipeline.query_stream(
                question=request.query,
                history=request.history,
                filter_type=request.filter_type,
                yield_json=True,
            ):
                yield {"data": json_str}
        except Exception as e:
            logger.error(f"SSE stream error: {e}", exc_info=True)
            import json
            error_payload = json.dumps({"event": "error", "error": str(e)})
            yield {"data": error_payload}
    
    return EventSourceResponse(event_generator())


# =============================================================================
# POST /v1/ingest/file — Upload File
# =============================================================================

UPLOAD_DIR = Path("data/04_misc")


def _run_ingest_file(file_path: Path):
    """Background task: ingest a single file into the pipeline."""
    try:
        pipeline = PipelineState.get()
        result = pipeline.ingest_pdf(file_path, force=False)
        
        if result.status == "processed":
            pipeline.save_index()
            logger.info(
                f"Ingestion complete: {result.filename} — "
                f"{result.chunks_added} chunks in {result.processing_time:.1f}s"
            )
        else:
            logger.warning(f"Ingestion result: {result.filename} — {result.status}: {result.message}")
            
    except Exception as e:
        logger.error(f"Background ingest failed for {file_path}: {e}", exc_info=True)


@router.post(
    "/v1/ingest/file",
    response_model=IngestResponse,
    summary="Upload & Ingest File",
    description="Upload a document file. Ingestion runs as a background task.",
    tags=["Ingest"],
)
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(
        ...,
        description="Document file to ingest (PDF, .py, .md, .tex, .ipynb, etc.)",
    ),
    _api_key: str = Depends(verify_api_key),
):
    """
    Accept a file upload, save to data/04_misc/, and queue ingestion.
    
    Supported formats: .pdf, .py, .ipynb, .md, .tex, .cpp, .cu, .c, .h, .txt
    """
    # Validate extension
    from src.rag.data_loader import UniversalLoader
    supported = UniversalLoader.supported_extensions()
    
    suffix = Path(file.filename).suffix.lower()
    if suffix not in supported:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: '{suffix}'. Supported: {supported}",
        )
    
    # Validate file is not empty
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    
    # Save file
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sanitize filename
    safe_name = Path(file.filename).name  # Strip directory traversal
    dest_path = UPLOAD_DIR / safe_name
    
    # Handle duplicate filenames
    if dest_path.exists():
        stem = dest_path.stem
        suffix_ext = dest_path.suffix
        counter = 1
        while dest_path.exists():
            dest_path = UPLOAD_DIR / f"{stem}_{counter}{suffix_ext}"
            counter += 1
    
    try:
        with open(dest_path, "wb") as f:
            f.write(contents)
    except IOError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {e}",
        )
    
    # Queue background ingestion
    background_tasks.add_task(_run_ingest_file, dest_path)
    
    logger.info(f"File saved: {dest_path} ({len(contents)} bytes) — ingestion queued")
    
    return IngestResponse(
        status="queued",
        message=f"File '{safe_name}' saved and queued for ingestion.",
        filename=str(dest_path),
    )


# =============================================================================
# POST /v1/ingest/url — Download & Ingest from URL
# =============================================================================

def _run_ingest_url(url: str, filename: Optional[str] = None):
    """Background task: download file from URL and ingest."""
    import requests
    
    try:
        # Determine filename
        if filename:
            safe_name = Path(filename).name
        else:
            # Extract from URL
            from urllib.parse import urlparse, unquote
            parsed = urlparse(url)
            url_filename = unquote(Path(parsed.path).name)
            safe_name = url_filename if url_filename and '.' in url_filename else "downloaded_paper.pdf"
        
        # Download
        logger.info(f"Downloading: {url}")
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()
        
        # Save
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest_path = UPLOAD_DIR / safe_name
        
        # Handle duplicates
        if dest_path.exists():
            stem = dest_path.stem
            suffix_ext = dest_path.suffix
            counter = 1
            while dest_path.exists():
                dest_path = UPLOAD_DIR / f"{stem}_{counter}{suffix_ext}"
                counter += 1
        
        with open(dest_path, "wb") as f:
            for data_chunk in response.iter_content(chunk_size=8192):
                f.write(data_chunk)
        
        file_size = dest_path.stat().st_size
        logger.info(f"Downloaded: {dest_path} ({file_size} bytes)")
        
        # Ingest
        _run_ingest_file(dest_path)
        
    except requests.RequestException as e:
        logger.error(f"Download failed for {url}: {e}")
    except Exception as e:
        logger.error(f"URL ingest failed for {url}: {e}", exc_info=True)


@router.post(
    "/v1/ingest/url",
    response_model=IngestResponse,
    summary="Download & Ingest from URL",
    description="Provide a URL to a document. Download and ingestion run as background tasks.",
    tags=["Ingest"],
)
async def ingest_url(
    request: IngestURLRequest,
    background_tasks: BackgroundTasks,
    _api_key: str = Depends(verify_api_key),
):
    """Download a file from URL and queue it for ingestion."""
    
    background_tasks.add_task(_run_ingest_url, request.url, request.filename)
    
    logger.info(f"URL ingest queued: {request.url}")
    
    return IngestResponse(
        status="queued",
        message=f"Download from '{request.url}' queued for ingestion.",
        filename=request.filename,
    )


# =============================================================================
# GET /health — System Health
# =============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health Check",
    description="Returns system status, index size, and backend connectivity.",
    tags=["System"],
)
async def health():
    """
    Public endpoint (no auth required).
    Returns pipeline stats and backend availability.
    """
    if not PipelineState.is_ready():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unavailable",
                "version": "2.0.0",
                "index_chunks": 0,
                "indexed_files": 0,
                "backends": {"groq": False, "ollama_fallback": False},
                "smart_routing": False,
            },
        )
    
    pipeline = PipelineState.get()
    
    try:
        stats = pipeline.get_stats()
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "degraded",
                "version": "2.0.0",
                "index_chunks": pipeline.index_size,
                "indexed_files": len(pipeline.get_processed_files()),
                "backends": {"groq": False, "ollama_fallback": False},
                "smart_routing": False,
                "error": str(e),
            },
        )
    
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        index_chunks=stats["total_chunks"],
        indexed_files=stats["processed_files"],
        backends=stats["backends"],
        smart_routing=stats["smart_routing"],
    )