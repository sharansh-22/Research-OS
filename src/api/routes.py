"""
Research-OS API Routes
========================
All HTTP endpoints for the Research-OS backend.

Endpoints:
    POST /v1/chat          - Streaming chat (SSE)
    POST /v1/ingest/file   - Upload and ingest a file
    POST /v1/ingest/url    - Download and ingest from URL
    GET  /v1/ingest/status - All ingestion task statuses
    GET  /v1/ingest/status/{task_id} - Single task status
    GET  /health           - System health and stats
"""

import os
import logging
import time
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
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from .dependencies import verify_api_key, PipelineState
from .ingestion_tracker import tracker, IngestionStage

logger = logging.getLogger(__name__)


# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    history: List[Dict[str, str]] = Field(default_factory=list)
    filter_type: Optional[str] = Field(default=None, pattern="^(code|theory|hybrid)$")


class IngestURLRequest(BaseModel):
    url: str = Field(...)
    filename: Optional[str] = Field(default=None)


class IngestResponse(BaseModel):
    status: str
    message: str
    filename: Optional[str] = None
    task_id: Optional[str] = None


class HealthResponse(BaseModel):
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
# POST /v1/chat
# =============================================================================

@router.post("/v1/chat", tags=["Chat"])
async def chat(request: ChatRequest, _api_key: str = Depends(verify_api_key)):
    pipeline = PipelineState.get()

    for i, msg in enumerate(request.history):
        if "role" not in msg or "content" not in msg:
            raise HTTPException(status_code=422, detail=f"history[{i}] must have 'role' and 'content'")
        if msg["role"] not in ("user", "assistant"):
            raise HTTPException(status_code=422, detail=f"history[{i}].role must be 'user' or 'assistant'")

    def event_generator():
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
            yield {"data": json.dumps({"event": "error", "error": str(e)})}

    return EventSourceResponse(event_generator())


# =============================================================================
# POST /v1/ingest/file
# =============================================================================

UPLOAD_DIR = Path("data/04_misc")


def _run_ingest_file(file_path: Path, task_id: str):
    try:
        tracker.update_task(task_id, IngestionStage.PARSING, 0.2)

        pipeline = PipelineState.get()

        tracker.update_task(task_id, IngestionStage.EMBEDDING, 0.4)

        result = pipeline.ingest_pdf(file_path, force=False)

        if result.status == "processed":
            tracker.update_task(task_id, IngestionStage.INDEXING, 0.8, chunks=result.chunks_added)
            pipeline.save_index()
            tracker.update_task(task_id, IngestionStage.COMPLETE, 1.0, chunks=result.chunks_added)
            logger.info(f"Ingestion complete: {result.filename} - {result.chunks_added} chunks")
        elif result.status == "skipped":
            tracker.update_task(task_id, IngestionStage.COMPLETE, 1.0)
            logger.info(f"Ingestion skipped: {result.filename} - already processed")
        else:
            tracker.update_task(task_id, IngestionStage.FAILED, error=result.message)
            logger.warning(f"Ingestion failed: {result.filename} - {result.message}")

    except Exception as e:
        tracker.update_task(task_id, IngestionStage.FAILED, error=str(e))
        logger.error(f"Background ingest failed for {file_path}: {e}", exc_info=True)


@router.post("/v1/ingest/file", response_model=IngestResponse, tags=["Ingest"])
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    _api_key: str = Depends(verify_api_key),
):
    from src.rag.data_loader import UniversalLoader
    supported = UniversalLoader.supported_extensions()

    suffix = Path(file.filename).suffix.lower()
    if suffix not in supported:
        raise HTTPException(status_code=415, detail=f"Unsupported: '{suffix}'. Supported: {supported}")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name
    dest_path = UPLOAD_DIR / safe_name

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
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    task_id = tracker.create_task(safe_name)
    background_tasks.add_task(_run_ingest_file, dest_path, task_id)

    logger.info(f"File saved: {dest_path} ({len(contents)} bytes) - task {task_id}")

    return IngestResponse(
        status="queued",
        message=f"File '{safe_name}' queued for ingestion.",
        filename=str(dest_path),
        task_id=task_id,
    )


# =============================================================================
# POST /v1/ingest/url
# =============================================================================

def _run_ingest_url(url: str, filename: Optional[str], task_id: str):
    import requests as req

    try:
        tracker.update_task(task_id, IngestionStage.DOWNLOADING, 0.1)

        if filename:
            safe_name = Path(filename).name
        else:
            from urllib.parse import urlparse, unquote
            parsed = urlparse(url)
            url_filename = unquote(Path(parsed.path).name)
            safe_name = url_filename if url_filename and '.' in url_filename else "downloaded_paper.pdf"

        logger.info(f"Downloading: {url}")
        response = req.get(url, timeout=120, stream=True)
        response.raise_for_status()

        tracker.update_task(task_id, IngestionStage.DOWNLOADING, 0.3)

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest_path = UPLOAD_DIR / safe_name

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

        logger.info(f"Downloaded: {dest_path} ({dest_path.stat().st_size} bytes)")

        _run_ingest_file(dest_path, task_id)

    except Exception as e:
        tracker.update_task(task_id, IngestionStage.FAILED, error=str(e))
        logger.error(f"URL ingest failed for {url}: {e}", exc_info=True)


@router.post("/v1/ingest/url", response_model=IngestResponse, tags=["Ingest"])
async def ingest_url(
    request: IngestURLRequest,
    background_tasks: BackgroundTasks,
    _api_key: str = Depends(verify_api_key),
):
    display_name = request.filename or request.url.split("/")[-1] or "download"
    task_id = tracker.create_task(display_name)
    background_tasks.add_task(_run_ingest_url, request.url, request.filename, task_id)

    logger.info(f"URL ingest queued: {request.url} - task {task_id}")

    return IngestResponse(
        status="queued",
        message=f"Download from '{request.url}' queued.",
        filename=request.filename,
        task_id=task_id,
    )


# =============================================================================
# GET /v1/ingest/status
# =============================================================================

@router.get("/v1/ingest/status", tags=["Ingest"])
async def ingest_status_all(_api_key: str = Depends(verify_api_key)):
    return {"tasks": tracker.get_all()}


@router.get("/v1/ingest/status/{task_id}", tags=["Ingest"])
async def ingest_status(task_id: str, _api_key: str = Depends(verify_api_key)):
    task = tracker.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return task.to_dict()


# =============================================================================
# GET /v1/index/files
# =============================================================================

@router.get("/v1/index/files", tags=["Index"])
async def index_files(_api_key: str = Depends(verify_api_key)):
    pipeline = PipelineState.get()
    return {
        "files": pipeline.get_processed_files(),
        "total": len(pipeline.get_processed_files()),
        "chunks": pipeline.index_size,
    }


# =============================================================================
# GET /health
# =============================================================================

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    if not PipelineState.is_ready():
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable", "version": "2.1.0",
                "index_chunks": 0, "indexed_files": 0,
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
            status_code=503,
            content={
                "status": "degraded", "version": "2.1.0",
                "index_chunks": pipeline.index_size,
                "indexed_files": len(pipeline.get_processed_files()),
                "backends": {"groq": False, "ollama_fallback": False},
                "smart_routing": False,
            },
        )

    return HealthResponse(
        status="healthy",
        version="2.1.0",
        index_chunks=stats["total_chunks"],
        indexed_files=stats["processed_files"],
        backends=stats["backends"],
        smart_routing=stats["smart_routing"],
    )
