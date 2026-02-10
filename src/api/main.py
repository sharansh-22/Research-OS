"""
Research-OS API — Application Factory
=======================================
Entry point for the FastAPI application.

Responsibilities:
    - Create FastAPI app with metadata
    - CORS middleware (configurable)
    - Lifespan management (pipeline init/shutdown)
    - Include all routers
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .dependencies import PipelineState
from .routes import router

logger = logging.getLogger(__name__)


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Startup:
        - Load the RAG pipeline (embedder, retriever, generator, verifier)
        - Load existing FAISS index from disk
    
    Shutdown:
        - Save index to disk
        - Release resources
    """
    # ---- STARTUP ----
    logger.info("=" * 60)
    logger.info("Research-OS API starting up...")
    logger.info("=" * 60)
    
    index_dir = os.environ.get("RESEARCH_OS_INDEX_DIR", "data/index")
    
    try:
        PipelineState.initialize(
            index_dir=index_dir,
            load_existing=True,
        )
    except Exception as e:
        logger.error(f"FATAL: Pipeline initialization failed: {e}", exc_info=True)
        # Still yield — health endpoint will report unavailable
        # This prevents the entire server from crashing on startup
    
    logger.info("Research-OS API ready")
    logger.info("=" * 60)
    
    yield  # ---- APP IS RUNNING ----
    
    # ---- SHUTDOWN ----
    logger.info("Research-OS API shutting down...")
    PipelineState.shutdown()
    logger.info("Shutdown complete")


# =============================================================================
# APP FACTORY
# =============================================================================

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI app instance.
    """
    app = FastAPI(
        title="Research-OS API",
        description=(
            "RAG-powered research assistant with smart query routing, "
            "streaming responses, and multi-format document ingestion."
        ),
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    
    # ----- CORS Middleware -----
    allowed_origins = os.environ.get(
        "RESEARCH_OS_CORS_ORIGINS", "*"
    ).split(",")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in allowed_origins],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ----- Routers -----
    app.include_router(router)
    
    # ----- Root redirect -----
    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": "Research-OS API",
            "version": "2.0.0",
            "docs": "/docs",
            "health": "/health",
        }
    
    return app
