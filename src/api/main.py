"""
Research-OS API - Application Factory
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .dependencies import PipelineState
from .routes import router
from ..rag.config import VERSION, DEFAULT_INDEX_DIR

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info(f"Research-OS API v{VERSION} starting up...")
    logger.info("=" * 60)

    index_dir = os.environ.get("RESEARCH_OS_INDEX_DIR", DEFAULT_INDEX_DIR)

    try:
        PipelineState.initialize(index_dir=index_dir, load_existing=True)
    except Exception as e:
        logger.error(f"FATAL: Pipeline initialization failed: {e}", exc_info=True)

    logger.info("Research-OS API ready")
    logger.info("=" * 60)

    yield

    logger.info("Research-OS API shutting down...")
    PipelineState.shutdown()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Research-OS API",
        description="RAG-powered research assistant",
        version=VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:5174",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5174",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/", include_in_schema=False)
    async def root():
        return {"service": "Research-OS API", "version": VERSION, "docs": "/docs", "health": "/health"}

    return app
