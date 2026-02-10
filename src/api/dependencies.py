"""
Research-OS API Dependencies
==============================
- API key verification (X-API-Key header)
- Global RAG pipeline singleton (loaded once at startup)
"""

import os
import logging
from typing import Optional

from fastapi import Header, HTTPException, status

from src.rag.pipeline import ResearchPipeline, PipelineConfig, create_pipeline

logger = logging.getLogger(__name__)


# =============================================================================
# PIPELINE SINGLETON
# =============================================================================

class PipelineState:
    """
    Holds the single RAG pipeline instance for the entire application.
    
    Lifecycle:
        - `initialize()` is called once during app startup (lifespan).
        - `get()` is called per-request via dependency injection.
        - `shutdown()` is called once during app teardown.
    """
    
    _instance: Optional[ResearchPipeline] = None
    
    @classmethod
    def initialize(
        cls,
        index_dir: str = "data/index",
        load_existing: bool = True,
    ) -> ResearchPipeline:
        """
        Create and store the pipeline singleton.
        Called exactly once during application startup.
        """
        if cls._instance is not None:
            logger.warning("Pipeline already initialized — returning existing instance")
            return cls._instance
        
        logger.info("Initializing RAG pipeline...")
        cls._instance = create_pipeline(
            index_dir=index_dir,
            load_existing=load_existing,
        )
        logger.info(
            f"Pipeline ready — {cls._instance.index_size} chunks indexed, "
            f"{len(cls._instance.get_processed_files())} files tracked"
        )
        return cls._instance
    
    @classmethod
    def get(cls) -> ResearchPipeline:
        """
        Return the pipeline singleton.
        Raises if called before `initialize()`.
        """
        if cls._instance is None:
            raise RuntimeError(
                "Pipeline not initialized. "
                "This should never happen — check app lifespan."
            )
        return cls._instance
    
    @classmethod
    def shutdown(cls) -> None:
        """
        Cleanup on application shutdown.
        Saves the index to persist any in-memory changes.
        """
        if cls._instance is not None:
            try:
                cls._instance.save_index()
                logger.info("Pipeline index saved on shutdown")
            except Exception as e:
                logger.error(f"Failed to save index on shutdown: {e}")
            cls._instance = None
    
    @classmethod
    def is_ready(cls) -> bool:
        """Check if pipeline is initialized and has data."""
        return cls._instance is not None


# =============================================================================
# SECURITY
# =============================================================================

def get_api_key_name() -> str:
    """Return the env var name used for the API key."""
    return "RESEARCH_OS_API_KEY"


async def verify_api_key(
    x_api_key: str = Header(
        ...,
        alias="X-API-Key",
        description="API key for authentication",
    ),
) -> str:
    """
    FastAPI dependency that validates the X-API-Key header.
    
    Compares against the RESEARCH_OS_API_KEY environment variable.
    If the env var is not set, ALL requests are rejected (fail-closed).
    
    Returns:
        The validated API key string.
        
    Raises:
        HTTPException 401: Missing or empty key
        HTTPException 403: Invalid key
        HTTPException 500: Server-side key not configured
    """
    # --- Server-side key ---
    server_key = os.environ.get("RESEARCH_OS_API_KEY", "").strip()
    
    if not server_key:
        logger.error(
            "RESEARCH_OS_API_KEY is not set. "
            "All authenticated requests will be rejected."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server API key not configured. Contact administrator.",
        )
    
    # --- Client-side key ---
    if not x_api_key or not x_api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # --- Comparison (constant-time) ---
    import hmac
    if not hmac.compare_digest(x_api_key.strip(), server_key):
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )
    
    return x_api_key