"""Response models for API."""
from pydantic import BaseModel
from typing import Optional, Dict, Any

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    total_chunks: int = 0
    processed_files: int = 0
    uptime: float

class IngestResponse(BaseModel):
    """Ingestion response."""
    status: str
    message: str
    filename: Optional[str] = None
