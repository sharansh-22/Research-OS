"""Request models for API."""
from pydantic import BaseModel, Field
from typing import Optional, List

class ChatMessage(BaseModel):
    """Single chat message."""
    role: str
    content: str

class ChatRequest(BaseModel):
    """Chat request."""
    query: str = Field(..., min_length=1)
    history: Optional[List[ChatMessage]] = Field(default=[])
    filter_type: Optional[str] = None

class IngestRequest(BaseModel):
    """Ingestion request."""
    source: str
    source_type: str = "path"
    force: bool = False
