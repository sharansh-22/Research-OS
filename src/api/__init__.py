"""
Research-OS API Package
========================
Modular FastAPI backend for the Research-OS RAG system.

Modules:
    - main: App factory, CORS, lifespan management
    - routes: HTTP endpoints (chat, ingest, health)
    - dependencies: Security, shared state (pipeline singleton)
"""

from .main import create_app

__all__ = ["create_app"]