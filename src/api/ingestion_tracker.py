"""
Ingestion Status Tracker
=========================
In-memory store for tracking background ingestion task progress.
Polled by frontend via GET /v1/ingest/status/<task_id>
"""

import time
import uuid
import threading
from typing import Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


class IngestionStage(str, Enum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    PARSING = "parsing"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class IngestionTask:
    task_id: str
    filename: str
    status: IngestionStage = IngestionStage.QUEUED
    progress: float = 0.0
    chunks_added: int = 0
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "filename": self.filename,
            "status": self.status.value,
            "progress": round(self.progress, 2),
            "chunks_added": self.chunks_added,
            "error": self.error,
            "elapsed": round(time.time() - self.created_at, 1),
        }

    def update(self, status: IngestionStage, progress: float = None):
        self.status = status
        if progress is not None:
            self.progress = progress
        self.updated_at = time.time()


class IngestionTracker:
    """Thread-safe in-memory tracker for ingestion tasks."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._tasks: Dict[str, IngestionTask] = {}
        return cls._instance

    def create_task(self, filename: str) -> str:
        task_id = uuid.uuid4().hex[:12]
        self._tasks[task_id] = IngestionTask(
            task_id=task_id,
            filename=filename,
        )
        return task_id

    def get_task(self, task_id: str) -> Optional[IngestionTask]:
        return self._tasks.get(task_id)

    def update_task(self, task_id: str, status: IngestionStage, progress: float = None, chunks: int = None, error: str = None):
        task = self._tasks.get(task_id)
        if task:
            task.update(status, progress)
            if chunks is not None:
                task.chunks_added = chunks
            if error is not None:
                task.error = error

    def get_all(self) -> list:
        cutoff = time.time() - 3600
        active = {k: v for k, v in self._tasks.items() if v.created_at > cutoff}
        self._tasks = active
        return [t.to_dict() for t in sorted(active.values(), key=lambda x: x.created_at, reverse=True)]

    def clear(self):
        self._tasks.clear()


tracker = IngestionTracker()
