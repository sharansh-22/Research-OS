"""Fast Embedder using FastEmbed (all-MiniLM-L6-v2)"""

import numpy as np
from typing import List, Optional, Union
import logging

from fastembed import TextEmbedding

logger = logging.getLogger(__name__)


class FastEmbedder:
    """Text embedding using FastEmbed."""
    
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._dimension = self.EMBEDDING_DIM
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = TextEmbedding(model_name=model_name)
        logger.info(f"Embedder ready. Dimension: {self._dimension}")
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    def embed(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """Generate embeddings for texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([]).reshape(0, self._dimension)
        
        embeddings = np.array(list(self.model.embed(texts, batch_size=self.batch_size)), dtype=np.float32)
        
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            embeddings = embeddings / norms
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed single query."""
        return self.embed([query])[0]
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """Embed multiple documents."""
        return self.embed(documents)


_embedder_instance: Optional[FastEmbedder] = None


def get_embedder() -> FastEmbedder:
    """Get or create singleton embedder."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = FastEmbedder()
    return _embedder_instance