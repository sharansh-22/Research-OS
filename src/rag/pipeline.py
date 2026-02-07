"""Research Pipeline - Full RAG orchestration"""

from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging

from .data_loader import ResearchDocumentLoader, Chunk, ChunkType
from .embedder import FastEmbedder, get_embedder
from .retriever import HybridRetriever, RetrievalResult
from .generator import ResearchArchitect, GenerationResult
from .verifier import ArchitectureVerifier

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    index_dir: str = "data/index"
    top_k: int = 5
    min_similarity: float = 0.25
    faiss_weight: float = 0.7
    bm25_weight: float = 0.3
    model: str = "qwen2.5-coder:7b"
    temperature: float = 0.3
    max_tokens: int = 2048
    verify_code: bool = True
    verification_timeout: int = 10
    
    def to_dict(self) -> Dict:
        return self.__dict__.copy()


@dataclass
class QueryResult:
    query: str
    response: str
    code_context: List[Dict]
    theory_context: List[Dict]
    verification_results: List[Dict]
    generation_metadata: Dict
    
    def to_dict(self) -> Dict:
        return {"query": self.query, "response": self.response, "context": {"code": self.code_context, "theory": self.theory_context}, "verification": self.verification_results, "metadata": self.generation_metadata}


class ResearchPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.embedder = get_embedder()
        self.loader = ResearchDocumentLoader()
        self.retriever = HybridRetriever(embedder=self.embedder, faiss_weight=self.config.faiss_weight, bm25_weight=self.config.bm25_weight, min_similarity=self.config.min_similarity)
        self.generator = ResearchArchitect(model=self.config.model, temperature=self.config.temperature, max_tokens=self.config.max_tokens)
        self.verifier = ArchitectureVerifier(timeout_seconds=self.config.verification_timeout)
        logger.info("Pipeline ready")
    
    def ingest_pdf(self, pdf_path: str | Path) -> int:
        chunks = self.loader.load_pdf(pdf_path)
        self.retriever.add_chunks(chunks)
        return len(chunks)
    
    def ingest_directory(self, dir_path: str | Path, recursive: bool = True) -> int:
        chunks = self.loader.load_directory(dir_path, recursive)
        self.retriever.add_chunks(chunks)
        return len(chunks)
    
    def query(self, question: str, verify: Optional[bool] = None) -> QueryResult:
        do_verify = verify if verify is not None else self.config.verify_code
        
        results = self.retriever.search_by_type(question, self.config.top_k)
        code_res = results.get("code", [])
        theory_res = results.get("theory", [])
        
        gen = self.generator.generate(question, code_res, theory_res)
        
        verifs = []
        if do_verify and "```" in gen.response:
            verifs = [v.to_dict() for v in self.verifier.verify_generated_response(gen.response)]
        
        return QueryResult(question, gen.response, [r.to_dict() for r in code_res], [r.to_dict() for r in theory_res], verifs, gen.to_dict())
    
    def save_index(self, path: Optional[str] = None) -> None:
        self.retriever.save(path or self.config.index_dir)
    
    def load_index(self, path: Optional[str] = None) -> None:
        self.retriever = HybridRetriever.load(path or self.config.index_dir, self.embedder)
    
    @property
    def index_size(self) -> int:
        return self.retriever.size
    
    def get_stats(self) -> Dict[str, Any]:
        types = {}
        for c in self.retriever.chunks:
            t = c.chunk_type.value
            types[t] = types.get(t, 0) + 1
        return {"total_chunks": self.index_size, "chunk_types": types, "config": self.config.to_dict()}


def create_pipeline(index_dir: str = "data/index", load_existing: bool = True) -> ResearchPipeline:
    cfg = PipelineConfig(index_dir=index_dir)
    pipe = ResearchPipeline(cfg)
    if load_existing and Path(index_dir).exists():
        try:
            pipe.load_index()
        except Exception as e:
            logger.warning(f"Could not load: {e}")
    return pipe
