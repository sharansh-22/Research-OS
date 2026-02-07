"""Research Architect Generator using Ollama"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

import ollama

from .retriever import RetrievalResult
from .data_loader import ChunkType

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Research Architect bridging mathematical theory and implementation.
Use [THEORY] for math foundations, [CODE] for implementations.
Use LaTeX for math. Be precise about tensor shapes."""


@dataclass
class GenerationResult:
    response: str
    model: str
    context_used: Dict[str, int]
    prompt_tokens: int
    response_tokens: int
    
    def to_dict(self) -> Dict:
        return {"response": self.response, "model": self.model, "context_used": self.context_used, "tokens": {"prompt": self.prompt_tokens, "response": self.response_tokens}}


class ResearchArchitect:
    DEFAULT_MODEL = "qwen2.5-coder:7b"
    
    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.3, max_tokens: int = 2048, host: str = "http://localhost:11434"):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = ollama.Client(host=host)
        logger.info(f"Generator: {model}")
    
    def _format_context(self, code_results: List[RetrievalResult], theory_results: List[RetrievalResult]) -> str:
        parts = []
        if theory_results:
            t = "## THEORY:\n"
            for i, r in enumerate(theory_results, 1):
                t += f"[T{i}] {r.chunk.content[:500]}...\n\n"
            parts.append(t)
        if code_results:
            c = "## CODE:\n"
            for i, r in enumerate(code_results, 1):
                c += f"[C{i}]\n```\n{r.chunk.content[:500]}\n```\n\n"
            parts.append(c)
        return "\n".join(parts)
    
    def generate(self, query: str, code_results: Optional[List[RetrievalResult]] = None, theory_results: Optional[List[RetrievalResult]] = None) -> GenerationResult:
        code_results = code_results or []
        theory_results = theory_results or []
        
        ctx = self._format_context(code_results, theory_results)
        msg = f"Context:\n{ctx}\n\nQuestion: {query}" if ctx else f"Question: {query}"
        
        try:
            resp = self.client.chat(model=self.model, messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": msg}], options={"temperature": self.temperature, "num_predict": self.max_tokens})
            return GenerationResult(resp['message']['content'], self.model, {"code": len(code_results), "theory": len(theory_results)}, resp.get('prompt_eval_count', 0), resp.get('eval_count', 0))
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return GenerationResult(f"Error: {e}", self.model, {"code": 0, "theory": 0}, 0, 0)
