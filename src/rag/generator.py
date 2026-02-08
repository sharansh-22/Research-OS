"""Research Architect Generator using Ollama"""

from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

import ollama

from .retriever import RetrievalResult
from .data_loader import ChunkType

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a Research Architect, an expert AI that bridges mathematical theory and practical implementation.

Use [THEORY] chunks for mathematical foundations and [CODE] chunks for implementations.
- Explain math first, then connect to code
- Use LaTeX for math: $inline$ or $$display$$
- Be precise about tensor shapes
- Cite your sources from the context"""


@dataclass
class GenerationResult:
    response: str
    model: str
    context_used: Dict[str, int]
    prompt_tokens: int
    response_tokens: int
    
    def to_dict(self) -> Dict:
        return {
            "response": self.response,
            "model": self.model,
            "context_used": self.context_used,
            "tokens": {"prompt": self.prompt_tokens, "response": self.response_tokens},
        }


class ResearchArchitect:
    """Research-aware LLM generator."""
    
    DEFAULT_MODEL = "qwen2.5-coder:7b"
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = ollama.Client(host=host)
        logger.info(f"ResearchArchitect initialized: {model}")
    
    def _format_context(
        self,
        code_results: List[RetrievalResult],
        theory_results: List[RetrievalResult],
    ) -> str:
        sections = []
        
        if theory_results:
            theory = "## THEORY Context:\n\n"
            for i, r in enumerate(theory_results, 1):
                src = r.chunk.metadata.get('source', 'unknown')
                theory += f"### [THEORY-{i}] ({src})\n{r.chunk.content}\n\n"
            sections.append(theory)
        
        if code_results:
            code = "## CODE Context:\n\n"
            for i, r in enumerate(code_results, 1):
                src = r.chunk.metadata.get('source', 'unknown')
                lang = r.chunk.metadata.get('language', 'python')
                code += f"### [CODE-{i}] ({src})\n```{lang}\n{r.chunk.content}\n```\n\n"
            sections.append(code)
        
        return "\n".join(sections)
    
    def generate(
        self,
        query: str,
        code_results: Optional[List[RetrievalResult]] = None,
        theory_results: Optional[List[RetrievalResult]] = None,
    ) -> GenerationResult:
        """Generate response."""
        code_results = code_results or []
        theory_results = theory_results or []
        
        context = self._format_context(code_results, theory_results)
        
        if context:
            user_msg = f"Context:\n{context}\n\nQuestion: {query}"
        else:
            user_msg = f"Question: {query}\n\n(No context retrieved)"
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                options={"temperature": self.temperature, "num_predict": self.max_tokens},
            )
            
            return GenerationResult(
                response=response['message']['content'],
                model=self.model,
                context_used={"code": len(code_results), "theory": len(theory_results)},
                prompt_tokens=response.get('prompt_eval_count', 0),
                response_tokens=response.get('eval_count', 0),
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return GenerationResult(
                response=f"Error: {e}",
                model=self.model,
                context_used={"code": 0, "theory": 0},
                prompt_tokens=0,
                response_tokens=0,
            )
            