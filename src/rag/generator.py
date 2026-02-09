"""
Research Architect Generator
============================
Primary: Groq API (llama-3.3-70b-versatile)
Fallback: Local Ollama (phi3:mini)
"""

import os
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from .retriever import RetrievalResult
from .data_loader import ChunkType

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a Research Architect, an expert AI that bridges mathematical theory and practical implementation.

Use [THEORY] chunks for mathematical foundations and [CODE] chunks for implementations.
- Explain the mathematical intuition FIRST using LaTeX: $inline$ or $$display$$
- Then show the implementation approach with code blocks
- Be precise about tensor shapes and dimensions
- If context is insufficient, say so explicitly."""


@dataclass
class GenerationResult:
    response: str
    model: str
    context_used: Dict[str, int]
    prompt_tokens: int
    response_tokens: int
    backend: str
    
    def to_dict(self) -> Dict:
        return {
            "response": self.response,
            "model": self.model,
            "backend": self.backend,
            "context_used": self.context_used,
            "tokens": {
                "prompt": self.prompt_tokens,
                "response": self.response_tokens,
            },
        }


class ResearchArchitect:
    """Research-aware LLM generator with Groq API and Ollama fallback."""
    
    GROQ_MODEL = "llama-3.3-70b-versatile"
    FALLBACK_MODEL = "phi3:mini"
    
    def __init__(
        self,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        enable_fallback: bool = True,
        groq_api_key: Optional[str] = None,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_fallback = enable_fallback
        
        # Get API key
        self.api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "\n" + "=" * 60 + "\n"
                "ERROR: GROQ_API_KEY not found!\n"
                "=" * 60 + "\n"
                "Set your Groq API key:\n\n"
                "  export GROQ_API_KEY='your-api-key-here'\n\n"
                "Get your free key at: https://console.groq.com/keys\n"
                "=" * 60
            )
        
        # Initialize Groq client
        try:
            from groq import Groq
            self.groq_client = Groq(api_key=self.api_key)
            logger.info(f"✓ Groq client ready ({self.GROQ_MODEL})")
        except ImportError:
            raise ImportError("Groq not installed. Run: pip install groq")
        
        # Initialize fallback
        self.ollama_client = None
        if self.enable_fallback:
            try:
                import ollama
                self.ollama_client = ollama.Client(host="http://localhost:11434")
                logger.info(f"✓ Fallback ready ({self.FALLBACK_MODEL})")
            except ImportError:
                logger.warning("Ollama not available for fallback")
    
    def _format_context(
        self,
        code_results: List[RetrievalResult],
        theory_results: List[RetrievalResult],
    ) -> str:
        sections = []
        
        if theory_results:
            t = "## THEORY Context:\n\n"
            for i, r in enumerate(theory_results, 1):
                src = r.chunk.metadata.get('source', 'unknown')
                content = r.chunk.content[:1500]
                t += f"### [THEORY-{i}] ({src})\n{content}\n\n"
            sections.append(t)
        
        if code_results:
            c = "## CODE Context:\n\n"
            for i, r in enumerate(code_results, 1):
                src = r.chunk.metadata.get('source', 'unknown')
                lang = r.chunk.metadata.get('language', 'python')
                content = r.chunk.content[:1500]
                c += f"### [CODE-{i}] ({src})\n```{lang}\n{content}\n```\n\n"
            sections.append(c)
        
        return "\n".join(sections)
    
    def _build_messages(self, query: str, context: str) -> List[Dict[str, str]]:
        if context:
            user_content = f"""Based on the context below, answer the question.

{context}

---
Question: {query}

Instructions:
1. Explain math using [THEORY] references
2. Show code using [CODE] references
3. Be precise about dimensions"""
        else:
            user_content = f"Question: {query}\n\n(No context retrieved)"
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    
    def _call_groq(self, messages: List[Dict[str, str]]) -> Dict:
        response = self.groq_client.chat.completions.create(
            model=self.GROQ_MODEL,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
    
    def _call_ollama_fallback(self, messages: List[Dict[str, str]]) -> Dict:
        if not self.ollama_client:
            raise RuntimeError("Ollama fallback not available")
        
        response = self.ollama_client.chat(
            model=self.FALLBACK_MODEL,
            messages=messages,
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
        )
        return {
            "content": response['message']['content'],
            "model": self.FALLBACK_MODEL,
            "prompt_tokens": response.get('prompt_eval_count', 0),
            "completion_tokens": response.get('eval_count', 0),
        }
    
    def generate(
        self,
        query: str,
        code_results: Optional[List[RetrievalResult]] = None,
        theory_results: Optional[List[RetrievalResult]] = None,
    ) -> GenerationResult:
        code_results = code_results or []
        theory_results = theory_results or []
        
        context = self._format_context(code_results, theory_results)
        messages = self._build_messages(query, context)
        
        # Try Groq first
        try:
            result = self._call_groq(messages)
            return GenerationResult(
                response=result["content"],
                model=result["model"],
                context_used={"code": len(code_results), "theory": len(theory_results)},
                prompt_tokens=result["prompt_tokens"],
                response_tokens=result["completion_tokens"],
                backend="groq",
            )
        except Exception as e:
            logger.error(f"Groq failed: {e}")
            
            # Try fallback
            if self.enable_fallback and self.ollama_client:
                try:
                    logger.info(f"Falling back to {self.FALLBACK_MODEL}...")
                    result = self._call_ollama_fallback(messages)
                    return GenerationResult(
                        response=result["content"],
                        model=result["model"],
                        context_used={"code": len(code_results), "theory": len(theory_results)},
                        prompt_tokens=result["prompt_tokens"],
                        response_tokens=result["completion_tokens"],
                        backend="ollama_fallback",
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback failed: {fallback_error}")
                    return self._error_result(
                        f"Both failed.\nGroq: {e}\nFallback: {fallback_error}",
                        code_results, theory_results
                    )
            
            return self._error_result(str(e), code_results, theory_results)
    
    def _error_result(self, error_msg: str, code_results: List, theory_results: List) -> GenerationResult:
        return GenerationResult(
            response=f"⚠️ Generation Failed\n\n{error_msg}",
            model="none",
            context_used={"code": len(code_results), "theory": len(theory_results)},
            prompt_tokens=0,
            response_tokens=0,
            backend="error",
        )
    
    def health_check(self) -> Dict[str, bool]:
        status = {"groq": False, "ollama_fallback": False}
        
        try:
            self.groq_client.chat.completions.create(
                model=self.GROQ_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
            )
            status["groq"] = True
        except Exception as e:
            logger.warning(f"Groq check failed: {e}")
        
        if self.ollama_client:
            try:
                self.ollama_client.chat(
                    model=self.FALLBACK_MODEL,
                    messages=[{"role": "user", "content": "hi"}],
                    options={"num_predict": 5},
                )
                status["ollama_fallback"] = True
            except Exception as e:
                logger.warning(f"Ollama check failed: {e}")
        
        return status
