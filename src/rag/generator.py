"""
Research Architect Generator
============================
Primary: Groq API (llama-3.3-70b-versatile) - Sub-second inference
Fallback: Local Ollama (phi3:mini) - When API fails

Environment:
    GROQ_API_KEY: Required for Groq API access
"""

import os
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from .retriever import RetrievalResult
from .data_loader import ChunkType

logger = logging.getLogger(__name__)


# =============================================================================
# SYSTEM PROMPT - Research Architect Persona
# =============================================================================

SYSTEM_PROMPT = """You are a Research Architect, an expert AI that bridges mathematical theory and practical implementation.

Your role is to synthesize information from two sources:
1. **[THEORY]** chunks: Mathematical foundations, theorems, definitions, proofs from research papers
2. **[CODE]** chunks: Working Python/PyTorch implementations from coding guides

When answering:
- Explain the mathematical intuition FIRST using LaTeX: $inline$ or $$display$$
- Then show the implementation approach with code blocks
- Be precise about tensor shapes and dimensions
- Highlight common pitfalls

If context is insufficient, say so explicitly rather than hallucinating."""


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GenerationResult:
    """Result container for generation output."""
    response: str
    model: str
    context_used: Dict[str, int]
    prompt_tokens: int
    response_tokens: int
    backend: str  # 'groq' or 'ollama'
    
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


# =============================================================================
# RESEARCH ARCHITECT - MAIN GENERATOR CLASS
# =============================================================================

class ResearchArchitect:
    """
    Research-aware LLM generator with Groq API (primary) and Ollama (fallback).
    
    Usage:
        >>> architect = ResearchArchitect()
        >>> result = architect.generate("Explain self-attention", code_results, theory_results)
        >>> print(result.response)
    """
    
    # Groq configuration
    GROQ_MODEL = "llama-3.3-70b-versatile"
    
    # Fallback local model (lightweight)
    FALLBACK_MODEL = "phi3:mini"
    
    def __init__(
        self,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        enable_fallback: bool = True,
        groq_api_key: Optional[str] = None,
    ):
        """
        Initialize the Research Architect.
        
        Args:
            temperature: Sampling temperature (0.0-1.0). Low = precise.
            max_tokens: Maximum tokens in response.
            enable_fallback: If True, fall back to local Ollama on Groq failure.
            groq_api_key: Groq API key. If None, reads from GROQ_API_KEY env var.
        
        Raises:
            ValueError: If GROQ_API_KEY is not set and no key provided.
        """
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
                "To fix this, set your Groq API key:\n\n"
                "  Option 1 - Export in terminal:\n"
                "    export GROQ_API_KEY='your-api-key-here'\n\n"
                "  Option 2 - Add to .env file:\n"
                "    echo 'GROQ_API_KEY=your-api-key-here' >> .env\n\n"
                "  Option 3 - Pass directly:\n"
                "    ResearchArchitect(groq_api_key='your-key')\n\n"
                "Get your free API key at: https://console.groq.com/keys\n"
                "=" * 60
            )
        
        # Initialize Groq client
        try:
            from groq import Groq
            self.groq_client = Groq(api_key=self.api_key)
            logger.info(f"✓ Groq client initialized (model: {self.GROQ_MODEL})")
        except ImportError:
            raise ImportError(
                "Groq package not installed. Run: pip install groq"
            )
        
        # Initialize fallback Ollama client (lazy load)
        self.ollama_client = None
        if self.enable_fallback:
            try:
                import ollama
                self.ollama_client = ollama.Client(host="http://localhost:11434")
                logger.info(f"✓ Fallback ready (model: {self.FALLBACK_MODEL})")
            except ImportError:
                logger.warning("Ollama not available for fallback")
    
    def _format_context(
        self,
        code_results: List[RetrievalResult],
        theory_results: List[RetrievalResult],
    ) -> str:
        """Format retrieved chunks into context string."""
        sections = []
        
        if theory_results:
            theory_section = "## Retrieved THEORY Context:\n\n"
            for i, result in enumerate(theory_results, 1):
                source = result.chunk.metadata.get('source', 'unknown')
                score = result.score
                content = result.chunk.content[:1500]  # Truncate for token limits
                theory_section += f"### [THEORY-{i}] (source: {source}, relevance: {score:.2f})\n"
                theory_section += f"{content}\n\n"
            sections.append(theory_section)
        
        if code_results:
            code_section = "## Retrieved CODE Context:\n\n"
            for i, result in enumerate(code_results, 1):
                source = result.chunk.metadata.get('source', 'unknown')
                lang = result.chunk.metadata.get('language', 'python')
                content = result.chunk.content[:1500]
                code_section += f"### [CODE-{i}] (source: {source})\n"
                code_section += f"```{lang}\n{content}\n```\n\n"
            sections.append(code_section)
        
        return "\n".join(sections)
    
    def _build_messages(
        self,
        query: str,
        context: str,
    ) -> List[Dict[str, str]]:
        """Build message list for API call."""
        if context:
            user_content = f"""Based on the retrieved context below, answer the research question.

{context}

---

## Research Question:
{query}

## Instructions:
1. Explain the mathematical foundation using [THEORY] references
2. Show implementation approach using [CODE] references  
3. Be precise about dimensions and shapes
4. Use LaTeX for math notation"""
        else:
            user_content = f"""Answer the following research question. Note that no specific context was retrieved.

## Research Question:
{query}

Provide a general answer based on your knowledge."""
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    
    def _call_groq(self, messages: List[Dict[str, str]]) -> Dict:
        """Make API call to Groq."""
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
        """Fallback to local Ollama."""
        if not self.ollama_client:
            raise RuntimeError("Ollama fallback not available")
        
        response = self.ollama_client.chat(
            model=self.FALLBACK_MODEL,
            messages=messages,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
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
        """
        Generate a research-aware response.
        
        Args:
            query: User's research question.
            code_results: Retrieved code chunks.
            theory_results: Retrieved theory chunks.
            
        Returns:
            GenerationResult with response and metadata.
        """
        code_results = code_results or []
        theory_results = theory_results or []
        
        # Build context and messages
        context = self._format_context(code_results, theory_results)
        messages = self._build_messages(query, context)
        
        # Try Groq first (primary)
        try:
            logger.debug("Calling Groq API...")
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
            error_msg = str(e)
            logger.error(f"Groq API failed: {error_msg}")
            
            # Classify error type
            if "rate_limit" in error_msg.lower():
                error_type = "Rate Limit Exceeded"
            elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower():
                error_type = "Authentication Error"
            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                error_type = "Network Error"
            else:
                error_type = "API Error"
            
            logger.warning(f"Groq failed ({error_type}): {error_msg[:100]}")
            
            # Try fallback if enabled
            if self.enable_fallback and self.ollama_client:
                try:
                    logger.info(f"Falling back to local Ollama ({self.FALLBACK_MODEL})...")
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
                    logger.error(f"Fallback also failed: {fallback_error}")
                    return self._error_result(
                        f"Both Groq and local fallback failed.\n"
                        f"Groq: {error_msg[:200]}\n"
                        f"Fallback: {str(fallback_error)[:200]}",
                        code_results,
                        theory_results,
                    )
            
            # No fallback - return error
            return self._error_result(
                f"Groq API Error ({error_type}): {error_msg[:300]}",
                code_results,
                theory_results,
            )
    
    def _error_result(
        self,
        error_msg: str,
        code_results: List,
        theory_results: List,
    ) -> GenerationResult:
        """Create an error result."""
        return GenerationResult(
            response=f"⚠️ Generation Failed\n\n{error_msg}",
            model="none",
            context_used={"code": len(code_results), "theory": len(theory_results)},
            prompt_tokens=0,
            response_tokens=0,
            backend="error",
        )
    
    def health_check(self) -> Dict[str, bool]:
        """Check connectivity to backends."""
        status = {"groq": False, "ollama_fallback": False}
        
        # Test Groq
        try:
            self.groq_client.chat.completions.create(
                model=self.GROQ_MODEL,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            status["groq"] = True
        except Exception as e:
            logger.warning(f"Groq health check failed: {e}")
        
        # Test Ollama fallback
        if self.ollama_client:
            try:
                self.ollama_client.chat(
                    model=self.FALLBACK_MODEL,
                    messages=[{"role": "user", "content": "ping"}],
                    options={"num_predict": 5},
                )
                status["ollama_fallback"] = True
            except Exception as e:
                logger.warning(f"Ollama health check failed: {e}")
        
        return status


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_generator(
    enable_fallback: bool = True,
    temperature: float = 0.3,
) -> ResearchArchitect:
    """
    Factory function to create a generator.
    
    Args:
        enable_fallback: Enable local Ollama fallback.
        temperature: Sampling temperature.
        
    Returns:
        Configured ResearchArchitect instance.
    """
    return ResearchArchitect(
        temperature=temperature,
        enable_fallback=enable_fallback,
    )
