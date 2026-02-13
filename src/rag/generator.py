"""
Research Architect Generator with Streaming & Memory
=====================================================
Primary: Groq API (llama-3.3-70b-versatile)
Fallback: Local Ollama (phi3:mini)

Features:
- Streaming responses for real-time output
- Short-term memory (conversation history)
- Non-streaming mode for batch processing
- Automatic fallback to local LLM
"""

import os
import logging
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass

from .retriever import RetrievalResult
from .data_loader import ChunkType

logger = logging.getLogger(__name__)


# =============================================================================
# SYSTEM PROMPT
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
- If context is insufficient, say so explicitly rather than hallucinating.

You have access to conversation history. Use it to understand follow-up questions and maintain context across turns."""


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GenerationResult:
    """Result container for non-streaming generation."""
    response: str
    model: str
    context_used: Dict[str, int]
    prompt_tokens: int
    response_tokens: int
    backend: str  # 'groq', 'ollama_fallback', or 'error'
    
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


@dataclass
class StreamMetadata:
    """Metadata returned after streaming completes."""
    model: str
    context_used: Dict[str, int]
    backend: str
    total_chunks: int


# =============================================================================
# RESEARCH ARCHITECT GENERATOR
# =============================================================================

class ResearchArchitect:
    """
    Research-aware LLM generator with streaming and memory support.
    
    Usage (Streaming with History):
        >>> architect = ResearchArchitect()
        >>> history = []
        >>> for chunk in architect.generate_stream("What is attention?", history=history):
        ...     print(chunk, end="", flush=True)
    
    Usage (Non-Streaming with History):
        >>> result = architect.generate("Explain further", history=history)
        >>> print(result.response)
    """
    
    # Model configuration
    GROQ_MODEL = "llama-3.3-70b-versatile"
    FALLBACK_MODEL = "phi3:mini"
    
    # Memory configuration
    MAX_HISTORY_TURNS = 3  # Keep last 3 user-assistant pairs (6 messages)
    
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
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            enable_fallback: Enable Ollama fallback if Groq fails
            groq_api_key: Groq API key (uses GROQ_API_KEY env var if None)
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
        
        # Initialize fallback (lazy)
        self.ollama_client = None

    
    # =========================================================================
    # CONTEXT FORMATTING
    # =========================================================================
    
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
                content = result.chunk.content[:1500]
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
    
    def _summarize_history(self, messages: List[Dict[str, str]]) -> str:
        """
        Summarize a list of messages into a concise 1-2 sentence context string.
        """
        if not messages:
            return ""
        
        # Format conversation for summarization
        conversation_text = ""
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            conversation_text += f"{role.upper()}: {content}\n"
        
        prompt = f"""Summarize the key mathematical concepts and code implementation details discussed in this conversation into 1-2 sentences. Focus on preserving context about theorems or algorithms mentioned.
        
        CONVERSATION:
        {conversation_text}
        
        SUMMARY:"""
        
        try:
            # Use Groq for summarization if available (fast)
            response = self.groq_client.chat.completions.create(
                model=self.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception:
            # Fallback to simple truncation if summarization fails
            return "Previous discussion involved research topics."

    def _build_messages(
        self,
        query: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Build message list for API call with conversation history.
        
        Message order:
        1. System prompt (with summarized LTM if applicable)
        2. Conversation history (recent turns)
        3. Current query with context
        """
        history = history or []
        
        # Handle LTM (Summarization)
        history_summary = ""
        recent_history = history
        
        max_messages = self.MAX_HISTORY_TURNS * 2
        if len(history) > max_messages:
            # Split into evicted (old) and kept (recent)
            num_evicted = len(history) - max_messages
            # Ensure we evict in pairs if possible, but strict count is fine
            evicted = history[:num_evicted]
            recent_history = history[num_evicted:]
            
            # Summarize evicted messages
            # Note: We do this synchronously. In production, this might be async/background.
            logger.info(f"Summarizing {len(evicted)} evicted messages for LTM...")
            summary_text = self._summarize_history(evicted)
            history_summary = f"\n\n[Past Context Summary]: {summary_text}"
        
        # Update System Prompt with Summary
        current_system_prompt = SYSTEM_PROMPT + history_summary
        
        messages = [
            {"role": "system", "content": current_system_prompt}
        ]
        
        # Add recent conversation history
        messages.extend(recent_history)
        
        # Build current user message with context
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
4. Use LaTeX for math notation
5. Consider our previous conversation for context"""
        else:
            user_content = f"""Answer the following research question. Consider our previous conversation for context.

## Research Question:
{query}

If no specific documents are retrieved, provide a general answer based on your knowledge."""
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    # =========================================================================
    # STREAMING GENERATION
    # =========================================================================
    
    def generate_stream(
        self,
        query: str,
        code_results: Optional[List[RetrievalResult]] = None,
        theory_results: Optional[List[RetrievalResult]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response with conversation history.
        
        Yields text chunks as they arrive from the API.
        
        Args:
            query: User's research question
            code_results: Retrieved code chunks
            theory_results: Retrieved theory chunks
            history: Conversation history [{"role": "user/assistant", "content": "..."}]
            
        Yields:
            str: Text chunks as they arrive
        """
        code_results = code_results or []
        theory_results = theory_results or []
        history = history or []
        
        # Build context and messages
        context = self._format_context(code_results, theory_results)
        messages = self._build_messages(query, context, history)
        
        # Try Groq streaming
        try:
            yield from self._stream_groq(messages)
            return
        except Exception as e:
            logger.error(f"Groq streaming failed: {e}")
            
            # Try Ollama fallback
            if self.enable_fallback:
                try:
                    logger.info(f"Falling back to {self.FALLBACK_MODEL}...")
                    yield from self._stream_ollama(messages)
                    return
                except Exception as fallback_error:
                    logger.error(f"Fallback streaming failed: {fallback_error}")
                    yield f"\n\n⚠️ Streaming failed: {e}\nFallback also failed: {fallback_error}"
            else:
                yield f"\n\n⚠️ Streaming failed: {e}"
    
    def _stream_groq(
        self,
        messages: List[Dict[str, str]],
    ) -> Generator[str, None, None]:
        """Stream from Groq API."""
        stream = self.groq_client.chat.completions.create(
            model=self.GROQ_MODEL,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def _stream_ollama(
        self,
        messages: List[Dict[str, str]],
    ) -> Generator[str, None, None]:
        """Stream from Ollama (fallback)."""
        if self.ollama_client is None:
            try:
                import ollama
                self.ollama_client = ollama.Client(host="http://localhost:11434")
            except ImportError:
                yield "⚠️ Ollama not installed."
                return
            except Exception as e:
                yield f"⚠️ Failed to connect to Ollama: {e}"
                return

        stream = self.ollama_client.chat(
            model=self.FALLBACK_MODEL,
            messages=messages,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            stream=True,
        )
        
        for chunk in stream:
            if chunk.get('message', {}).get('content'):
                yield chunk['message']['content']
    
    # =========================================================================
    # NON-STREAMING GENERATION
    # =========================================================================
    
    def generate(
        self,
        query: str,
        code_results: Optional[List[RetrievalResult]] = None,
        theory_results: Optional[List[RetrievalResult]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> GenerationResult:
        """
        Generate a complete response (non-streaming) with conversation history.
        
        Args:
            query: User's research question
            code_results: Retrieved code chunks
            theory_results: Retrieved theory chunks
            history: Conversation history [{"role": "user/assistant", "content": "..."}]
            
        Returns:
            GenerationResult with full response
        """
        code_results = code_results or []
        theory_results = theory_results or []
        history = history or []
        
        # Build context and messages
        context = self._format_context(code_results, theory_results)
        messages = self._build_messages(query, context, history)
        
        # Try Groq first
        try:
            response = self.groq_client.chat.completions.create(
                model=self.GROQ_MODEL,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
            )
            
            return GenerationResult(
                response=response.choices[0].message.content,
                model=response.model,
                context_used={"code": len(code_results), "theory": len(theory_results)},
                prompt_tokens=response.usage.prompt_tokens,
                response_tokens=response.usage.completion_tokens,
                backend="groq",
            )
            
        except Exception as e:
            logger.error(f"Groq failed: {e}")
            
            # Try fallback
            if self.enable_fallback:
                try:
                    if self.ollama_client is None:
                        import ollama
                        self.ollama_client = ollama.Client(host="http://localhost:11434")

                    response = self.ollama_client.chat(
                        model=self.FALLBACK_MODEL,
                        messages=messages,
                        options={
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens,
                        },
                        stream=False,
                    )
                    
                    return GenerationResult(
                        response=response['message']['content'],
                        model=self.FALLBACK_MODEL,
                        context_used={"code": len(code_results), "theory": len(theory_results)},
                        prompt_tokens=response.get('prompt_eval_count', 0),
                        response_tokens=response.get('eval_count', 0),
                        backend="ollama_fallback",
                    )
                    
                except Exception as fallback_error:
                    return self._error_result(
                        f"Both failed.\nGroq: {e}\nFallback: {fallback_error}",
                        code_results,
                        theory_results,
                    )
            
            return self._error_result(str(e), code_results, theory_results)
    
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
    
    # =========================================================================
    # HEALTH CHECK
    # =========================================================================
    
    def health_check(self) -> Dict[str, bool]:
        """Check connectivity to backends."""
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
        
        # Ollama check skipped to prevent model loading (Lazy Load)
        if self.ollama_client:
             status["ollama_fallback"] = True # Assume true if client exists (formerly checked)
        else:
             status["ollama_fallback"] = False # Not loaded yet
        
        return status
