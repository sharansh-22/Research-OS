import ollama
from typing import List, Tuple, Dict, Any

class Generator:
    def __init__(self, model_name: str = "qwen2.5-coder:7b"):
        # Default to a specialized coding / research model,
        # but allow override via constructor.
        self.model_name = model_name
        self.system_prompt = (
            "You are a Research Architect specializing in machine learning and mathematics.\n"
            "You answer strictly based on the retrieved context and clearly separate theory from implementation.\n"
            "\n"
            "CONTEXT INTERPRETATION RULES:\n"
            "- The context will contain lines starting with [CODE] or [THEORY].\n"
            "- If a line is tagged [CODE], treat it as implementation reference: focus on APIs, control flow, and usage patterns.\n"
            "- If a line is tagged [THEORY], treat it as mathematical or conceptual explanation: focus on theorems, assumptions, and derivations.\n"
            "- Never invent details that are not supported by the provided context; if information is missing, say so explicitly.\n"
            "\n"
            "FORMATTING RULES:\n"
            "- For any mathematical equations or important expressions, use LaTeX delimited by $$, e.g. $$a^2 + b^2 = c^2$$.\n"
            "- For code examples, respond using fenced Python code blocks with syntax highlighting, e.g. ```python ... ```.\n"
            "- Keep explanations concise but precise, and tie every claim back to the given context.\n"
            "\n"
            "INSTRUCTIONS:\n"
            "- First, read all context carefully.\n"
            "- Then, integrate [THEORY] for the conceptual narrative and use [CODE] snippets as concrete implementation guidance.\n"
            "- If there is a mismatch between theory and code, call it out explicitly and suggest a resolution."
        )

    def generate(self, query: str, context_chunks: List[Tuple[Dict[str, Any], float]]) -> str:
        """
        Generate an answer using retrieved context chunks with metadata.
        
        Args:
            query: The user's question
            context_chunks: List of (chunk_dict, distance) tuples from retriever
                            chunk_dict must have 'text' and 'metadata' keys
            
        Returns:
            Generated answer string
        """
        # Build context from chunks
        context_parts = []
        for i, (chunk, dist) in enumerate(context_chunks, 1):
            text = chunk.get("text", "")
            meta = chunk.get("metadata", {})
            source = meta.get("source", f"Source {i}")
            chunk_type = meta.get("type", "theory").upper()
            context_parts.append(
                f"[{chunk_type}] Source {source}: {text.strip()}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Build the prompt
        user_prompt = f"""Context:
{context}

Question: {query}

Answer based on the context above:"""

        # Call Ollama
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating response: {str(e)}"
