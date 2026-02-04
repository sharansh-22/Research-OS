
import ollama
from typing import List, Tuple

class Generator:
    def __init__(self, model_name: str = "phi3:mini"):
        self.model_name = model_name
        self.system_prompt = """You are a helpful AI assistant. Answer the user's question based ONLY on the provided context. 
If the context doesn't contain enough information to answer, say so honestly.
Keep your answers concise and relevant."""

    def generate(self, query: str, context_chunks: List[Tuple[str, float]]) -> str:
        """
        Generate an answer using retrieved context chunks.
        
        Args:
            query: The user's question
            context_chunks: List of (chunk_text, distance) tuples from retriever
            
        Returns:
            Generated answer string
        """
        # Build context from chunks
        context_parts = []
        for i, (chunk, dist) in enumerate(context_chunks, 1):
            context_parts.append(f"[Source {i}]: {chunk.strip()}")
        
        context = "\n\n".join(context_parts)
        
        # Build the prompt
        user_prompt = f"""Context:
{context}

Question: {query}

Answer based on the context above:"""

        # Call Ollama
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return response['message']['content']
