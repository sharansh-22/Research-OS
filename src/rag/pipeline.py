import os
import json
from datetime import datetime

from src.rag.data_loader import load_and_chunk_pdf
from src.rag.embedder import Embedder
from src.rag.retriever import Retriever
from src.rag.generator import Generator

class RAGPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", llm_model: str = "llama3.2:3b", log_dir: str = "logs"):
        # Initialize embedder, retriever, and generator
        self.embedder = Embedder(model_name=model_name)
        self.retriever = None
        self.generator = Generator(model_name=llm_model)

        # Setup logging directory
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def build_index(self, pdf_path: str, chunk_size: int = 500, overlap: int = 50):
        # Step 1: Load and chunk
        chunks = load_and_chunk_pdf(pdf_path, chunk_size=chunk_size, overlap=overlap)

        # Step 2: Embed
        embeddings = self.embedder.encode(chunks)

        # Step 3: Build retriever
        self.retriever = Retriever(dim=embeddings.shape[1])
        self.retriever.add(embeddings, chunks)

        print(f"Index built with {len(chunks)} chunks.")

    def query(self, text: str, k: int = 5):
        if not self.retriever:
            raise ValueError("Retriever not built. Call build_index() first.")

        query_emb = self.embedder.encode_single(text)
        results = self.retriever.search(query_emb, k=k)

        # --- Logging block ---
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": text,
            "results": [
                {"doc": doc, "distance": dist}
                for doc, dist in results
            ]
        }
        log_file = os.path.join(self.log_dir, "queries.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        # ---------------------

        return results

    def ask(self, question: str, k: int = 3) -> dict:
        """
        Full RAG pipeline: retrieve relevant chunks and generate an answer.
        
        Args:
            question: User's question
            k: Number of context chunks to retrieve
            
        Returns:
            Dict with 'answer' and 'sources' keys
        """
        if not self.retriever:
            raise ValueError("Retriever not built. Call build_index() first.")

        # Step 1: Retrieve relevant chunks
        results = self.query(question, k=k)

        # Step 2: Generate answer using LLM
        answer = self.generator.generate(question, results)

        return {
            "answer": answer,
            "sources": [{"text": doc[:200] + "...", "distance": dist} for doc, dist in results]
        }

