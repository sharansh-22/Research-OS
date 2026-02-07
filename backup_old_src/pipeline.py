import os
import json
import re
from datetime import datetime

from src.rag.data_loader import MathLoader
from src.rag.embedder import Embedder
from src.rag.retriever import Retriever
from src.rag.generator import Generator
from src.rag.verifier import ArchitectureVerifier

class RAGPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", llm_model: str = "llama3.2:3b", log_dir: str = "logs", vector_store: str = "vector_store"):
        # Initialize embedder, retriever, and generator
        self.embedder = Embedder(model_name=model_name)
        self.generator = Generator(model_name=llm_model)

        # Setup paths
        self.log_dir = log_dir
        self.vector_store = vector_store
        os.makedirs(log_dir, exist_ok=True)

        # Try to load existing index
        self.retriever = None
        if os.path.exists(os.path.join(vector_store, "index.faiss")):
            try:
                # Initialize with dummy dim, will be replaced by load
                self.retriever = Retriever(dim=384) 
                self.retriever.load(vector_store)
                print(f"Loaded existing index from {vector_store}")
            except Exception as e:
                print(f"Failed to load index: {e}")
                self.retriever = None

    def build_index(self, pdf_path: str, chunk_size: int = 1000, overlap: int = 2):
        # Step 1: Load and chunk (Marker-PDF OCR)
        loader = MathLoader()
        try:
            chunks = loader.load_and_chunk(pdf_path, chunk_size=chunk_size, overlap=overlap)
        except Exception as e:
            # Robust handling: marker-pdf can fail on some PDFs.
            print(f"Failed to ingest {pdf_path}: {e}")
            return

        # Step 2: Embed (extract text from chunk dicts)
        texts = [c["text"] for c in chunks]
        if not texts:
            print(f"No chunks produced for {pdf_path} (skipping).")
            return
        embeddings = self.embedder.encode(texts)

        # Step 3: Build or extend retriever
        if self.retriever is None:
            self.retriever = Retriever(dim=embeddings.shape[1])

        # Add to retriever (builds both FAISS and BM25)
        self.retriever.add(embeddings, chunks)
        
        # Step 4: Save index
        self.retriever.save(self.vector_store)

        print(f"Index built with {len(chunks)} chunks and saved to {self.vector_store}.")

    def query(self, text: str, k: int = 5):
        if not self.retriever:
            raise ValueError("Retriever not built. Call build_index() first.")

        query_emb = self.embedder.encode_single(text)
        # Pass text for BM25 hybrid search
        results = self.retriever.search(query_emb, query_text=text, k=k)

        # --- Logging block ---
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": text,
            "results": [
                {"doc": r.get("text", "")[:100], "score": r.get("score")}
                for r in results
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
        # Generator expects list of (chunk_dict, score) tuples.
        generator_inputs = [
            ({"text": r.get("text", ""), "metadata": r.get("metadata", {})}, r.get("score", 0.0))
            for r in results
        ]
        answer = self.generator.generate(question, generator_inputs)

        # Step 3: Automated verification of generated PyTorch code (local only).
        # SECURITY WARNING:
        # - This uses `exec()` internally and will execute arbitrary code.
        # - Do NOT enable this in production without proper sandboxing.
        verifier = ArchitectureVerifier()
        code_block_match = re.search(r"```python\s*[\s\S]*?```", answer, flags=re.IGNORECASE)
        if code_block_match:
            is_valid, message = verifier.verify_dimensions(code_block_match.group(0))
            if is_valid:
                answer = (
                    f"{answer}\n\n"
                    f"Automated Verification Passed: {message}"
                )
            else:
                answer = (
                    f"{answer}\n\n"
                    f"Warning: The generated code threw a runtime error: {message}"
                )

        return {
            "answer": answer,
            "sources": [
                {
                    "text": r.get("text", "")[:200] + "...",
                    "score": r.get("score"),
                    "metadata": r.get("metadata"),
                }
                for r in results
            ]
        }

