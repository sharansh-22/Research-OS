import os
import json
from datetime import datetime

from src.rag.data_loader import load_and_chunk_pdf
from src.rag.embedder import Embedder
from src.rag.retriever import Retriever

class RAGPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", log_dir: str = "logs"):
        # Initialize embedder and retriever
        self.embedder = Embedder(model_name=model_name)
        self.retriever = None

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
