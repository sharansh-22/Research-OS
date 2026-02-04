
import faiss
import numpy as np

class Retriever:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []

    def add(self, embeddings: np.ndarray, docs: list[str]):
        self.index.add(embeddings)
        self.documents.extend(docs)

    def search(self, query_embedding: np.ndarray, k: int = 5):
        D, I = self.index.search(np.array([query_embedding]), k)
        results = [(self.documents[i], float(D[0][j])) for j, i in enumerate(I[0])]
        return results
