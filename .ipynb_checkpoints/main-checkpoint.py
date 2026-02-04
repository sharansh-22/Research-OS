from src.rag.pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Build index from a sample PDF
pipeline.build_index("data/terminal.pdf")

# Run a query
query = input('Enter Your query :')
results = pipeline.query(query, k=3)

print("Query:", query)
for doc, dist in results:
    print(f"Match: {doc[:80]}... (distance={dist:.4f})")



