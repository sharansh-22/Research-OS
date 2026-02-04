from src.rag.pipeline import RAGPipeline

# Initialize pipeline with Phi-3 Mini for generation (CPU-optimized)
pipeline = RAGPipeline()

# Build index from a sample PDF
pipeline.build_index("data/terminal.pdf")

# Run a query with RAG
print("\n" + "="*60)
print("AI Knowledge Assistant (powered by Phi-3 Mini)")
print("="*60)

query = input('\nEnter your question: ')
print("\nThinking...")

result = pipeline.ask(query, k=3)

print("\n" + "-"*60)
print("ANSWER:")
print("-"*60)
print(result["answer"])

print("\n" + "-"*60)
print("SOURCES:")
print("-"*60)
for i, source in enumerate(result["sources"], 1):
    print(f"\n[{i}] (relevance: {1/(1+source['distance']):.2%})")
    print(f"    {source['text']}")
