
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from rag.generator import Generator
except ImportError as e:
    print(f"Error importing Generator: {e}")
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
    from src.rag.generator import Generator

def test_generation():
    print("Testing Gemini Generator...")
    
    # Check for API key
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your_gemini_api_key_here":
        print("ERROR: GOOGLE_API_KEY not found or default value in .env")
        print("Please set your valid Google API key in the .env file.")
        return

    try:
        generator = Generator()
        print("Generator initialized successfully.")
        
        query = "What is the capital of France?"
        context = [("Paris is the capital and most populous city of France.", 0.9)]
        
        print(f"\nQuery: {query}")
        print("Generating answer...")
        
        answer = generator.generate(query, context)
        
        print(f"\nAnswer: {answer}")
        print("\nSUCCESS: Gemini API request completed.")
        
    except Exception as e:
        print(f"\nFAILURE: An error occurred: {e}")

if __name__ == "__main__":
    test_generation()
