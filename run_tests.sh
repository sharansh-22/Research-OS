#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           RESEARCH-OS PRE-COMMIT TEST SUITE                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

PASS=0
FAIL=0

check() {
    if eval "$1" > /dev/null 2>&1; then
        echo "✓ $2"
        ((PASS++))
    else
        echo "✗ $2"
        ((FAIL++))
    fi
}

echo "=== PHASE 1: Environment ==="
check "python --version" "Python installed"
check "[ -s src/rag/__init__.py ]" "src/rag/__init__.py"
check "[ -s src/rag/data_loader.py ]" "src/rag/data_loader.py"
check "[ -s src/rag/embedder.py ]" "src/rag/embedder.py"
check "[ -s src/rag/retriever.py ]" "src/rag/retriever.py"
check "[ -s src/rag/generator.py ]" "src/rag/generator.py"
check "[ -s src/rag/verifier.py ]" "src/rag/verifier.py"
check "[ -s src/rag/pipeline.py ]" "src/rag/pipeline.py"
check "[ -s main.py ]" "main.py"
check "[ -s requirements.txt ]" "requirements.txt"

echo ""
echo "=== PHASE 2: Dependencies ==="
check "python -c 'import pymupdf4llm'" "pymupdf4llm"
check "python -c 'import fastembed'" "fastembed"
check "python -c 'import faiss'" "faiss"
check "python -c 'import rank_bm25'" "rank_bm25"
check "python -c 'import groq'" "groq"
check "python -c 'import ollama'" "ollama"

echo ""
echo "=== PHASE 3: Imports ==="
check "python -c 'from src.rag import ResearchDocumentLoader'" "ResearchDocumentLoader"
check "python -c 'from src.rag import FastEmbedder'" "FastEmbedder"
check "python -c 'from src.rag import HybridRetriever'" "HybridRetriever"
check "python -c 'from src.rag import ResearchArchitect'" "ResearchArchitect"
check "python -c 'from src.rag import ArchitectureVerifier'" "ArchitectureVerifier"
check "python -c 'from src.rag import ResearchPipeline'" "ResearchPipeline"

echo ""
echo "=== PHASE 4: API Key ==="
if [ -n "$GROQ_API_KEY" ]; then
    echo "✓ GROQ_API_KEY set"
    ((PASS++))
else
    echo "✗ GROQ_API_KEY not set"
    ((FAIL++))
fi

echo ""
echo "=== PHASE 5: Syntax Check ==="
check "python -m py_compile main.py" "main.py syntax"
check "python -m py_compile src/rag/pipeline.py" "pipeline.py syntax"
check "python -m py_compile src/rag/generator.py" "generator.py syntax"

echo ""
echo "=== PHASE 6: Component Tests ==="
check "python -c 'from src.rag import get_embedder; get_embedder().embed_query(\"test\")'" "Embedder"
check "python -c 'from src.rag import HybridRetriever; HybridRetriever()'" "Retriever"
check "python -c 'from src.rag import ArchitectureVerifier; ArchitectureVerifier()'" "Verifier"

echo ""
echo "=== PHASE 7: Groq API ==="
python -c "
import os
from groq import Groq
try:
    Groq(api_key=os.environ.get('GROQ_API_KEY')).chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role':'user','content':'hi'}],
        max_tokens=3
    )
    print('✓ Groq API connected')
except Exception as e:
    print(f'✗ Groq API: {e}')
"

echo ""
echo "=== PHASE 8: Pipeline ==="
check "python -c 'from src.rag import ResearchPipeline; ResearchPipeline()'" "Pipeline init"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "                        RESULTS"
echo "════════════════════════════════════════════════════════════════"
echo "  ✓ Passed: $PASS"
echo "  ✗ Failed: $FAIL"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "  🎉 ALL TESTS PASSED - Ready to commit!"
    echo ""
    echo "  Next steps:"
    echo "    git add ."
    echo "    git commit -m 'feat: Add Groq API integration'"
    echo "    git push"
else
    echo "  ⚠️  FIX FAILURES BEFORE COMMITTING"
fi
echo "════════════════════════════════════════════════════════════════"
