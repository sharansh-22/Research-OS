#!/usr/bin/env python3
"""
Research-OS — System Verification
====================================
Run this BEFORE starting CLI or API.
Checks every dependency, file, env var, and index.

Usage:
    python verify_setup.py
"""

import os
import sys
import importlib
from pathlib import Path

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

passed = 0
failed = 0
warned = 0


def ok(msg, detail=""):
    global passed
    passed += 1
    print(f"  {GREEN}OK{RESET}    {msg}  {detail}")


def fail(msg, detail=""):
    global failed
    failed += 1
    print(f"  {RED}FAIL{RESET}  {msg}  {detail}")


def warn(msg, detail=""):
    global warned
    warned += 1
    print(f"  {YELLOW}WARN{RESET}  {msg}  {detail}")


def section(title):
    print(f"\n{BOLD}{CYAN}{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}{RESET}\n")


# =============================================================================
# 1. PYTHON VERSION
# =============================================================================

def check_python():
    section("1. Python Version")
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if v.major == 3 and v.minor >= 10:
        ok("Python version", version_str)
    elif v.major == 3 and v.minor >= 8:
        warn("Python version", f"{version_str} (3.10+ recommended)")
    else:
        fail("Python version", f"{version_str} (need 3.8+)")


# =============================================================================
# 2. ENVIRONMENT VARIABLES
# =============================================================================

def check_env_vars():
    section("2. Environment Variables")

    # Load .env if python-dotenv available
    try:
        from dotenv import load_dotenv
        load_dotenv()
        ok("python-dotenv", "loaded .env file")
    except ImportError:
        warn("python-dotenv", "not installed — using shell env only")

    # GROQ_API_KEY — required for CLI and API
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key and groq_key.startswith("gsk_"):
        ok("GROQ_API_KEY", f"set ({groq_key[:12]}...)")
    elif groq_key:
        warn("GROQ_API_KEY", f"set but unexpected format ({groq_key[:8]}...)")
    else:
        fail("GROQ_API_KEY", "not set — required for LLM generation")

    # RESEARCH_OS_API_KEY — required for API only
    api_key = os.environ.get("RESEARCH_OS_API_KEY", "")
    if api_key and len(api_key) >= 8:
        ok("RESEARCH_OS_API_KEY", f"set ({api_key[:8]}...)")
    elif api_key:
        warn("RESEARCH_OS_API_KEY", "set but very short (use 16+ chars)")
    else:
        warn("RESEARCH_OS_API_KEY", "not set — needed for API auth (not needed for CLI)")


# =============================================================================
# 3. PYTHON PACKAGES
# =============================================================================

def check_package(name, import_name=None, required=True):
    """Try importing a package and report status."""
    import_name = import_name or name
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        ok(name, version)
        return True
    except ImportError:
        if required:
            fail(name, "not installed")
        else:
            warn(name, "not installed (optional)")
        return False


def check_packages():
    section("3. Core Packages (RAG Pipeline)")
    check_package("pymupdf4llm")
    check_package("pymupdf", "fitz")
    check_package("fastembed")
    check_package("faiss-cpu", "faiss")
    check_package("rank-bm25", "rank_bm25")
    check_package("numpy")
    check_package("groq")
    check_package("pydantic")
    check_package("python-dotenv", "dotenv", required=False)

    section("4. Reranking (FlashRank)")
    check_package("flashrank")

    section("5. API Packages")
    check_package("fastapi")
    check_package("uvicorn")
    check_package("sse-starlette", "sse_starlette")
    check_package("python-multipart", "multipart")

    section("6. Optional Packages")
    check_package("ollama", required=False)
    check_package("requests", required=False)
    check_package("httpx", required=False)
    check_package("rich", required=False)
    check_package("tqdm", required=False)


# =============================================================================
# 4. PROJECT STRUCTURE
# =============================================================================

def check_file(path, required=True, description=""):
    """Check if a file exists."""
    p = Path(path)
    label = f"{path}"
    if description:
        label = f"{path} ({description})"

    if p.exists():
        if p.is_file():
            size = p.stat().st_size
            if size == 0:
                warn(label, "exists but EMPTY")
            else:
                size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
                ok(label, size_str)
        else:
            ok(label, "directory exists")
        return True
    else:
        if required:
            fail(label, "MISSING")
        else:
            warn(label, "missing (optional)")
        return False


def check_dir(path, required=True, description=""):
    """Check if a directory exists."""
    p = Path(path)
    label = f"{path}/"
    if description:
        label = f"{path}/ ({description})"

    if p.exists() and p.is_dir():
        count = len(list(p.iterdir()))
        ok(label, f"{count} items")
        return True
    else:
        if required:
            fail(label, "MISSING")
        else:
            warn(label, "missing (optional)")
        return False


def check_structure():
    section("7. Project Structure — Source Code")
    check_file("src/rag/__init__.py", True, "RAG package init")
    check_file("src/rag/data_loader.py", True, "document parsing")
    check_file("src/rag/embedder.py", True, "embedding model")
    check_file("src/rag/retriever.py", True, "FAISS + BM25 search")
    check_file("src/rag/generator.py", True, "LLM generation")
    check_file("src/rag/pipeline.py", True, "RAG orchestration")
    check_file("src/rag/verifier.py", True, "code verification")

    section("8. Project Structure — API")
    check_file("src/api/__init__.py", True, "API package init")
    check_file("src/api/main.py", True, "FastAPI app factory")
    check_file("src/api/routes.py", True, "HTTP endpoints")
    check_file("src/api/dependencies.py", True, "auth + singleton")

    section("9. Project Structure — Entry Points")
    check_file("main.py", True, "CLI entry point")
    check_file("run_api.py", True, "API server launcher")
    check_file("requirements.txt", True, "dependencies")
    check_file(".env", False, "environment variables")

    section("10. Project Structure — Data Directories")
    check_dir("data", True, "root data directory")
    check_dir("data/index", False, "FAISS index storage")
    check_dir("data/01_fundamentals", False, "textbook PDFs")
    check_dir("data/02_papers", False, "research papers")
    check_dir("data/03_implementation", False, "coding guides")
    check_dir("data/04_misc", False, "uploaded files")

    section("11. Project Structure — Test Files")
    check_file("test_api_complete.py", False, "API test suite")
    check_file("test_curl.sh", False, "curl smoke tests")
    check_file("test_stream_visual.py", False, "visual SSE test")


# =============================================================================
# 5. FAISS INDEX
# =============================================================================

def check_index():
    section("12. FAISS Index")

    index_dir = Path("data/index")

    if not index_dir.exists():
        warn("data/index/", "directory missing — run ingestion first")
        return

    check_file("data/index/faiss.index", True, "FAISS vector index")
    check_file("data/index/chunks.pkl", True, "chunk storage")
    check_file("data/index/chunk_texts.pkl", True, "text storage")
    check_file("data/index/bm25.pkl", True, "BM25 sparse index")
    check_file("data/index/config.json", True, "index configuration")
    check_file("data/index/processed_files.json", False, "ingestion ledger")

    # Read config to show stats
    config_path = index_dir / "config.json"
    if config_path.exists():
        try:
            import json
            with open(config_path) as f:
                config = json.load(f)
            n_chunks = config.get("n_chunks", "?")
            dimension = config.get("dimension", "?")
            ok("Index config", f"chunks={n_chunks} dimension={dimension}")
        except Exception as e:
            warn("Index config", f"could not read: {e}")

    # Read ledger to show file count
    ledger_path = index_dir / "processed_files.json"
    if ledger_path.exists():
        try:
            import json
            with open(ledger_path) as f:
                ledger = json.load(f)
            n_files = ledger.get("total_files", "?")
            ok("Ingestion ledger", f"{n_files} files tracked")
        except Exception as e:
            warn("Ingestion ledger", f"could not read: {e}")


# =============================================================================
# 6. FLASHRANK CACHE
# =============================================================================

def check_flashrank_cache():
    section("13. FlashRank Reranker Cache")

    cache_dir = Path(".cache/flashrank/ms-marco-TinyBERT-L-2-v2")

    if not cache_dir.exists():
        warn("FlashRank cache", "not found — will download on first run")
        return

    check_file(
        ".cache/flashrank/ms-marco-TinyBERT-L-2-v2/flashrank-TinyBERT-L-2-v2.onnx",
        False, "ONNX model"
    )
    check_file(
        ".cache/flashrank/ms-marco-TinyBERT-L-2-v2/tokenizer.json",
        False, "tokenizer"
    )
    check_file(
        ".cache/flashrank/ms-marco-TinyBERT-L-2-v2/config.json",
        False, "model config"
    )


# =============================================================================
# 7. IMPORT SMOKE TEST
# =============================================================================

def check_imports():
    section("14. Import Smoke Test")

    # Test RAG package import
    try:
        from src.rag import (
            ResearchPipeline,
            PipelineConfig,
            create_pipeline,
            FastEmbedder,
            HybridRetriever,
            ResearchArchitect,
            ArchitectureVerifier,
            UniversalLoader,
            StreamChunk,
        )
        ok("src.rag", "all core classes importable")
    except ImportError as e:
        fail("src.rag", f"import failed: {e}")
    except Exception as e:
        fail("src.rag", f"unexpected error: {e}")

    # Test API package import
    try:
        from src.api import create_app
        ok("src.api", "create_app importable")
    except ImportError as e:
        fail("src.api", f"import failed: {e}")
    except Exception as e:
        fail("src.api", f"unexpected error: {e}")

    # Test that CLI main is importable
    try:
        import main as cli_main
        if hasattr(cli_main, "main"):
            ok("main.py", "main() function found")
        else:
            warn("main.py", "importable but no main() function")
    except ImportError as e:
        fail("main.py", f"import failed: {e}")
    except Exception as e:
        warn("main.py", f"import triggered side effect: {type(e).__name__}")


# =============================================================================
# 8. NETWORK CONNECTIVITY (Optional)
# =============================================================================

def check_network():
    section("15. Network Connectivity (Optional)")

    # Groq API reachability
    try:
        import requests
        r = requests.get("https://api.groq.com", timeout=5)
        ok("Groq API reachable", f"HTTP {r.status_code}")
    except ImportError:
        warn("Groq API check", "requests not installed — skipping")
    except Exception as e:
        warn("Groq API reachable", f"cannot reach: {e}")

    # Ollama local
    try:
        import requests
        r = requests.get("http://localhost:11434/api/version", timeout=2)
        if r.status_code == 200:
            ok("Ollama local", f"running — {r.json().get('version', '?')}")
        else:
            warn("Ollama local", f"responded HTTP {r.status_code}")
    except ImportError:
        warn("Ollama check", "requests not installed — skipping")
    except Exception:
        warn("Ollama local", "not running (optional fallback)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"""
{BOLD}{CYAN}{'=' * 60}
  RESEARCH-OS  SYSTEM  VERIFICATION
{'=' * 60}{RESET}
""")

    check_python()
    check_env_vars()
    check_packages()
    check_structure()
    check_index()
    check_flashrank_cache()
    check_imports()
    check_network()

    # Summary
    total = passed + failed + warned
    print(f"""
{BOLD}{CYAN}{'=' * 60}
  VERIFICATION RESULTS
{'=' * 60}{RESET}

  {GREEN}OK:       {passed}{RESET}
  {RED}FAIL:     {failed}{RESET}
  {YELLOW}WARN:     {warned}{RESET}
  Total:    {total}
""")

    if failed > 0:
        print(f"  {RED}{BOLD}SYSTEM NOT READY{RESET}")
        print(f"  Fix the FAIL items above before running CLI or API.\n")
        sys.exit(1)
    elif warned > 0:
        print(f"  {YELLOW}{BOLD}SYSTEM READY (with warnings){RESET}")
        print(f"  WARN items are optional but recommended.\n")
        sys.exit(0)
    else:
        print(f"  {GREEN}{BOLD}SYSTEM FULLY READY{RESET}")
        print(f"  All checks passed. You can run CLI or API.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
