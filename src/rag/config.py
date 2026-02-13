"""
Research-OS Configuration
=========================
Centralized settings for the entire project.
Supports environment variable overrides.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# System Metadata
VERSION = "2.1.0"

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DEFAULT_INDEX_DIR = os.getenv("RESEARCH_OS_INDEX_DIR", "data/index")
DEFAULT_UPLOAD_DIR = os.getenv("RESEARCH_OS_UPLOAD_DIR", "data/04_misc")
LEDGER_FILENAME = "processed_files.json"

# Models
# Groq Models (Generation & Auditing)
DEFAULT_GENERATION_MODEL = os.getenv("RESEARCH_OS_GEN_MODEL", "llama-3.3-70b-versatile")
AUDITOR_MODEL = os.getenv("RESEARCH_OS_AUDIT_MODEL", "llama-3.3-70b-versatile")

# Encoder Models (Evaluation)
FAITHFULNESS_MODEL_ID = "cross-encoder/nli-deberta-v3-xsmall"
RELEVANCY_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# API Security
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
RESEARCH_OS_API_KEY = os.getenv("RESEARCH_OS_API_KEY")

# Pipeline Defaults
DEFAULT_TOP_K = 5
DEFAULT_MIN_SIMILARITY = 0.25
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_HISTORY = 3
CACHE_SIMILARITY_THRESHOLD = 0.95
