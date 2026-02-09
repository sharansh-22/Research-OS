#!/usr/bin/env python3
"""
Auto Download & Classify Agent
==============================
Intelligent document acquisition with LLM-based classification.

Features:
- Smart GitHub URL conversion (blob → raw)
- Content sniffing (PDF pages / text preview)
- Groq LLM classification into folders
- Automatic file organization

Usage:
    python scripts/auto_download.py --url "https://github.com/user/repo/blob/main/file.pdf"
    python scripts/auto_download.py --url "https://arxiv.org/pdf/1706.03762.pdf"
    python scripts/auto_download.py --url "https://example.com/code.py"
    
    # Batch mode (file with URLs)
    python scripts/auto_download.py --batch urls.txt
    
    # Override classification
    python scripts/auto_download.py --url "..." --folder 02_papers
"""

import os
import sys
import re
import json
import shutil
import tempfile
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse, unquote

import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("auto-download")


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("data")

FOLDERS = {
    "01_fundamentals": "Math, Theory, Textbooks, Fundamentals",
    "02_papers": "Research Papers, Academic PDFs, ArXiv",
    "03_implementation": "Code, Documentation, Tutorials, Notebooks",
    "04_misc": "Uncategorized, Mixed content",
}

SUPPORTED_EXTENSIONS = {
    '.pdf', '.py', '.ipynb', '.md', '.tex', 
    '.cpp', '.cu', '.h', '.hpp', '.c',
    '.txt', '.rst', '.json'
}

# Groq configuration
GROQ_MODEL = "llama-3.3-70b-versatile"

CLASSIFICATION_PROMPT = """You are a document classifier for a Machine Learning research system.

Classify this document into exactly ONE category based on its content:

CATEGORIES:
- 01_fundamentals: Math foundations, linear algebra, calculus, probability theory, textbook content, theoretical foundations
- 02_papers: Research papers, academic publications, arXiv papers, conference papers (NeurIPS, ICML, etc.)
- 03_implementation: Code files, tutorials, documentation, Jupyter notebooks, implementation guides, PyTorch/TensorFlow code
- 04_misc: Cannot determine, mixed content, or doesn't fit other categories

DOCUMENT PREVIEW:
---
{content}
---

FILENAME: {filename}

Respond with ONLY the folder name (e.g., "02_papers"). No explanation."""


# =============================================================================
# URL UTILITIES
# =============================================================================

def convert_github_url(url: str) -> str:
    """
    Convert GitHub blob URLs to raw content URLs.
    
    Examples:
        https://github.com/user/repo/blob/main/file.py
        → https://raw.githubusercontent.com/user/repo/main/file.py
    """
    # Pattern for GitHub blob URLs
    github_blob_pattern = r'https?://github\.com/([^/]+)/([^/]+)/blob/(.+)'
    match = re.match(github_blob_pattern, url)
    
    if match:
        user, repo, path = match.groups()
        raw_url = f"https://raw.githubusercontent.com/{user}/{repo}/{path}"
        logger.info(f"Converted GitHub URL: {url[:50]}... → raw.githubusercontent.com")
        return raw_url
    
    # Pattern for GitHub raw URLs (already correct)
    if 'raw.githubusercontent.com' in url:
        return url
    
    # Pattern for GitHub gist
    gist_pattern = r'https?://gist\.github\.com/([^/]+)/([^/]+)'
    gist_match = re.match(gist_pattern, url)
    if gist_match:
        # Gist raw URLs need different handling
        logger.warning("GitHub Gist URLs may need manual raw URL extraction")
    
    return url


def extract_filename(url: str, response: requests.Response) -> str:
    """Extract filename from URL or Content-Disposition header."""
    # Try Content-Disposition header first
    cd = response.headers.get('Content-Disposition', '')
    if 'filename=' in cd:
        match = re.search(r'filename[*]?=["\']?([^"\';\n]+)', cd)
        if match:
            return unquote(match.group(1))
    
    # Fall back to URL path
    parsed = urlparse(url)
    path = unquote(parsed.path)
    filename = Path(path).name
    
    # Handle URLs without extension
    if '.' not in filename:
        content_type = response.headers.get('Content-Type', '')
        if 'pdf' in content_type:
            filename += '.pdf'
        elif 'python' in content_type or 'x-python' in content_type:
            filename += '.py'
        elif 'json' in content_type:
            filename += '.json'
        elif 'text' in content_type:
            filename += '.txt'
    
    # Sanitize filename
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    return filename or 'downloaded_file'


# =============================================================================
# CONTENT EXTRACTION
# =============================================================================

def extract_pdf_preview(file_path: Path, max_pages: int = 2) -> str:
    """Extract text from first N pages of PDF."""
    try:
        import fitz  # pymupdf
        
        doc = fitz.open(str(file_path))
        text_parts = []
        
        for page_num in range(min(max_pages, len(doc))):
            page = doc[page_num]
            text = page.get_text()
            text_parts.append(text)
        
        doc.close()
        
        full_text = "\n\n".join(text_parts)
        # Limit to ~3000 chars for LLM
        return full_text[:3000]
        
    except Exception as e:
        logger.error(f"Failed to extract PDF text: {e}")
        return f"[PDF file, extraction failed: {e}]"


def extract_text_preview(file_path: Path, max_chars: int = 2000) -> str:
    """Extract text preview from text-based files."""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read(max_chars)
            return content
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            return f"[Text file, read failed: {e}]"
    
    return "[Binary or unreadable file]"


def extract_notebook_preview(file_path: Path, max_cells: int = 5) -> str:
    """Extract preview from Jupyter notebook."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        cells = notebook.get('cells', [])[:max_cells]
        preview_parts = []
        
        for cell in cells:
            cell_type = cell.get('cell_type', '')
            source = ''.join(cell.get('source', []))
            preview_parts.append(f"[{cell_type.upper()}]\n{source[:500]}")
        
        return "\n\n".join(preview_parts)[:2000]
        
    except Exception as e:
        logger.error(f"Failed to parse notebook: {e}")
        return f"[Jupyter notebook, parse failed: {e}]"


def get_content_preview(file_path: Path) -> str:
    """Get content preview based on file type."""
    ext = file_path.suffix.lower()
    
    if ext == '.pdf':
        return extract_pdf_preview(file_path)
    elif ext == '.ipynb':
        return extract_notebook_preview(file_path)
    else:
        return extract_text_preview(file_path)


# =============================================================================
# LLM CLASSIFICATION
# =============================================================================

def classify_with_llm(
    content: str,
    filename: str,
    api_key: str,
) -> Tuple[str, Optional[str]]:
    """
    Classify document using Groq LLM.
    
    Returns:
        (folder_name, reasoning)
    """
    try:
        from groq import Groq
        
        client = Groq(api_key=api_key)
        
        prompt = CLASSIFICATION_PROMPT.format(
            content=content[:3000],
            filename=filename,
        )
        
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=50,
        )
        
        result = response.choices[0].message.content.strip()
        
        # Clean up response (in case LLM adds extra text)
        for folder in FOLDERS.keys():
            if folder in result:
                return folder, None
        
        # Default fallback
        logger.warning(f"Unexpected LLM response: {result}")
        return "04_misc", f"Unexpected response: {result}"
        
    except Exception as e:
        logger.error(f"LLM classification failed: {e}")
        return "04_misc", str(e)


def classify_by_extension(filename: str) -> str:
    """Fallback classification by file extension."""
    ext = Path(filename).suffix.lower()
    
    extension_map = {
        '.pdf': '02_papers',  # Assume papers by default
        '.py': '03_implementation',
        '.ipynb': '03_implementation',
        '.cpp': '03_implementation',
        '.cu': '03_implementation',
        '.c': '03_implementation',
        '.h': '03_implementation',
        '.hpp': '03_implementation',
        '.tex': '02_papers',
        '.md': '03_implementation',
        '.rst': '03_implementation',
        '.txt': '04_misc',
    }
    
    return extension_map.get(ext, '04_misc')


# =============================================================================
# MAIN DOWNLOAD & CLASSIFY FUNCTION
# =============================================================================

def download_and_classify(
    url: str,
    api_key: Optional[str] = None,
    override_folder: Optional[str] = None,
    dry_run: bool = False,
) -> Tuple[bool, str, str]:
    """
    Download file, classify with LLM, and organize.
    
    Args:
        url: URL to download
        api_key: Groq API key (uses env var if None)
        override_folder: Skip LLM, use this folder
        dry_run: Don't actually move file
        
    Returns:
        (success, filename, folder)
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    
    print(f"\n{'='*60}")
    print(f"📥 URL: {url[:70]}{'...' if len(url) > 70 else ''}")
    print(f"{'='*60}")
    
    # Step 1: Convert GitHub URLs
    download_url = convert_github_url(url)
    
    # Step 2: Download to temp file
    print("⏳ Downloading...")
    try:
        response = requests.get(
            download_url,
            timeout=60,
            headers={'User-Agent': 'Research-OS/1.0'},
            stream=True,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Download failed: {e}")
        return False, "", ""
    
    # Extract filename
    filename = extract_filename(url, response)
    ext = Path(filename).suffix.lower()
    
    # Check if supported
    if ext not in SUPPORTED_EXTENSIONS:
        logger.warning(f"Unsupported extension: {ext}")
        # Still allow download to misc
    
    # Save to temp file
    temp_dir = Path(tempfile.mkdtemp())
    temp_file = temp_dir / filename
    
    with open(temp_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    file_size = temp_file.stat().st_size
    print(f"✓ Downloaded: {filename} ({file_size / 1024:.1f} KB)")
    
    # Step 3: Extract content preview
    print("🔍 Analyzing content...")
    preview = get_content_preview(temp_file)
    print(f"   Preview: {preview[:100].replace(chr(10), ' ')}...")
    
    # Step 4: Classify
    if override_folder:
        folder = override_folder
        reasoning = "User override"
        print(f"📁 Folder (override): {folder}")
    elif api_key:
        print("🤖 Asking LLM...")
        folder, reasoning = classify_with_llm(preview, filename, api_key)
        print(f"📁 LLM Classification: {folder}")
        if reasoning:
            print(f"   Note: {reasoning}")
    else:
        folder = classify_by_extension(filename)
        reasoning = "Extension-based (no API key)"
        print(f"📁 Fallback Classification: {folder}")
    
    # Step 5: Move to destination
    dest_dir = DATA_DIR / folder
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / filename
    
    # Handle duplicate filenames
    if dest_file.exists():
        base = dest_file.stem
        ext = dest_file.suffix
        counter = 1
        while dest_file.exists():
            dest_file = dest_dir / f"{base}_{counter}{ext}"
            counter += 1
        print(f"⚠️ File exists, renamed to: {dest_file.name}")
    
    if dry_run:
        print(f"🔸 DRY RUN: Would move to {dest_file}")
        # Cleanup temp
        shutil.rmtree(temp_dir)
    else:
        shutil.move(str(temp_file), str(dest_file))
        print(f"✓ Saved to: {dest_file}")
        # Cleanup temp dir
        shutil.rmtree(temp_dir)
    
    print(f"{'='*60}\n")
    
    return True, filename, folder


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_batch(
    urls_file: Path,
    api_key: Optional[str] = None,
    dry_run: bool = False,
):
    """Process multiple URLs from a file."""
    if not urls_file.exists():
        logger.error(f"File not found: {urls_file}")
        return
    
    with open(urls_file, 'r') as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"\n📋 Batch processing {len(urls)} URLs...\n")
    
    results = {"success": 0, "failed": 0}
    
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}]")
        success, filename, folder = download_and_classify(
            url=url,
            api_key=api_key,
            dry_run=dry_run,
        )
        
        if success:
            results["success"] += 1
        else:
            results["failed"] += 1
    
    print(f"\n{'='*60}")
    print(f"📊 BATCH COMPLETE")
    print(f"   Success: {results['success']}")
    print(f"   Failed: {results['failed']}")
    print(f"{'='*60}\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Auto-download and classify documents for Research-OS"
    )
    
    parser.add_argument(
        "--url", "-u",
        type=str,
        help="URL to download"
    )
    
    parser.add_argument(
        "--batch", "-b",
        type=Path,
        help="File containing URLs (one per line)"
    )
    
    parser.add_argument(
        "--folder", "-f",
        type=str,
        choices=list(FOLDERS.keys()),
        help="Override LLM classification with specific folder"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and classify but don't move files"
    )
    
    parser.add_argument(
        "--list-folders",
        action="store_true",
        help="Show available folders and exit"
    )
    
    args = parser.parse_args()
    
    # List folders
    if args.list_folders:
        print("\n📁 Available Folders:")
        for folder, desc in FOLDERS.items():
            print(f"   {folder}: {desc}")
        print("")
        return
    
    # Check for API key
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\n⚠️ GROQ_API_KEY not set - using extension-based classification")
        print("   For smart classification: export GROQ_API_KEY='your-key'\n")
    
    # Process
    if args.batch:
        process_batch(
            urls_file=args.batch,
            api_key=api_key,
            dry_run=args.dry_run,
        )
    elif args.url:
        download_and_classify(
            url=args.url,
            api_key=api_key,
            override_folder=args.folder,
            dry_run=args.dry_run,
        )
    else:
        parser.print_help()
        print("\n❌ Please provide --url or --batch\n")


if __name__ == "__main__":
    main()
