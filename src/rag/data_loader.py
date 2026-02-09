"""
Universal Document Loader
=========================
Multi-format document parsing with unified output.

Supported Formats:
- PDF (.pdf) - via pymupdf4llm
- Python (.py) - split by functions/classes
- Jupyter (.ipynb) - code/markdown cells
- Markdown (.md) - split by headers
- LaTeX (.tex) - split by sections
- C++/CUDA (.cpp, .cu, .c, .h) - split by functions

Output Format:
    {
        "text": "content...",
        "metadata": {
            "source": "filename.ext",
            "type": "code" | "theory",
            "chunk_id": "abc123"
        }
    }
"""

import re
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Union
from enum import Enum
from abc import ABC, abstractmethod
import logging

import pymupdf4llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================

class ChunkType(Enum):
    """Classification of chunk content type."""
    CODE = "code"
    THEORY = "theory"
    MATH = "math"
    ALGORITHM = "algorithm"
    DEFINITION = "definition"
    THEOREM = "theorem"
    PROOF = "proof"
    MARKDOWN = "markdown"
    MIXED = "mixed"


@dataclass
class Chunk:
    """A piece of document content with metadata."""
    content: str
    chunk_type: ChunkType
    metadata: Dict = field(default_factory=dict)
    chunk_id: str = ""
    
    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(self.content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict:
        """Convert to standardized output format."""
        return {
            "text": self.content,
            "metadata": {
                "source": self.metadata.get("source", "unknown"),
                "type": self.chunk_type.value,
                "chunk_id": self.chunk_id,
                **{k: v for k, v in self.metadata.items() if k != "source"}
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Chunk":
        """Reconstruct from dictionary."""
        # Handle both old and new format
        if "text" in data:
            content = data["text"]
            metadata = data.get("metadata", {})
            chunk_type_str = metadata.pop("type", "theory")
        else:
            content = data.get("content", "")
            metadata = data.get("metadata", {})
            chunk_type_str = data.get("type", "theory")
        
        return cls(
            content=content,
            chunk_type=ChunkType(chunk_type_str),
            metadata=metadata,
            chunk_id=metadata.get("chunk_id", ""),
        )


# =============================================================================
# BASE PARSER
# =============================================================================

class BaseParser(ABC):
    """Abstract base class for all parsers."""
    
    SUPPORTED_EXTENSIONS: List[str] = []
    
    def __init__(self, min_chunk_chars: int = 50, max_chunk_chars: int = 3000):
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
    
    @abstractmethod
    def parse(self, file_path: Path) -> List[Chunk]:
        """Parse file and return chunks."""
        pass
    
    def _read_file(self, file_path: Path) -> str:
        """Read file with encoding fallback."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode file: {file_path}")
    
    def _create_chunk(
        self,
        content: str,
        chunk_type: ChunkType,
        source: str,
        **extra_metadata
    ) -> Optional[Chunk]:
        """Create chunk if content meets minimum size."""
        content = content.strip()
        if len(content) < self.min_chunk_chars:
            return None
        
        return Chunk(
            content=content,
            chunk_type=chunk_type,
            metadata={"source": source, **extra_metadata}
        )
    
    def _split_large_text(self, text: str, separator: str = "\n\n") -> List[str]:
        """Split large text into smaller chunks."""
        if len(text) <= self.max_chunk_chars:
            return [text]
        
        parts = text.split(separator)
        chunks = []
        current = []
        current_len = 0
        
        for part in parts:
            if current_len + len(part) > self.max_chunk_chars and current:
                chunks.append(separator.join(current))
                current = []
                current_len = 0
            current.append(part)
            current_len += len(part) + len(separator)
        
        if current:
            chunks.append(separator.join(current))
        
        return chunks


# =============================================================================
# PDF PARSER
# =============================================================================

class PDFParser(BaseParser):
    """Parse PDF files using pymupdf4llm."""
    
    SUPPORTED_EXTENSIONS = ['.pdf']
    
    CODE_PATTERN = re.compile(r'```[\w]*\n[\s\S]*?```', re.MULTILINE)
    MATH_PATTERN = re.compile(r'\$\$[\s\S]+?\$\$', re.MULTILINE)
    SECTION_PATTERN = re.compile(r'^(#{1,6})\s+(.+?)$', re.MULTILINE)
    
    def parse(self, file_path: Path) -> List[Chunk]:
        logger.info(f"Parsing PDF: {file_path.name}")
        
        try:
            markdown = pymupdf4llm.to_markdown(str(file_path))
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            return []
        
        return self._parse_markdown(markdown, file_path.name)
    
    def _parse_markdown(self, markdown: str, source: str) -> List[Chunk]:
        chunks = []
        
        # Extract code blocks
        for match in self.CODE_PATTERN.finditer(markdown):
            code = self._clean_code_block(match.group(0))
            chunk = self._create_chunk(code, ChunkType.CODE, source, language=self._detect_language(match.group(0)))
            if chunk:
                chunks.append(chunk)
        
        # Remove code blocks for theory processing
        theory_text = self.CODE_PATTERN.sub('', markdown)
        
        # Split by sections
        sections = self._split_by_sections(theory_text)
        
        for title, content in sections:
            if len(content.strip()) < self.min_chunk_chars:
                continue
            
            chunk_type = self._classify_content(content)
            
            for text in self._split_large_text(content):
                chunk = self._create_chunk(text, chunk_type, source, section=title)
                if chunk:
                    chunks.append(chunk)
        
        logger.info(f"Extracted {len(chunks)} chunks from {source}")
        return chunks
    
    def _split_by_sections(self, text: str) -> List[Tuple[str, str]]:
        sections = []
        current_title = ""
        current_content = []
        
        for line in text.split('\n'):
            match = self.SECTION_PATTERN.match(line)
            if match:
                if current_content:
                    sections.append((current_title, '\n'.join(current_content)))
                current_title = match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)
        
        if current_content:
            sections.append((current_title, '\n'.join(current_content)))
        
        return sections if sections else [("", text)]
    
    def _classify_content(self, text: str) -> ChunkType:
        math_matches = self.MATH_PATTERN.findall(text)
        math_ratio = sum(len(m) for m in math_matches) / len(text) if text else 0
        
        if math_ratio > 0.3:
            return ChunkType.MATH
        
        lower = text.lower()
        if '**theorem' in lower or 'theorem ' in lower[:100]:
            return ChunkType.THEOREM
        if '**definition' in lower:
            return ChunkType.DEFINITION
        if '**proof' in lower:
            return ChunkType.PROOF
        
        return ChunkType.THEORY
    
    def _clean_code_block(self, code: str) -> str:
        lines = code.strip().split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        return '\n'.join(lines)
    
    def _detect_language(self, code_block: str) -> str:
        first_line = code_block.split('\n')[0]
        match = re.match(r'```(\w+)', first_line)
        if match:
            return match.group(1).lower()
        if 'import torch' in code_block or 'def ' in code_block:
            return 'python'
        return 'unknown'


# =============================================================================
# PYTHON PARSER
# =============================================================================

class PythonParser(BaseParser):
    """Parse Python files, splitting by functions and classes."""
    
    SUPPORTED_EXTENSIONS = ['.py']
    
    # Pattern to match top-level function/class definitions
    DEFINITION_PATTERN = re.compile(
        r'^((?:@\w+(?:\([^)]*\))?\s*\n)*)'  # Decorators
        r'^((?:async\s+)?(?:def|class)\s+\w+[^:]*:)',  # Definition
        re.MULTILINE
    )
    
    def parse(self, file_path: Path) -> List[Chunk]:
        logger.info(f"Parsing Python: {file_path.name}")
        
        try:
            content = self._read_file(file_path)
        except Exception as e:
            logger.error(f"Failed to read Python file: {e}")
            return []
        
        chunks = []
        source = file_path.name
        
        # Find all top-level definitions
        definitions = self._extract_definitions(content)
        
        if definitions:
            for name, code, def_type in definitions:
                chunk = self._create_chunk(
                    code, 
                    ChunkType.CODE, 
                    source,
                    definition=name,
                    definition_type=def_type,
                    language="python"
                )
                if chunk:
                    chunks.append(chunk)
        else:
            # No definitions found, treat whole file as one chunk
            for text in self._split_large_text(content):
                chunk = self._create_chunk(text, ChunkType.CODE, source, language="python")
                if chunk:
                    chunks.append(chunk)
        
        # Also extract module-level docstring as theory
        docstring = self._extract_module_docstring(content)
        if docstring:
            chunk = self._create_chunk(docstring, ChunkType.THEORY, source)
            if chunk:
                chunks.insert(0, chunk)
        
        logger.info(f"Extracted {len(chunks)} chunks from {source}")
        return chunks
    
    def _extract_definitions(self, content: str) -> List[Tuple[str, str, str]]:
        """Extract function and class definitions with their bodies."""
        lines = content.split('\n')
        definitions = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for decorators
            decorators = []
            while line.strip().startswith('@'):
                decorators.append(line)
                i += 1
                if i < len(lines):
                    line = lines[i]
                else:
                    break
            
            # Check for def or class at start of line (top-level)
            match = re.match(r'^(async\s+)?(def|class)\s+(\w+)', line)
            if match:
                async_prefix = match.group(1) or ''
                def_type = match.group(2)
                name = match.group(3)
                
                # Collect the full definition
                def_lines = decorators + [line]
                i += 1
                
                # Get indented body
                while i < len(lines):
                    next_line = lines[i]
                    # Empty lines or indented lines are part of body
                    if next_line.strip() == '' or (next_line and next_line[0] in ' \t'):
                        def_lines.append(next_line)
                        i += 1
                    else:
                        break
                
                # Remove trailing empty lines
                while def_lines and def_lines[-1].strip() == '':
                    def_lines.pop()
                
                code = '\n'.join(def_lines)
                if len(code) >= self.min_chunk_chars:
                    definitions.append((name, code, def_type))
            else:
                i += 1
        
        return definitions
    
    def _extract_module_docstring(self, content: str) -> Optional[str]:
        """Extract module-level docstring."""
        match = re.match(r'^[\s]*["\']["\']["\'](.+?)["\']["\']["\']', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        match = re.match(r'^[\s]*["\'](.+?)["\']', content)
        if match:
            return match.group(1).strip()
        
        return None


# =============================================================================
# JUPYTER NOTEBOOK PARSER
# =============================================================================

class JupyterParser(BaseParser):
    """Parse Jupyter notebooks, extracting code and markdown cells."""
    
    SUPPORTED_EXTENSIONS = ['.ipynb']
    
    def parse(self, file_path: Path) -> List[Chunk]:
        logger.info(f"Parsing Jupyter: {file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
        except Exception as e:
            logger.error(f"Failed to parse notebook: {e}")
            return []
        
        chunks = []
        source = file_path.name
        
        cells = notebook.get('cells', [])
        
        for idx, cell in enumerate(cells):
            cell_type = cell.get('cell_type', '')
            source_lines = cell.get('source', [])
            
            # Handle both list and string formats
            if isinstance(source_lines, list):
                content = ''.join(source_lines)
            else:
                content = source_lines
            
            if not content.strip():
                continue
            
            if cell_type == 'code':
                chunk_type = ChunkType.CODE
            elif cell_type == 'markdown':
                chunk_type = ChunkType.THEORY
            else:
                chunk_type = ChunkType.MIXED
            
            # Split large cells
            for text in self._split_large_text(content):
                chunk = self._create_chunk(
                    text,
                    chunk_type,
                    source,
                    cell_index=idx,
                    cell_type=cell_type,
                    language="python" if cell_type == 'code' else None
                )
                if chunk:
                    chunks.append(chunk)
        
        logger.info(f"Extracted {len(chunks)} chunks from {source}")
        return chunks


# =============================================================================
# MARKDOWN PARSER
# =============================================================================

class MarkdownParser(BaseParser):
    """Parse Markdown files, splitting by headers."""
    
    SUPPORTED_EXTENSIONS = ['.md', '.markdown']
    
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+?)$', re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r'```[\w]*\n[\s\S]*?```', re.MULTILINE)
    
    def parse(self, file_path: Path) -> List[Chunk]:
        logger.info(f"Parsing Markdown: {file_path.name}")
        
        try:
            content = self._read_file(file_path)
        except Exception as e:
            logger.error(f"Failed to read Markdown: {e}")
            return []
        
        chunks = []
        source = file_path.name
        
        # Extract code blocks first
        for match in self.CODE_BLOCK_PATTERN.finditer(content):
            code = self._clean_code_block(match.group(0))
            chunk = self._create_chunk(code, ChunkType.CODE, source, language=self._detect_language(match.group(0)))
            if chunk:
                chunks.append(chunk)
        
        # Remove code blocks for theory
        theory_text = self.CODE_BLOCK_PATTERN.sub('', content)
        
        # Split by headers
        sections = self._split_by_headers(theory_text)
        
        for header, text in sections:
            for part in self._split_large_text(text):
                chunk = self._create_chunk(part, ChunkType.THEORY, source, section=header)
                if chunk:
                    chunks.append(chunk)
        
        logger.info(f"Extracted {len(chunks)} chunks from {source}")
        return chunks
    
    def _split_by_headers(self, text: str) -> List[Tuple[str, str]]:
        sections = []
        current_header = ""
        current_content = []
        
        for line in text.split('\n'):
            match = self.HEADER_PATTERN.match(line)
            if match:
                if current_content:
                    sections.append((current_header, '\n'.join(current_content)))
                current_header = match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)
        
        if current_content:
            sections.append((current_header, '\n'.join(current_content)))
        
        return sections if sections else [("", text)]
    
    def _clean_code_block(self, code: str) -> str:
        lines = code.strip().split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        return '\n'.join(lines)
    
    def _detect_language(self, code_block: str) -> str:
        first_line = code_block.split('\n')[0]
        match = re.match(r'```(\w+)', first_line)
        return match.group(1).lower() if match else 'unknown'


# =============================================================================
# LATEX PARSER
# =============================================================================

class LaTeXParser(BaseParser):
    """Parse LaTeX files, splitting by sections."""
    
    SUPPORTED_EXTENSIONS = ['.tex']
    
    SECTION_PATTERN = re.compile(
        r'\\(section|subsection|subsubsection|chapter|paragraph)\{([^}]+)\}',
        re.MULTILINE
    )
    
    def parse(self, file_path: Path) -> List[Chunk]:
        logger.info(f"Parsing LaTeX: {file_path.name}")
        
        try:
            content = self._read_file(file_path)
        except Exception as e:
            logger.error(f"Failed to read LaTeX: {e}")
            return []
        
        chunks = []
        source = file_path.name
        
        # Find section boundaries
        matches = list(self.SECTION_PATTERN.finditer(content))
        
        if not matches:
            # No sections, treat as one chunk
            for text in self._split_large_text(content):
                chunk = self._create_chunk(text, ChunkType.THEORY, source)
                if chunk:
                    chunks.append(chunk)
        else:
            # Content before first section
            if matches[0].start() > 0:
                preamble = content[:matches[0].start()]
                chunk = self._create_chunk(preamble, ChunkType.THEORY, source, section="preamble")
                if chunk:
                    chunks.append(chunk)
            
            # Process each section
            for i, match in enumerate(matches):
                section_type = match.group(1)
                section_title = match.group(2)
                
                start = match.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
                section_content = content[start:end]
                
                for text in self._split_large_text(section_content):
                    chunk = self._create_chunk(
                        text, 
                        ChunkType.THEORY, 
                        source, 
                        section=section_title,
                        section_type=section_type
                    )
                    if chunk:
                        chunks.append(chunk)
        
        logger.info(f"Extracted {len(chunks)} chunks from {source}")
        return chunks


# =============================================================================
# C/C++/CUDA PARSER
# =============================================================================

class CppParser(BaseParser):
    """Parse C/C++/CUDA files, splitting by functions."""
    
    SUPPORTED_EXTENSIONS = ['.cpp', '.cu', '.c', '.h', '.hpp', '.cuh']
    
    # Pattern to match function definitions (simplified)
    FUNCTION_PATTERN = re.compile(
        r'^[\w\s\*&:<>,]+\s+'  # Return type
        r'(\w+)\s*'  # Function name
        r'\([^)]*\)\s*'  # Parameters
        r'(?:const\s*)?'  # Optional const
        r'(?:override\s*)?'  # Optional override
        r'\{',  # Opening brace
        re.MULTILINE
    )
    
    def parse(self, file_path: Path) -> List[Chunk]:
        logger.info(f"Parsing C++: {file_path.name}")
        
        try:
            content = self._read_file(file_path)
        except Exception as e:
            logger.error(f"Failed to read C++ file: {e}")
            return []
        
        chunks = []
        source = file_path.name
        language = "cuda" if file_path.suffix in ['.cu', '.cuh'] else "cpp"
        
        # Find functions
        functions = self._extract_functions(content)
        
        if functions:
            for name, code in functions:
                chunk = self._create_chunk(
                    code, 
                    ChunkType.CODE, 
                    source, 
                    function=name,
                    language=language
                )
                if chunk:
                    chunks.append(chunk)
        else:
            # No functions found, chunk by size
            for text in self._split_large_text(content):
                chunk = self._create_chunk(text, ChunkType.CODE, source, language=language)
                if chunk:
                    chunks.append(chunk)
        
        # Extract header comments as theory
        header_comment = self._extract_header_comment(content)
        if header_comment:
            chunk = self._create_chunk(header_comment, ChunkType.THEORY, source)
            if chunk:
                chunks.insert(0, chunk)
        
        logger.info(f"Extracted {len(chunks)} chunks from {source}")
        return chunks
    
    def _extract_functions(self, content: str) -> List[Tuple[str, str]]:
        """Extract function definitions with their bodies."""
        functions = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Look for function start (line ending with {)
            if '{' in line and not line.strip().startswith('//'):
                # Try to match function pattern
                match = self.FUNCTION_PATTERN.search(line)
                if match or ('(' in line and ')' in line):
                    # Get function name
                    func_match = re.search(r'(\w+)\s*\(', line)
                    name = func_match.group(1) if func_match else "unknown"
                    
                    # Skip common non-function patterns
                    if name in ['if', 'for', 'while', 'switch', 'catch']:
                        i += 1
                        continue
                    
                    # Extract function body using brace matching
                    func_lines = [line]
                    brace_count = line.count('{') - line.count('}')
                    i += 1
                    
                    while i < len(lines) and brace_count > 0:
                        func_lines.append(lines[i])
                        brace_count += lines[i].count('{') - lines[i].count('}')
                        i += 1
                    
                    code = '\n'.join(func_lines)
                    if len(code) >= self.min_chunk_chars:
                        functions.append((name, code))
                    continue
            
            i += 1
        
        return functions
    
    def _extract_header_comment(self, content: str) -> Optional[str]:
        """Extract file header comment."""
        # Look for /* */ style comment at start
        match = re.match(r'^\s*/\*(.+?)\*/', content, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Look for // style comments at start
        lines = content.split('\n')
        comment_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('//'):
                comment_lines.append(stripped[2:].strip())
            elif stripped and not stripped.startswith('#'):
                break
        
        if comment_lines:
            return '\n'.join(comment_lines)
        
        return None


# =============================================================================
# PLAIN TEXT PARSER
# =============================================================================

class TextParser(BaseParser):
    """Parse plain text files."""
    
    SUPPORTED_EXTENSIONS = ['.txt', '.rst']
    
    def parse(self, file_path: Path) -> List[Chunk]:
        logger.info(f"Parsing Text: {file_path.name}")
        
        try:
            content = self._read_file(file_path)
        except Exception as e:
            logger.error(f"Failed to read text file: {e}")
            return []
        
        chunks = []
        source = file_path.name
        
        for text in self._split_large_text(content):
            chunk = self._create_chunk(text, ChunkType.THEORY, source)
            if chunk:
                chunks.append(chunk)
        
        logger.info(f"Extracted {len(chunks)} chunks from {source}")
        return chunks


# =============================================================================
# UNIVERSAL LOADER (FACTORY)
# =============================================================================

class UniversalLoader:
    """
    Universal document loader with factory pattern.
    
    Automatically selects the appropriate parser based on file extension.
    
    Usage:
        >>> loader = UniversalLoader()
        >>> chunks = loader.load("paper.pdf")
        >>> chunks = loader.load("model.py")
        >>> chunks = loader.load_directory("data/")
    """
    
    # Parser registry
    PARSERS: Dict[str, type] = {
        '.pdf': PDFParser,
        '.py': PythonParser,
        '.ipynb': JupyterParser,
        '.md': MarkdownParser,
        '.markdown': MarkdownParser,
        '.tex': LaTeXParser,
        '.cpp': CppParser,
        '.cu': CppParser,
        '.c': CppParser,
        '.h': CppParser,
        '.hpp': CppParser,
        '.cuh': CppParser,
        '.txt': TextParser,
        '.rst': TextParser,
    }
    
    def __init__(self, min_chunk_chars: int = 50, max_chunk_chars: int = 3000):
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self._parser_cache: Dict[str, BaseParser] = {}
    
    def _get_parser(self, extension: str) -> Optional[BaseParser]:
        """Get or create parser for extension."""
        ext = extension.lower()
        
        if ext not in self.PARSERS:
            return None
        
        if ext not in self._parser_cache:
            parser_class = self.PARSERS[ext]
            self._parser_cache[ext] = parser_class(
                min_chunk_chars=self.min_chunk_chars,
                max_chunk_chars=self.max_chunk_chars
            )
        
        return self._parser_cache[ext]
    
    def load(self, file_path: Union[str, Path]) -> List[Chunk]:
        """
        Load and parse a single file.
        
        Args:
            file_path: Path to file
            
        Returns:
            List of Chunk objects
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        parser = self._get_parser(file_path.suffix)
        
        if parser is None:
            logger.warning(f"No parser for extension: {file_path.suffix}")
            return []
        
        return parser.parse(file_path)
    
    def load_directory(
        self,
        dir_path: Union[str, Path],
        recursive: bool = True,
    ) -> List[Chunk]:
        """
        Load all supported files from directory.
        
        Args:
            dir_path: Directory path
            recursive: Search subdirectories
            
        Returns:
            List of all Chunk objects
        """
        dir_path = Path(dir_path)
        
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        
        all_chunks = []
        extensions = set(self.PARSERS.keys())
        
        for ext in extensions:
            pattern = f"**/*{ext}" if recursive else f"*{ext}"
            files = list(dir_path.glob(pattern))
            
            for file_path in files:
                try:
                    chunks = self.load(file_path)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
        
        return all_chunks
    
    @classmethod
    def supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions."""
        return list(cls.PARSERS.keys())


# =============================================================================
# BACKWARD COMPATIBILITY
# =============================================================================

class ResearchDocumentLoader(UniversalLoader):
    """
    Backward-compatible alias for UniversalLoader.
    
    Maintains API compatibility with existing code.
    """
    
    def load_pdf(self, pdf_path: Union[str, Path]) -> List[Chunk]:
        """Load a PDF file (backward compatibility)."""
        return self.load(pdf_path)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_file(path: Union[str, Path]) -> List[Chunk]:
    """Quick function to load any supported file."""
    return UniversalLoader().load(path)


def load_research_pdf(path: Union[str, Path]) -> List[Chunk]:
    """Quick function to load a PDF (backward compatibility)."""
    return UniversalLoader().load(path)
