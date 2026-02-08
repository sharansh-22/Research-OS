"""
Research Document Loader - Uses pymupdf4llm (NOT marker-pdf)
"""

import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from enum import Enum
import logging

import pymupdf4llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkType(Enum):
    CODE = "code"
    THEORY = "theory"
    MATH = "math"
    ALGORITHM = "algorithm"
    DEFINITION = "definition"
    THEOREM = "theorem"
    PROOF = "proof"


@dataclass
class Chunk:
    content: str
    chunk_type: ChunkType
    metadata: Dict = field(default_factory=dict)
    chunk_id: str = ""
    
    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(self.content.encode()).hexdigest()[:12]
    
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "type": self.chunk_type.value,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Chunk":
        return cls(
            content=data["content"],
            chunk_type=ChunkType(data["type"]),
            metadata=data.get("metadata", {}),
            chunk_id=data.get("chunk_id", ""),
        )


class ResearchDocumentLoader:
    """Loads PDFs using pymupdf4llm and splits into semantic chunks."""
    
    CODE_PATTERN = re.compile(r'```[\w]*\n[\s\S]*?```', re.MULTILINE)
    MATH_PATTERN = re.compile(r'\$\$[\s\S]+?\$\$', re.MULTILINE)
    SECTION_PATTERN = re.compile(r'^(#{1,6})\s+(.+?)$', re.MULTILINE)
    
    def __init__(self, min_chunk_chars: int = 100, max_chunk_chars: int = 3000):
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
    
    def load_pdf(self, pdf_path: str | Path) -> List[Chunk]:
        """Load PDF and return chunks."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        logger.info(f"Loading PDF: {pdf_path.name}")
        
        # Convert PDF to Markdown using pymupdf4llm
        markdown = pymupdf4llm.to_markdown(str(pdf_path))
        
        base_metadata = {
            "source": pdf_path.name,
            "source_path": str(pdf_path.absolute()),
        }
        
        chunks = self._parse_markdown(markdown, base_metadata)
        logger.info(f"Extracted {len(chunks)} chunks from {pdf_path.name}")
        return chunks
    
    def load_directory(self, dir_path: str | Path, recursive: bool = True) -> List[Chunk]:
        """Load all PDFs from directory."""
        dir_path = Path(dir_path)
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(dir_path.glob(pattern))
        
        logger.info(f"Found {len(pdf_files)} PDFs in {dir_path}")
        
        all_chunks = []
        for pdf in pdf_files:
            try:
                chunks = self.load_pdf(pdf)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to load {pdf}: {e}")
        
        return all_chunks
    
    def _parse_markdown(self, markdown: str, base_metadata: Dict) -> List[Chunk]:
        """Parse markdown into chunks."""
        chunks = []
        
        # Extract code blocks
        for match in self.CODE_PATTERN.finditer(markdown):
            code = self._clean_code_block(match.group(0))
            if len(code.strip()) >= 50:
                chunks.append(Chunk(
                    content=code,
                    chunk_type=ChunkType.CODE,
                    metadata={**base_metadata, "language": self._detect_language(match.group(0))}
                ))
        
        # Remove code blocks for theory processing
        theory_text = self.CODE_PATTERN.sub('', markdown)
        
        # Split by sections
        sections = self._split_by_sections(theory_text)
        
        for title, content in sections:
            if len(content.strip()) < self.min_chunk_chars:
                continue
            
            # Classify content
            chunk_type = self._classify_content(content)
            
            # Split large sections
            if len(content) > self.max_chunk_chars:
                sub_chunks = self._split_large_text(content)
                for sub in sub_chunks:
                    chunks.append(Chunk(
                        content=sub.strip(),
                        chunk_type=chunk_type,
                        metadata={**base_metadata, "section": title}
                    ))
            else:
                chunks.append(Chunk(
                    content=content.strip(),
                    chunk_type=chunk_type,
                    metadata={**base_metadata, "section": title}
                ))
        
        return chunks
    
    def _split_by_sections(self, text: str) -> List[Tuple[str, str]]:
        """Split text by headers."""
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
    
    def _split_large_text(self, text: str) -> List[str]:
        """Split large text into smaller chunks."""
        paragraphs = text.split('\n\n')
        chunks = []
        current = []
        current_len = 0
        
        for para in paragraphs:
            if current_len + len(para) > self.max_chunk_chars and current:
                chunks.append('\n\n'.join(current))
                current = []
                current_len = 0
            current.append(para)
            current_len += len(para)
        
        if current:
            chunks.append('\n\n'.join(current))
        
        return chunks
    
    def _classify_content(self, text: str) -> ChunkType:
        """Classify chunk type based on content."""
        math_matches = self.MATH_PATTERN.findall(text)
        math_ratio = sum(len(m) for m in math_matches) / len(text) if text else 0
        
        if math_ratio > 0.3:
            return ChunkType.MATH
        
        lower = text.lower()
        if '**theorem' in lower or 'theorem ' in lower[:100]:
            return ChunkType.THEOREM
        if '**definition' in lower or 'definition ' in lower[:100]:
            return ChunkType.DEFINITION
        if '**proof' in lower:
            return ChunkType.PROOF
        if '**algorithm' in lower:
            return ChunkType.ALGORITHM
        
        return ChunkType.THEORY
    
    def _clean_code_block(self, code: str) -> str:
        """Remove markdown code fences."""
        lines = code.strip().split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        return '\n'.join(lines)
    
    def _detect_language(self, code_block: str) -> str:
        """Detect programming language."""
        first_line = code_block.split('\n')[0]
        match = re.match(r'```(\w+)', first_line)
        if match:
            return match.group(1).lower()
        if 'import torch' in code_block or 'def ' in code_block:
            return 'python'
        return 'unknown'


def load_research_pdf(path: str | Path) -> List[Chunk]:
    """Quick function to load a PDF."""
    return ResearchDocumentLoader().load_pdf(path)
    