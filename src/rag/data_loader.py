
from typing import List
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    raw_text = extract_text_from_pdf(pdf_path)
    return chunk_text(raw_text, chunk_size, overlap)
