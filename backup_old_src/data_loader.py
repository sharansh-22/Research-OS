import os
import re
from typing import Any, Dict, List

from marker.convert import convert_single_pdf
from marker.models import load_all_models


def split_text_semantic(text: str, target_chars: int = 1000, overlap_sentences: int = 2) -> List[str]:
    """
    Sentence-based "semantic" splitting.

    Applied ONLY to theory blocks. Code blocks are kept intact.
    """
    paragraphs = re.split(r"\n{2,}", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # Robust sentence splitting regex
    sentence_split_regex = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s"

    sentences: List[str] = []
    for para in paragraphs:
        sents = re.split(sentence_split_regex, para)
        sents = [s.strip() for s in sents if s.strip()]
        sentences.extend(sents)

    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        current.append(sent)
        current_len += len(sent)

        if current_len >= target_chars:
            chunks.append(" ".join(current).strip())

            if overlap_sentences > 0:
                current = current[-overlap_sentences:]
                current_len = sum(len(s) for s in current)
            else:
                current = []
                current_len = 0

    if current:
        chunks.append(" ".join(current).strip())

    return [c for c in chunks if c]


class MathLoader:
    """
    Marker-PDF (OCR) based loader that preserves math (LaTeX) and code blocks.
    """

    def __init__(self):
        # Marker model loading can be heavy; do it once per loader instance.
        self.model_lst = load_all_models()

    def load_and_chunk(
        self,
        pdf_path: str,
        chunk_size: int = 1000,
        overlap: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Convert PDF -> Markdown and split into chunks:
        - Split markdown by code block delimiter ``` (triple backticks).
        - Inside delimiters => type: code (kept intact).
        - Outside delimiters => type: theory (sentence-split only).

        Returns chunks of shape:
        {
          "text": "...",
          "metadata": {"source": filename, "type": "code"|"theory", "chunk_id": int}
        }
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        filename = os.path.basename(pdf_path)

        # OCR/convert can fail on some PDFs; keep robust error handling.
        try:
            markdown_text, _images, _out_meta = convert_single_pdf(pdf_path, self.model_lst)
        except Exception as e:
            # Marker-PDF can fail on certain PDFs; return no chunks rather than crashing the pipeline.
            print(f"Marker OCR/convert failed for {filename}: {e}")
            return []

        if not markdown_text or not str(markdown_text).strip():
            return []

        # Split by delimiter ```; odd indices are "inside code", even are "outside".
        parts = str(markdown_text).split("```")

        chunks: List[Dict[str, Any]] = []
        chunk_id = 0

        for idx, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue

            is_code = idx % 2 == 1
            if is_code:
                # Put the delimiters back to keep it a proper fenced block.
                text = f"```{part}```"
                chunks.append(
                    {
                        "text": text,
                        "metadata": {"source": filename, "type": "code", "chunk_id": chunk_id},
                    }
                )
                chunk_id += 1
            else:
                theory_chunks = split_text_semantic(
                    part,
                    target_chars=chunk_size,
                    overlap_sentences=overlap,
                )
                for t in theory_chunks:
                    chunks.append(
                        {
                            "text": t,
                            "metadata": {"source": filename, "type": "theory", "chunk_id": chunk_id},
                        }
                    )
                    chunk_id += 1

        return chunks
