"""
converters/pdf.py — Convert a PDF file to a normalized RawDocument.

Uses PyMuPDF (fitz) for text extraction. Heading levels are inferred from
span font sizes relative to the document's body (most common) font size:

    ratio ≥ 1.4 × body  →  H1
    ratio ≥ 1.2 × body  →  H2
    ratio ≥ 1.05 × body AND bold  →  H3
    bold short numbered line  →  H3
    everything else  →  body paragraph
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF

from ontograph.models.document import PageMapEntry, RawDocument

# Matches "1.", "1.2", "1.2.3" at the start of a heading candidate
_NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)*\s+\S")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _estimate_body_size(doc: fitz.Document) -> float:
    """
    Return the most common span font size across the first 5 pages.

    This is used as the 'body' size so we can express heading sizes as a
    ratio, making detection resilient to different base font sizes.
    """
    size_counts: dict[float, int] = {}
    sample_pages = min(5, len(doc))
    for i in range(sample_pages):
        page = doc[i]
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:  # skip image blocks
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = round(span.get("size", 0), 1)
                    if size > 4:  # ignore tiny footnote / sub-script sizes
                        size_counts[size] = size_counts.get(size, 0) + 1
    if not size_counts:
        return 10.0
    return max(size_counts, key=lambda s: size_counts[s])


def _heading_level(
    span_size: float, body_size: float, is_bold: bool, text: str
) -> int | None:
    """
    Return heading level 1–3, or None if this looks like body text.

    A span is never promoted to a heading if it is longer than 120 chars
    (likely a body sentence that happens to be in a large/bold font).
    """
    stripped = text.strip()
    if not stripped or len(stripped) > 120:
        return None
    if body_size <= 0:
        return None

    ratio = span_size / body_size
    if ratio >= 1.4:
        return 1
    if ratio >= 1.2:
        return 2
    if ratio >= 1.05 and is_bold:
        return 3
    # Bold + short + numbered-section pattern → H3
    if is_bold and _NUMBERED_HEADING.match(stripped) and len(stripped) < 80:
        return 3
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_pdf(path: str | Path) -> RawDocument:
    """
    Convert a PDF file to a RawDocument with page_map.

    Each PDF page becomes a block of Markdown text. Headings are detected
    from font-size heuristics. The page_map records which character range in
    the output Markdown corresponds to each page so downstream components
    (chunker, source locator) can recover exact page numbers.
    """
    path = Path(path).resolve()
    source_sha256 = _sha256_path(path)
    doc_id = hashlib.sha256(
        f"{path}:{source_sha256}".encode()
    ).hexdigest()[:16]

    fitz_doc = fitz.open(str(path))
    body_size = _estimate_body_size(fitz_doc)

    md_parts: list[str] = []
    page_map: list[PageMapEntry] = []
    char_offset = 0

    for page_idx in range(len(fitz_doc)):
        page = fitz_doc[page_idx]
        page_char_start = char_offset
        page_paragraphs: list[str] = []

        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue

            # Collect all lines in this block
            line_texts: list[str] = []
            block_level: int | None = None

            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                # Use the largest span in the line for heading classification
                dominant = max(spans, key=lambda s: s.get("size", 0))
                span_size = dominant.get("size", body_size)
                is_bold = bool(dominant.get("flags", 0) & 16)
                line_text = "".join(s.get("text", "") for s in spans).strip()
                if not line_text:
                    continue

                level = _heading_level(span_size, body_size, is_bold, line_text)
                if level is not None and block_level is None:
                    block_level = level
                line_texts.append(line_text)

            if not line_texts:
                continue

            full_text = " ".join(line_texts)
            if block_level is not None:
                page_paragraphs.append("#" * block_level + " " + full_text)
            else:
                page_paragraphs.append(full_text)

        page_md = "\n\n".join(page_paragraphs) + "\n\n" if page_paragraphs else "\n"
        md_parts.append(page_md)
        char_offset += len(page_md)

        page_map.append(PageMapEntry(
            char_start=page_char_start,
            char_end=char_offset,
            page=page_idx + 1,
        ))

    fitz_doc.close()

    return RawDocument(
        id=doc_id,
        source_path=str(path),
        source_format="pdf",
        source_sha256=source_sha256,
        markdown="".join(md_parts),
        page_map=page_map,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
