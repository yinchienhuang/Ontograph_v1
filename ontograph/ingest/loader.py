"""
ingest/loader.py — Route a source file to the correct converter by extension.

Supported formats:
    .pdf            → convert_pdf   (PyMuPDF)
    .txt  / .text   → convert_text
    .md   / .markdown → convert_markdown
"""

from __future__ import annotations

from pathlib import Path

from ontograph.models.document import RawDocument

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".txt", ".text", ".md", ".markdown"}
)


def load_document(path: str | Path) -> RawDocument:
    """
    Convert *path* to a :class:`~ontograph.models.document.RawDocument`.

    The converter is chosen by file extension (case-insensitive). The
    returned RawDocument is ready to be passed to the chunker.

    Raises:
        FileNotFoundError: if *path* does not exist.
        ValueError: if the extension is not in :data:`SUPPORTED_EXTENSIONS`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        from ontograph.ingest.converters.pdf import convert_pdf
        return convert_pdf(path)

    if ext in (".txt", ".text"):
        from ontograph.ingest.converters.text import convert_text
        return convert_text(path)

    if ext in (".md", ".markdown"):
        from ontograph.ingest.converters.markdown import convert_markdown
        return convert_markdown(path)

    raise ValueError(
        f"Unsupported format '{ext}'. "
        f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )
