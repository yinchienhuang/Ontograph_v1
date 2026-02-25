"""
document.py — DocumentArtifact and supporting types.

Represents the output of ingestion: a source file that has been converted to
normalized Markdown and split into hierarchy-aware chunks with full source
provenance.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field, computed_field


# ---------------------------------------------------------------------------
# Source location — enough to open the exact spot in the original file
# ---------------------------------------------------------------------------

class SourceLocator(BaseModel):
    """Points back to the exact location in the original source file."""

    source_id: str = Field(description="sha256 of the original file bytes")
    source_path: str = Field(description="Absolute path at ingest time")
    source_format: Literal["pdf", "txt", "md"]

    # PDF-specific (None for txt/md)
    page: int | None = Field(default=None, description="1-indexed page number")
    bbox: tuple[float, float, float, float] | None = Field(
        default=None,
        description="Bounding box (x0, y0, x1, y1) in PDF points",
    )

    # Text/MD-specific (None for pdf)
    line_start: int | None = Field(default=None, description="1-indexed start line")
    line_end: int | None = Field(default=None, description="1-indexed end line (inclusive)")

    def to_uri(self) -> str:
        """Returns a URI the CLI/frontend can open directly."""
        if self.source_format == "pdf" and self.page is not None:
            return f"file://{self.source_path}#page={self.page}"
        if self.line_start is not None:
            return f"file://{self.source_path}#L{self.line_start}"
        return f"file://{self.source_path}"


# ---------------------------------------------------------------------------
# Section hierarchy
# ---------------------------------------------------------------------------

class HeadingNode(BaseModel):
    """One node in the section heading stack."""

    level: int = Field(ge=1, le=6, description="Heading depth: 1=H1 … 6=H6")
    title: str = Field(description="Heading text, stripped of markdown markers")
    anchor: str = Field(description="URL-safe slug, e.g. '3-2-thruster-subsystem'")


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    """
    A contiguous slice of a document.

    The `section_path` captures the full heading hierarchy from the document
    root to the immediate parent heading of this chunk.  LLM calls should use
    `to_llm_context()` so the model always sees the hierarchy.
    """

    id: str = Field(description="sha256(source_id + char_start + char_end)")
    text: str = Field(description="Raw chunk text (no heading prefix)")
    source_locator: SourceLocator

    # Character offsets within the normalized Markdown of the RawDocument
    char_start: int
    char_end: int
    token_count: int

    # Hierarchy snapshot at the moment this chunk was created
    section_path: list[HeadingNode] = Field(
        default_factory=list,
        description="Heading stack from root to immediate parent (ordered)",
    )

    @computed_field  # type: ignore[misc]
    @property
    def section_context(self) -> str:
        """Human-readable breadcrumb: '3. Propulsion > 3.2 Thruster Subsystem'"""
        if not self.section_path:
            return "(no section)"
        return " > ".join(h.title for h in self.section_path)

    def to_llm_context(self) -> str:
        """
        Text to inject into an LLM prompt.

        Prepends the full section breadcrumb so the model always has hierarchy
        context, even when the chunk itself doesn't repeat the heading.
        """
        return f"[Section: {self.section_context}]\n\n{self.text}"

    @staticmethod
    def make_id(source_id: str, char_start: int, char_end: int) -> str:
        raw = f"{source_id}:{char_start}:{char_end}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# RawDocument — intermediate product of any converter, before chunking
# ---------------------------------------------------------------------------

class PageMapEntry(BaseModel):
    """Maps a character range in the normalized Markdown to a PDF page."""

    char_start: int
    char_end: int
    page: int  # 1-indexed


class RawDocument(BaseModel):
    """
    The output of a format converter (PDF/TXT/MD → normalized Markdown).

    The chunker consumes this type — it never touches the original file.
    """

    id: str = Field(description="sha256(source_path + mtime at ingest)")
    source_path: str
    source_format: Literal["pdf", "txt", "md"]
    source_sha256: str = Field(description="sha256 of original file bytes")
    markdown: str = Field(description="Full normalized Markdown text")

    # Only populated for PDF; maps markdown char offsets → page numbers.
    # None for txt/md (use line numbers instead).
    page_map: list[PageMapEntry] | None = None

    created_at: str = Field(description="ISO 8601 timestamp")


# ---------------------------------------------------------------------------
# DocumentArtifact — final product after chunking
# ---------------------------------------------------------------------------

class DocumentArtifact(BaseModel):
    """
    A source document represented as an ordered list of hierarchy-aware chunks.

    This is the primary input to the extraction step.
    """

    id: str = Field(description="sha256(source_path + source_sha256)")
    raw_document_id: str
    source_path: str
    source_sha256: str
    chunks: list[Chunk]
    created_at: str = Field(description="ISO 8601 timestamp")

    def chunk_by_id(self, chunk_id: str) -> Chunk | None:
        for c in self.chunks:
            if c.id == chunk_id:
                return c
        return None
