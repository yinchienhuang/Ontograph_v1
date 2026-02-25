"""
ingest/chunker.py — Split a RawDocument into hierarchy-aware Chunks.

Algorithm
---------
Lines are scanned in order. A heading-stack is maintained; when a heading
is detected, the stack is updated (shallower levels pop deeper ones) and
the accumulated text buffer is flushed.

Body text accumulates in a buffer. The buffer is flushed at a paragraph
boundary (blank line) once it exceeds ``max_tokens``. If a single paragraph
exceeds 2× ``max_tokens`` it is force-split at a word boundary.

Each flushed Chunk receives a *snapshot* of the heading stack at the moment
of flushing, giving every chunk its full section breadcrumb.

Token count is approximated as ``len(text) // 4`` (≈ 4 chars / token for
English prose). This avoids a tokenizer dependency while being accurate
enough for chunking decisions.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

from ontograph.models.document import (
    Chunk,
    DocumentArtifact,
    HeadingNode,
    PageMapEntry,
    RawDocument,
    SourceLocator,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_TOKENS: int = 512

_HEADING_RE   = re.compile(r"^(#{1,6})\s+(.+)$")
_SLUG_NONWORD = re.compile(r"[^\w\s-]")
_SLUG_SPACES  = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Convert heading text to a URL-safe anchor slug."""
    s = _SLUG_NONWORD.sub("", text.lower())
    s = _SLUG_SPACES.sub("-", s).strip("-")
    return s or "section"


def _estimate_tokens(text: str) -> int:
    """Approximate token count — 4 characters ≈ 1 token (English prose)."""
    return max(1, len(text) // 4)


def _update_stack(
    stack: list[HeadingNode], new_node: HeadingNode
) -> list[HeadingNode]:
    """
    Return a new heading stack with *new_node* pushed at the correct depth.

    All existing nodes at the same level or deeper are discarded so that
    sibling and sub-heading changes are handled correctly.
    """
    return [h for h in stack if h.level < new_node.level] + [new_node]


def _page_for_char(
    page_map: list[PageMapEntry] | None, char_pos: int
) -> int | None:
    """Return the 1-indexed page that contains *char_pos*, or None."""
    if not page_map:
        return None
    for entry in page_map:
        if entry.char_start <= char_pos < entry.char_end:
            return entry.page
    return page_map[-1].page  # fall back to last page


def _line_range(
    markdown: str, char_start: int, char_end: int
) -> tuple[int, int]:
    """Return 1-indexed (line_start, line_end) for *[char_start, char_end)*."""
    before = markdown[:char_start]
    line_start = before.count("\n") + 1
    span = markdown[char_start:char_end]
    line_end = line_start + span.count("\n")
    return line_start, max(line_start, line_end)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk(
    raw: RawDocument,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> DocumentArtifact:
    """
    Split *raw* into a :class:`~ontograph.models.document.DocumentArtifact`.

    Args:
        raw:        Output of any format converter.
        max_tokens: Target (soft) maximum tokens per chunk.  The buffer is
                    flushed at the next paragraph boundary once this is
                    exceeded.  Very long paragraphs (>2× max_tokens) are
                    force-split at a word boundary.

    Returns:
        A :class:`~ontograph.models.document.DocumentArtifact` whose
        ``chunks`` list preserves document order.
    """
    markdown   = raw.markdown
    sha        = raw.source_sha256
    page_map   = raw.page_map
    fmt        = raw.source_format

    # Stable artifact ID = sha256(source_path:source_sha256)
    artifact_id = hashlib.sha256(
        f"{raw.source_path}:{sha}".encode()
    ).hexdigest()[:16]

    # ── State ────────────────────────────────────────────────────────────────
    heading_stack: list[HeadingNode] = []
    buf_lines:  list[str] = []
    buf_start:  int = 0          # absolute char offset of buffer start
    chunks:     list[Chunk] = []

    # ── Helpers that close over state ────────────────────────────────────────

    def _make_locator(char_start: int, char_end: int) -> SourceLocator:
        if fmt == "pdf":
            return SourceLocator(
                source_id=sha,
                source_path=raw.source_path,
                source_format="pdf",
                page=_page_for_char(page_map, char_start),
            )
        ls, le = _line_range(markdown, char_start, char_end)
        return SourceLocator(
            source_id=sha,
            source_path=raw.source_path,
            source_format=fmt,  # type: ignore[arg-type]
            line_start=ls,
            line_end=le,
        )

    def _emit(text: str, char_start: int, char_end: int) -> None:
        """Validate, build, and append one Chunk."""
        text = text.strip()
        if not text:
            return
        chunks.append(Chunk(
            id=Chunk.make_id(sha, char_start, char_end),
            text=text,
            source_locator=_make_locator(char_start, char_end),
            char_start=char_start,
            char_end=char_end,
            token_count=_estimate_tokens(text),
            section_path=list(heading_stack),   # snapshot — independent copy
        ))

    def _flush(up_to_pos: int) -> None:
        """Emit the current buffer and reset it."""
        nonlocal buf_lines, buf_start
        if buf_lines:
            _emit("".join(buf_lines), buf_start, up_to_pos)
            buf_lines = []
            buf_start = up_to_pos

    # ── Main scan ────────────────────────────────────────────────────────────
    pos = 0
    for raw_line in markdown.splitlines(keepends=True):
        line_end = pos + len(raw_line)
        stripped = raw_line.rstrip("\r\n")

        m = _HEADING_RE.match(stripped)

        if m:
            # ── Heading line ────────────────────────────────────────────────
            _flush(pos)                          # flush body before heading
            level = len(m.group(1))
            title = m.group(2).strip()
            new_node = HeadingNode(
                level=level, title=title, anchor=_slugify(title)
            )
            heading_stack = _update_stack(heading_stack, new_node)
            buf_start = line_end                 # next body starts after heading

        elif stripped == "":
            # ── Blank line (paragraph boundary) ────────────────────────────
            buf_lines.append(raw_line)
            if _estimate_tokens("".join(buf_lines)) >= max_tokens:
                _flush(line_end)

        else:
            # ── Body text ───────────────────────────────────────────────────
            if not buf_lines:
                buf_start = pos               # buffer starts here
            buf_lines.append(raw_line)

            # Force-split an oversized paragraph before the next blank line
            current_text = "".join(buf_lines)
            if _estimate_tokens(current_text) >= max_tokens * 2:
                # Cut near the max_tokens boundary at a word boundary
                target_chars = max_tokens * 4
                cut = current_text.rfind(" ", 0, target_chars)
                if cut <= 0:
                    cut = target_chars
                part1 = current_text[:cut]
                part2 = current_text[cut:].lstrip()
                _emit(part1, buf_start, buf_start + cut)
                buf_start = buf_start + cut
                buf_lines = [part2] if part2 else []

        pos = line_end

    # Flush any remaining text
    _flush(pos)

    return DocumentArtifact(
        id=artifact_id,
        raw_document_id=raw.id,
        source_path=raw.source_path,
        source_sha256=sha,
        chunks=chunks,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
