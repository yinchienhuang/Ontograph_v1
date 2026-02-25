"""
tests/unit/test_ingest_chunker.py — Tests for ontograph/ingest/chunker.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from ontograph.ingest.chunker import (
    DEFAULT_MAX_TOKENS,
    _estimate_tokens,
    _slugify,
    _update_stack,
    chunk,
)
from ontograph.models.document import (
    HeadingNode,
    RawDocument,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_slugify_basic(self):
        assert _slugify("Thruster Module") == "thruster-module"

    def test_slugify_strips_special_chars(self):
        assert _slugify("3.2 Thruster!") == "32-thruster"

    def test_slugify_empty_returns_section(self):
        assert _slugify("") == "section"

    def test_estimate_tokens_positive(self):
        assert _estimate_tokens("hello world") >= 1

    def test_estimate_tokens_empty_is_one(self):
        assert _estimate_tokens("") == 1

    def test_estimate_tokens_scales_with_length(self):
        short = _estimate_tokens("x" * 40)
        long_ = _estimate_tokens("x" * 400)
        assert long_ > short

    def test_update_stack_push_deeper(self):
        h1 = HeadingNode(level=1, title="H1", anchor="h1")
        h2 = HeadingNode(level=2, title="H2", anchor="h2")
        stack = _update_stack([h1], h2)
        assert stack == [h1, h2]

    def test_update_stack_same_level_replaces(self):
        h2a = HeadingNode(level=2, title="A", anchor="a")
        h2b = HeadingNode(level=2, title="B", anchor="b")
        stack = _update_stack([h2a], h2b)
        assert stack == [h2b]

    def test_update_stack_shallower_pops_deeper(self):
        h1 = HeadingNode(level=1, title="H1", anchor="h1")
        h2 = HeadingNode(level=2, title="H2", anchor="h2")
        new_h1 = HeadingNode(level=1, title="H1b", anchor="h1b")
        stack = _update_stack([h1, h2], new_h1)
        assert stack == [new_h1]

    def test_update_stack_empty_start(self):
        h1 = HeadingNode(level=1, title="H1", anchor="h1")
        assert _update_stack([], h1) == [h1]


# ---------------------------------------------------------------------------
# Fixture: build a RawDocument from a markdown string
# ---------------------------------------------------------------------------

def _make_raw(markdown: str, source_format: str = "md") -> RawDocument:
    import hashlib
    sha = hashlib.sha256(markdown.encode()).hexdigest()
    return RawDocument(
        id="test-raw-id",
        source_path="/fake/path/doc." + source_format,
        source_format=source_format,  # type: ignore[arg-type]
        source_sha256=sha,
        markdown=markdown,
        page_map=None,
        created_at="2025-01-01T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# Chunker — structural tests
# ---------------------------------------------------------------------------

class TestChunkerStructure:
    def test_empty_document_produces_no_chunks(self):
        art = chunk(_make_raw(""))
        assert art.chunks == []

    def test_blank_only_document_produces_no_chunks(self):
        art = chunk(_make_raw("\n\n\n"))
        assert art.chunks == []

    def test_single_paragraph_produces_one_chunk(self):
        art = chunk(_make_raw("Some body text.\n"))
        assert len(art.chunks) == 1

    def test_chunk_text_is_stripped(self):
        art = chunk(_make_raw("\n\nSome text.\n\n"))
        assert art.chunks[0].text == "Some text."

    def test_heading_only_no_body_produces_no_chunks(self):
        art = chunk(_make_raw("# Introduction\n"))
        assert art.chunks == []

    def test_heading_followed_by_body(self):
        art = chunk(_make_raw("# Intro\n\nBody text.\n"))
        assert len(art.chunks) == 1

    def test_section_path_reflects_heading(self):
        art = chunk(_make_raw("# Intro\n\nBody text.\n"))
        assert len(art.chunks[0].section_path) == 1
        assert art.chunks[0].section_path[0].title == "Intro"
        assert art.chunks[0].section_path[0].level == 1

    def test_no_heading_means_empty_section_path(self):
        art = chunk(_make_raw("Just a body paragraph.\n"))
        assert art.chunks[0].section_path == []

    def test_two_sections_two_chunks(self):
        md = "# Section A\n\nBody A.\n\n# Section B\n\nBody B.\n"
        art = chunk(_make_raw(md))
        assert len(art.chunks) == 2

    def test_section_paths_are_independent(self):
        md = "# Section A\n\nBody A.\n\n# Section B\n\nBody B.\n"
        art = chunk(_make_raw(md))
        assert art.chunks[0].section_path[0].title == "Section A"
        assert art.chunks[1].section_path[0].title == "Section B"


# ---------------------------------------------------------------------------
# Heading stack depth
# ---------------------------------------------------------------------------

class TestHeadingStack:
    def test_h2_under_h1_stack_has_both(self):
        md = "# Top\n\n## Sub\n\nBody.\n"
        art = chunk(_make_raw(md))
        assert len(art.chunks) == 1
        path = art.chunks[0].section_path
        assert path[0].level == 1
        assert path[1].level == 2

    def test_new_h1_resets_stack(self):
        md = "# A\n\n## A.1\n\nBody A.1.\n\n# B\n\nBody B.\n"
        art = chunk(_make_raw(md))
        # chunk 0 = "Body A.1." under [H1(A), H2(A.1)]
        # chunk 1 = "Body B."  under [H1(B)]
        assert len(art.chunks) == 2
        assert len(art.chunks[0].section_path) == 2
        assert len(art.chunks[1].section_path) == 1
        assert art.chunks[1].section_path[0].title == "B"

    def test_sibling_h2_pops_previous_h2(self):
        md = "# Top\n\n## First\n\nBody 1.\n\n## Second\n\nBody 2.\n"
        art = chunk(_make_raw(md))
        assert art.chunks[0].section_path[-1].title == "First"
        assert art.chunks[1].section_path[-1].title == "Second"
        # Both should still have H1 in path
        assert art.chunks[0].section_path[0].level == 1
        assert art.chunks[1].section_path[0].level == 1

    def test_heading_anchor_is_slugified(self):
        md = "# Propulsion System!\n\nText.\n"
        art = chunk(_make_raw(md))
        assert art.chunks[0].section_path[0].anchor == "propulsion-system"


# ---------------------------------------------------------------------------
# Splitting behaviour
# ---------------------------------------------------------------------------

class TestChunkSplitting:
    def test_short_content_not_split(self):
        md = "# S\n\n" + "Short line.\n\n" * 3
        art = chunk(_make_raw(md), max_tokens=512)
        # All content fits in one chunk (far below 512 tokens)
        assert len(art.chunks) == 1

    def test_split_at_paragraph_boundary(self):
        # Create content that exceeds max_tokens across two paragraphs
        para = "word " * 30 + "\n\n"       # ~150 chars ≈ 37 tokens
        md = "# S\n\n" + para * 20          # ~740 tokens total
        art = chunk(_make_raw(md), max_tokens=100)
        assert len(art.chunks) > 1

    def test_chunks_cover_all_content(self):
        para = "word " * 20 + "\n\n"
        md = "# S\n\n" + para * 10
        art = chunk(_make_raw(md), max_tokens=50)
        all_text = " ".join(c.text for c in art.chunks)
        # Every "word" in the source should appear in some chunk
        assert "word" in all_text

    def test_force_split_very_long_paragraph(self):
        # Single paragraph > 2× max_tokens
        long_para = ("word " * 600).rstrip() + "\n"  # ~3000 chars ≈ 750 tokens
        md = "# S\n\n" + long_para
        art = chunk(_make_raw(md), max_tokens=100)
        assert len(art.chunks) > 1

    def test_split_chunks_share_section_path(self):
        """Chunks split from the same section retain the same heading path."""
        long_para = ("word " * 600).rstrip() + "\n"
        md = "# Long Section\n\n" + long_para
        art = chunk(_make_raw(md), max_tokens=100)
        for c in art.chunks:
            assert c.section_path[0].title == "Long Section"


# ---------------------------------------------------------------------------
# Chunk metadata
# ---------------------------------------------------------------------------

class TestChunkMetadata:
    def test_chunk_id_is_16_hex(self):
        art = chunk(_make_raw("# H\n\nText.\n"))
        assert len(art.chunks[0].id) == 16
        assert art.chunks[0].id.isalnum()

    def test_chunk_ids_are_unique(self):
        para = "word " * 20 + "\n\n"
        art = chunk(_make_raw("# S\n\n" + para * 10), max_tokens=50)
        ids = [c.id for c in art.chunks]
        assert len(ids) == len(set(ids))

    def test_char_offsets_non_overlapping(self):
        para = "word " * 20 + "\n\n"
        art = chunk(_make_raw("# S\n\n" + para * 10), max_tokens=50)
        for i in range(len(art.chunks) - 1):
            assert art.chunks[i].char_end <= art.chunks[i + 1].char_start

    def test_token_count_positive(self):
        art = chunk(_make_raw("# H\n\nSome text here.\n"))
        assert art.chunks[0].token_count >= 1

    def test_section_context_computed(self):
        art = chunk(_make_raw("# Intro\n\n## Sub\n\nText.\n"))
        ctx = art.chunks[0].section_context
        assert "Intro" in ctx
        assert "Sub" in ctx

    def test_to_llm_context_includes_section(self):
        art = chunk(_make_raw("# Intro\n\nText.\n"))
        llm = art.chunks[0].to_llm_context()
        assert "[Section:" in llm
        assert "Intro" in llm
        assert "Text." in llm


# ---------------------------------------------------------------------------
# Source locator — text/md
# ---------------------------------------------------------------------------

class TestSourceLocatorText:
    def test_txt_locator_has_line_numbers(self):
        art = chunk(_make_raw("# H\n\nBody text.\n", source_format="txt"))
        loc = art.chunks[0].source_locator
        assert loc.line_start is not None
        assert loc.line_end is not None
        assert loc.page is None

    def test_md_locator_has_line_numbers(self):
        art = chunk(_make_raw("# H\n\nBody.\n", source_format="md"))
        loc = art.chunks[0].source_locator
        assert loc.line_start is not None

    def test_locator_source_format_matches(self):
        art = chunk(_make_raw("Text.\n", source_format="txt"))
        assert art.chunks[0].source_locator.source_format == "txt"


# ---------------------------------------------------------------------------
# Source locator — PDF (with synthetic page_map)
# ---------------------------------------------------------------------------

class TestSourceLocatorPdf:
    def _make_pdf_raw(self, markdown: str) -> RawDocument:
        import hashlib
        from ontograph.models.document import PageMapEntry

        sha = hashlib.sha256(markdown.encode()).hexdigest()
        mid = len(markdown) // 2
        return RawDocument(
            id="pdf-raw",
            source_path="/fake/doc.pdf",
            source_format="pdf",
            source_sha256=sha,
            markdown=markdown,
            page_map=[
                PageMapEntry(char_start=0,   char_end=mid,          page=1),
                PageMapEntry(char_start=mid, char_end=len(markdown), page=2),
            ],
            created_at="2025-01-01T00:00:00+00:00",
        )

    def test_pdf_locator_has_page(self):
        md = "# H\n\nFirst chunk.\n"
        raw = self._make_pdf_raw(md)
        art = chunk(raw)
        loc = art.chunks[0].source_locator
        assert loc.page is not None
        assert loc.page in (1, 2)

    def test_pdf_locator_no_line_numbers(self):
        md = "# H\n\nFirst chunk.\n"
        raw = self._make_pdf_raw(md)
        art = chunk(raw)
        loc = art.chunks[0].source_locator
        assert loc.line_start is None
        assert loc.line_end is None

    def test_pdf_locator_source_format_is_pdf(self):
        md = "# H\n\nText.\n"
        raw = self._make_pdf_raw(md)
        art = chunk(raw)
        assert art.chunks[0].source_locator.source_format == "pdf"


# ---------------------------------------------------------------------------
# DocumentArtifact metadata
# ---------------------------------------------------------------------------

class TestArtifactMetadata:
    def test_artifact_id_16_hex(self):
        art = chunk(_make_raw("# H\n\nText.\n"))
        assert len(art.id) == 16

    def test_artifact_raw_document_id_matches(self):
        raw = _make_raw("# H\n\nText.\n")
        art = chunk(raw)
        assert art.raw_document_id == raw.id

    def test_artifact_chunk_by_id(self):
        art = chunk(_make_raw("# H\n\nText.\n"))
        c = art.chunks[0]
        assert art.chunk_by_id(c.id) is c

    def test_artifact_chunk_by_id_missing_returns_none(self):
        art = chunk(_make_raw("# H\n\nText.\n"))
        assert art.chunk_by_id("does-not-exist") is None
