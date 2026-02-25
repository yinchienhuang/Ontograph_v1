"""
tests/unit/test_ingest_converters.py — Tests for the three format converters
and the loader dispatch function.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ontograph.ingest.converters.markdown import convert_markdown
from ontograph.ingest.converters.text import _heading_level, convert_text
from ontograph.ingest.loader import load_document


# ---------------------------------------------------------------------------
# Markdown converter
# ---------------------------------------------------------------------------

class TestConvertMarkdown:
    def test_returns_raw_document(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_text("# Hello\n\nSome text.\n", encoding="utf-8")
        raw = convert_markdown(f)
        assert raw.source_format == "md"

    def test_source_sha256_matches_file(self, tmp_path: Path):
        content = "# Title\n\nBody.\n"
        f = tmp_path / "doc.md"
        f.write_bytes(content.encode())
        raw = convert_markdown(f)
        expected = hashlib.sha256(content.encode()).hexdigest()
        assert raw.source_sha256 == expected

    def test_page_map_is_none(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_text("# Title\n", encoding="utf-8")
        assert convert_markdown(f).page_map is None

    def test_markdown_content_preserved(self, tmp_path: Path):
        content = "# H1\n\n## H2\n\nParagraph.\n"
        f = tmp_path / "doc.md"
        f.write_text(content, encoding="utf-8")
        raw = convert_markdown(f)
        assert "# H1" in raw.markdown
        assert "## H2" in raw.markdown

    def test_excess_blank_lines_collapsed(self, tmp_path: Path):
        content = "# Title\n\n\n\n\nParagraph.\n"
        f = tmp_path / "doc.md"
        f.write_text(content, encoding="utf-8")
        raw = convert_markdown(f)
        assert "\n\n\n" not in raw.markdown

    def test_id_is_16_hex_chars(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_text("hello\n", encoding="utf-8")
        raw = convert_markdown(f)
        assert len(raw.id) == 16
        assert raw.id.isalnum()

    def test_source_path_is_absolute(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_text("x\n", encoding="utf-8")
        raw = convert_markdown(f)
        assert Path(raw.source_path).is_absolute()

    def test_same_content_same_id(self, tmp_path: Path):
        content = "# Title\n\nBody.\n"
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text(content, encoding="utf-8")
        f2.write_text(content, encoding="utf-8")
        # IDs differ (they include the path) but sha256s are the same
        r1 = convert_markdown(f1)
        r2 = convert_markdown(f2)
        assert r1.source_sha256 == r2.source_sha256
        assert r1.id != r2.id


# ---------------------------------------------------------------------------
# Text heading detection helper
# ---------------------------------------------------------------------------

class TestHeadingLevel:
    def test_numbered_h1(self):
        assert _heading_level("1 Introduction") == 1

    def test_numbered_h2(self):
        assert _heading_level("1.2 Background") == 2

    def test_numbered_h3(self):
        assert _heading_level("1.2.3 Details") == 3

    def test_numbered_depth_capped_at_3(self):
        assert _heading_level("1.2.3.4 Deep") == 3

    def test_all_caps_short(self):
        assert _heading_level("INTRODUCTION") == 2

    def test_all_caps_with_spaces(self):
        assert _heading_level("PROPULSION SYSTEM") == 2

    def test_body_sentence_is_none(self):
        assert _heading_level("This is a regular sentence.") is None

    def test_long_line_is_none(self):
        assert _heading_level("1 " + "A" * 90) is None

    def test_all_caps_with_terminal_period_is_none(self):
        assert _heading_level("SENTENCE.") is None

    def test_empty_line_is_none(self):
        assert _heading_level("") is None


# ---------------------------------------------------------------------------
# Text converter
# ---------------------------------------------------------------------------

class TestConvertText:
    def test_returns_raw_document(self, tmp_path: Path):
        f = tmp_path / "doc.txt"
        f.write_text("1 Introduction\nSome text.\n", encoding="utf-8")
        raw = convert_text(f)
        assert raw.source_format == "txt"

    def test_page_map_is_none(self, tmp_path: Path):
        f = tmp_path / "doc.txt"
        f.write_text("Hello.\n", encoding="utf-8")
        assert convert_text(f).page_map is None

    def test_numbered_section_becomes_h1(self, tmp_path: Path):
        f = tmp_path / "doc.txt"
        f.write_text("1 Introduction\nBody text.\n", encoding="utf-8")
        raw = convert_text(f)
        assert "# 1 Introduction" in raw.markdown

    def test_numbered_subsection_becomes_h2(self, tmp_path: Path):
        f = tmp_path / "doc.txt"
        f.write_text("1.2 Background\nBody text.\n", encoding="utf-8")
        raw = convert_text(f)
        assert "## 1.2 Background" in raw.markdown

    def test_all_caps_becomes_h2(self, tmp_path: Path):
        f = tmp_path / "doc.txt"
        f.write_text("PROPULSION\nBody.\n", encoding="utf-8")
        raw = convert_text(f)
        assert "## PROPULSION" in raw.markdown

    def test_body_lines_pass_through(self, tmp_path: Path):
        f = tmp_path / "doc.txt"
        f.write_text("Regular line.\nAnother line.\n", encoding="utf-8")
        raw = convert_text(f)
        assert "Regular line." in raw.markdown
        assert "#" not in raw.markdown

    def test_source_sha256_matches_file(self, tmp_path: Path):
        content = b"1 Intro\nText.\n"
        f = tmp_path / "doc.txt"
        f.write_bytes(content)
        raw = convert_text(f)
        assert raw.source_sha256 == hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------------
# PDF converter (smoke test — uses a PyMuPDF-generated test PDF)
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_pdf(tmp_path: Path) -> Path:
    """Create a minimal PDF with two pages for testing."""
    import fitz  # type: ignore[import]
    path = tmp_path / "test.pdf"
    doc = fitz.open()
    # Page 1
    p1 = doc.new_page()
    p1.insert_text((50, 72), "1 Introduction", fontsize=18)
    p1.insert_text((50, 100), "This is introductory text about propulsion.", fontsize=11)
    # Page 2
    p2 = doc.new_page()
    p2.insert_text((50, 72), "2 Propulsion", fontsize=18)
    p2.insert_text((50, 100), "This section covers the thruster.", fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


class TestConvertPdf:
    def test_returns_pdf_raw_document(self, tiny_pdf: Path):
        from ontograph.ingest.converters.pdf import convert_pdf
        raw = convert_pdf(tiny_pdf)
        assert raw.source_format == "pdf"

    def test_page_map_has_two_entries(self, tiny_pdf: Path):
        from ontograph.ingest.converters.pdf import convert_pdf
        raw = convert_pdf(tiny_pdf)
        assert raw.page_map is not None
        assert len(raw.page_map) == 2

    def test_page_map_pages_are_sequential(self, tiny_pdf: Path):
        from ontograph.ingest.converters.pdf import convert_pdf
        raw = convert_pdf(tiny_pdf)
        pages = [e.page for e in raw.page_map]  # type: ignore[union-attr]
        assert pages == [1, 2]

    def test_page_map_ranges_are_contiguous(self, tiny_pdf: Path):
        from ontograph.ingest.converters.pdf import convert_pdf
        raw = convert_pdf(tiny_pdf)
        pm = raw.page_map  # type: ignore[union-attr]
        assert pm[0].char_start == 0
        assert pm[0].char_end == pm[1].char_start

    def test_markdown_contains_text(self, tiny_pdf: Path):
        from ontograph.ingest.converters.pdf import convert_pdf
        raw = convert_pdf(tiny_pdf)
        assert "Introduction" in raw.markdown or "introduction" in raw.markdown.lower()

    def test_source_sha256_stable(self, tiny_pdf: Path):
        from ontograph.ingest.converters.pdf import convert_pdf
        r1 = convert_pdf(tiny_pdf)
        r2 = convert_pdf(tiny_pdf)
        assert r1.source_sha256 == r2.source_sha256

    def test_id_is_deterministic(self, tiny_pdf: Path):
        from ontograph.ingest.converters.pdf import convert_pdf
        r1 = convert_pdf(tiny_pdf)
        r2 = convert_pdf(tiny_pdf)
        assert r1.id == r2.id


# ---------------------------------------------------------------------------
# Loader dispatch
# ---------------------------------------------------------------------------

class TestLoadDocument:
    def test_md_extension(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_text("# Title\n", encoding="utf-8")
        raw = load_document(f)
        assert raw.source_format == "md"

    def test_markdown_extension(self, tmp_path: Path):
        f = tmp_path / "doc.markdown"
        f.write_text("# Title\n", encoding="utf-8")
        raw = load_document(f)
        assert raw.source_format == "md"

    def test_txt_extension(self, tmp_path: Path):
        f = tmp_path / "doc.txt"
        f.write_text("Hello.\n", encoding="utf-8")
        raw = load_document(f)
        assert raw.source_format == "txt"

    def test_text_extension(self, tmp_path: Path):
        f = tmp_path / "doc.text"
        f.write_text("Hello.\n", encoding="utf-8")
        raw = load_document(f)
        assert raw.source_format == "txt"

    def test_pdf_extension(self, tiny_pdf: Path):
        raw = load_document(tiny_pdf)
        assert raw.source_format == "pdf"

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_document(tmp_path / "nonexistent.md")

    def test_unsupported_extension_raises(self, tmp_path: Path):
        f = tmp_path / "doc.docx"
        f.write_bytes(b"fake docx")
        with pytest.raises(ValueError, match="Unsupported"):
            load_document(f)

    def test_case_insensitive_extension(self, tmp_path: Path):
        f = tmp_path / "doc.MD"
        f.write_text("# Title\n", encoding="utf-8")
        raw = load_document(f)
        assert raw.source_format == "md"
