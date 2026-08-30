"""
Tests for document parsers.

Covers text/markdown parsing, auto-detection by extension,
and error handling for missing files and missing dependencies.
PDF and DOCX tests are skipped if the dependencies aren't
installed since they're optional.
"""

import pytest
from pathlib import Path

from simplev.parsers import (
    parse_text, parse_pdf, parse_docx, auto_parse, ParserError
)


class TestTextParser:
    """Test plain text parsing."""

    def test_read_txt_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        text = parse_text(f)
        assert text == "hello world"

    def test_read_markdown_file(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# Title\n\nSome content.", encoding="utf-8")
        text = parse_text(f)
        assert "# Title" in text
        assert "Some content." in text

    def test_read_multiline(self, tmp_path):
        content = "line one\nline two\nline three"
        f = tmp_path / "multi.txt"
        f.write_text(content, encoding="utf-8")
        text = parse_text(f)
        assert text == content

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ParserError, match="not found"):
            parse_text(tmp_path / "nope.txt")

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        text = parse_text(f)
        assert text == ""

    def test_unicode_content(self, tmp_path):
        content = "Hello Mundo Welt"
        f = tmp_path / "unicode.txt"
        f.write_text(content, encoding="utf-8")
        text = parse_text(f)
        assert text == content


class TestAutoParser:
    """Test auto-detection by file extension."""

    def test_auto_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content", encoding="utf-8")
        text = auto_parse(f)
        assert text == "content"

    def test_auto_md(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Heading", encoding="utf-8")
        text = auto_parse(f)
        assert "Heading" in text

    def test_unsupported_extension(self, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_text("stuff", encoding="utf-8")
        with pytest.raises(ParserError, match="Unsupported"):
            auto_parse(f)


class TestPDFParser:
    """Test PDF parsing -- skipped if pymupdf is not installed."""

    @pytest.fixture
    def has_pymupdf(self):
        try:
            import fitz
            return True
        except ImportError:
            pytest.skip("pymupdf not installed")

    def test_missing_file_raises(self, has_pymupdf, tmp_path):
        with pytest.raises(ParserError, match="not found"):
            parse_pdf(tmp_path / "nope.pdf")

    def test_import_error_message(self, tmp_path):
        """If pymupdf is missing, error message should say how to install."""
        try:
            import fitz
            pytest.skip("pymupdf is installed, can't test import error")
        except ImportError:
            f = tmp_path / "test.pdf"
            f.write_bytes(b"%PDF-1.4 fake")
            with pytest.raises(ParserError, match="pymupdf"):
                parse_pdf(f)


class TestDOCXParser:
    """Test DOCX parsing -- skipped if python-docx is not installed."""

    @pytest.fixture
    def has_docx(self):
        try:
            from docx import Document
            return True
        except ImportError:
            pytest.skip("python-docx not installed")

    def test_missing_file_raises(self, has_docx, tmp_path):
        with pytest.raises(ParserError, match="not found"):
            parse_docx(tmp_path / "nope.docx")

    def test_import_error_message(self, tmp_path):
        """If python-docx is missing, error message should say how to install."""
        try:
            from docx import Document
            pytest.skip("python-docx is installed, can't test import error")
        except ImportError:
            f = tmp_path / "test.docx"
            f.write_bytes(b"fake docx")
            with pytest.raises(ParserError, match="python-docx"):
                parse_docx(f)
