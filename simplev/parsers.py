"""
Document parsers for SimpleV.

Extracts plain text from common file formats so users can
feed documents directly into the database without manually
reading and converting them first.

Supported formats:
  - .txt and .md (plain text, read as-is)
  - .pdf (via pymupdf/fitz)
  - .docx (via python-docx)

Each parser is a standalone function that takes a file path
and returns the extracted text as a string. We keep them
simple and independent -- if a user doesn't have pymupdf
installed, the txt parser still works fine.
"""

import logging
from pathlib import Path
from typing import Union

from simplev.exceptions import SimpleVError


logger = logging.getLogger(__name__)


class ParserError(SimpleVError):
    """Something went wrong while parsing a document."""
    pass


def parse_text(path: Union[str, Path]) -> str:
    """Read a plain text or markdown file.

    Works for .txt, .md, .csv, or really any utf-8 text file.
    Nothing fancy here, just reads the whole thing.

    Args:
        path: Path to the text file.

    Returns:
        The file contents as a string.
    """
    path = Path(path)
    if not path.exists():
        raise ParserError(f"File not found: {path}")

    try:
        text = path.read_text(encoding="utf-8")
        logger.debug(f"Parsed text file: {path} ({len(text)} chars)")
        return text
    except UnicodeDecodeError:
        # try latin-1 as a fallback for older files
        try:
            text = path.read_text(encoding="latin-1")
            return text
        except Exception as e:
            raise ParserError(f"Could not read {path}: {e}") from e
    except Exception as e:
        raise ParserError(f"Failed to read {path}: {e}") from e


def parse_pdf(path: Union[str, Path]) -> str:
    """Extract text from a PDF file.

    Uses pymupdf (imported as fitz) under the hood. You need
    to install it separately: pip install pymupdf

    Args:
        path: Path to the PDF file.

    Returns:
        All text from all pages concatenated together.
    """
    path = Path(path)
    if not path.exists():
        raise ParserError(f"File not found: {path}")

    try:
        import fitz  # pymupdf
    except ImportError:
        raise ParserError(
            "pymupdf is required for PDF parsing. "
            "Install it with: pip install pymupdf"
        )

    try:
        doc = fitz.open(str(path))
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                pages.append(text)

        doc.close()

        full_text = "\n\n".join(pages)
        logger.debug(
            f"Parsed PDF: {path} ({len(doc)} pages, {len(full_text)} chars)"
        )
        return full_text

    except ParserError:
        raise
    except Exception as e:
        raise ParserError(f"Failed to parse PDF '{path}': {e}") from e


def parse_docx(path: Union[str, Path]) -> str:
    """Extract text from a DOCX file.

    Uses python-docx under the hood. Install it with:
    pip install python-docx

    Args:
        path: Path to the DOCX file.

    Returns:
        All paragraphs concatenated with newlines.
    """
    path = Path(path)
    if not path.exists():
        raise ParserError(f"File not found: {path}")

    try:
        from docx import Document
    except ImportError:
        raise ParserError(
            "python-docx is required for DOCX parsing. "
            "Install it with: pip install python-docx"
        )

    try:
        doc = Document(str(path))
        paragraphs = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)

        full_text = "\n".join(paragraphs)
        logger.debug(
            f"Parsed DOCX: {path} ({len(paragraphs)} paragraphs, "
            f"{len(full_text)} chars)"
        )
        return full_text

    except ParserError:
        raise
    except Exception as e:
        raise ParserError(f"Failed to parse DOCX '{path}': {e}") from e


def auto_parse(path: Union[str, Path]) -> str:
    """Automatically pick the right parser based on file extension.

    Convenience function so the user doesn't have to think about
    which parser to call. Just pass any supported file and we
    figure it out.

    Args:
        path: Path to any supported document.

    Returns:
        Extracted text content.

    Raises:
        ParserError: If the format is not supported.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    parsers = {
        ".txt": parse_text,
        ".md": parse_text,
        ".markdown": parse_text,
        ".csv": parse_text,
        ".pdf": parse_pdf,
        ".docx": parse_docx,
    }

    if suffix not in parsers:
        raise ParserError(
            f"Unsupported file format: '{suffix}'. "
            f"Supported: {list(parsers.keys())}"
        )

    return parsers[suffix](path)
