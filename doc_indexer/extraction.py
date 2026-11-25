"""Extract clean plain text from PDF and DOCX files."""

import re
import unicodedata
from pathlib import Path

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from doc_indexer.errors import (
    DocumentError,
    DocumentNotFoundError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)

SUPPORTED_EXTENSIONS = (".pdf", ".docx")

_CONTROL_CHARS = re.compile(r"[^\S\n]|[\x00-\x08\x0b-\x1f\x7f]")
_HYPHEN_LINEBREAK = re.compile(r"(?<=[^\W\d_])-\n(?=[^\W\d_])")
_BLANK_LINES = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)*")


def extract_text(path: str | Path) -> str:
    """Return cleaned text of a PDF/DOCX file given as a str or Path.

    Paragraphs in the result are separated by a blank line ("\\n\\n"); line
    breaks inside a paragraph are collapsed to single spaces.
    Raises a DocumentError subclass for invalid, missing, unsupported,
    unreadable or text-less files, before any expensive work (embeddings, DB)
    happens.
    """
    try:
        path = Path(path)
    except TypeError as exc:
        raise DocumentError("Invalid file path: expected a string or a path object.") from exc

    if not path.is_file():
        raise DocumentNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    raw = _read_pdf(path) if suffix == ".pdf" else _read_docx(path)
    text = clean_text(raw)
    if not text:
        raise EmptyDocumentError(
            "No extractable text found. The file may be empty or a scanned "
            "document (images only), which would require OCR."
        )
    return text


def clean_text(raw: str) -> str:
    """Normalize raw extracted text while preserving paragraph boundaries."""
    text = unicodedata.normalize("NFKC", raw).replace("\r\n", "\n").replace("\r", "\n")
    # Rejoin words that the PDF layout hyphenated across a line break.
    text = _HYPHEN_LINEBREAK.sub("", text)
    text = _CONTROL_CHARS.sub(" ", text)

    paragraphs = []
    for block in _BLANK_LINES.split(text):
        # A single newline is a soft wrap, not a paragraph boundary.
        paragraph = " ".join(block.split())
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def _read_pdf(path: Path) -> str:
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise DocumentError("The PDF is password-protected and cannot be read.")
        # Default mode drops vertical gaps, so every paragraph would merge into one;
        # "layout" keeps them as blank lines (its space padding is removed by clean_text).
        return "\n\n".join(
            page.extract_text(extraction_mode="layout") or "" for page in reader.pages
        )
    except DocumentError:
        raise
    except (PyPdfError, OSError, ValueError) as exc:
        raise DocumentError("The PDF file is corrupted or could not be read.") from exc


def _read_docx(path: Path) -> str:
    # Only body paragraphs are read; text inside tables is not extracted.
    try:
        document = Document(str(path))
    except (PackageNotFoundError, KeyError, ValueError, OSError) as exc:
        raise DocumentError("The DOCX file is corrupted or could not be read.") from exc
    return "\n\n".join(p.text for p in document.paragraphs)
