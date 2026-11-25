"""Unit tests for text extraction and cleaning, including all document error cases."""

from pathlib import Path

import pytest
from docx import Document
from pypdf import PdfWriter

from doc_indexer.errors import (
    DocumentError,
    DocumentNotFoundError,
    EmptyDocumentError,
    UnsupportedFileTypeError,
)
from doc_indexer.extraction import clean_text, extract_text

EXAMPLES = Path(__file__).resolve().parent.parent / "docs"


def test_clean_text_normalizes_whitespace_and_keeps_paragraphs():
    raw = "Hello   world,\nsoft wrapped line.\r\n\r\n\r\nSecond\tparagraph.\x00"
    assert clean_text(raw) == "Hello world, soft wrapped line.\n\nSecond paragraph."


def test_clean_text_rejoins_hyphenated_line_breaks():
    assert clean_text("inter-\nnational") == "international"


def test_extract_docx_paragraphs(tmp_path):
    path = tmp_path / "doc.docx"
    document = Document()
    document.add_paragraph("First paragraph.")
    document.add_paragraph("Second paragraph.")
    document.save(path)
    assert extract_text(path) == "First paragraph.\n\nSecond paragraph."


def test_extract_example_files_keep_paragraphs():
    for name in ("example.pdf", "example.docx"):
        text = extract_text(EXAMPLES / name)
        assert "Login issues" in text
        assert text.count("\n\n") >= 5


def test_missing_file(tmp_path):
    with pytest.raises(DocumentNotFoundError):
        extract_text(tmp_path / "nope.pdf")


def test_accepts_str_path():
    assert "Login issues" in extract_text(str(EXAMPLES / "example.docx"))


@pytest.mark.parametrize("bad_path", [None, 123, 4.5])
def test_non_path_input_is_domain_error(bad_path):
    with pytest.raises(DocumentError):
        extract_text(bad_path)


def test_empty_string_path_is_not_found():
    with pytest.raises(DocumentNotFoundError):
        extract_text("")


def test_unsupported_type(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(UnsupportedFileTypeError):
        extract_text(path)


def test_pdf_without_text(tmp_path):
    path = tmp_path / "scan.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with path.open("wb") as f:
        writer.write(f)
    with pytest.raises(EmptyDocumentError):
        extract_text(path)


def test_empty_docx(tmp_path):
    path = tmp_path / "empty.docx"
    Document().save(path)
    with pytest.raises(EmptyDocumentError):
        extract_text(path)


@pytest.mark.parametrize("name", ["bad.pdf", "bad.docx"])
def test_corrupted_file(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"this is not a real document")
    with pytest.raises(DocumentError):
        extract_text(path)
