"""Tests for the two CLI entry points: exit codes, messages, output format and secret hygiene."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

import index_documents
import search
from doc_indexer.db import SearchResult
from doc_indexer.errors import DatabaseError, EmbeddingError

EXAMPLE_PDF = str(Path(__file__).resolve().parent.parent / "docs" / "example.pdf")
SECRETS = ("SECRET-KEY-123", "SECRET-PW", "postgresql://")


class FakeEmbedder:
    def __init__(self, api_key):
        pass

    def embed_documents(self, texts):
        return [[0.1] for _ in texts]

    def embed_query(self, text):
        return [0.1]


class FakeRepository:
    """Context-manager stand-in for VectorRepository; behaviour is set per test via class attributes."""

    connect_error: Exception | None = None
    results: list = []

    def __init__(self, url):
        pass

    def __enter__(self):
        if self.connect_error:
            raise self.connect_error
        return self

    def __exit__(self, *exc):
        return None

    def ensure_schema(self):
        pass

    def replace_chunks(self, filename, strategy, chunks, vectors):
        return len(chunks)

    def search(self, query_vector, top_k, strategy):
        return self.results


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    # A real .env in the project must not leak into these tests.
    monkeypatch.setattr("doc_indexer.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET-KEY-123")
    monkeypatch.setenv("POSTGRES_URL", "postgresql://user:SECRET-PW@host/db")
    FakeRepository.connect_error = None
    FakeRepository.results = []
    for module in (index_documents, search):
        monkeypatch.setattr(module, "GeminiEmbedder", FakeEmbedder)
        monkeypatch.setattr(module, "VectorRepository", FakeRepository)


def _assert_no_secrets(captured):
    for secret in SECRETS:
        assert secret not in captured.out and secret not in captured.err


def _result(text, rank_score=0.9, index=0):
    return SearchResult(
        id=1, filename="example.pdf", split_strategy="paragraph",
        chunk_index=index, chunk_text=text, score=rank_score,
        created_at=datetime.now(timezone.utc),
    )


# --- index_documents.py ---------------------------------------------------------------

def test_index_success_prints_summary(capsys):
    assert index_documents.main(["--file", EXAMPLE_PDF, "--strategy", "paragraph"]) == 0
    captured = capsys.readouterr()
    assert "Indexed 'example.pdf' with strategy 'paragraph'" in captured.out
    _assert_no_secrets(captured)


def test_index_missing_file(capsys, tmp_path):
    assert index_documents.main(["--file", str(tmp_path / "missing.pdf")]) == 3
    assert "File not found" in capsys.readouterr().err


def test_index_unsupported_file(capsys, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    assert index_documents.main(["--file", str(path)]) == 3
    assert "Unsupported file type" in capsys.readouterr().err


def test_index_missing_configuration(capsys, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    assert index_documents.main(["--file", EXAMPLE_PDF]) == 2
    captured = capsys.readouterr()
    assert "GEMINI_API_KEY" in captured.err
    _assert_no_secrets(captured)


def test_index_embedding_failure(capsys, monkeypatch):
    def fail(self, texts):
        raise EmbeddingError("Gemini rejected the request (HTTP 401).")

    monkeypatch.setattr(FakeEmbedder, "embed_documents", fail)
    assert index_documents.main(["--file", EXAMPLE_PDF]) == 4
    assert "Gemini rejected" in capsys.readouterr().err


def test_index_database_failure(capsys):
    FakeRepository.connect_error = DatabaseError("Could not connect to PostgreSQL. Check POSTGRES_URL.")
    assert index_documents.main(["--file", EXAMPLE_PDF]) == 5
    captured = capsys.readouterr()
    assert "POSTGRES_URL" in captured.err
    _assert_no_secrets(captured)


def test_unexpected_error_hides_message_and_traceback(capsys, monkeypatch):
    def boom(self, texts):
        raise RuntimeError("connection string postgresql://user:SECRET-PW@host/db")

    monkeypatch.setattr(FakeEmbedder, "embed_documents", boom)
    assert index_documents.main(["--file", EXAMPLE_PDF, "--verbose"]) == 1
    captured = capsys.readouterr()
    assert "RuntimeError" in captured.err
    assert "Traceback" not in captured.err
    _assert_no_secrets(captured)


# --- search.py ------------------------------------------------------------------------

def test_search_with_no_results_prints_friendly_message(capsys):
    assert search.main(["--query", "login issue"]) == 0
    assert "No results found" in capsys.readouterr().out


def test_search_prints_all_fields_and_full_untruncated_text(capsys):
    long_text = "First paragraph. " * 80 + "\n\nSecond paragraph ends here."
    FakeRepository.results = [_result(long_text, 0.9123, index=4), _result("Other chunk", 0.5, index=7)]
    assert search.main(["--query", "login issue"]) == 0
    out = capsys.readouterr().out
    for expected in (
        "Rank:      1", "Rank:      2", "Score:     0.9123", "File:      example.pdf",
        "Strategy:  paragraph", "Chunk:     #4", "Chunk:     #7",
    ):
        assert expected in out
    assert len(long_text) > 1000
    assert "Second paragraph ends here." in out  # nothing cut off
    assert out.count("First paragraph.") == 80


def test_search_empty_query_is_rejected(capsys):
    assert search.main(["--query", "   "]) == 2
    assert "must not be empty" in capsys.readouterr().err


def test_search_database_failure(capsys):
    FakeRepository.connect_error = DatabaseError("Could not connect to PostgreSQL. Check POSTGRES_URL.")
    assert search.main(["--query", "x"]) == 5
    _assert_no_secrets(capsys.readouterr())
