"""Unit tests for the indexing and search services using recording fakes (no network, no DB)."""

from pathlib import Path

import pytest
from docx import Document

from doc_indexer.errors import (
    ConfigError,
    DatabaseError,
    DocumentNotFoundError,
    EmbeddingError,
    EmptyDocumentError,
)
from doc_indexer.services.indexing_service import IndexingService
from doc_indexer.services.search_service import SearchService

EXAMPLE_DOCX = Path(__file__).resolve().parent.parent / "docs" / "example.docx"


class FakeEmbedder:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        if self.fail:
            raise EmbeddingError("embedding failed")
        return [[float(i)] for i, _ in enumerate(texts)]

    def embed_query(self, text):
        self.query_calls.append(text)
        return [0.5]


class FakeRepository:
    def __init__(self, schema_error: Exception | None = None, results=None) -> None:
        self.schema_error = schema_error
        self.results = results or []
        self.calls: list[str] = []
        self.stored: tuple | None = None
        self.search_args: tuple | None = None

    def ensure_schema(self):
        self.calls.append("ensure_schema")
        if self.schema_error:
            raise self.schema_error

    def replace_chunks(self, filename, strategy, chunks, vectors):
        self.calls.append("replace_chunks")
        self.stored = (filename, strategy, chunks, vectors)
        return len(chunks)

    def search(self, query_vector, top_k, strategy):
        self.calls.append("search")
        self.search_args = (query_vector, top_k, strategy)
        return self.results


def _index(embedder, repository, path=EXAMPLE_DOCX, strategy="paragraph", **kwargs):
    return IndexingService(embedder, repository).index_file(path, strategy, **kwargs)


def test_indexing_happy_path_stores_one_vector_per_chunk():
    embedder, repository = FakeEmbedder(), FakeRepository()
    result = _index(embedder, repository)
    filename, strategy, chunks, vectors = repository.stored
    assert (filename, strategy) == ("example.docx", "paragraph")
    assert len(chunks) == len(vectors) == result.chunk_count > 0
    assert embedder.document_calls == [chunks]
    assert repository.calls == ["ensure_schema", "replace_chunks"]


def test_filename_is_stored_without_directories():
    repository = FakeRepository()
    _index(FakeEmbedder(), repository, path=str(EXAMPLE_DOCX))
    assert repository.stored[0] == "example.docx"


def test_missing_file_touches_neither_api_nor_database(tmp_path):
    embedder, repository = FakeEmbedder(), FakeRepository()
    with pytest.raises(DocumentNotFoundError):
        _index(embedder, repository, path=tmp_path / "nope.pdf")
    assert embedder.document_calls == [] and repository.calls == []


def test_document_without_text_touches_neither_api_nor_database(tmp_path):
    empty = tmp_path / "empty.docx"
    Document().save(empty)
    embedder, repository = FakeEmbedder(), FakeRepository()
    with pytest.raises(EmptyDocumentError):
        _index(embedder, repository, path=empty)
    assert embedder.document_calls == [] and repository.calls == []


def test_invalid_chunk_parameters_touch_neither_api_nor_database():
    embedder, repository = FakeEmbedder(), FakeRepository()
    with pytest.raises(ConfigError):
        _index(embedder, repository, strategy="fixed", chunk_size=100, overlap=100)
    assert embedder.document_calls == [] and repository.calls == []


def test_schema_failure_happens_before_paying_for_embeddings():
    embedder = FakeEmbedder()
    repository = FakeRepository(schema_error=DatabaseError("no permission"))
    with pytest.raises(DatabaseError):
        _index(embedder, repository)
    assert embedder.document_calls == []


def test_embedding_failure_writes_nothing_to_the_database():
    repository = FakeRepository()
    with pytest.raises(EmbeddingError):
        _index(FakeEmbedder(fail=True), repository)
    assert "replace_chunks" not in repository.calls


def test_search_embeds_query_and_passes_arguments_through():
    embedder, repository = FakeEmbedder(), FakeRepository(results=["r1", "r2"])
    results = SearchService(embedder, repository).search("  login issue  ", top_k=3, strategy="fixed")
    assert results == ["r1", "r2"]
    assert embedder.query_calls == ["login issue"]
    assert repository.search_args == ([0.5], 3, "fixed")


def test_search_with_no_matches_returns_empty_list():
    assert SearchService(FakeEmbedder(), FakeRepository(results=[])).search("anything") == []


@pytest.mark.parametrize(
    "query, top_k, strategy",
    [("   ", 5, None), ("ok", 0, None), ("ok", 5, "semantic")],
)
def test_invalid_search_input_is_rejected_before_embedding(query, top_k, strategy):
    embedder, repository = FakeEmbedder(), FakeRepository()
    with pytest.raises(ConfigError):
        SearchService(embedder, repository).search(query, top_k, strategy)
    assert embedder.query_calls == [] and repository.calls == []
