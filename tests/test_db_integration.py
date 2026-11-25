"""Integration tests for VectorRepository against a real PostgreSQL with pgvector.

Skipped unless TEST_POSTGRES_URL is set. Only rows whose filename starts with
'itest_' are created and removed, so an existing database is left intact.
"""

import os
import uuid

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from doc_indexer.db import VectorRepository
from doc_indexer.embeddings import EMBEDDING_DIM
from doc_indexer.errors import DatabaseError

TEST_URL = os.getenv("TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not TEST_URL, reason="TEST_POSTGRES_URL is not set")

PREFIX = "itest_"


def basis(index: int) -> list[float]:
    """Unit vector with a single 1.0, so cosine similarity between distinct ones is exactly 0."""
    vector = [0.0] * EMBEDDING_DIM
    vector[index] = 1.0
    return vector


def _ours(results):
    return [r for r in results if r.filename.startswith(PREFIX)]


@pytest.fixture
def repo():
    with VectorRepository(TEST_URL) as repository:
        repository.ensure_schema()
        yield repository
    with psycopg.connect(TEST_URL, autocommit=True) as conn:
        conn.execute("DELETE FROM document_chunks WHERE filename LIKE %s", (PREFIX + "%",))


@pytest.fixture
def isolated_repo():
    """Repository bound to a throwaway schema, so counts are exact even on a shared database."""
    schema = PREFIX + uuid.uuid4().hex[:8]
    with psycopg.connect(TEST_URL, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            # `public` stays on the path so the pgvector `vector` type is still found.
            url = make_conninfo(TEST_URL, options=f"-c search_path={schema},public")
            with VectorRepository(url) as repository:
                yield repository
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def _count(filename: str, strategy: str | None = None) -> int:
    query = "SELECT count(*) FROM document_chunks WHERE filename = %s"
    params: tuple = (filename,)
    if strategy:
        query += " AND split_strategy = %s"
        params += (strategy,)
    with psycopg.connect(TEST_URL) as conn:
        return conn.execute(query, params).fetchone()[0]


def test_ensure_schema_is_idempotent_and_creates_hnsw_index(repo):
    repo.ensure_schema()
    with psycopg.connect(TEST_URL) as conn:
        indexes = [row[0] for row in conn.execute(
            "SELECT indexdef FROM pg_indexes WHERE tablename = 'document_chunks'"
        )]
    assert any("hnsw" in definition for definition in indexes)


def test_replace_chunks_leaves_no_duplicates(repo):
    name = PREFIX + "dup.pdf"
    repo.replace_chunks(name, "fixed", ["a", "b", "c"], [basis(0), basis(1), basis(2)])
    stored = repo.replace_chunks(name, "fixed", ["x", "y"], [basis(3), basis(4)])
    assert stored == 2
    assert _count(name) == 2


def test_search_orders_by_similarity_and_scores_identical_vector_as_one(repo):
    name = PREFIX + "rank.pdf"
    repo.replace_chunks(name, "paragraph", ["zero", "one", "two"], [basis(0), basis(1), basis(2)])
    results = _ours(repo.search(basis(1), top_k=50))
    assert results[0].chunk_text == "one"
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.0, abs=1e-6)
    assert results[0].filename == name and results[0].split_strategy == "paragraph"


def test_strategy_filter_only_returns_that_strategy(repo):
    name = PREFIX + "filter.pdf"
    repo.replace_chunks(name, "fixed", ["fixed text"], [basis(5)])
    repo.replace_chunks(name, "sentence", ["sentence text"], [basis(5)])
    results = _ours(repo.search(basis(5), top_k=50, strategy="sentence"))
    assert [r.chunk_text for r in results] == ["sentence text"]


def test_search_with_no_results_returns_empty_list(isolated_repo):
    assert isolated_repo.search(basis(0), top_k=5) == []  # table does not exist yet
    isolated_repo.ensure_schema()
    assert isolated_repo.search(basis(0), top_k=5) == []  # table exists but is empty


def test_top_k_limits_the_number_of_results(isolated_repo):
    isolated_repo.ensure_schema()
    isolated_repo.replace_chunks(
        PREFIX + "limit.pdf", "fixed", [str(i) for i in range(5)], [basis(i) for i in range(5)]
    )
    assert len(isolated_repo.search(basis(0), top_k=3)) == 3
    assert len(isolated_repo.search(basis(0), top_k=10)) == 5  # fewer rows than top_k


def test_replace_chunks_does_not_touch_other_strategies_or_files(repo):
    file_a, file_b = PREFIX + "scope_a.pdf", PREFIX + "scope_b.pdf"
    repo.replace_chunks(file_a, "fixed", ["a1", "a2"], [basis(10), basis(11)])
    repo.replace_chunks(file_a, "sentence", ["s1", "s2", "s3"], [basis(12), basis(13), basis(14)])
    repo.replace_chunks(file_b, "fixed", ["b1"], [basis(15)])

    repo.replace_chunks(file_a, "fixed", ["a-new"], [basis(16)])

    assert _count(file_a, "fixed") == 1
    assert _count(file_a, "sentence") == 3  # same file, other strategy
    assert _count(file_b, "fixed") == 1  # other file, same strategy


def test_failed_write_rolls_back_and_keeps_previous_chunks(repo):
    name = PREFIX + "atomic.pdf"
    repo.replace_chunks(name, "fixed", ["keep me"], [basis(6)])
    wrong_dimension = [1.0, 2.0, 3.0]
    with pytest.raises(DatabaseError):
        repo.replace_chunks(name, "fixed", ["new one", "new two"], [basis(7), wrong_dimension])
    assert _count(name) == 1
    assert _ours(repo.search(basis(6), top_k=50))[0].chunk_text == "keep me"
