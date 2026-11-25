"""Tests for VectorRepository error paths that need no running database."""

import pytest

from doc_indexer.db import VectorRepository
from doc_indexer.errors import DatabaseError

# Nothing listens on port 1, so connecting fails immediately and locally.
UNREACHABLE_URL = "postgresql://some-user:FAKE-PASSWORD@127.0.0.1:1/some-db"
SENSITIVE = ("FAKE-PASSWORD", "some-user", "127.0.0.1", "some-db")


def test_connection_failure_gives_clean_message_without_leaking_details():
    with pytest.raises(DatabaseError) as info:
        with VectorRepository(UNREACHABLE_URL):
            pass
    error = info.value
    assert "POSTGRES_URL" in str(error)
    assert not any(s in str(error) for s in SENSITIVE)
    # The original driver error (which names the host) must not be chained or printed.
    assert error.__cause__ is None
    assert error.__suppress_context__ is True


def test_using_repository_outside_with_block_is_rejected():
    repo = VectorRepository(UNREACHABLE_URL)
    with pytest.raises(DatabaseError):
        repo.search([0.1], top_k=5)


def test_mismatched_chunks_and_vectors_are_rejected_before_writing():
    repo = VectorRepository(UNREACHABLE_URL)
    with pytest.raises(DatabaseError):
        repo.replace_chunks("a.pdf", "fixed", ["one", "two"], [[0.1]])
