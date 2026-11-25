"""Unit tests for the Gemini embedder using a scripted fake client (no network)."""

import math
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors
from tenacity import wait_none

from doc_indexer import embeddings
from doc_indexer.embeddings import EMBEDDING_DIM, GeminiEmbedder, is_transient, normalize_l2
from doc_indexer.errors import EmbeddingError

SECRET = "super-secret-key"


def _ok(count: int, value: float = 1.0):
    vectors = [SimpleNamespace(values=[value] * EMBEDDING_DIM) for _ in range(count)]
    return SimpleNamespace(embeddings=vectors)


class FakeClient:
    """Plays back a script: each item is a response to return or an exception to raise."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.models = self

    def embed_content(self, *, model, contents, config):
        self.calls.append((list(contents), config.task_type))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(embeddings._embed_with_retry.retry, "wait", wait_none())


def _embedder(script):
    client = FakeClient(script)
    return GeminiEmbedder(SECRET, client=client), client


def test_normalize_l2_gives_unit_length():
    vec = normalize_l2([3.0, 4.0])
    assert vec == pytest.approx([0.6, 0.8])
    assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0)


def test_zero_vector_is_rejected():
    with pytest.raises(EmbeddingError):
        normalize_l2([0.0] * 4)


def test_documents_and_query_use_different_task_types():
    embedder, client = _embedder([_ok(2), _ok(1)])
    docs = embedder.embed_documents(["a", "b"])
    query = embedder.embed_query("q")
    assert [c[1] for c in client.calls] == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
    assert len(docs) == 2 and len(query) == EMBEDDING_DIM
    assert math.isclose(sum(x * x for x in query), 1.0)


def test_batches_requests_and_keeps_order():
    embedder, client = _embedder([_ok(50), _ok(50), _ok(20)])
    assert len(embedder.embed_documents([str(i) for i in range(120)])) == 120
    assert [len(c[0]) for c in client.calls] == [50, 50, 20]


def test_empty_input_makes_no_api_call():
    embedder, client = _embedder([])
    assert embedder.embed_documents([]) == []
    assert client.calls == []


@pytest.mark.parametrize("code", [429, 503])
def test_transient_errors_are_retried(code):
    error_cls = genai_errors.ClientError if code == 429 else genai_errors.ServerError
    embedder, client = _embedder([error_cls(code, {}), error_cls(code, {}), _ok(1)])
    assert len(embedder.embed_documents(["a"])) == 1
    assert len(client.calls) == 3


@pytest.mark.parametrize("code", [400, 401, 403])
def test_permanent_errors_fail_immediately(code):
    embedder, client = _embedder([genai_errors.ClientError(code, {}), _ok(1)])
    with pytest.raises(EmbeddingError) as info:
        embedder.embed_documents(["a"])
    assert len(client.calls) == 1
    assert str(code) in str(info.value)
    assert SECRET not in str(info.value)


def test_gives_up_after_max_attempts():
    embedder, client = _embedder([genai_errors.ServerError(503, {})] * 10)
    with pytest.raises(EmbeddingError):
        embedder.embed_documents(["a"])
    assert len(client.calls) == 4


def test_network_error_is_retried_then_reported():
    embedder, client = _embedder([ConnectionError("boom")] * 10)
    with pytest.raises(EmbeddingError):
        embedder.embed_documents(["a"])
    assert len(client.calls) == 4


def test_wrong_vector_count_is_rejected():
    embedder, _ = _embedder([_ok(1)])
    with pytest.raises(EmbeddingError):
        embedder.embed_documents(["a", "b"])


def test_wrong_dimension_is_rejected():
    bad = SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0] * 10)])
    embedder, _ = _embedder([bad])
    with pytest.raises(EmbeddingError):
        embedder.embed_documents(["a"])


def test_zero_vector_from_api_is_rejected():
    embedder, _ = _embedder([_ok(1, value=0.0)])
    with pytest.raises(EmbeddingError):
        embedder.embed_documents(["a"])


def test_is_transient_classification():
    assert is_transient(genai_errors.ServerError(500, {}))
    assert is_transient(genai_errors.ClientError(429, {}))
    assert not is_transient(genai_errors.ClientError(403, {}))
    assert not is_transient(ValueError("x"))
