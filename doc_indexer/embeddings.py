"""Gemini embeddings: batching, selective retry on transient errors, L2 normalization."""

import math

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from doc_indexer.errors import EmbeddingError

MODEL = "gemini-embedding-001"
# The model defaults to 3072 dims, above pgvector's 2000-dim limit for HNSW indexes.
EMBEDDING_DIM = 768
BATCH_SIZE = 50
_MAX_ATTEMPTS = 4


class GeminiEmbedder:
    """Turns texts into unit-length 768-dim vectors using gemini-embedding-001."""

    def __init__(self, api_key: str, client: genai.Client | None = None) -> None:
        # The SDK retries by default; disable it so only our selective policy applies.
        self._client = client or genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks for indexing (RETRIEVAL_DOCUMENT)."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start:start + BATCH_SIZE]
            vectors.extend(self._embed(batch, "RETRIEVAL_DOCUMENT"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (RETRIEVAL_QUERY)."""
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        try:
            response = _embed_with_retry(self._client, texts, task_type)
        except genai_errors.ClientError as exc:
            # Message is built from the status code only: SDK error text may echo request details.
            if exc.code == 429:
                raise EmbeddingError(
                    "Gemini rate limit exceeded after several retries. Try again later."
                ) from None
            raise EmbeddingError(
                f"Gemini rejected the request (HTTP {exc.code}). Check that "
                f"GEMINI_API_KEY is valid and has access to {MODEL}."
            ) from None
        except Exception:
            raise EmbeddingError(
                "Failed to generate embeddings because the Gemini service is "
                "unavailable or unreachable. Check your network and try again."
            ) from None

        embeddings = response.embeddings or []
        if len(embeddings) != len(texts):
            raise EmbeddingError("Gemini returned an unexpected number of embeddings.")
        vectors = [list(e.values or []) for e in embeddings]
        if any(len(v) != EMBEDDING_DIM for v in vectors):
            raise EmbeddingError("Gemini returned embeddings with an unexpected dimension.")
        return [normalize_l2(v) for v in vectors]


def normalize_l2(vector: list[float]) -> list[float]:
    """Scale a vector to unit length. Needed because truncated (768-dim) outputs are not normalized."""
    norm = math.sqrt(math.fsum(x * x for x in vector))
    if norm == 0:
        raise EmbeddingError("Received a zero-length embedding vector.")
    return [x / norm for x in vector]


def is_transient(exc: BaseException) -> bool:
    """True for errors worth retrying: HTTP 429, 5xx and network/timeout failures."""
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return exc.code == 429
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    # The SDK's HTTP transport library differs between versions (httpx / httpx2), so
    # match its transport errors by name instead of importing a specific package.
    return any(
        cls.__module__.startswith("httpx") and cls.__name__ == "TransportError"
        for cls in type(exc).__mro__
    )


@retry(
    retry=retry_if_exception(is_transient),
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    wait=wait_exponential_jitter(initial=1, max=20),
    reraise=True,
)
def _embed_with_retry(client: genai.Client, texts: list[str], task_type: str):
    return client.models.embed_content(
        model=MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=EMBEDDING_DIM
        ),
    )
