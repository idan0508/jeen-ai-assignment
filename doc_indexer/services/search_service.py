"""Search workflow: embed the query, then run a vector similarity search."""

from doc_indexer.chunking import STRATEGIES
from doc_indexer.db import SearchResult, VectorRepository
from doc_indexer.embeddings import GeminiEmbedder
from doc_indexer.errors import ConfigError

DEFAULT_TOP_K = 5


class SearchService:
    """Semantic search over indexed chunks. Collaborators are injected so they can be faked in tests."""

    def __init__(self, embedder: GeminiEmbedder, repository: VectorRepository) -> None:
        self._embedder = embedder
        self._repository = repository

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        strategy: str | None = None,
    ) -> list[SearchResult]:
        # Validate before the (paid) embedding call.
        query = query.strip()
        if not query:
            raise ConfigError("The search query must not be empty.")
        if top_k < 1:
            raise ConfigError("top_k must be at least 1.")
        if strategy is not None and strategy not in STRATEGIES:
            raise ConfigError(f"Unknown strategy '{strategy}'. Choose from: {', '.join(STRATEGIES)}")

        query_vector = self._embedder.embed_query(query)
        return self._repository.search(query_vector, top_k, strategy)
