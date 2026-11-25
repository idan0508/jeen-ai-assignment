"""Indexing workflow: extract -> chunk -> embed -> store."""

import logging
from dataclasses import dataclass
from pathlib import Path

from doc_indexer.chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_text
from doc_indexer.db import VectorRepository
from doc_indexer.embeddings import GeminiEmbedder
from doc_indexer.extraction import extract_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexingResult:
    filename: str
    strategy: str
    char_count: int
    chunk_count: int


class IndexingService:
    """Indexes one document at a time. Collaborators are injected so they can be faked in tests."""

    def __init__(self, embedder: GeminiEmbedder, repository: VectorRepository) -> None:
        self._embedder = embedder
        self._repository = repository

    def index_file(
        self,
        path: str | Path,
        strategy: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_OVERLAP,
    ) -> IndexingResult:
        # Steps run from cheapest to most expensive, so bad input fails before any
        # API call is paid for or any database write happens.
        text = extract_text(path)
        chunks = chunk_text(text, strategy, chunk_size, overlap)
        logger.debug("Split %d characters into %d chunks", len(text), len(chunks))

        self._repository.ensure_schema()
        vectors = self._embedder.embed_documents(chunks)
        # Only reached when every embedding succeeded: the write is all-or-nothing.
        stored = self._repository.replace_chunks(Path(path).name, strategy, chunks, vectors)
        return IndexingResult(Path(path).name, strategy, len(text), stored)
