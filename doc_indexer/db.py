"""PostgreSQL + pgvector access layer (schema setup, chunk storage, similarity search)."""

import logging
from dataclasses import dataclass
from datetime import datetime

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from doc_indexer.embeddings import EMBEDDING_DIM
from doc_indexer.errors import DatabaseError

logger = logging.getLogger(__name__)

_CONNECTION_MESSAGE = (
    "Could not connect to PostgreSQL. Check POSTGRES_URL in your .env file "
    "and make sure the database is running and reachable."
)

# Static SQL only: user-supplied values always travel as bound parameters.
_CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS document_chunks (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    filename        TEXT        NOT NULL,
    split_strategy  TEXT        NOT NULL
                    CHECK (split_strategy IN ('fixed', 'sentence', 'paragraph')),
    chunk_index     INT         NOT NULL,
    chunk_text      TEXT        NOT NULL,
    embedding       vector({EMBEDDING_DIM}) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (filename, split_strategy, chunk_index)
)
"""
_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw
ON document_chunks USING hnsw (embedding vector_cosine_ops)
"""
_DELETE = "DELETE FROM document_chunks WHERE filename = %s AND split_strategy = %s"
_INSERT = """
INSERT INTO document_chunks (filename, split_strategy, chunk_index, chunk_text, embedding)
VALUES (%s, %s, %s, %s, %s)
"""
_SEARCH = """
SELECT id, filename, split_strategy, chunk_index, chunk_text,
       1 - (embedding <=> %(query)s) AS score, created_at
FROM document_chunks
{where}
ORDER BY embedding <=> %(query)s
LIMIT %(limit)s
"""


@dataclass(frozen=True)
class SearchResult:
    """One chunk returned by a similarity search; `score` is cosine similarity (1 = identical)."""

    id: int
    filename: str
    split_strategy: str
    chunk_index: int
    chunk_text: str
    score: float
    created_at: datetime


class VectorRepository:
    """Stores and retrieves embedded chunks. Use as a context manager."""

    def __init__(self, postgres_url: str) -> None:
        self._url = postgres_url
        self._conn: psycopg.Connection | None = None
        self._vector_registered = False

    def __enter__(self) -> "VectorRepository":
        try:
            # autocommit lets us scope real transactions explicitly with conn.transaction().
            self._conn = psycopg.connect(self._url, autocommit=True, connect_timeout=5)
        except psycopg.Error as exc:
            # Original message may contain the host/user, so it is neither shown nor chained.
            logger.debug("Connection failed: %s", type(exc).__name__)
            raise DatabaseError(_CONNECTION_MESSAGE) from None
        # Lets searches work on an already-initialized DB; ensure_schema registers again after creating it.
        self._register_vector(required=False)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def ensure_schema(self) -> None:
        """Create the pgvector extension (if absent), the table and the vector index."""
        conn = self._connection()
        try:
            # Checking first avoids needing superuser rights when an admin already enabled it.
            exists = conn.execute(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            if exists is None:
                self._create_extension()
            # Must come after the extension exists: it looks up the `vector` type's OID.
            self._register_vector(required=True)
            with conn.transaction():
                conn.execute(_CREATE_TABLE)
                conn.execute(_CREATE_INDEX)
        except psycopg.Error as exc:
            logger.debug("Schema setup failed: %s", type(exc).__name__)
            raise DatabaseError("Failed to initialize the database schema.") from None

    def replace_chunks(
        self,
        filename: str,
        strategy: str,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> int:
        """Atomically replace all stored chunks of (filename, strategy). Returns the number stored."""
        if len(chunks) != len(vectors):
            raise DatabaseError("Internal error: chunk and embedding counts differ.")
        rows = [
            (filename, strategy, index, text, Vector(vector))
            for index, (text, vector) in enumerate(zip(chunks, vectors))
        ]
        conn = self._connection()
        try:
            with conn.transaction(), conn.cursor() as cur:
                cur.execute(_DELETE, (filename, strategy))
                cur.executemany(_INSERT, rows)
        except psycopg.Error as exc:
            logger.debug("Insert failed: %s", type(exc).__name__)
            raise DatabaseError("Failed to store the document chunks in the database.") from None
        return len(rows)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        strategy: str | None = None,
    ) -> list[SearchResult]:
        """Return the `top_k` most similar chunks, optionally restricted to one strategy."""
        conn = self._connection()  # first, so misuse outside `with` is always reported
        if not self._vector_registered:
            return []  # the pgvector extension does not exist yet, so nothing can be indexed
        where = "WHERE split_strategy = %(strategy)s" if strategy else ""
        params = {"query": Vector(query_vector), "limit": top_k, "strategy": strategy}
        try:
            rows = conn.execute(_SEARCH.format(where=where), params).fetchall()
        except psycopg.errors.UndefinedTable:
            return []  # nothing has been indexed yet
        except psycopg.Error as exc:
            logger.debug("Search failed: %s", type(exc).__name__)
            raise DatabaseError("Failed to run the similarity search.") from None
        return [SearchResult(*row) for row in rows]

    def _connection(self) -> psycopg.Connection:
        if self._conn is None:
            raise DatabaseError("Internal error: repository used outside a 'with' block.")
        return self._conn

    def _create_extension(self) -> None:
        try:
            self._connection().execute("CREATE EXTENSION vector")
        except psycopg.errors.InsufficientPrivilege:
            raise DatabaseError(
                "The pgvector extension is not enabled and the database user is not "
                "allowed to create it. Ask a database admin to run: CREATE EXTENSION vector;"
            ) from None
        except (psycopg.errors.FeatureNotSupported, psycopg.errors.UndefinedFile):
            raise DatabaseError(
                "pgvector is not installed on the PostgreSQL server. Install it, or "
                "use the pgvector/pgvector Docker image (see README)."
            ) from None

    def _register_vector(self, required: bool) -> None:
        try:
            register_vector(self._connection())
            self._vector_registered = True
        except psycopg.ProgrammingError:
            if required:
                raise
