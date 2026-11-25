"""CLI: index a PDF/DOCX file into PostgreSQL (pgvector) for semantic search."""

import argparse
import sys
from collections.abc import Sequence

from doc_indexer.chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, STRATEGIES
from doc_indexer.cli import configure_logging, execute
from doc_indexer.config import load_config
from doc_indexer.db import VectorRepository
from doc_indexer.embeddings import GeminiEmbedder
from doc_indexer.services.indexing_service import IndexingService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF/DOCX file, split it, embed it with Gemini "
        "and store it in PostgreSQL/pgvector."
    )
    parser.add_argument("--file", required=True, help="path to a .pdf or .docx file")
    parser.add_argument(
        "--strategy", choices=list(STRATEGIES), default="paragraph",
        help="chunking strategy (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help="max characters per chunk (default: %(default)s)",
    )
    parser.add_argument(
        "--overlap", type=int, default=DEFAULT_OVERLAP,
        help="characters shared by neighbouring chunks, 'fixed' strategy only (default: %(default)s)",
    )
    parser.add_argument("--verbose", action="store_true", help="show debug output")
    return parser


def _run(args: argparse.Namespace) -> None:
    config = load_config()
    embedder = GeminiEmbedder(config.gemini_api_key)
    with VectorRepository(config.postgres_url) as repository:
        result = IndexingService(embedder, repository).index_file(
            args.file, args.strategy, args.chunk_size, args.overlap
        )
    # The parameters are not stored in the database, so the run's output is their only record.
    print(
        f"Indexed '{result.filename}' with strategy '{result.strategy}' "
        f"(chunk_size={args.chunk_size}, overlap={args.overlap}): "
        f"{result.char_count} characters -> {result.chunk_count} chunks stored."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    return execute(lambda: _run(args), args.verbose)


if __name__ == "__main__":
    sys.exit(main())
