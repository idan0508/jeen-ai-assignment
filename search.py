"""CLI: semantic search over the chunks indexed in PostgreSQL (pgvector)."""

import argparse
import sys
import textwrap
from collections.abc import Sequence

from doc_indexer.chunking import STRATEGIES
from doc_indexer.cli import configure_logging, execute
from doc_indexer.config import load_config
from doc_indexer.db import SearchResult, VectorRepository
from doc_indexer.embeddings import GeminiEmbedder
from doc_indexer.services.search_service import DEFAULT_TOP_K, SearchService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Semantic search over indexed documents.")
    parser.add_argument("--query", required=True, help="what to look for, in natural language")
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K,
        help="number of results to return (default: %(default)s)",
    )
    parser.add_argument(
        "--strategy", choices=list(STRATEGIES), default=None,
        help="only search chunks created with this strategy",
    )
    parser.add_argument("--verbose", action="store_true", help="show debug output")
    return parser


def format_results(query: str, results: list[SearchResult]) -> str:
    """Render every result as a labelled block with the full, unmodified chunk text."""
    if not results:
        return "No results found. Have you indexed any documents yet?"
    separator = "-" * 60
    lines = [f'Top {len(results)} results for "{query}":', separator]
    for rank, result in enumerate(results, start=1):
        lines += [
            f"Rank:      {rank}",
            f"Score:     {result.score:.4f}",
            f"File:      {result.filename}",
            f"Strategy:  {result.split_strategy}",
            f"Chunk:     #{result.chunk_index}",
            "Text:",
            textwrap.indent(result.chunk_text, "    "),
            separator,
        ]
    return "\n".join(lines)


def _run(args: argparse.Namespace) -> None:
    config = load_config()
    embedder = GeminiEmbedder(config.gemini_api_key)
    with VectorRepository(config.postgres_url) as repository:
        results = SearchService(embedder, repository).search(args.query, args.top_k, args.strategy)
    print(format_results(args.query.strip(), results))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    return execute(lambda: _run(args), args.verbose)


if __name__ == "__main__":
    sys.exit(main())
