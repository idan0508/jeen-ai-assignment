"""Shared CLI plumbing: logging setup and mapping of errors to messages and exit codes."""

import logging
import sys
from collections.abc import Callable

from doc_indexer.errors import DocIndexerError


def configure_logging(verbose: bool) -> None:
    """Show DEBUG output of this package only; third-party loggers stay at WARNING."""
    # Enabling DEBUG globally would turn on HTTP/driver logs that can contain URLs or keys.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("doc_indexer").setLevel(logging.DEBUG if verbose else logging.WARNING)


def execute(action: Callable[[], None], verbose: bool) -> int:
    """Run `action`, returning a process exit code and printing errors to stderr."""
    try:
        action()
        return 0
    except DocIndexerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        # Deliberately no traceback or message: third-party errors can embed connection details.
        hint = type(exc).__name__ if verbose else "re-run with --verbose to see the error type"
        print(f"Unexpected error ({hint}).", file=sys.stderr)
        return 1
