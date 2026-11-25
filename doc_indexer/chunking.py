"""Text chunking strategies: fixed-size with overlap, sentence-based, paragraph-based."""

import re
from collections.abc import Callable

from doc_indexer.errors import ConfigError

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# A period after these does not end a sentence ("Dr. Smith", "e.g. this").
_ABBREVIATIONS = frozenset(
    {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.", "etc.",
     "e.g.", "i.e.", "no.", "fig."}
)


def fixed_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split into windows of `chunk_size` characters, each sharing `overlap` characters with the previous."""
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break  # the last window already reaches the end; avoid a tail made only of overlap
    return chunks


def sentence_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Pack whole sentences into chunks of up to `chunk_size` characters."""
    return _pack(_split_sentences(text), " ", chunk_size, overlap)


def paragraph_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Pack whole paragraphs into chunks of up to `chunk_size` characters."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    return _pack(paragraphs, "\n\n", chunk_size, overlap)


# Single source of truth for available strategies; the CLI derives `choices` from it.
STRATEGIES: dict[str, Callable[[str, int, int], list[str]]] = {
    "fixed": fixed_chunks,
    "sentence": sentence_chunks,
    "paragraph": paragraph_chunks,
}


def chunk_text(
    text: str,
    strategy: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Split `text` with the named strategy. `overlap` only affects 'fixed' and oversized-unit splitting."""
    if strategy not in STRATEGIES:
        raise ConfigError(f"Unknown strategy '{strategy}'. Choose from: {', '.join(STRATEGIES)}")
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ConfigError("Invalid chunking parameters: need chunk_size > 0 and 0 <= overlap < chunk_size.")
    return STRATEGIES[strategy](text, chunk_size, overlap)


def _split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for piece in _SENTENCE_END.split(text.replace("\n\n", " ")):
        # Re-attach a piece to the previous one when the split happened after an abbreviation.
        if sentences and sentences[-1].rsplit(" ", 1)[-1].lower() in _ABBREVIATIONS:
            sentences[-1] += " " + piece
        elif piece:
            sentences.append(piece)
    return sentences


def _pack(units: list[str], joiner: str, chunk_size: int, overlap: int) -> list[str]:
    """Greedily group units without ever cutting one, except units larger than a whole chunk."""
    chunks: list[str] = []
    current = ""
    for unit in units:
        if len(unit) > chunk_size:
            # Cannot fit even alone: flush, then fall back to fixed-size windows.
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(fixed_chunks(unit, chunk_size, overlap))
        elif not current:
            current = unit
        elif len(current) + len(joiner) + len(unit) <= chunk_size:
            current += joiner + unit
        else:
            chunks.append(current)
            current = unit
    if current:
        chunks.append(current)
    return chunks
