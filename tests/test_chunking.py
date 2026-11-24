"""Unit tests for the three chunking strategies."""

import pytest

from doc_indexer.chunking import STRATEGIES, chunk_text, fixed_chunks
from doc_indexer.errors import ConfigError


def test_fixed_chunks_overlap_between_neighbours():
    text = "abcdefghij" * 5  # 50 chars
    chunks = fixed_chunks(text, chunk_size=20, overlap=5)
    assert all(len(c) <= 20 for c in chunks)
    assert chunks[0][-5:] == chunks[1][:5]
    assert chunks[-1].endswith(text[-5:])


def test_fixed_chunks_short_text_is_single_chunk():
    assert fixed_chunks("short", 100, 10) == ["short"]


def test_sentence_chunks_never_cut_a_sentence():
    text = "First sentence here. Second one follows! Is this the third? Yes it is."
    chunks = chunk_text(text, "sentence", chunk_size=45, overlap=0)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk[-1] in ".!?"


def test_sentence_split_ignores_abbreviations():
    chunks = chunk_text("Ask Dr. Smith about it. Then call us.", "sentence", chunk_size=30, overlap=0)
    assert chunks == ["Ask Dr. Smith about it.", "Then call us."]


def test_paragraph_chunks_keep_paragraphs_whole():
    text = "Para one is here.\n\nPara two is here.\n\nPara three is here."
    chunks = chunk_text(text, "paragraph", chunk_size=40, overlap=0)
    assert chunks == ["Para one is here.\n\nPara two is here.", "Para three is here."]


def test_oversized_paragraph_falls_back_to_fixed_windows():
    chunks = chunk_text("x" * 250, "paragraph", chunk_size=100, overlap=20)
    assert len(chunks) == 3
    assert all(len(c) <= 100 for c in chunks)


@pytest.mark.parametrize("strategy", list(STRATEGIES))
def test_empty_text_yields_no_chunks(strategy):
    assert chunk_text("", strategy) == []


@pytest.mark.parametrize("size, overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters_rejected(size, overlap):
    with pytest.raises(ConfigError):
        chunk_text("text", "fixed", chunk_size=size, overlap=overlap)


def test_unknown_strategy_rejected():
    with pytest.raises(ConfigError):
        chunk_text("text", "semantic")
