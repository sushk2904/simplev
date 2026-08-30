"""
Tests for text chunking utilities.

Covers character-based chunking, separator-based chunking,
overlap behavior, edge cases, and the chunk_document helper.
"""

import pytest

from simplev.chunking import (
    chunk_text, chunk_document, ChunkingError
)


class TestChunkText:
    """Test basic character-based chunking."""

    def test_short_text_no_split(self):
        text = "hello world"
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    def test_splits_into_chunks(self):
        text = "a" * 100
        chunks = chunk_text(text, chunk_size=30, overlap=0)
        # 100 / 30 = 3 full chunks + 1 partial = 4
        assert len(chunks) == 4
        assert all(len(c) <= 30 for c in chunks)

    def test_overlap_works(self):
        text = "abcdefghijklmnopqrstuvwxyz"
        chunks = chunk_text(text, chunk_size=10, overlap=3)
        # with overlap, chunks should share characters at boundaries
        # chunk 0 starts at 0, chunk 1 starts at 7 (10-3), etc
        assert chunks[0][:10] == "abcdefghij"
        assert chunks[1][:3] == "hij"  # overlap from previous

    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_none_like_empty(self):
        assert chunk_text("") == []

    def test_exact_chunk_size(self):
        text = "a" * 50
        chunks = chunk_text(text, chunk_size=50, overlap=0)
        assert len(chunks) == 1


class TestChunkValidation:
    """Test parameter validation."""

    def test_zero_chunk_size_raises(self):
        with pytest.raises(ChunkingError, match="positive"):
            chunk_text("hello", chunk_size=0)

    def test_negative_chunk_size_raises(self):
        with pytest.raises(ChunkingError, match="positive"):
            chunk_text("hello", chunk_size=-10)

    def test_negative_overlap_raises(self):
        with pytest.raises(ChunkingError, match="non-negative"):
            chunk_text("hello", chunk_size=10, overlap=-1)

    def test_overlap_equals_chunk_size_raises(self):
        with pytest.raises(ChunkingError, match="less than"):
            chunk_text("hello", chunk_size=10, overlap=10)

    def test_overlap_exceeds_chunk_size_raises(self):
        with pytest.raises(ChunkingError, match="less than"):
            chunk_text("hello", chunk_size=10, overlap=15)


class TestSeparatorChunking:
    """Test separator-based chunking."""

    def test_split_by_newline(self):
        text = "line one\nline two\nline three\nline four"
        chunks = chunk_text(text, chunk_size=25, overlap=0, separator="\n")
        # each chunk should contain complete lines, not cut mid-line
        for chunk in chunks:
            # no line should be cut in the middle of a word
            assert "line" in chunk

    def test_split_by_sentence(self):
        text = "First sentence. Second sentence. Third sentence. Fourth one."
        chunks = chunk_text(text, chunk_size=40, overlap=0, separator=". ")
        assert len(chunks) >= 2

    def test_split_by_paragraph(self):
        text = "Para one text here.\n\nPara two over here.\n\nPara three at the end."
        chunks = chunk_text(text, chunk_size=30, overlap=0, separator="\n\n")
        assert len(chunks) >= 2

    def test_single_segment_larger_than_chunk(self):
        # if one segment is bigger than chunk_size, it still gets included
        text = "short\nvery long segment that exceeds the chunk size limit"
        chunks = chunk_text(text, chunk_size=15, overlap=0, separator="\n")
        assert len(chunks) >= 2


class TestChunkDocument:
    """Test the convenience function for Client.add_many()."""

    def test_basic_chunking(self):
        text = "a" * 200
        docs = chunk_document(text, "mydoc", chunk_size=80, overlap=10)
        assert len(docs) >= 2

        # check the format is right for add_many()
        for doc in docs:
            assert "doc_id" in doc
            assert "text" in doc
            assert "metadata" in doc

    def test_doc_ids_are_indexed(self):
        docs = chunk_document("hello " * 100, "paper", chunk_size=50, overlap=0)
        for i, doc in enumerate(docs):
            assert doc["doc_id"] == f"paper_chunk_{i}"

    def test_metadata_includes_chunk_info(self):
        docs = chunk_document(
            "some long text " * 50,
            "report",
            chunk_size=100,
            overlap=0,
            metadata={"author": "alice"},
        )
        for i, doc in enumerate(docs):
            meta = doc["metadata"]
            assert meta["chunk_index"] == i
            assert meta["total_chunks"] == len(docs)
            assert meta["source_doc"] == "report"
            assert meta["author"] == "alice"

    def test_short_text_single_chunk(self):
        docs = chunk_document("short text", "tiny")
        assert len(docs) == 1
        assert docs[0]["doc_id"] == "tiny_chunk_0"
        assert docs[0]["text"] == "short text"
