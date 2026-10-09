"""
Tests for Okapi BM25 sparse keyword indexing.

Covers tokenization, single/batch document indexing, inverted index
maintenance, document deletion, ranking calculation, and edge cases.
"""

import pytest

from simplev.sparse import BM25Index


class TestBM25Init:
    """Test BM25Index initialization and attributes."""

    def test_default_params(self):
        idx = BM25Index()
        assert idx.k1 == 1.5
        assert idx.b == 0.75
        assert idx.count == 0
        assert idx.doc_count == 0

    def test_custom_params(self):
        idx = BM25Index(k1=2.0, b=0.5)
        assert idx.k1 == 2.0
        assert idx.b == 0.5

    def test_repr(self):
        idx = BM25Index()
        assert "BM25Index(docs=0, k1=1.5, b=0.75)" in repr(idx)


class TestBM25Tokenize:
    """Test tokenization logic."""

    def test_basic_tokenization(self):
        tokens = BM25Index.tokenize("Hello, World! SimpleV is great.")
        assert tokens == ["hello", "world", "simplev", "is", "great"]

    def test_empty_string(self):
        assert BM25Index.tokenize("") == []
        assert BM25Index.tokenize("    ") == []

    def test_punctuation_only(self):
        assert BM25Index.tokenize("!@#$%^&*()+-=") == []


class TestBM25IndexOperations:
    """Test adding, updating, removing, and clearing documents."""

    def test_add_document(self):
        idx = BM25Index()
        idx.add_document("doc1", "The quick brown fox jumps over the lazy dog.")
        assert idx.count == 1
        assert "doc1" in idx.doc_lengths
        assert idx.doc_lengths["doc1"] == 9

    def test_add_duplicate_replaces(self):
        idx = BM25Index()
        idx.add_document("doc1", "apple banana")
        assert idx.doc_lengths["doc1"] == 2
        idx.add_document("doc1", "apple orange grape peach")
        assert idx.count == 1
        assert idx.doc_lengths["doc1"] == 4

    def test_remove_document(self):
        idx = BM25Index()
        idx.add_document("doc1", "database vector search")
        idx.add_document("doc2", "relational database table")
        assert idx.count == 2

        removed = idx.remove_document("doc1")
        assert removed is True
        assert idx.count == 1
        assert "doc1" not in idx.doc_lengths

        removed_again = idx.remove_document("doc1")
        assert removed_again is False

    def test_build_and_clear(self):
        idx = BM25Index()
        docs = [
            ("d1", "vector retrieval systems"),
            ("d2", "deep learning embeddings"),
            ("d3", "information retrieval systems"),
        ]
        idx.build(docs)
        assert idx.count == 3

        idx.clear()
        assert idx.count == 0
        assert len(idx.doc_lengths) == 0


class TestBM25Search:
    """Test BM25 query scoring and filtering."""

    @pytest.fixture
    def indexed_store(self):
        idx = BM25Index()
        idx.add_document("d1", "Python programming language for machine learning")
        idx.add_document(
            "d2", "Rust systems programming language with zero cost abstractions"
        )
        idx.add_document("d3", "Machine learning and deep neural networks in Python")
        idx.add_document("d4", "SQLite lightweight SQL database engine")
        return idx

    def test_exact_keyword_match(self, indexed_store):
        results = indexed_store.search("Rust", top_k=2)
        assert len(results) >= 1
        assert results[0][0] == "d2"
        assert results[0][1] > 0.0

    def test_term_frequency_ranking(self, indexed_store):
        results = indexed_store.search("Python machine learning", top_k=2)
        top_ids = [doc_id for doc_id, _ in results]
        assert "d1" in top_ids or "d3" in top_ids
        assert "d4" not in top_ids

    def test_search_empty_query(self, indexed_store):
        assert indexed_store.search("") == []
        assert indexed_store.search("    ") == []

    def test_search_no_matches(self, indexed_store):
        assert indexed_store.search("astronomy telescope galaxies") == []

    def test_search_with_allowed_doc_ids(self, indexed_store):
        # Restrict to d2 and d4
        results = indexed_store.search("Python", allowed_doc_ids={"d2", "d4"})
        assert results == []

        results2 = indexed_store.search("programming", allowed_doc_ids={"d2"})
        assert len(results2) == 1
        assert results2[0][0] == "d2"

    def test_search_empty_index(self):
        idx = BM25Index()
        assert idx.search("query") == []
