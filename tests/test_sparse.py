"""
Tests for BM25Index (Okapi BM25 sparse lexical search).
"""

from simplev.sparse import BM25Index


class TestBM25Index:
    """Test suite for BM25Index."""

    def test_empty_index(self):
        index = BM25Index()
        assert index.doc_count == 0
        assert index.search("anything") == []

    def test_tokenize_basic(self):
        tokens = BM25Index.tokenize("Hello, World! 123 Vector-Search")
        assert tokens == ["hello", "world", "123", "vector", "search"]

    def test_add_and_search(self):
        index = BM25Index()
        index.add_document("d1", "The quick brown fox jumps over the lazy dog")
        index.add_document("d2", "A fast brown fox leaped across a sleeping hound")
        index.add_document("d3", "Artificial intelligence and vector databases")

        assert index.doc_count == 3

        # Search for fox
        results = index.search("brown fox", top_k=2)
        assert len(results) == 2
        doc_ids = [doc_id for doc_id, _ in results]
        assert "d1" in doc_ids
        assert "d2" in doc_ids
        assert "d3" not in doc_ids
        # Scores should be positive
        assert all(score > 0.0 for _, score in results)

    def test_term_frequency_weighting(self):
        index = BM25Index()
        index.add_document("d1", "python python python code")
        index.add_document("d2", "python code")

        results = index.search("python", top_k=2)
        # d1 mentions python 3 times, d2 mentions once -> d1 should rank higher or equal
        assert results[0][0] == "d1"

    def test_remove_document(self):
        index = BM25Index()
        index.add_document("d1", "machine learning algorithms")
        index.add_document("d2", "deep learning neural networks")
        assert index.doc_count == 2

        # Remove d1
        removed = index.remove_document("d1")
        assert removed is True
        assert index.doc_count == 1
        assert "d1" not in index.doc_lengths

        results = index.search("machine learning")
        assert len(results) == 1
        assert results[0][0] == "d2"

        # Removing non-existent returns False
        assert index.remove_document("nonexistent") is False

    def test_remove_and_readd_document(self):
        index = BM25Index()
        index.add_document("d1", "initial content")
        index.remove_document("d1")
        index.add_document("d1", "updated unique keywords")

        results = index.search("unique keywords")
        assert len(results) == 1
        assert results[0][0] == "d1"

    def test_reset(self):
        index = BM25Index()
        index.add_document("d1", "sample document text")
        assert index.doc_count == 1
        index.reset()
        assert index.doc_count == 0
        assert index.search("sample") == []

    def test_top_k_limiting(self):
        index = BM25Index()
        for i in range(10):
            index.add_document(f"doc_{i}", f"common word specific_{i}")

        results = index.search("common word", top_k=3)
        assert len(results) == 3

    def test_search_no_match(self):
        index = BM25Index()
        index.add_document("d1", "apples and oranges")
        results = index.search("banana cherry")
        assert results == []
