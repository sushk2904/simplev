"""
Tests for the flat index search engine.

Tests cosine similarity and L2 distance search, masking behavior,
edge cases like empty stores, and result ordering.
"""

import numpy as np
import pytest

from simplev.indexing import FlatIndex, IndexingError


class TestFlatIndexInit:
    """Basic creation and config."""

    def test_default_metric_is_cosine(self):
        idx = FlatIndex()
        assert idx.metric == "cosine"

    def test_l2_metric(self):
        idx = FlatIndex(metric="l2")
        assert idx.metric == "l2"

    def test_invalid_metric_raises(self):
        with pytest.raises(IndexingError, match="Unknown metric"):
            FlatIndex(metric="hamming")

    def test_repr(self):
        idx = FlatIndex(metric="cosine")
        assert "cosine" in repr(idx)


class TestCosineSearch:
    """Test cosine similarity search."""

    @pytest.fixture
    def index(self):
        return FlatIndex(metric="cosine")

    @pytest.fixture
    def simple_vectors(self):
        # 4 vectors, 3 dimensions
        # vec 0 points in x direction
        # vec 1 points in y direction
        # vec 2 points in x+y (similar to query)
        # vec 3 points in z direction (least similar to x-y queries)
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

    def test_basic_search(self, index, simple_vectors):
        query = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        results = index.search(query, simple_vectors, top_k=2)
        assert len(results) == 2
        # vec 2 should be most similar (it's the same direction)
        assert results[0][0] == 2

    def test_returns_scores(self, index, simple_vectors):
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, simple_vectors, top_k=4)
        # first result should be vec 0 (exact match = similarity 1.0)
        idx, score = results[0]
        assert idx == 0
        assert abs(score - 1.0) < 1e-6

    def test_top_k_limits_results(self, index, simple_vectors):
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, simple_vectors, top_k=1)
        assert len(results) == 1

    def test_top_k_larger_than_dataset(self, index, simple_vectors):
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = index.search(query, simple_vectors, top_k=100)
        # should just return all 4, not crash
        assert len(results) == 4

    def test_cosine_order_is_descending(self, index, simple_vectors):
        query = np.array([1.0, 0.5, 0.0], dtype=np.float32)
        results = index.search(query, simple_vectors, top_k=4)
        scores = [s for _, s in results]
        # should be sorted highest to lowest
        assert scores == sorted(scores, reverse=True)


class TestL2Search:
    """Test L2 (Euclidean) distance search."""

    @pytest.fixture
    def index(self):
        return FlatIndex(metric="l2")

    def test_basic_l2_search(self, index):
        vectors = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [10.0, 10.0],
        ], dtype=np.float32)
        query = np.array([0.5, 0.0], dtype=np.float32)

        results = index.search(query, vectors, top_k=2)
        # closest should be vec 0 (dist=0.5) or vec 1 (dist=0.5)
        # then the other one. vec 2 should be far away.
        returned_indices = [idx for idx, _ in results]
        assert 2 not in returned_indices

    def test_l2_order_is_ascending(self, index):
        vectors = np.array([
            [0.0, 0.0],
            [5.0, 5.0],
            [1.0, 1.0],
        ], dtype=np.float32)
        query = np.array([0.0, 0.0], dtype=np.float32)
        results = index.search(query, vectors, top_k=3)
        scores = [s for _, s in results]
        # L2 distances should be ascending (closest first)
        assert scores == sorted(scores)

    def test_exact_match_returns_zero_distance(self, index):
        vectors = np.array([[3.0, 4.0]], dtype=np.float32)
        query = np.array([3.0, 4.0], dtype=np.float32)
        results = index.search(query, vectors, top_k=1)
        _, score = results[0]
        assert abs(score) < 1e-6


class TestMaskedSearch:
    """Test that the boolean mask correctly filters results."""

    @pytest.fixture
    def index(self):
        return FlatIndex(metric="cosine")

    def test_mask_excludes_vectors(self, index):
        vectors = np.array([
            [1.0, 0.0, 0.0],  # index 0 - best match but masked out
            [0.0, 1.0, 0.0],  # index 1
            [0.8, 0.2, 0.0],  # index 2 - second best
        ], dtype=np.float32)
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # mask out the best match
        mask = np.array([False, True, True])
        results = index.search(query, vectors, top_k=2, mask=mask)

        returned_indices = [idx for idx, _ in results]
        assert 0 not in returned_indices  # should be excluded
        assert 2 in returned_indices       # should be the best remaining

    def test_all_masked_returns_empty(self, index):
        vectors = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        query = np.array([1.0, 0.0], dtype=np.float32)
        mask = np.array([False, False])
        results = index.search(query, vectors, top_k=5, mask=mask)
        assert results == []

    def test_mask_length_mismatch_raises(self, index):
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        query = np.array([1.0, 0.0], dtype=np.float32)
        bad_mask = np.array([True])  # wrong length
        with pytest.raises(IndexingError, match="Mask length"):
            index.search(query, vectors, top_k=1, mask=bad_mask)


class TestEdgeCases:
    """Various edge cases and error handling."""

    @pytest.fixture
    def index(self):
        return FlatIndex()

    def test_empty_vectors_returns_empty(self, index):
        query = np.array([1.0, 0.0], dtype=np.float32)
        results = index.search(query, np.array([]).reshape(0, 2), top_k=5)
        assert results == []

    def test_none_vectors_returns_empty(self, index):
        query = np.array([1.0, 0.0], dtype=np.float32)
        results = index.search(query, None, top_k=5)
        assert results == []

    def test_dimension_mismatch_raises(self, index):
        vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        query = np.array([1.0, 0.0], dtype=np.float32)  # wrong dim
        with pytest.raises(IndexingError, match="Dimension mismatch"):
            index.search(query, vectors, top_k=1)

    def test_2d_query_raises(self, index):
        vectors = np.array([[1.0, 0.0]], dtype=np.float32)
        query = np.array([[1.0, 0.0]], dtype=np.float32)  # 2D, should be 1D
        with pytest.raises(IndexingError, match="1-D"):
            index.search(query, vectors, top_k=1)

    def test_single_vector_in_store(self, index):
        vectors = np.array([[0.5, 0.5]], dtype=np.float32)
        query = np.array([1.0, 0.0], dtype=np.float32)
        results = index.search(query, vectors, top_k=1)
        assert len(results) == 1
        assert results[0][0] == 0

    def test_zero_vector_doesnt_crash(self, index):
        vectors = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # the zero vector should get similarity 0, not cause a division error
        results = index.search(query, vectors, top_k=2)
        assert len(results) == 2
