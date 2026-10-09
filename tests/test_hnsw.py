"""
Tests for HNSW (Hierarchical Navigable Small World) indexing engine.

Covers graph construction, cosine and L2 distance metrics, approximate
nearest neighbor recall vs exact search, masking, and edge cases.
"""

import numpy as np
import pytest

from simplev.indexing import FlatIndex, HNSWIndex, IndexingError


@pytest.fixture
def sample_vectors():
    """Deterministic cluster of vectors for testing graph connectivity."""
    rng = np.random.RandomState(42)
    # 50 vectors of dimension 8
    vecs = rng.randn(50, 8).astype(np.float32)
    # normalize for clean cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


class TestHNSWInit:
    """Test HNSWIndex configuration and parameter validation."""

    def test_default_config(self):
        hnsw = HNSWIndex()
        assert hnsw.metric == "cosine"
        assert hnsw._m == 16
        assert hnsw._ef_construction == 200
        assert hnsw._ef_search == 50

    def test_l2_metric(self):
        hnsw = HNSWIndex(metric="l2")
        assert hnsw.metric == "l2"

    def test_invalid_metric_raises(self):
        with pytest.raises(IndexingError, match="Unknown metric"):
            HNSWIndex(metric="hamming")


class TestHNSWSearch:
    """Test nearest neighbor search accuracy and edge cases."""

    def test_empty_vectors_returns_empty(self):
        hnsw = HNSWIndex()
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert hnsw.search(query, np.empty((0, 3), dtype=np.float32)) == []

    def test_single_vector(self):
        hnsw = HNSWIndex()
        vec = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        hnsw.build(vec)
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        res = hnsw.search(query, vec, top_k=1)
        assert len(res) == 1
        assert res[0][0] == 0
        assert pytest.approx(res[0][1], abs=1e-4) == 1.0

    def test_hnsw_matches_flat_top1_cosine(self, sample_vectors):
        flat = FlatIndex(metric="cosine")
        hnsw = HNSWIndex(metric="cosine", m=16, ef_construction=100, ef_search=50)
        hnsw.build(sample_vectors)

        query = sample_vectors[10]  # Exact match with vector 10
        flat_results = flat.search(query, sample_vectors, top_k=5)
        hnsw_results = hnsw.search(query, sample_vectors, top_k=5)

        # Vector 10 should be at rank 0 for both
        assert flat_results[0][0] == 10
        assert hnsw_results[0][0] == 10
        assert pytest.approx(flat_results[0][1], abs=1e-4) == hnsw_results[0][1]

    def test_hnsw_l2_metric_search(self, sample_vectors):
        flat = FlatIndex(metric="l2")
        hnsw = HNSWIndex(metric="l2", m=16, ef_construction=100, ef_search=50)
        hnsw.build(sample_vectors)

        query = sample_vectors[5]
        flat_results = flat.search(query, sample_vectors, top_k=3)
        hnsw_results = hnsw.search(query, sample_vectors, top_k=3)

        assert flat_results[0][0] == 5
        assert hnsw_results[0][0] == 5

    def test_hnsw_mask_skips_tombstoned_vectors(self, sample_vectors):
        hnsw = HNSWIndex()
        hnsw.build(sample_vectors)

        query = sample_vectors[10]
        # Mask out vector 10 (simulate deletion)
        mask = np.ones(len(sample_vectors), dtype=bool)
        mask[10] = False

        results = hnsw.search(query, sample_vectors, top_k=5, mask=mask)
        returned_indices = [idx for idx, _ in results]
        assert 10 not in returned_indices
