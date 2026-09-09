"""
Tests for the HNSW index and ANN search.

Covers graph initialization, single/batch insertion, cosine and L2 searches,
masking, recall comparison against FlatIndex, and edge cases.
"""

from unittest.mock import patch

import numpy as np
import pytest

from simplev.client import Client
from simplev.indexing import FlatIndex, HNSWIndex, IndexingError

DIM = 8


def make_vector(dim: int = DIM) -> np.ndarray:
    vec = np.random.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm if norm > 0 else 1.0)


class TestHNSWInit:
    """Test HNSW initialization."""

    def test_default_metric(self):
        idx = HNSWIndex()
        assert idx.metric == "cosine"
        assert idx.count == 0

    def test_l2_metric(self):
        idx = HNSWIndex(metric="l2")
        assert idx.metric == "l2"

    def test_invalid_metric_raises(self):
        with pytest.raises(IndexingError, match="Unknown metric"):
            HNSWIndex(metric="chebyshev")

    def test_repr(self):
        idx = HNSWIndex(metric="cosine")
        r = repr(idx)
        assert "HNSWIndex" in r
        assert "cosine" in r


class TestHNSWSearchCosine:
    """Test Cosine similarity search with HNSW."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 50
        vectors = np.array([make_vector(DIM) for _ in range(n)], dtype=np.float32)
        return vectors

    def test_build_and_search(self, sample_data):
        idx = HNSWIndex(metric="cosine", seed=42)
        idx.build(sample_data)
        assert idx.count == len(sample_data)

        # Query is identical to vector 5
        query = sample_data[5].copy()
        results = idx.search(query, sample_data, top_k=3)
        assert len(results) == 3
        # First result must be index 5 with similarity ~1.0
        assert results[0][0] == 5
        assert abs(results[0][1] - 1.0) < 1e-4

    def test_cosine_descending_order(self, sample_data):
        idx = HNSWIndex(metric="cosine", seed=42)
        idx.build(sample_data)

        query = make_vector(DIM)
        results = idx.search(query, sample_data, top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_add_incremental(self):
        idx = HNSWIndex(metric="cosine", seed=42)
        v0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v1 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.7071, 0.7071, 0.0, 0.0], dtype=np.float32)

        idx.add(v0)
        idx.add(v1)
        idx.add(v2)
        assert len(idx) == 3

        all_vecs = np.vstack([v0, v1, v2])
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = idx.search(query, all_vecs, top_k=2)
        assert results[0][0] == 0


class TestHNSWSearchL2:
    """Test L2 Euclidean distance search with HNSW."""

    def test_basic_l2_search(self):
        vectors = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [10.0, 10.0],
        ], dtype=np.float32)
        idx = HNSWIndex(metric="l2", seed=42)
        idx.build(vectors)

        query = np.array([0.1, 0.0], dtype=np.float32)
        results = idx.search(query, vectors, top_k=2)
        # Closest should be index 0
        assert results[0][0] == 0
        # Scores should be ascending (lower distance is better)
        assert results[0][1] <= results[1][1]


class TestHNSWMasking:
    """Test pre-filtering with boolean bitmask."""

    def test_mask_filters_candidate(self):
        vectors = np.array([
            [1.0, 0.0, 0.0],  # 0: exact match
            [0.0, 1.0, 0.0],  # 1
            [0.8, 0.2, 0.0],  # 2: second best
        ], dtype=np.float32)
        idx = HNSWIndex(metric="cosine", seed=42)
        idx.build(vectors)

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # Mask out 0
        mask = np.array([False, True, True])
        results = idx.search(query, vectors, top_k=2, mask=mask)
        returned = [r[0] for r in results]
        assert 0 not in returned
        assert 2 in returned

    def test_all_masked_returns_empty(self):
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        idx = HNSWIndex(metric="cosine")
        query = np.array([1.0, 0.0], dtype=np.float32)
        mask = np.array([False, False])
        results = idx.search(query, vectors, top_k=2, mask=mask)
        assert results == []

    def test_mask_length_mismatch_raises(self):
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        idx = HNSWIndex()
        query = np.array([1.0, 0.0], dtype=np.float32)
        bad_mask = np.array([True])
        with pytest.raises(IndexingError, match="Mask length"):
            idx.search(query, vectors, top_k=1, mask=bad_mask)


class TestHNSWRecallVsFlat:
    """Verify that HNSW achieves high recall compared to exact FlatIndex."""

    def test_high_recall_on_random_vectors(self):
        np.random.seed(123)
        n = 100
        dim = 16
        vectors = np.random.randn(n, dim).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms

        flat = FlatIndex(metric="cosine")
        hnsw = HNSWIndex(
            metric="cosine", m=16, ef_construction=100, ef_search=50, seed=123
        )
        hnsw.build(vectors)

        top_k = 5
        recalls = []

        for _ in range(10):
            q = np.random.randn(dim).astype(np.float32)
            q = q / np.linalg.norm(q)

            flat_res = flat.search(q, vectors, top_k=top_k)
            hnsw_res = hnsw.search(q, vectors, top_k=top_k)

            flat_ids = set(idx for idx, _ in flat_res)
            hnsw_ids = set(idx for idx, _ in hnsw_res)

            overlap = len(flat_ids & hnsw_ids)
            recalls.append(overlap / top_k)

        avg_recall = np.mean(recalls)
        # HNSW should achieve >= 80% recall on 100 random vectors
        assert avg_recall >= 0.80


class TestHNSWClientIntegration:
    """Test Client SDK configured with index_type='hnsw'."""

    @pytest.fixture
    def mock_embeddings(self):
        with patch("simplev.client.EmbeddingManager") as MockEmb:
            instance = MockEmb.return_value
            instance.dimension = 4
            vec4 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            instance.embed.side_effect = lambda text: vec4
            instance.embed_batch.side_effect = (
                lambda texts: np.stack([vec4 for _ in texts])
            )
            yield instance

    def test_client_with_hnsw(self, mock_embeddings):
        db = Client(index_type="hnsw")
        assert db.info()["index_type"] == "hnsw"

        # Test insert alias and add
        db.insert("d1", "hello world", metadata={"tag": "greet"})
        db.add("d2", "farewell world", metadata={"tag": "bye"})
        assert db.count == 2

        results = db.search("greeting", top_k=2)
        assert len(results) == 2
        assert results[0].doc_id in ("d1", "d2")

    def test_client_info_method(self, tmp_path, mock_embeddings):
        db_path = tmp_path / "test_info.sv"
        db = Client(path=db_path, index_type="hnsw")
        db.insert("doc1", "sample content")
        db.commit()

        info = db.info()
        assert info["count"] == 1
        assert info["active_count"] == 1
        assert info["dimension"] == 4
        assert info["index_type"] == "hnsw"
        assert info["file_exists"] is True
        assert info["file_size_bytes"] > 0
