"""
Tests for hybrid search (dense semantic + BM25 sparse lexical with RRF fusion).
"""

from unittest.mock import patch

import numpy as np
import pytest

from simplev.client import Client

DIM = 4


def make_vector(dim: int = DIM) -> np.ndarray:
    return np.random.randn(dim).astype(np.float32)


@pytest.fixture
def mock_embeddings():
    """Patch EmbeddingManager so we don't load a real model."""
    with patch("simplev.client.EmbeddingManager") as MockEmb:
        instance = MockEmb.return_value
        instance.dimension = DIM
        # Stable mock embedding mapping
        instance.embed.side_effect = lambda text: np.array(
            [float(len(text) % 5), 1.0, 0.0, 2.0], dtype=np.float32
        )
        instance.embed_batch.side_effect = lambda texts: np.stack(
            [
                np.array([float(len(t) % 5), 1.0, 0.0, 2.0], dtype=np.float32)
                for t in texts
            ]
        )
        yield instance


class TestHybridSearch:
    """Test suite for hybrid search combining BM25 and vector search."""

    @pytest.fixture(autouse=True)
    def setup_corpus(self, mock_embeddings):
        self.db = Client()
        self.db.add(
            "doc_ml",
            "Deep neural networks and machine learning architectures",
            metadata={"category": "ai"},
        )
        self.db.add(
            "doc_py",
            "Python programming language guide and syntax reference",
            metadata={"category": "dev"},
        )
        self.db.add(
            "doc_db",
            "Relational databases, indexing, and SQL query optimization",
            metadata={"category": "database"},
        )
        self.db.add(
            "doc_vec",
            "Vector database and similarity search algorithms in Python",
            metadata={"category": "ai"},
        )

    def test_hybrid_search_basic(self):
        results = self.db.hybrid_search("Python vector similarity", top_k=2)
        assert len(results) == 2
        res_ids = {r.doc_id for r in results}
        assert "doc_vec" in res_ids
        assert "doc_py" in res_ids

    def test_hybrid_search_with_alpha_weights(self):
        # alpha=0.0 (pure BM25)
        res_bm25 = self.db.hybrid_search("syntax reference", alpha=0.0, top_k=1)
        assert len(res_bm25) == 1
        assert res_bm25[0].doc_id == "doc_py"

        # alpha=1.0 (pure dense rank fusion)
        res_dense = self.db.hybrid_search("machine learning", alpha=1.0, top_k=1)
        assert len(res_dense) == 1

    def test_hybrid_search_with_filter(self):
        # Search Python across AI category only
        results = self.db.hybrid_search(
            "Python",
            filters={"category": "ai"},
            top_k=5,
        )
        ids = [r.doc_id for r in results]
        assert "doc_vec" in ids
        assert "doc_py" not in ids  # filtered out by category

    def test_hybrid_search_after_delete_and_compact(self):
        self.db.delete("doc_vec")
        results = self.db.hybrid_search("vector database", top_k=5)
        ids = [r.doc_id for r in results]
        assert "doc_vec" not in ids

        self.db.compact()
        results2 = self.db.hybrid_search("neural networks", top_k=1)
        assert results2[0].doc_id == "doc_ml"

    def test_empty_db_hybrid_search(self, mock_embeddings):
        empty_db = Client()
        assert empty_db.hybrid_search("anything") == []
