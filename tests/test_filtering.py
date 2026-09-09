"""
Tests for rich metadata filtering operators in QueryEngine.

Supports:
- Scalar equality: {"category": "news"}
- Explicit $eq: {"category": {"$eq": "news"}}
- Inequality $ne: {"status": {"$ne": "archived"}}
- Comparisons: $gt, $gte, $lt, $lte on numeric values
- Set membership: $in, $nin on list of values
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
        instance.embed.side_effect = lambda text: make_vector(DIM)
        instance.embed_batch.side_effect = (
            lambda texts: np.stack([make_vector(DIM) for _ in texts])
        )
        yield instance


class TestMetadataFiltering:
    """Test rich metadata filter operators."""

    @pytest.fixture(autouse=True)
    def setup_db(self, mock_embeddings):
        self.db = Client()
        docs = [
            {
                "doc_id": "item1",
                "text": "first article about python",
                "metadata": {
                    "price": 10,
                    "tag": "tech",
                    "rating": 4.5,
                    "status": "active",
                },
            },
            {
                "doc_id": "item2",
                "text": "second article about gardening",
                "metadata": {
                    "price": 25,
                    "tag": "home",
                    "rating": 3.8,
                    "status": "active",
                },
            },
            {
                "doc_id": "item3",
                "text": "third article about machine learning",
                "metadata": {
                    "price": 50,
                    "tag": "tech",
                    "rating": 4.9,
                    "status": "archived",
                },
            },
            {
                "doc_id": "item4",
                "text": "fourth article about cooking recipes",
                "metadata": {
                    "price": 100,
                    "tag": "food",
                    "rating": 2.5,
                    "status": "pending",
                },
            },
        ]
        self.db.add_many(docs)

    def test_scalar_equality(self):
        results = self.db.search("article", filters={"tag": "tech"})
        ids = {r.doc_id for r in results}
        assert ids == {"item1", "item3"}

    def test_explicit_eq(self):
        results = self.db.search("article", filters={"tag": {"$eq": "home"}})
        ids = {r.doc_id for r in results}
        assert ids == {"item2"}

    def test_ne_operator(self):
        results = self.db.search("article", filters={"status": {"$ne": "active"}})
        ids = {r.doc_id for r in results}
        assert ids == {"item3", "item4"}

    def test_gt_and_gte_operators(self):
        # price > 25 -> item3 (50), item4 (100)
        res_gt = self.db.search("article", filters={"price": {"$gt": 25}})
        assert {r.doc_id for r in res_gt} == {"item3", "item4"}

        # price >= 25 -> item2 (25), item3 (50), item4 (100)
        res_gte = self.db.search("article", filters={"price": {"$gte": 25}})
        assert {r.doc_id for r in res_gte} == {"item2", "item3", "item4"}

    def test_lt_and_lte_operators(self):
        # price < 50 -> item1 (10), item2 (25)
        res_lt = self.db.search("article", filters={"price": {"$lt": 50}})
        assert {r.doc_id for r in res_lt} == {"item1", "item2"}

        # price <= 50 -> item1 (10), item2 (25), item3 (50)
        res_lte = self.db.search("article", filters={"price": {"$lte": 50}})
        assert {r.doc_id for r in res_lte} == {"item1", "item2", "item3"}

    def test_in_operator(self):
        results = self.db.search(
            "article", filters={"tag": {"$in": ["tech", "food"]}}
        )
        assert {r.doc_id for r in results} == {"item1", "item3", "item4"}

    def test_nin_operator(self):
        results = self.db.search(
            "article", filters={"tag": {"$nin": ["tech", "food"]}}
        )
        assert {r.doc_id for r in results} == {"item2"}

    def test_combined_operators(self):
        # tech tag and price > 20 -> item3
        results = self.db.search(
            "article",
            filters={
                "tag": "tech",
                "price": {"$gt": 20},
            },
        )
        assert {r.doc_id for r in results} == {"item3"}

    def test_missing_metadata_key_filtered_out(self):
        results = self.db.search("article", filters={"nonexistent": "value"})
        assert len(results) == 0

    def test_range_filter(self):
        # rating between 3.0 and 4.6 -> item1 (4.5), item2 (3.8)
        results = self.db.search(
            "article",
            filters={
                "rating": {"$gte": 3.0, "$lte": 4.6},
            },
        )
        assert {r.doc_id for r in results} == {"item1", "item2"}
