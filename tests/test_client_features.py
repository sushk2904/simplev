"""
Tests for newly added Client features:
- Primary key document retrieval (get, db[doc_id])
- In-place document updates and upserts
- Pythonic collection protocols (__contains__, __len__, __iter__)
- Batch queries (search_batch)
- HNSW index synchronization across compactions
- WAL crash recovery for update operations
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from simplev.client import Client
from simplev.exceptions import SimpleVError

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


class TestClientGetAndDunders:
    """Test get() and collection dunder protocols."""

    def test_get_existing(self, mock_embeddings):
        db = Client()
        db.add("d1", "hello world", metadata={"author": "alice"})

        record = db.get("d1")
        assert record is not None
        assert record["doc_id"] == "d1"
        assert record["text"] == "hello world"
        assert record["metadata"] == {"author": "alice"}
        assert "vector" not in record

    def test_get_with_vector(self, mock_embeddings):
        db = Client()
        db.add("d1", "hello world")
        record = db.get("d1", include_vector=True)
        assert record is not None
        assert "vector" in record
        assert isinstance(record["vector"], np.ndarray)
        assert record["vector"].shape == (DIM,)

    def test_get_missing_and_deleted(self, mock_embeddings):
        db = Client()
        assert db.get("nonexistent") is None

        db.add("d1", "text")
        db.delete("d1")
        assert db.get("d1") is None

    def test_getitem_dunder(self, mock_embeddings):
        db = Client()
        db.add("d1", "text content", metadata={"lang": "en"})

        rec = db["d1"]
        assert rec["doc_id"] == "d1"
        assert rec["text"] == "text content"

        with pytest.raises(KeyError):
            _ = db["missing"]

    def test_contains_dunder(self, mock_embeddings):
        db = Client()
        assert "d1" not in db

        db.add("d1", "text")
        assert "d1" in db

        db.delete("d1")
        assert "d1" not in db

    def test_len_and_iter_dunders(self, mock_embeddings):
        db = Client()
        assert len(db) == 0
        assert list(db) == []

        db.add("d1", "text 1")
        db.add("d2", "text 2")
        db.add("d3", "text 3")
        assert len(db) == 3

        records = list(db)
        assert len(records) == 3
        ids = [r["doc_id"] for r in records]
        assert ids == ["d1", "d2", "d3"]

        db.delete("d2")
        assert len(db) == 2
        assert [r["doc_id"] for r in list(db)] == ["d1", "d3"]


class TestClientUpdateAndUpsert:
    """Test updating and upserting documents."""

    def test_update_text_and_reembed(self, mock_embeddings):
        db = Client()
        db.add("d1", "original text", metadata={"ver": 1})

        db.update("d1", text="updated text")
        rec = db.get("d1")
        assert rec["text"] == "updated text"
        assert rec["metadata"] == {"ver": 1}

    def test_update_metadata_only(self, mock_embeddings):
        db = Client()
        db.add("d1", "stable text", metadata={"v": 1})

        mock_embeddings.embed.reset_mock()
        db.update("d1", metadata={"v": 2, "extra": True})

        # Should not re-embed when text is unchanged
        mock_embeddings.embed.assert_not_called()
        rec = db.get("d1")
        assert rec["text"] == "stable text"
        assert rec["metadata"] == {"v": 2, "extra": True}

    def test_update_with_custom_vector(self, mock_embeddings):
        db = Client()
        db.add("d1", "text")
        custom_vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        db.update("d1", vector=custom_vec)
        rec = db.get("d1", include_vector=True)
        np.testing.assert_array_almost_equal(rec["vector"], custom_vec)

    def test_update_missing_or_deleted_raises(self, mock_embeddings):
        db = Client()
        with pytest.raises(SimpleVError):
            db.update("missing", text="hello")

        db.add("d1", "text")
        db.delete("d1")
        with pytest.raises(SimpleVError):
            db.update("d1", text="new text")

    def test_upsert_new_and_existing(self, mock_embeddings):
        db = Client()
        # Upsert brand new
        db.upsert("d1", "new content", metadata={"a": 1})
        assert db.get("d1")["text"] == "new content"

        # Upsert existing active
        db.upsert("d1", "modified content", metadata={"a": 2})
        assert db.get("d1")["text"] == "modified content"
        assert db.get("d1")["metadata"] == {"a": 2}

        # Upsert tombstoned document (reactivates it)
        db.delete("d1")
        assert "d1" not in db

        db.upsert("d1", "revived content", metadata={"a": 3})
        assert "d1" in db
        assert db.get("d1")["text"] == "revived content"
        assert db.get("d1")["metadata"] == {"a": 3}


class TestClientBatchSearch:
    """Test batch search functionality."""

    def test_search_batch_empty_db(self, mock_embeddings):
        db = Client()
        res = db.search_batch(["query 1", "query 2"])
        assert res == [[], []]

    def test_search_batch_results(self, mock_embeddings):
        db = Client()
        db.add("d1", "first document")
        db.add("d2", "second document")

        batch_results = db.search_batch(["doc 1", "doc 2"], top_k=2)
        assert len(batch_results) == 2
        for r_list in batch_results:
            assert len(r_list) == 2


class TestHNSWCompactionSync:
    """Verify HNSW index stays synchronized when documents are compacted."""

    def test_hnsw_compaction_recovery(self, mock_embeddings):
        db = Client(index_type="hnsw")
        # Add 5 documents
        for i in range(5):
            db.add(f"doc_{i}", f"content of doc {i}")

        assert db.count == 5

        # Delete doc_1 and doc_3
        db.delete("doc_1")
        db.delete("doc_3")
        assert db.count == 3

        # Search should only return remaining active documents
        res_before = db.search("content", top_k=5)
        before_ids = {r.doc_id for r in res_before}
        assert "doc_1" not in before_ids
        assert "doc_3" not in before_ids

        # Compact database (alters internal vector offsets)
        removed = db.compact()
        assert removed == 2
        assert db.count == 3

        # Search after compaction should cleanly find documents without index error
        res_after = db.search("content", top_k=5)
        after_ids = {r.doc_id for r in res_after}
        assert after_ids == {"doc_0", "doc_2", "doc_4"}


class TestWALUpdateRecovery:
    """Test that WAL correctly logs and recovers update operations."""

    def test_wal_update_recovery(self, tmp_path, mock_embeddings):
        db_path = tmp_path / "test_wal_update.sv"

        # 1. Initialize and insert
        db = Client(path=db_path, use_wal=True)
        db.add("d1", "original text", metadata={"v": 1})
        db.commit()

        # 2. Update without committing (simulates crash)
        db.update("d1", text="crashed updated text", metadata={"v": 2})

        # Ensure WAL contains the update
        wal_path = Path(str(db_path) + ".wal")
        assert wal_path.exists()
        assert wal_path.stat().st_size > 0

        # 3. Simulate process restart by loading a new Client instance
        db2 = Client(path=db_path, use_wal=True)
        rec = db2.get("d1")
        assert rec is not None
        assert rec["text"] == "crashed updated text"
        assert rec["metadata"] == {"v": 2}
