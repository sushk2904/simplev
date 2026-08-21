"""
Tests for the Client SDK.

We mock the embedding manager here so tests run fast without
needing the actual sentence-transformers model. The Client is
a thin facade so we're mainly testing that it wires things
together correctly and that save/load works end-to-end.
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from simplev.client import Client
from simplev.exceptions import SimpleVError, StorageError


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


class TestClientInit:
    """Test client creation."""

    def test_create_in_memory(self, mock_embeddings):
        db = Client()
        assert db.count == 0
        assert db.dimension is None  # not initialized until first add

    def test_repr(self, mock_embeddings):
        db = Client()
        r = repr(db)
        assert "docs=0" in r
        assert "in-memory" in r


class TestClientAdd:
    """Test adding documents."""

    def test_add_single(self, mock_embeddings):
        db = Client()
        result = db.add("d1", "hello world")
        assert result == "d1"
        assert db.count == 1

    def test_add_with_metadata(self, mock_embeddings):
        db = Client()
        db.add("d1", "text", metadata={"key": "val"})
        assert db.count == 1

    def test_add_with_precomputed_vector(self, mock_embeddings):
        db = Client()
        vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        db.add("d1", "text", vector=vec)
        assert db.count == 1
        # embed() should NOT have been called since we gave a vector
        mock_embeddings.embed.assert_not_called()

    def test_add_duplicate_raises(self, mock_embeddings):
        db = Client()
        db.add("dup", "first")
        with pytest.raises(StorageError, match="already exists"):
            db.add("dup", "second")

    def test_add_many(self, mock_embeddings):
        db = Client()
        docs = [
            {"doc_id": "a", "text": "alpha"},
            {"doc_id": "b", "text": "beta"},
            {"doc_id": "c", "text": "gamma"},
        ]
        ids = db.add_many(docs)
        assert ids == ["a", "b", "c"]
        assert db.count == 3

    def test_add_many_empty_list(self, mock_embeddings):
        db = Client()
        ids = db.add_many([])
        assert ids == []

    def test_add_many_bad_format_raises(self, mock_embeddings):
        db = Client()
        with pytest.raises(SimpleVError, match="doc_id"):
            db.add_many([{"text": "missing doc_id"}])


class TestClientSearch:
    """Test search through the client."""

    def test_search_empty_db(self, mock_embeddings):
        db = Client()
        results = db.search("anything")
        assert results == []

    def test_search_returns_results(self, mock_embeddings):
        db = Client()
        db.add("d1", "hello")
        db.add("d2", "world")
        results = db.search("test", top_k=2)
        # should get something back (exact order depends on random vectors)
        assert len(results) <= 2

    def test_search_with_filters(self, mock_embeddings):
        db = Client()
        db.add("d1", "cat", metadata={"type": "animal"})
        db.add("d2", "car", metadata={"type": "vehicle"})

        results = db.search("test", filters={"type": "animal"})
        doc_ids = [r.doc_id for r in results]
        assert "d2" not in doc_ids


class TestClientDelete:
    """Test deletion through the client."""

    def test_delete_existing(self, mock_embeddings):
        db = Client()
        db.add("d1", "text")
        result = db.delete("d1")
        assert result is True
        assert db.count == 0  # active count

    def test_delete_nonexistent_raises(self, mock_embeddings):
        db = Client()
        with pytest.raises(StorageError, match="not found"):
            db.delete("nope")

    def test_compact(self, mock_embeddings):
        db = Client()
        db.add("d1", "text")
        db.delete("d1")
        removed = db.compact()
        assert removed == 1


class TestClientPersistence:
    """Test save/load through the client."""

    def test_save_and_load(self, tmp_path, mock_embeddings):
        path = tmp_path / "test_db.sv"

        # create and save
        db = Client()
        db.add("d1", "hello world", metadata={"page": 1})
        db.add("d2", "foo bar", metadata={"page": 2})
        db.save(path)

        assert path.exists()

        # load into a new client
        db2 = Client(path=path)
        assert db2.count == 2

    def test_save_no_path_raises(self, mock_embeddings):
        db = Client()  # no path given
        db.add("d1", "text")
        with pytest.raises(SimpleVError, match="No save path"):
            db.save()

    def test_save_empty_raises(self, mock_embeddings):
        db = Client()
        with pytest.raises(SimpleVError, match="empty"):
            db.save(Path("doesnt_matter.sv"))

    def test_len(self, mock_embeddings):
        db = Client()
        assert len(db) == 0
        db.add("d1", "text")
        assert len(db) == 1
