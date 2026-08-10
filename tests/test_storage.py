"""
Tests for the in-memory storage engine.

Covers adding documents, soft deletion with tombstones,
metadata lookups, compaction, and edge cases.
"""

import numpy as np
import pytest

from simplev.storage import StorageEngine
from simplev.exceptions import StorageError


# helper to make a random vector of the right dimension
def make_vector(dim: int = 384) -> np.ndarray:
    return np.random.randn(dim).astype(np.float32)


class TestStorageInit:
    """Basic initialization checks."""

    def test_creates_with_dimension(self):
        se = StorageEngine(dimension=384)
        assert se.dimension == 384
        assert se.count == 0
        assert se.active_count == 0

    def test_zero_dimension_raises(self):
        with pytest.raises(StorageError):
            StorageEngine(dimension=0)

    def test_negative_dimension_raises(self):
        with pytest.raises(StorageError):
            StorageEngine(dimension=-5)

    def test_repr(self):
        se = StorageEngine(dimension=128)
        r = repr(se)
        assert "dim=128" in r
        assert "docs=0/0" in r


class TestAddDocuments:
    """Test inserting documents into storage."""

    @pytest.fixture
    def storage(self):
        return StorageEngine(dimension=4)

    def test_add_single_doc(self, storage):
        vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        idx = storage.add("doc1", "hello world", vec)
        assert idx == 0
        assert storage.count == 1
        assert storage.active_count == 1

    def test_add_multiple_docs(self, storage):
        for i in range(5):
            vec = make_vector(4)
            storage.add(f"doc_{i}", f"text {i}", vec)
        assert storage.count == 5
        assert storage.active_count == 5

    def test_vectors_are_contiguous(self, storage):
        v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        storage.add("a", "text a", v1)
        storage.add("b", "text b", v2)
        vecs = storage.get_vectors()
        assert vecs.shape == (2, 4)
        np.testing.assert_array_equal(vecs[0], v1)
        np.testing.assert_array_equal(vecs[1], v2)

    def test_duplicate_doc_id_raises(self, storage):
        storage.add("dup", "first", make_vector(4))
        with pytest.raises(StorageError, match="already exists"):
            storage.add("dup", "second", make_vector(4))

    def test_wrong_dimension_raises(self, storage):
        bad_vec = np.array([1.0, 2.0], dtype=np.float32)
        with pytest.raises(StorageError, match="shape mismatch"):
            storage.add("bad", "text", bad_vec)

    def test_auto_converts_to_float32(self, storage):
        vec_f64 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        storage.add("f64", "float64 input", vec_f64)
        stored = storage.get_vectors()
        assert stored.dtype == np.float32

    def test_metadata_stored_correctly(self, storage):
        meta = {"source": "test", "page": 3}
        storage.add("m1", "with metadata", make_vector(4), metadata=meta)
        result = storage.get_metadata("m1")
        assert result["doc_id"] == "m1"
        assert result["text"] == "with metadata"
        assert result["metadata"]["source"] == "test"
        assert result["metadata"]["page"] == 3

    def test_metadata_defaults_to_empty_dict(self, storage):
        storage.add("no_meta", "plain text", make_vector(4))
        result = storage.get_metadata("no_meta")
        assert result["metadata"] == {}


class TestSoftDeletion:
    """Test tombstone-based soft deletion."""

    @pytest.fixture
    def storage_with_data(self):
        se = StorageEngine(dimension=4)
        for i in range(5):
            se.add(f"doc_{i}", f"text {i}", make_vector(4))
        return se

    def test_delete_marks_tombstone(self, storage_with_data):
        result = storage_with_data.mark_deleted("doc_2")
        assert result is True
        assert storage_with_data.active_count == 4
        assert storage_with_data.count == 5  # total unchanged

    def test_mask_reflects_deletion(self, storage_with_data):
        storage_with_data.mark_deleted("doc_1")
        mask = storage_with_data.get_active_mask()
        assert mask[0] == True   # doc_0
        assert mask[1] == False  # doc_1 deleted
        assert mask[2] == True   # doc_2

    def test_delete_nonexistent_raises(self, storage_with_data):
        with pytest.raises(StorageError, match="not found"):
            storage_with_data.mark_deleted("nope")

    def test_double_delete_returns_false(self, storage_with_data):
        storage_with_data.mark_deleted("doc_3")
        result = storage_with_data.mark_deleted("doc_3")
        assert result is False

    def test_is_active_after_delete(self, storage_with_data):
        assert storage_with_data.is_active("doc_0") is True
        storage_with_data.mark_deleted("doc_0")
        assert storage_with_data.is_active("doc_0") is False

    def test_has_document_still_true_after_delete(self, storage_with_data):
        storage_with_data.mark_deleted("doc_4")
        # doc still exists, just tombstoned
        assert storage_with_data.has_document("doc_4") is True


class TestCompaction:
    """Test that compaction actually removes tombstoned records."""

    def test_compact_removes_deleted(self):
        se = StorageEngine(dimension=4)
        se.add("keep_0", "t0", make_vector(4))
        se.add("delete_me", "t1", make_vector(4))
        se.add("keep_1", "t2", make_vector(4))

        se.mark_deleted("delete_me")
        removed = se.compact()

        assert removed == 1
        assert se.count == 2
        assert se.active_count == 2
        assert not se.has_document("delete_me")
        assert se.has_document("keep_0")
        assert se.has_document("keep_1")

    def test_compact_reindexes_correctly(self):
        se = StorageEngine(dimension=4)
        v0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v1 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

        se.add("a", "text a", v0)
        se.add("b", "text b", v1)  # will delete
        se.add("c", "text c", v2)

        se.mark_deleted("b")
        se.compact()

        # vectors should now be [v0, v2]
        vecs = se.get_vectors()
        assert vecs.shape == (2, 4)
        np.testing.assert_array_equal(vecs[0], v0)
        np.testing.assert_array_equal(vecs[1], v2)

        # metadata should be accessible with correct doc_ids
        meta_a = se.get_metadata("a")
        assert meta_a["text"] == "text a"
        meta_c = se.get_metadata("c")
        assert meta_c["text"] == "text c"

    def test_compact_nothing_to_remove(self):
        se = StorageEngine(dimension=4)
        se.add("x", "text", make_vector(4))
        removed = se.compact()
        assert removed == 0

    def test_compact_empty_storage(self):
        se = StorageEngine(dimension=4)
        removed = se.compact()
        assert removed == 0

    def test_compact_all_deleted(self):
        se = StorageEngine(dimension=4)
        se.add("a", "t", make_vector(4))
        se.add("b", "t", make_vector(4))
        se.mark_deleted("a")
        se.mark_deleted("b")
        removed = se.compact()
        assert removed == 2
        assert se.count == 0
        assert se.get_vectors() is None


class TestClear:
    """Test the nuclear option."""

    def test_clear_wipes_everything(self):
        se = StorageEngine(dimension=4)
        for i in range(10):
            se.add(f"d{i}", f"t{i}", make_vector(4))
        se.clear()
        assert se.count == 0
        assert se.active_count == 0
        assert se.get_vectors() is None
        assert se.get_active_mask() is None
