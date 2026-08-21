"""
Tests for the .sv file persistence layer.

Tests saving and loading the binary format, header validation,
tombstone bitmap packing, and round-trip consistency.
"""

import json
import struct
import numpy as np
import pytest
from pathlib import Path

from simplev.persistence import (
    FileManager, MAGIC, FORMAT_VERSION,
    HEADER_SIZE, ENDIAN_MARKER,
)
from simplev.storage import StorageEngine
from simplev.exceptions import StorageError


def make_vector(dim: int = 4) -> np.ndarray:
    return np.random.randn(dim).astype(np.float32)


def build_storage(n_docs: int = 5, dim: int = 4) -> StorageEngine:
    """Build a storage engine with some test data."""
    se = StorageEngine(dimension=dim)
    for i in range(n_docs):
        se.add(
            f"doc_{i}",
            f"text for document {i}",
            make_vector(dim),
            metadata={"index": i, "tag": f"tag_{i}"},
        )
    return se


class TestSaveAndLoad:
    """Test the full save/load round trip."""

    def test_basic_round_trip(self, tmp_path):
        original = build_storage(n_docs=3, dim=4)
        path = tmp_path / "test.sv"

        fm = FileManager()
        fm.save(path, original)

        assert path.exists()
        assert path.stat().st_size > HEADER_SIZE

        loaded = fm.load(path)
        assert loaded.count == 3
        assert loaded.dimension == 4
        assert loaded.active_count == 3

    def test_vectors_survive_round_trip(self, tmp_path):
        se = StorageEngine(dimension=3)
        v1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        v2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        se.add("a", "text a", v1)
        se.add("b", "text b", v2)

        path = tmp_path / "vecs.sv"
        fm = FileManager()
        fm.save(path, se)
        loaded = fm.load(path)

        vecs = loaded.get_vectors()
        np.testing.assert_array_almost_equal(vecs[0], v1)
        np.testing.assert_array_almost_equal(vecs[1], v2)

    def test_metadata_survives_round_trip(self, tmp_path):
        se = StorageEngine(dimension=4)
        meta = {"author": "alice", "pages": 42, "tags": ["a", "b"]}
        se.add("m1", "metadata test", make_vector(4), metadata=meta)

        path = tmp_path / "meta.sv"
        fm = FileManager()
        fm.save(path, se)
        loaded = fm.load(path)

        result = loaded.get_metadata("m1")
        assert result["text"] == "metadata test"
        assert result["metadata"]["author"] == "alice"
        assert result["metadata"]["pages"] == 42
        assert result["metadata"]["tags"] == ["a", "b"]

    def test_tombstones_survive_round_trip(self, tmp_path):
        se = build_storage(n_docs=5)
        se.mark_deleted("doc_1")
        se.mark_deleted("doc_3")

        path = tmp_path / "tomb.sv"
        fm = FileManager()
        fm.save(path, se)
        loaded = fm.load(path)

        mask = loaded.get_active_mask()
        assert mask[0] == True   # doc_0 active
        assert mask[1] == False  # doc_1 deleted
        assert mask[2] == True   # doc_2 active
        assert mask[3] == False  # doc_3 deleted
        assert mask[4] == True   # doc_4 active
        assert loaded.active_count == 3

    def test_doc_ids_preserved(self, tmp_path):
        se = build_storage(n_docs=3)
        path = tmp_path / "ids.sv"

        fm = FileManager()
        fm.save(path, se)
        loaded = fm.load(path)

        assert loaded.has_document("doc_0")
        assert loaded.has_document("doc_1")
        assert loaded.has_document("doc_2")

    def test_large_dataset_round_trip(self, tmp_path):
        """Make sure it works with more docs than fit in one bitmap byte."""
        se = StorageEngine(dimension=4)
        for i in range(100):
            se.add(f"d{i}", f"text {i}", make_vector(4))

        # delete some scattered across byte boundaries
        se.mark_deleted("d7")
        se.mark_deleted("d8")
        se.mark_deleted("d15")
        se.mark_deleted("d63")
        se.mark_deleted("d99")

        path = tmp_path / "large.sv"
        fm = FileManager()
        fm.save(path, se)
        loaded = fm.load(path)

        assert loaded.count == 100
        assert loaded.active_count == 95
        assert not loaded.is_active("d7")
        assert not loaded.is_active("d63")
        assert loaded.is_active("d50")


class TestSaveEdgeCases:
    """Test save error handling."""

    def test_save_empty_raises(self, tmp_path):
        se = StorageEngine(dimension=4)
        fm = FileManager()
        with pytest.raises(StorageError, match="empty"):
            fm.save(tmp_path / "empty.sv", se)


class TestLoadValidation:
    """Test that load rejects bad files."""

    def test_load_nonexistent_raises(self, tmp_path):
        fm = FileManager()
        with pytest.raises(StorageError, match="not found"):
            fm.load(tmp_path / "nope.sv")

    def test_load_wrong_magic_raises(self, tmp_path):
        path = tmp_path / "bad_magic.sv"
        with open(path, "wb") as f:
            f.write(b"NOPE" + b"\x00" * 60)

        fm = FileManager()
        with pytest.raises(StorageError, match="magic bytes"):
            fm.load(path)

    def test_load_truncated_header_raises(self, tmp_path):
        path = tmp_path / "short.sv"
        with open(path, "wb") as f:
            f.write(b"SV01")  # only 4 bytes, need 64

        fm = FileManager()
        with pytest.raises(StorageError, match="too small"):
            fm.load(path)

    def test_load_bad_version_raises(self, tmp_path):
        path = tmp_path / "bad_ver.sv"
        header = bytearray(HEADER_SIZE)
        header[0:4] = MAGIC
        struct.pack_into("<I", header, 4, ENDIAN_MARKER)
        struct.pack_into("<H", header, 8, 99)  # fake version
        struct.pack_into("<I", header, 12, 4)   # dimension
        struct.pack_into("<I", header, 16, 1)   # count

        with open(path, "wb") as f:
            f.write(header)

        fm = FileManager()
        with pytest.raises(StorageError, match="version"):
            fm.load(path)


class TestTombstoneBitmap:
    """Test the bitmap packing/unpacking directly."""

    def test_pack_unpack_all_active(self):
        fm = FileManager()
        mask = np.ones(8, dtype=bool)
        packed = fm._pack_tombstones(mask, 8)
        assert packed == bytes([0xFF])
        unpacked = fm._unpack_tombstones(packed, 8)
        np.testing.assert_array_equal(mask, unpacked)

    def test_pack_unpack_all_deleted(self):
        fm = FileManager()
        mask = np.zeros(8, dtype=bool)
        packed = fm._pack_tombstones(mask, 8)
        assert packed == bytes([0x00])
        unpacked = fm._unpack_tombstones(packed, 8)
        np.testing.assert_array_equal(mask, unpacked)

    def test_pack_unpack_mixed(self):
        fm = FileManager()
        # active at positions 0, 2, 5
        mask = np.array([True, False, True, False, False, True, False, False])
        packed = fm._pack_tombstones(mask, 8)
        unpacked = fm._unpack_tombstones(packed, 8)
        np.testing.assert_array_equal(mask, unpacked)

    def test_pack_unpack_non_byte_aligned(self):
        """Test when doc count isn't a multiple of 8."""
        fm = FileManager()
        mask = np.array([True, False, True, True, False])  # 5 docs
        packed = fm._pack_tombstones(mask, 5)
        assert len(packed) == 1  # ceil(5/8) = 1 byte
        unpacked = fm._unpack_tombstones(packed, 5)
        np.testing.assert_array_equal(mask, unpacked)
