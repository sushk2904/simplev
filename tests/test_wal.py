"""
Tests for the Write-Ahead Log.

Covers the JSONL format, append behavior, crash recovery
replay, truncation, and corrupted entry handling.
"""

import json
import numpy as np
import pytest
from pathlib import Path

from simplev.wal import WriteAheadLog, WALEntry


class TestWALEntry:
    """Test the WAL entry serialization."""

    def test_insert_to_json(self):
        entry = WALEntry(
            operation="insert",
            doc_id="d1",
            text="hello world",
            vector=[1.0, 2.0, 3.0],
            metadata={"key": "val"},
        )
        line = entry.to_json()
        data = json.loads(line)
        assert data["op"] == "insert"
        assert data["doc_id"] == "d1"
        assert data["text"] == "hello world"
        assert data["vector"] == [1.0, 2.0, 3.0]
        assert data["metadata"]["key"] == "val"

    def test_delete_to_json(self):
        entry = WALEntry(operation="delete", doc_id="d2")
        line = entry.to_json()
        data = json.loads(line)
        assert data["op"] == "delete"
        assert data["doc_id"] == "d2"
        assert "text" not in data
        assert "vector" not in data

    def test_round_trip(self):
        original = WALEntry(
            operation="insert",
            doc_id="abc",
            text="test text",
            vector=[0.1, 0.2],
            metadata={"page": 3},
        )
        line = original.to_json()
        restored = WALEntry.from_json(line)
        assert restored.operation == "insert"
        assert restored.doc_id == "abc"
        assert restored.text == "test text"
        assert restored.vector == [0.1, 0.2]
        assert restored.metadata["page"] == 3

    def test_repr(self):
        entry = WALEntry(operation="insert", doc_id="x")
        r = repr(entry)
        assert "insert" in r
        assert "x" in r


class TestWriteAheadLog:
    """Test the WAL file operations."""

    def test_log_insert(self, tmp_path):
        wal = WriteAheadLog(tmp_path / "test.wal")
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        wal.log_insert("d1", "hello", vec, metadata={"k": "v"})
        wal.close()

        # read back
        entries = wal.read_entries()
        assert len(entries) == 1
        assert entries[0].operation == "insert"
        assert entries[0].doc_id == "d1"
        assert entries[0].text == "hello"

    def test_log_delete(self, tmp_path):
        wal = WriteAheadLog(tmp_path / "test.wal")
        wal.log_delete("d1")
        wal.close()

        entries = wal.read_entries()
        assert len(entries) == 1
        assert entries[0].operation == "delete"

    def test_multiple_entries(self, tmp_path):
        wal = WriteAheadLog(tmp_path / "test.wal")
        vec = np.array([1.0, 0.0], dtype=np.float32)

        wal.log_insert("a", "text a", vec)
        wal.log_insert("b", "text b", vec)
        wal.log_delete("a")
        wal.log_insert("c", "text c", vec)
        wal.close()

        entries = wal.read_entries()
        assert len(entries) == 4
        assert entries[0].doc_id == "a"
        assert entries[1].doc_id == "b"
        assert entries[2].operation == "delete"
        assert entries[3].doc_id == "c"

    def test_truncate_clears_entries(self, tmp_path):
        wal = WriteAheadLog(tmp_path / "test.wal")
        wal.log_insert("d1", "text", np.array([1.0], dtype=np.float32))
        wal.close()

        assert wal.has_entries()

        wal.truncate()
        assert not wal.has_entries()

        entries = wal.read_entries()
        assert entries == []

    def test_has_entries_empty_file(self, tmp_path):
        wal_path = tmp_path / "empty.wal"
        wal_path.touch()
        wal = WriteAheadLog(wal_path)
        assert not wal.has_entries()

    def test_has_entries_no_file(self, tmp_path):
        wal = WriteAheadLog(tmp_path / "missing.wal")
        assert not wal.has_entries()

    def test_read_entries_no_file(self, tmp_path):
        wal = WriteAheadLog(tmp_path / "missing.wal")
        entries = wal.read_entries()
        assert entries == []

    def test_corrupted_entry_is_skipped(self, tmp_path):
        wal_path = tmp_path / "corrupt.wal"
        with open(wal_path, "w") as f:
            # good entry
            entry = WALEntry("insert", "d1", "text", [1.0])
            f.write(entry.to_json() + "\n")
            # corrupted line
            f.write("this is not valid json{{{}\n")
            # another good entry
            entry2 = WALEntry("delete", "d2")
            f.write(entry2.to_json() + "\n")

        wal = WriteAheadLog(wal_path)
        entries = wal.read_entries()
        # should get 2 entries, skipping the corrupted one
        assert len(entries) == 2
        assert entries[0].doc_id == "d1"
        assert entries[1].doc_id == "d2"

    def test_append_after_reopen(self, tmp_path):
        wal = WriteAheadLog(tmp_path / "test.wal")
        wal.log_insert("d1", "first", np.array([1.0], dtype=np.float32))
        wal.close()

        # reopen and append more
        wal2 = WriteAheadLog(tmp_path / "test.wal")
        wal2.log_insert("d2", "second", np.array([2.0], dtype=np.float32))
        wal2.close()

        entries = wal2.read_entries()
        assert len(entries) == 2

    def test_vector_roundtrip_accuracy(self, tmp_path):
        wal = WriteAheadLog(tmp_path / "test.wal")
        original_vec = np.array([1.5, -0.3, 0.0, 99.9], dtype=np.float32)
        wal.log_insert("d1", "text", original_vec)
        wal.close()

        entries = wal.read_entries()
        recovered = np.array(entries[0].vector, dtype=np.float32)
        np.testing.assert_array_almost_equal(recovered, original_vec, decimal=5)
