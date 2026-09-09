"""
Write-Ahead Log for SimpleV.

Every insert or delete is written to a .wal file (JSONL format)
before the in-memory state is touched. If the process crashes,
we replay the log on next startup to recover any operations
that didn't make it into the .sv file.

The lifecycle is:
  1. User calls add() or delete()
  2. We append a JSONL entry to the .wal file
  3. The in-memory storage is updated
  4. On commit(), we save the .sv file and truncate the .wal

The .wal file lives next to the .sv file:
  my_database.sv      <- main data file
  my_database.sv.wal  <- the log
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class WALEntry:
    """One operation in the write-ahead log.

    Each entry is a single line in the JSONL file.
    """
    INSERT = "insert"
    DELETE = "delete"
    UPDATE = "update"

    def __init__(
        self,
        operation: str,
        doc_id: str,
        text: Optional[str] = None,
        vector: Optional[list] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self.operation = operation
        self.doc_id = doc_id
        self.text = text
        self.vector = vector
        self.metadata = metadata

    def to_json(self) -> str:
        """Serialize to a JSON string (one line, no newline at end)."""
        data = {
            "op": self.operation,
            "doc_id": self.doc_id,
        }
        if self.text is not None:
            data["text"] = self.text
        if self.vector is not None:
            data["vector"] = self.vector
        if self.metadata is not None:
            data["metadata"] = self.metadata

        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "WALEntry":
        """Deserialize from a JSON string."""
        data = json.loads(line.strip())
        return cls(
            operation=data["op"],
            doc_id=data["doc_id"],
            text=data.get("text"),
            vector=data.get("vector"),
            metadata=data.get("metadata"),
        )

    def __repr__(self) -> str:
        return f"WALEntry(op={self.operation}, doc_id={self.doc_id})"


class WriteAheadLog:
    """Append-only JSONL log for crash recovery.

    Sits between the client and the storage engine. Logs
    every mutation to disk before it hits memory.

    Args:
        wal_path: Path to the .wal file. Usually derived from
            the .sv path by appending '.wal'.
    """

    def __init__(self, wal_path: Union[str, Path]) -> None:
        self._path = Path(wal_path)
        self._file = None

        # make sure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def open(self) -> None:
        """Open the WAL file for appending.

        Creates the file if it doesn't exist. Opening in append
        mode means we never overwrite existing entries.
        """
        if self._file is not None:
            return
        self._file = open(self._path, "a", encoding="utf-8")
        logger.debug(f"WAL opened: {self._path}")

    def close(self) -> None:
        """Close the WAL file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def log_insert(
        self,
        doc_id: str,
        text: str,
        vector: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log an insert operation.

        The vector is converted to a plain list for JSON serialization.
        This is intentional -- the WAL is a recovery mechanism, not a
        performance-critical path, so readability matters more than
        binary packing here.
        """
        self._ensure_open()

        entry = WALEntry(
            operation=WALEntry.INSERT,
            doc_id=doc_id,
            text=text,
            vector=vector.tolist(),
            metadata=metadata,
        )

        self._write_entry(entry)

    def log_delete(self, doc_id: str) -> None:
        """Log a delete operation."""
        self._ensure_open()

        entry = WALEntry(
            operation=WALEntry.DELETE,
            doc_id=doc_id,
        )

        self._write_entry(entry)

    def log_update(
        self,
        doc_id: str,
        text: str,
        vector: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log an update operation."""
        self._ensure_open()

        entry = WALEntry(
            operation=WALEntry.UPDATE,
            doc_id=doc_id,
            text=text,
            vector=vector.tolist(),
            metadata=metadata,
        )

        self._write_entry(entry)

    def _write_entry(self, entry: WALEntry) -> None:
        """Write a single entry and flush to disk."""
        line = entry.to_json() + "\n"
        self._file.write(line)
        # flush immediately so the entry survives a crash
        self._file.flush()
        os.fsync(self._file.fileno())

    def _ensure_open(self) -> None:
        if self._file is None:
            self.open()

    def read_entries(self) -> list[WALEntry]:
        """Read all entries from the WAL file.

        Used during crash recovery to replay operations.
        Returns an empty list if the file doesn't exist or is empty.
        """
        if not self._path.exists():
            return []

        entries = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = WALEntry.from_json(line)
                    entries.append(entry)
                except (json.JSONDecodeError, KeyError) as e:
                    # corrupted line -- log it and skip
                    # partial writes from a crash could leave a
                    # truncated json line at the end
                    logger.warning(
                        f"Skipping corrupted WAL entry at line {line_num}: {e}"
                    )

        return entries

    def truncate(self) -> None:
        """Clear the WAL file.

        Called after a successful commit to the .sv file.
        The file is kept but emptied so we don't have to
        recreate it on the next write.
        """
        # close any open handle first
        self.close()

        # open in write mode to truncate, then close
        with open(self._path, "w", encoding="utf-8"):
            pass  # just truncate, write nothing

        logger.debug(f"WAL truncated: {self._path}")

    def has_entries(self) -> bool:
        """Check if the WAL has any pending entries."""
        if not self._path.exists():
            return False
        return self._path.stat().st_size > 0

    @property
    def path(self) -> Path:
        return self._path

    def __repr__(self) -> str:
        exists = self._path.exists()
        size = self._path.stat().st_size if exists else 0
        return f"WriteAheadLog(path='{self._path}', size={size})"

    def __del__(self) -> None:
        self.close()
