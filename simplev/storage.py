"""
Storage engine for SimpleV.

This is the in-memory data layer. It holds vectors in a contiguous
numpy array, metadata in a plain dict, and tracks deletions with
a boolean tombstone mask.

For now this is purely in-memory -- disk persistence (.sv files)
and the WAL come in a later phase. The interface is designed so
that adding those later won't require changing how the rest of
the codebase interacts with storage.
"""

import logging
from typing import Optional

import numpy as np

from simplev.exceptions import StorageError


logger = logging.getLogger(__name__)


class StorageEngine:
    """In-memory storage for vectors and their associated metadata.

    Vectors are stored in a contiguous numpy array for fast math.
    Metadata (doc_id, text, user dict) lives in a python dict
    keyed by an internal integer index. Deletions are soft --
    we flip a bit in the tombstone mask and skip those during search.

    Args:
        dimension: The size of each vector. Once set, every vector
            added must match this dimension exactly.
    """

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise StorageError(
                f"Dimension must be a positive integer, got {dimension}"
            )

        self._dimension = dimension

        # the main vector store -- starts empty, grows as docs are added.
        # shape will be (n_docs, dimension) once we have data
        self._vectors: Optional[np.ndarray] = None

        # maps internal index -> document info
        # each entry is {"doc_id": str, "text": str, "metadata": dict}
        self._metadata: dict[int, dict] = {}

        # reverse lookup: doc_id string -> internal index
        # so we can find things by doc_id without scanning
        self._id_map: dict[str, int] = {}

        # tracks which slots are alive vs soft-deleted
        # True = active, False = tombstoned
        self._tombstones: Optional[np.ndarray] = None

        # how many docs we currently have (including tombstoned ones)
        self._count = 0

        logger.info(f"Storage engine initialized with dimension={dimension}")

    @property
    def dimension(self) -> int:
        """Vector dimension size this storage was created for."""
        return self._dimension

    @property
    def count(self) -> int:
        """Total number of documents including tombstoned ones."""
        return self._count

    @property
    def active_count(self) -> int:
        """Number of non-deleted documents."""
        if self._tombstones is None:
            return 0
        return int(np.sum(self._tombstones))

    def add(
        self,
        doc_id: str,
        text: str,
        vector: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> int:
        """Add a document and its vector to the store.

        Args:
            doc_id: Unique string identifier for this document.
            text: The original text that was embedded.
            vector: The embedding vector, must be float32 with
                shape (dimension,).
            metadata: Optional user-provided dict of extra info.

        Returns:
            The internal integer index assigned to this document.

        Raises:
            StorageError: If doc_id already exists or vector shape is wrong.
        """
        # don't allow duplicate doc ids
        if doc_id in self._id_map:
            raise StorageError(
                f"Document '{doc_id}' already exists. "
                "Delete it first if you want to replace it."
            )

        # make sure the vector is the right shape
        if vector.shape != (self._dimension,):
            raise StorageError(
                f"Vector shape mismatch: expected ({self._dimension},), "
                f"got {vector.shape}"
            )

        # force float32 for consistency
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)

        idx = self._count

        # grow the arrays
        vec_row = vector.reshape(1, -1)
        if self._vectors is None:
            self._vectors = vec_row.copy()
            self._tombstones = np.array([True])
        else:
            self._vectors = np.vstack([self._vectors, vec_row])
            self._tombstones = np.append(self._tombstones, True)

        # store the metadata
        self._metadata[idx] = {
            "doc_id": doc_id,
            "text": text,
            "metadata": metadata if metadata is not None else {},
        }
        self._id_map[doc_id] = idx

        self._count += 1
        return idx

    def mark_deleted(self, doc_id: str) -> bool:
        """Soft-delete a document by flipping its tombstone bit.

        The vector data stays in memory until compaction runs.
        The query engine should check the tombstone mask and skip
        these during search.

        Args:
            doc_id: The document to delete.

        Returns:
            True if the document was found and deleted.

        Raises:
            StorageError: If doc_id doesn't exist.
        """
        if doc_id not in self._id_map:
            raise StorageError(f"Document '{doc_id}' not found.")

        idx = self._id_map[doc_id]

        # already deleted? that's fine, just return
        if not self._tombstones[idx]:
            logger.warning(f"Document '{doc_id}' is already deleted.")
            return False

        self._tombstones[idx] = False
        logger.debug(f"Soft-deleted document '{doc_id}' at index {idx}")
        return True

    def get_metadata(self, doc_id: str) -> dict:
        """Get the stored metadata for a document.

        Args:
            doc_id: The document to look up.

        Returns:
            Dict with keys 'doc_id', 'text', 'metadata'.

        Raises:
            StorageError: If doc_id doesn't exist.
        """
        if doc_id not in self._id_map:
            raise StorageError(f"Document '{doc_id}' not found.")
        idx = self._id_map[doc_id]
        return self._metadata[idx]

    def get_metadata_by_index(self, index: int) -> dict:
        """Get metadata by internal index. Used during result hydration."""
        if index not in self._metadata:
            raise StorageError(f"No document at index {index}")
        return self._metadata[index]

    def get_vectors(self) -> Optional[np.ndarray]:
        """Return the full vector array.

        Returns None if the store is empty. The caller should NOT
        modify this array directly.
        """
        return self._vectors

    def get_active_mask(self) -> Optional[np.ndarray]:
        """Return the tombstone bitmask.

        True means the document is active, False means deleted.
        Returns None if the store is empty.
        """
        return self._tombstones

    def get_all_metadata(self) -> dict[int, dict]:
        """Return the full metadata dictionary.

        Used by the query engine for pre-filtering. Keys are
        internal indices, values are the metadata dicts.
        """
        return self._metadata

    def has_document(self, doc_id: str) -> bool:
        """Check if a document exists (even if tombstoned)."""
        return doc_id in self._id_map

    def is_active(self, doc_id: str) -> bool:
        """Check if a document exists and is not tombstoned."""
        if doc_id not in self._id_map:
            return False
        idx = self._id_map[doc_id]
        return bool(self._tombstones[idx])

    def compact(self) -> int:
        """Remove tombstoned records from memory.

        Rebuilds the vector array and metadata dict without the
        deleted entries. This is expensive but reclaims memory
        and keeps the arrays tight.

        Returns:
            Number of records that were removed.
        """
        if self._tombstones is None or self._count == 0:
            return 0

        # figure out which ones to keep
        keep_mask = self._tombstones.astype(bool)
        removed = int(np.sum(~keep_mask))

        if removed == 0:
            return 0

        logger.info(f"Compacting: removing {removed} tombstoned records")

        # rebuild vectors
        new_vectors = self._vectors[keep_mask]

        # rebuild metadata and id_map with new indices
        new_metadata: dict[int, dict] = {}
        new_id_map: dict[str, int] = {}
        new_idx = 0
        for old_idx in range(self._count):
            if keep_mask[old_idx]:
                entry = self._metadata[old_idx]
                new_metadata[new_idx] = entry
                new_id_map[entry["doc_id"]] = new_idx
                new_idx += 1

        # swap everything
        self._vectors = new_vectors if len(new_vectors) > 0 else None
        self._metadata = new_metadata
        self._id_map = new_id_map
        self._count = new_idx
        self._tombstones = (
            np.ones(new_idx, dtype=bool) if new_idx > 0 else None
        )

        logger.info(f"Compaction done. {self._count} records remaining.")
        return removed

    def clear(self) -> None:
        """Wipe everything. Mostly useful for testing."""
        self._vectors = None
        self._metadata = {}
        self._id_map = {}
        self._tombstones = None
        self._count = 0

    def __repr__(self) -> str:
        active = self.active_count
        total = self._count
        return (
            f"StorageEngine(dim={self._dimension}, "
            f"docs={active}/{total})"
        )
