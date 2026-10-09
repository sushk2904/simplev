"""
Client SDK for SimpleV.

This is what users actually import. It's a thin facade that wires
together the storage engine, embedding manager, indexing, and
query engine behind a dead-simple API.

The goal is that someone can go from zero to vector search in
about 5 lines of code:

    from simplev import Client
    db = Client()
    db.add("doc1", "the cat sat on the mat")
    results = db.search("animals sitting")
    print(results[0].text)

Everything else -- model loading, vector math, tombstones --
is handled internally and the user never has to think about it.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

from simplev.embeddings import EmbeddingManager
from simplev.exceptions import SimpleVError, StorageError
from simplev.indexing import FlatIndex, HNSWIndex
from simplev.persistence import FileManager
from simplev.query import QueryEngine, QueryResult
from simplev.sparse import BM25Index
from simplev.storage import StorageEngine
from simplev.wal import WriteAheadLog

logger = logging.getLogger(__name__)


class Client:
    """The main entry point for SimpleV.

    Creates and manages a vector database instance. Handles
    adding documents, searching, deleting, and persisting
    to disk.

    Args:
        path: Optional path to a .sv file. If provided and the
            file exists, the database will be loaded from disk.
            If not provided, the database runs purely in memory.
        model_name: The sentence-transformers model to use.
            Defaults to all-MiniLM-L6-v2 (384 dimensions).
        metric: Distance metric for search. Either 'cosine'
            or 'l2'. Defaults to 'cosine'.
        use_wal: Whether to enable write-ahead logging for
            crash recovery. Only works when a path is provided.
            Defaults to True.
        index_type: Which index to use for search. Either 'flat'
            for brute-force exact search or 'hnsw' for approximate
            nearest neighbor. Defaults to 'flat'.

    Example:
        >>> db = Client()
        >>> db.add("greeting", "hello world")
        >>> results = db.search("hi there")
        >>> results[0].doc_id
        'greeting'
    """

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        model_name: str = "all-MiniLM-L6-v2",
        metric: str = "cosine",
        use_wal: bool = True,
        index_type: str = "flat",
    ) -> None:
        self._path = Path(path) if path else None
        self._model_name = model_name
        self._metric = metric
        self._use_wal = use_wal
        self._index_type = index_type

        # set up the embedding manager
        self._embeddings = EmbeddingManager(model_name=model_name)

        # we don't know the dimension until the model loads,
        # but we might know it from a loaded .sv file
        self._storage: Optional[StorageEngine] = None
        self._index: Optional[Union[FlatIndex, HNSWIndex]] = None
        self._query_engine: Optional[QueryEngine] = None
        self._sparse_index: Optional[BM25Index] = None
        self._dimension: Optional[int] = None
        self._wal: Optional[WriteAheadLog] = None

        # if a path was given, set up the WAL and load
        if self._path:
            if use_wal:
                wal_path = Path(str(self._path) + ".wal")
                self._wal = WriteAheadLog(wal_path)

            if self._path.exists():
                self._load_from_disk()
                # check for crash recovery after loading
                self._replay_wal()

    def _create_index(self, metric: str = "cosine") -> Union[FlatIndex, HNSWIndex]:
        """Create the appropriate index based on index_type config."""
        if self._index_type == "hnsw":
            return HNSWIndex(metric=metric)
        return FlatIndex(metric=metric)

    def _ensure_initialized(self, dimension: Optional[int] = None) -> None:
        """Make sure the internal engines are set up.

        On first add() we need to know the vector dimension, which
        means the embedding model has to load. After that the
        dimension is locked in.
        """
        if self._storage is not None:
            return

        if dimension is None:
            # force the model to load so we know the dimension
            dimension = self._embeddings.dimension

        self._dimension = dimension
        self._storage = StorageEngine(dimension=dimension)
        self._index = self._create_index(metric=self._metric)
        self._sparse_index = BM25Index()
        self._query_engine = QueryEngine(
            storage=self._storage,
            embeddings=self._embeddings,
            index=self._index,
        )

        logger.info(
            f"Client initialized: dim={dimension}, "
            f"metric={self._metric}, index={self._index_type}"
        )

    def add(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[dict] = None,
        vector: Optional[np.ndarray] = None,
    ) -> str:
        """Add a document to the database.

        Embeds the text automatically unless you provide a
        pre-computed vector.

        Args:
            doc_id: Unique identifier for this document.
            text: The text content. Will be embedded and stored.
            metadata: Optional dict of extra info to attach.
            vector: Optional pre-computed embedding vector.
                If provided, text is stored but not re-embedded.

        Returns:
            The doc_id that was added.

        Raises:
            SimpleVError: If doc_id already exists or embedding fails.
        """
        if vector is not None:
            # user provided their own vector
            if vector.ndim != 1:
                raise SimpleVError(f"Vector must be 1-D, got shape {vector.shape}")
            self._ensure_initialized(dimension=vector.shape[0])
            vec = vector.astype(np.float32, copy=False)
        else:
            # embed the text ourselves
            self._ensure_initialized()
            vec = self._embeddings.embed(text)

        # log to WAL before touching memory
        if self._wal is not None:
            self._wal.log_insert(doc_id, text, vec, metadata)

        self._storage.add(doc_id, text, vec, metadata=metadata)

        # update the sparse index for hybrid search
        if self._sparse_index is not None:
            self._sparse_index.add_document(doc_id, text)

        return doc_id

    def insert(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[dict] = None,
        vector: Optional[np.ndarray] = None,
    ) -> bool:
        """Insert a document into the database.

        Alias for add() conforming to API_SPEC.md and quickstart examples.

        Args:
            doc_id: Unique identifier for this document.
            text: The text content to embed and store.
            metadata: Optional dict of extra info to attach.
            vector: Optional pre-computed embedding vector.

        Returns:
            True indicating successful insertion.
        """
        self.add(doc_id=doc_id, text=text, metadata=metadata, vector=vector)
        return True

    def add_many(
        self,
        documents: list[dict],
    ) -> list[str]:
        """Add multiple documents in one call.

        Batches the embedding step for better performance.

        Args:
            documents: List of dicts, each with at least 'doc_id'
                and 'text' keys. Optional 'metadata' key.

        Returns:
            List of doc_ids that were added.

        Raises:
            SimpleVError: If any document is invalid.
        """
        if not documents:
            return []

        # validate the format first
        for i, doc in enumerate(documents):
            if "doc_id" not in doc or "text" not in doc:
                raise SimpleVError(
                    f"Document at index {i} must have 'doc_id' "
                    f"and 'text' keys. "
                    f"Got keys: {list(doc.keys())}"
                )

        self._ensure_initialized()

        # batch embed all texts at once
        texts = [d["text"] for d in documents]
        vectors = self._embeddings.embed_batch(texts)

        added_ids = []
        for i, doc in enumerate(documents):
            vec = vectors[i]

            # log each insert to WAL
            if self._wal is not None:
                self._wal.log_insert(
                    doc["doc_id"],
                    doc["text"],
                    vec,
                    doc.get("metadata"),
                )

            self._storage.add(
                doc["doc_id"],
                doc["text"],
                vec,
                metadata=doc.get("metadata"),
            )

            # update sparse index
            if self._sparse_index is not None:
                self._sparse_index.add_document(doc["doc_id"], doc["text"])

            added_ids.append(doc["doc_id"])

        return added_ids

    def update(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[dict] = None,
        vector: Optional[np.ndarray] = None,
    ) -> str:
        """Update an existing document in place.

        Replaces the text, vector, and metadata without changing
        the document's internal index. The text is re-embedded
        unless a pre-computed vector is provided.

        Args:
            doc_id: Identifier of the document to update.
            text: New text content.
            metadata: Optional new metadata dict.
            vector: Optional pre-computed replacement vector.

        Returns:
            The doc_id that was updated.

        Raises:
            StorageError: If doc_id does not exist.
        """
        if self._storage is None:
            raise StorageError(f"Document '{doc_id}' not found.")

        if vector is not None:
            vec = vector.astype(np.float32, copy=False)
        else:
            vec = self._embeddings.embed(text)

        # log to WAL before touching memory
        if self._wal is not None:
            self._wal.log_update(doc_id, text, vec, metadata)

        self._storage.update(doc_id, text, vec, metadata=metadata)

        # update the sparse index
        if self._sparse_index is not None:
            self._sparse_index.remove_document(doc_id)
            self._sparse_index.add_document(doc_id, text)

        return doc_id

    def upsert(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[dict] = None,
        vector: Optional[np.ndarray] = None,
    ) -> str:
        """Insert or update a document.

        If doc_id already exists and is active, updates it in place.
        Otherwise, adds it as a new document.

        Args:
            doc_id: Unique identifier for the document.
            text: The text content.
            metadata: Optional metadata dict.
            vector: Optional pre-computed embedding vector.

        Returns:
            The doc_id that was added or updated.
        """
        self._ensure_initialized(
            dimension=vector.shape[0] if vector is not None else None
        )

        if self._storage.has_document(doc_id):
            if self._storage.is_active(doc_id):
                return self.update(doc_id, text, metadata=metadata, vector=vector)
            # tombstoned -- compact to clear, then add fresh
            # (we can't just update a tombstoned doc through add
            # because add rejects existing doc_ids)
            self._storage.compact()

        return self.add(doc_id, text, metadata=metadata, vector=vector)

    def get(
        self,
        doc_id: str,
        include_vector: bool = False,
    ) -> Optional[dict]:
        """Retrieve a document by its doc_id.

        Args:
            doc_id: The document to look up.
            include_vector: Whether to include the embedding vector.

        Returns:
            Dict with 'doc_id', 'text', 'metadata' (and optional
            'vector'), or None if not found / deleted.
        """
        if self._storage is None:
            return None
        return self._storage.get_record(doc_id, include_vector=include_vector)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[QueryResult]:
        """Search the database with a natural language query.

        Args:
            query: The search text.
            top_k: Maximum number of results.
            filters: Optional metadata filters. Supports exact matching
                and comparison operators ($eq, $ne, $gt, $gte, $lt,
                $lte, $in, $nin).

        Returns:
            List of QueryResult objects sorted by relevance.
        """
        if self._query_engine is None:
            # nothing has been added yet
            return []

        return self._query_engine.search(query=query, top_k=top_k, filters=filters)

    def search_batch(
        self,
        queries: list[str],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[list[QueryResult]]:
        """Run multiple search queries in a single batch.

        Embeds all queries at once for better throughput.

        Args:
            queries: List of search strings.
            top_k: Maximum number of results per query.
            filters: Optional metadata filters.

        Returns:
            List of QueryResult lists, one per query.
        """
        if self._query_engine is None:
            return [[] for _ in queries]
        return self._query_engine.search_batch(
            queries=queries, top_k=top_k, filters=filters
        )

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        filters: Optional[dict] = None,
    ) -> list[QueryResult]:
        """Run hybrid search combining dense and BM25 sparse search.

        Uses Reciprocal Rank Fusion (RRF) to combine dense semantic
        search rankings with sparse keyword (BM25) rankings.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.
            alpha: Weight balancing dense vs sparse relevance.
                1.0 = pure dense search, 0.0 = pure BM25.
            filters: Optional metadata filters.

        Returns:
            List of QueryResult objects sorted by fused RRF score.
        """
        if self._query_engine is None or self._sparse_index is None:
            return []
        return self._query_engine.search_hybrid(
            query=query,
            sparse_index=self._sparse_index,
            top_k=top_k,
            alpha=alpha,
            filters=filters,
        )

    def delete(self, doc_id: str) -> bool:
        """Soft-delete a document.

        The document is marked as deleted and excluded from
        future searches. Call compact() to actually free the
        memory.

        Args:
            doc_id: The document to delete.

        Returns:
            True if the document was deleted.
        """
        if self._storage is None:
            raise StorageError(f"Document '{doc_id}' not found.")

        # log to WAL before touching memory
        if self._wal is not None:
            self._wal.log_delete(doc_id)

        deleted = self._storage.mark_deleted(doc_id)
        if deleted and self._sparse_index is not None:
            self._sparse_index.remove_document(doc_id)
        return deleted

    def compact(self) -> int:
        """Remove soft-deleted documents from memory.

        Rebuilds vector storage and synchronizes dense (HNSW)
        and sparse (BM25) search indices.

        Returns:
            The number of records that were removed.
        """
        if self._storage is None:
            return 0
        removed = self._storage.compact()
        if removed > 0:
            # rebuild HNSW index if applicable
            if self._index_type == "hnsw" and hasattr(self._index, "build"):
                vecs = self._storage.get_vectors()
                if vecs is not None and len(vecs) > 0:
                    self._index.build(vecs)
                elif hasattr(self._index, "_reset"):
                    self._index._reset()

            # rebuild sparse index from scratch
            self._sparse_index = BM25Index()
            for rec in self._storage.get_active_records():
                self._sparse_index.add_document(rec["doc_id"], rec["text"])

        return removed

    def commit(self) -> Path:
        """Save to disk and clear the WAL.

        This is the 'safe checkpoint' -- after commit() returns,
        everything is persisted in the .sv file and the WAL is
        empty. If the process crashes after this, no data is lost.

        Returns:
            The path the file was saved to.

        Raises:
            SimpleVError: If no path is set or database is empty.
        """
        if self._path is None:
            raise SimpleVError(
                "Cannot commit an in-memory database. "
                "Provide a path when creating the Client."
            )

        if self._storage is None or self._storage.count == 0:
            raise SimpleVError("Nothing to commit -- database is empty.")

        fm = FileManager()
        fm.save(self._path, self._storage)

        # truncate the WAL now that the .sv file is up to date
        if self._wal is not None:
            self._wal.truncate()

        logger.info(f"Committed to {self._path}")
        return self._path

    def save(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Save the database to a .sv file.

        Args:
            path: Where to save. Uses the path from __init__
                if not specified.

        Returns:
            The path the file was saved to.

        Raises:
            SimpleVError: If no path is available or save fails.
        """
        save_path = Path(path) if path else self._path
        if save_path is None:
            raise SimpleVError(
                "No save path specified. Pass a path to save() "
                "or provide one when creating the Client."
            )

        if self._storage is None or self._storage.count == 0:
            raise SimpleVError("Nothing to save -- database is empty.")

        fm = FileManager()
        fm.save(save_path, self._storage)

        self._path = save_path
        logger.info(f"Database saved to {save_path}")
        return save_path

    def _load_from_disk(self) -> None:
        """Load database state from a .sv file."""
        fm = FileManager()
        storage = fm.load(self._path)

        self._storage = storage
        self._dimension = storage.dimension
        self._index = self._create_index(metric=self._metric)
        self._sparse_index = BM25Index()
        for rec in storage.get_active_records():
            self._sparse_index.add_document(rec["doc_id"], rec["text"])

        if self._index_type == "hnsw" and storage.active_count > 0:
            vecs = storage.get_vectors()
            if vecs is not None:
                self._index.build(vecs)

        self._query_engine = QueryEngine(
            storage=self._storage,
            embeddings=self._embeddings,
            index=self._index,
        )

        logger.info(f"Loaded {storage.active_count} documents " f"from {self._path}")

    def _replay_wal(self) -> None:
        """Replay WAL entries for crash recovery.

        If there's a non-empty .wal file, it means the process
        crashed before the last commit. We replay those operations
        and immediately commit to get back to a consistent state.
        """
        if self._wal is None or not self._wal.has_entries():
            return

        entries = self._wal.read_entries()
        if not entries:
            return

        logger.info(f"Crash recovery: replaying {len(entries)} WAL entries")

        if self._storage is None:
            # shouldn't happen if .sv was loaded, but be safe
            return

        replayed = 0
        for entry in entries:
            try:
                if entry.operation == "insert":
                    # skip if the doc was already loaded from .sv
                    if self._storage.has_document(entry.doc_id):
                        continue

                    vec = np.array(entry.vector, dtype=np.float32)
                    self._storage.add(
                        entry.doc_id,
                        entry.text,
                        vec,
                        metadata=entry.metadata,
                    )
                    if self._sparse_index is not None:
                        self._sparse_index.add_document(entry.doc_id, entry.text)
                    replayed += 1

                elif entry.operation == "update":
                    vec = np.array(entry.vector, dtype=np.float32)
                    if self._storage.has_document(entry.doc_id):
                        self._storage.update(
                            entry.doc_id,
                            entry.text,
                            vec,
                            metadata=entry.metadata,
                        )
                    else:
                        self._storage.add(
                            entry.doc_id,
                            entry.text,
                            vec,
                            metadata=entry.metadata,
                        )
                    if self._sparse_index is not None:
                        self._sparse_index.remove_document(entry.doc_id)
                        self._sparse_index.add_document(entry.doc_id, entry.text)
                    replayed += 1

                elif entry.operation == "delete":
                    if self._storage.has_document(
                        entry.doc_id
                    ) and self._storage.is_active(entry.doc_id):
                        self._storage.mark_deleted(entry.doc_id)
                        if self._sparse_index is not None:
                            self._sparse_index.remove_document(entry.doc_id)
                        replayed += 1

            except Exception as e:
                logger.warning(f"Failed to replay WAL entry {entry}: {e}")

        if replayed > 0:
            logger.info(f"Replayed {replayed} operations, committing")
            # rebuild HNSW if needed
            if self._index_type == "hnsw" and hasattr(self._index, "build"):
                vecs = self._storage.get_vectors()
                if vecs is not None:
                    self._index.build(vecs)

            # immediately commit to persist the recovered state
            try:
                fm = FileManager()
                fm.save(self._path, self._storage)
                self._wal.truncate()
            except Exception as e:
                logger.error(f"Failed to commit after WAL replay: {e}")

    def info(self) -> dict:
        """Return diagnostic metadata about the database.

        Returns:
            Dict with keys: path, count, active_count, dimension,
            model_name, metric, index_type, file_size_bytes,
            file_exists, wal_size_bytes, wal_exists.
        """
        wal_size = 0
        wal_exists = False
        if self._wal is not None and self._wal.path.exists():
            wal_exists = True
            wal_size = self._wal.path.stat().st_size

        file_size = 0
        file_exists = False
        if self._path is not None and self._path.exists():
            file_exists = True
            file_size = self._path.stat().st_size

        return {
            "path": str(self._path) if self._path else None,
            "count": self.count,
            "active_count": (self._storage.active_count if self._storage else 0),
            "dimension": self._dimension,
            "model_name": self._model_name,
            "metric": self._metric,
            "index_type": self._index_type,
            "file_size_bytes": file_size,
            "file_exists": file_exists,
            "wal_size_bytes": wal_size,
            "wal_exists": wal_exists,
        }

    @property
    def count(self) -> int:
        """Number of active (non-deleted) documents."""
        if self._storage is None:
            return 0
        return self._storage.active_count

    @property
    def dimension(self) -> Optional[int]:
        """Vector dimension, or None if not yet initialized."""
        return self._dimension

    def __repr__(self) -> str:
        n = self.count
        p = self._path or "in-memory"
        return f"Client(docs={n}, path='{p}')"

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, doc_id: str) -> dict:
        """Retrieve a document record by doc_id like db['doc1']."""
        rec = self.get(doc_id)
        if rec is None:
            raise KeyError(f"Document '{doc_id}' not found.")
        return rec

    def __contains__(self, doc_id: str) -> bool:
        """Check if an active document exists ('doc1' in db)."""
        if self._storage is None:
            return False
        return self._storage.is_active(doc_id)

    def __iter__(self):
        """Iterate over all active document records."""
        if self._storage is None:
            return iter([])
        return iter(self._storage.get_active_records())
