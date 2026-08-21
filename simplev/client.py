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
from simplev.indexing import FlatIndex
from simplev.persistence import FileManager
from simplev.query import QueryEngine, QueryResult
from simplev.storage import StorageEngine


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
    ) -> None:
        self._path = Path(path) if path else None
        self._model_name = model_name
        self._metric = metric

        # set up the embedding manager
        self._embeddings = EmbeddingManager(model_name=model_name)

        # we don't know the dimension until the model loads,
        # but we might know it from a loaded .sv file
        self._storage: Optional[StorageEngine] = None
        self._index: Optional[FlatIndex] = None
        self._query_engine: Optional[QueryEngine] = None
        self._dimension: Optional[int] = None

        # if a path was given and the file exists, load it
        if self._path and self._path.exists():
            self._load_from_disk()

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
        self._index = FlatIndex(metric=self._metric)
        self._query_engine = QueryEngine(
            storage=self._storage,
            embeddings=self._embeddings,
            index=self._index,
        )

        logger.info(
            f"Client initialized: dim={dimension}, metric={self._metric}"
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
                raise SimpleVError(
                    f"Vector must be 1-D, got shape {vector.shape}"
                )
            self._ensure_initialized(dimension=vector.shape[0])
            vec = vector.astype(np.float32, copy=False)
        else:
            # embed the text ourselves
            self._ensure_initialized()
            vec = self._embeddings.embed(text)

        self._storage.add(doc_id, text, vec, metadata=metadata)
        return doc_id

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
                    f"Document at index {i} must have 'doc_id' and 'text' keys. "
                    f"Got keys: {list(doc.keys())}"
                )

        self._ensure_initialized()

        # batch embed all texts at once
        texts = [d["text"] for d in documents]
        vectors = self._embeddings.embed_batch(texts)

        added_ids = []
        for i, doc in enumerate(documents):
            self._storage.add(
                doc["doc_id"],
                doc["text"],
                vectors[i],
                metadata=doc.get("metadata"),
            )
            added_ids.append(doc["doc_id"])

        return added_ids

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
            filters: Optional metadata filters (all must match).

        Returns:
            List of QueryResult objects sorted by relevance.
        """
        if self._query_engine is None:
            # nothing has been added yet
            return []

        return self._query_engine.search(
            query=query, top_k=top_k, filters=filters
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
        return self._storage.mark_deleted(doc_id)

    def compact(self) -> int:
        """Remove soft-deleted documents from memory.

        Returns the number of records that were removed.
        """
        if self._storage is None:
            return 0
        return self._storage.compact()

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
        self._index = FlatIndex(metric=self._metric)
        self._query_engine = QueryEngine(
            storage=self._storage,
            embeddings=self._embeddings,
            index=self._index,
        )

        logger.info(
            f"Loaded {storage.active_count} documents from {self._path}"
        )

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
