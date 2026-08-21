"""
Query engine for SimpleV.

This is the orchestrator. It doesn't store data or compute distances
itself -- it coordinates the pipeline between embeddings, storage,
and the index to turn a user's text query into ranked results.

The search pipeline has 6 steps:
  1. Validate the request
  2. Embed the query text
  3. Pre-filter by metadata (build a boolean mask)
  4. Run the index search
  5. Hydrate results with metadata from storage
  6. Format and return

Keeping this logic separate from storage and indexing means
we can swap out the index algorithm later without touching
any of the orchestration code.
"""

import logging
from typing import Optional

import numpy as np

from simplev.embeddings import EmbeddingManager
from simplev.exceptions import QueryError
from simplev.indexing import FlatIndex
from simplev.storage import StorageEngine


logger = logging.getLogger(__name__)


class QueryResult:
    """A single search result with all the info a user would want.

    Just a plain data container. Nothing fancy.
    """

    def __init__(
        self,
        doc_id: str,
        text: str,
        score: float,
        metadata: dict,
    ) -> None:
        self.doc_id = doc_id
        self.text = text
        self.score = score
        self.metadata = metadata

    def to_dict(self) -> dict:
        """Convert to a plain dict for serialization or display."""
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return (
            f"QueryResult(doc_id='{self.doc_id}', "
            f"score={self.score:.4f}, text='{text_preview}')"
        )


class QueryEngine:
    """Orchestrates the full search pipeline.

    Ties together the embedding manager, storage engine, and
    flat index to handle end-to-end search requests.

    Args:
        storage: The storage engine holding all vectors and metadata.
        embeddings: The embedding manager for converting query text.
        index: The search index to use. Defaults to a FlatIndex
            with cosine similarity if not provided.
    """

    def __init__(
        self,
        storage: StorageEngine,
        embeddings: EmbeddingManager,
        index: Optional[FlatIndex] = None,
    ) -> None:
        self._storage = storage
        self._embeddings = embeddings
        self._index = index or FlatIndex(metric="cosine")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[QueryResult]:
        """Run the full search pipeline.

        Takes a natural language query, embeds it, optionally
        filters by metadata, searches the index, and returns
        hydrated results.

        Args:
            query: The search text.
            top_k: Maximum number of results to return.
            filters: Optional dict of metadata key-value pairs
                to filter on. Only documents whose metadata
                contains all specified key-value pairs will
                be included in the search.

        Returns:
            List of QueryResult objects sorted by relevance.

        Raises:
            QueryError: If the query is invalid or search fails.
        """
        # -- step 1: validate --
        self._validate_request(query, top_k)

        # -- step 2: embed the query --
        try:
            query_vector = self._embeddings.embed(query)
        except Exception as e:
            raise QueryError(f"Failed to embed query: {e}") from e

        # -- step 3: pre-filter by metadata --
        mask = self._build_filter_mask(filters)

        # -- step 4: run the index search --
        vectors = self._storage.get_vectors()
        if vectors is None:
            return []

        raw_results = self._index.search(
            query_vector=query_vector,
            all_vectors=vectors,
            top_k=top_k,
            mask=mask,
        )

        # -- step 5: hydrate results --
        results = self._hydrate_results(raw_results)

        # -- step 6: return --
        logger.debug(
            f"Search for '{query[:40]}' returned {len(results)} results"
        )
        return results

    def _validate_request(self, query: str, top_k: int) -> None:
        """Check that the search request makes sense."""
        if not isinstance(query, str) or not query.strip():
            raise QueryError(
                "Search query must be a non-empty string. "
                f"Got type={type(query).__name__}"
            )

        if not isinstance(top_k, int) or top_k < 1:
            raise QueryError(
                f"top_k must be a positive integer, got {top_k}"
            )

        # clamp top_k to what we actually have
        # no need to error here, just return fewer results
        active = self._storage.active_count
        if active == 0:
            logger.debug("Search on empty database, will return no results")

    def _build_filter_mask(
        self, filters: Optional[dict]
    ) -> Optional[np.ndarray]:
        """Build a boolean mask combining tombstones and metadata filters.

        Starts with the tombstone mask from storage (so we never
        return deleted docs), then narrows it down further if
        the user specified metadata filters.
        """
        # always start with the active mask from storage
        base_mask = self._storage.get_active_mask()
        if base_mask is None:
            return None

        # if no filters, just use the tombstone mask as-is
        if not filters:
            return base_mask

        # build a metadata filter mask
        # we go through every document and check if its metadata
        # matches all the filter criteria
        all_metadata = self._storage.get_all_metadata()
        filter_mask = np.zeros(len(base_mask), dtype=bool)

        for idx in range(len(base_mask)):
            if idx not in all_metadata:
                continue

            doc_meta = all_metadata[idx].get("metadata", {})

            # document must match ALL filter key-value pairs
            matches = True
            for key, value in filters.items():
                if key not in doc_meta or doc_meta[key] != value:
                    matches = False
                    break

            if matches:
                filter_mask[idx] = True

        # combine: must be active AND pass filters
        combined = base_mask & filter_mask
        return combined

    def _hydrate_results(
        self, raw_results: list[tuple[int, float]]
    ) -> list[QueryResult]:
        """Turn raw (index, score) pairs into full QueryResult objects."""
        results = []

        for idx, score in raw_results:
            try:
                meta = self._storage.get_metadata_by_index(idx)
                result = QueryResult(
                    doc_id=meta["doc_id"],
                    text=meta["text"],
                    score=round(score, 6),
                    metadata=meta.get("metadata", {}),
                )
                results.append(result)
            except Exception as e:
                # if we can't hydrate a result, skip it and log
                # rather than crashing the whole search
                logger.warning(f"Failed to hydrate result at index {idx}: {e}")

        return results

    def __repr__(self) -> str:
        return (
            f"QueryEngine(storage={self._storage}, "
            f"index={self._index})"
        )
