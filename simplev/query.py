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
from typing import Any, Optional, Union

import numpy as np

from simplev.embeddings import EmbeddingManager
from simplev.exceptions import QueryError
from simplev.indexing import FlatIndex, HNSWIndex
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
        index: Optional[Union[FlatIndex, HNSWIndex]] = None,
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

    def search_batch(
        self,
        queries: list[str],
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> list[list[QueryResult]]:
        """Run semantic search for multiple queries in a batch.

        Args:
            queries: List of query strings.
            top_k: Number of results to return per query.
            filters: Optional metadata filters.

        Returns:
            List of QueryResult lists, one per query.
        """
        if not queries:
            return []

        for q in queries:
            self._validate_request(q, top_k)

        try:
            query_vectors = self._embeddings.embed_batch(queries)
        except Exception as e:
            raise QueryError(f"Failed to embed query batch: {e}") from e

        mask = self._build_filter_mask(filters)
        vectors = self._storage.get_vectors()
        if vectors is None or len(vectors) == 0:
            return [[] for _ in queries]

        batch_results = []
        for q_vec in query_vectors:
            raw = self._index.search(
                query_vector=q_vec,
                all_vectors=vectors,
                top_k=top_k,
                mask=mask,
            )
            batch_results.append(self._hydrate_results(raw))

        return batch_results

    def search_hybrid(
        self,
        query: str,
        sparse_index: Any,
        top_k: int = 5,
        alpha: float = 0.5,
        filters: Optional[dict] = None,
    ) -> list[QueryResult]:
        """Combine dense vector and sparse BM25 search via Reciprocal Rank Fusion."""
        self._validate_request(query, top_k)

        candidate_k = max(top_k * 3, 20)
        dense_results = self.search(
            query=query, top_k=candidate_k, filters=filters
        )

        mask = self._build_filter_mask(filters)
        allowed_doc_ids = None
        if mask is not None:
            all_metadata = self._storage.get_all_metadata()
            allowed_doc_ids = {
                all_metadata[i]["doc_id"]
                for i in np.where(mask)[0]
                if i in all_metadata
            }

        sparse_raw = sparse_index.search(
            query=query, top_k=candidate_k, allowed_doc_ids=allowed_doc_ids
        )

        # Reciprocal Rank Fusion: RRF = a/(60+rank_dense) + (1-a)/(60+rank_sparse)
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, QueryResult] = {}

        for rank, res in enumerate(dense_results, 1):
            rrf_scores[res.doc_id] = rrf_scores.get(res.doc_id, 0.0) + (
                alpha / (60.0 + rank)
            )
            doc_map[res.doc_id] = res

        for rank, (doc_id, s_score) in enumerate(sparse_raw, 1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
                (1.0 - alpha) / (60.0 + rank)
            )
            if doc_id not in doc_map:
                rec = self._storage.get_record(doc_id)
                if rec is not None:
                    doc_map[doc_id] = QueryResult(
                        doc_id=rec["doc_id"],
                        text=rec["text"],
                        score=round(s_score, 6),
                        metadata=rec.get("metadata", {}),
                    )

        sorted_docs = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_k]

        hybrid_results = []
        for doc_id, score in sorted_docs:
            original = doc_map[doc_id]
            hybrid_results.append(
                QueryResult(
                    doc_id=original.doc_id,
                    text=original.text,
                    score=round(score, 6),
                    metadata=original.metadata,
                )
            )
        return hybrid_results

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

    @staticmethod
    def _match_condition(actual: Any, expected: Any) -> bool:
        """Evaluate if an actual field value satisfies expected filter condition."""
        if not isinstance(expected, dict):
            return actual == expected

        for op, target in expected.items():
            if op == "$eq":
                if actual != target:
                    return False
            elif op == "$ne":
                if actual == target:
                    return False
            elif op == "$gt":
                if actual is None or not (actual > target):
                    return False
            elif op == "$gte":
                if actual is None or not (actual >= target):
                    return False
            elif op == "$lt":
                if actual is None or not (actual < target):
                    return False
            elif op == "$lte":
                if actual is None or not (actual <= target):
                    return False
            elif op == "$in":
                if not isinstance(target, (list, tuple, set)) or actual not in target:
                    return False
            elif op == "$nin":
                if isinstance(target, (list, tuple, set)) and actual in target:
                    return False
            else:
                if actual != expected:
                    return False
        return True

    def _build_filter_mask(
        self, filters: Optional[dict]
    ) -> Optional[np.ndarray]:
        """Build a boolean mask combining tombstones and metadata filters.

        Starts with the tombstone mask from storage (so we never
        return deleted docs), then narrows it down further if
        the user specified metadata filters. Supports comparison operators
        such as $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin.
        """
        # always start with the active mask from storage
        base_mask = self._storage.get_active_mask()
        if base_mask is None:
            return None

        # if no filters, just use the tombstone mask as-is
        if not filters:
            return base_mask

        # build a metadata filter mask
        all_metadata = self._storage.get_all_metadata()
        filter_mask = np.zeros(len(base_mask), dtype=bool)

        for idx in range(len(base_mask)):
            if idx not in all_metadata:
                continue

            doc_meta = all_metadata[idx].get("metadata", {})

            matches = True
            for key, condition in filters.items():
                if key not in doc_meta:
                    matches = False
                    break
                if not self._match_condition(doc_meta[key], condition):
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
