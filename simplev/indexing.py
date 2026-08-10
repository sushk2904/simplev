"""
Indexing engine for SimpleV.

This module handles the actual vector similarity search. For v0.1
we only have a flat index which does brute-force comparison against
every vector. It's simple but surprisingly fast for small datasets
thanks to numpy doing the heavy lifting.

The flat index supports cosine similarity (default) and L2 distance.
It accepts a boolean mask so the query engine can pre-filter results
before we waste cycles computing distances on ineligible documents.
"""

import logging
from typing import Optional

import numpy as np

from simplev.exceptions import SimpleVError


logger = logging.getLogger(__name__)


class IndexingError(SimpleVError):
    """Something went wrong during index search."""
    pass


class FlatIndex:
    """Brute-force exact nearest neighbor search.

    Compares the query vector against every vector in the dataset
    using the specified distance metric. No training, no graph
    building, no overhead beyond the raw vectors.

    Good for datasets up to ~50k vectors. Beyond that you'll
    want the HNSW index (coming in v0.4).

    Args:
        metric: Distance metric to use. Either 'cosine' or 'l2'.
            Cosine is the default since it works best with text embeddings.
    """

    SUPPORTED_METRICS = ("cosine", "l2")

    def __init__(self, metric: str = "cosine") -> None:
        if metric not in self.SUPPORTED_METRICS:
            raise IndexingError(
                f"Unknown metric '{metric}'. "
                f"Supported: {self.SUPPORTED_METRICS}"
            )
        self._metric = metric
        logger.info(f"FlatIndex created with metric={metric}")

    @property
    def metric(self) -> str:
        """The distance metric being used."""
        return self._metric

    def search(
        self,
        query_vector: np.ndarray,
        all_vectors: np.ndarray,
        top_k: int = 5,
        mask: Optional[np.ndarray] = None,
    ) -> list[tuple[int, float]]:
        """Find the top_k most similar vectors.

        Args:
            query_vector: The query embedding, shape (dimension,).
            all_vectors: The full vector array from storage,
                shape (n_docs, dimension).
            top_k: How many results to return.
            mask: Optional boolean array of shape (n_docs,).
                True = include this vector in search.
                False = skip it (tombstoned or filtered out).
                If None, searches everything.

        Returns:
            List of (index, score) tuples sorted by relevance.
            For cosine, higher is better. For L2, lower is better.
            The list will have at most top_k entries.

        Raises:
            IndexingError: If inputs are invalid.
        """
        # some quick sanity checks
        if all_vectors is None or len(all_vectors) == 0:
            return []

        if query_vector.ndim != 1:
            raise IndexingError(
                f"Query vector must be 1-D, got shape {query_vector.shape}"
            )

        if query_vector.shape[0] != all_vectors.shape[1]:
            raise IndexingError(
                f"Dimension mismatch: query has {query_vector.shape[0]}, "
                f"vectors have {all_vectors.shape[1]}"
            )

        n_docs = all_vectors.shape[0]

        # if a mask is provided, figure out which indices to actually search
        if mask is not None:
            if mask.shape[0] != n_docs:
                raise IndexingError(
                    f"Mask length ({mask.shape[0]}) doesn't match "
                    f"vector count ({n_docs})"
                )
            active_indices = np.where(mask)[0]
        else:
            active_indices = np.arange(n_docs)

        if len(active_indices) == 0:
            return []

        # grab just the vectors we care about
        candidates = all_vectors[active_indices]

        # compute distances
        if self._metric == "cosine":
            scores = self._cosine_similarity(query_vector, candidates)
            # for cosine, higher = more similar, so we want descending
            best_local = np.argsort(scores)[::-1]
        else:
            scores = self._l2_distance(query_vector, candidates)
            # for l2, lower = closer, so ascending
            best_local = np.argsort(scores)

        # clamp to top_k
        k = min(top_k, len(best_local))
        top_local = best_local[:k]

        # map back to global indices and build results
        results = []
        for local_idx in top_local:
            global_idx = int(active_indices[local_idx])
            score = float(scores[local_idx])
            results.append((global_idx, score))

        return results

    def _cosine_similarity(
        self, query: np.ndarray, vectors: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity between query and all vectors.

        cosine_sim = dot(a, b) / (norm(a) * norm(b))

        We compute this with numpy vectorized ops so it runs fast
        even on large arrays. No python loops over individual vectors.
        """
        # dot product of query with each vector
        dots = vectors @ query

        # norms
        query_norm = np.linalg.norm(query)
        vector_norms = np.linalg.norm(vectors, axis=1)

        # avoid division by zero -- if a vector has zero norm
        # we just give it zero similarity
        denominator = query_norm * vector_norms
        denominator = np.where(denominator == 0, 1.0, denominator)

        similarities = dots / denominator
        return similarities

    def _l2_distance(
        self, query: np.ndarray, vectors: np.ndarray
    ) -> np.ndarray:
        """Compute L2 (Euclidean) distance between query and all vectors.

        This is the straight line distance. Lower values mean the
        vectors are closer together.
        """
        # numpy broadcasting handles this nicely
        diff = vectors - query
        distances = np.linalg.norm(diff, axis=1)
        return distances

    def __repr__(self) -> str:
        return f"FlatIndex(metric='{self._metric}')"
