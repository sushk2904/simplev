"""
Indexing engine for SimpleV.

This module handles the actual vector similarity search. It provides
two index implementations:

  - FlatIndex: Brute-force exact nearest neighbor search. Simple and
    fast for small datasets (up to ~50k vectors) thanks to numpy.

  - HNSWIndex: Approximate nearest neighbor search using Hierarchical
    Navigable Small World graphs. Much faster for large datasets at
    the cost of slightly reduced recall.

Both indexes support cosine similarity and L2 distance, and accept
a boolean mask so the query engine can pre-filter results before
computing distances on ineligible documents.
"""

import heapq
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
    want the HNSWIndex.

    Args:
        metric: Distance metric to use. Either 'cosine' or 'l2'.
            Cosine is the default since it works best with text embeddings.
    """

    SUPPORTED_METRICS = ("cosine", "l2")

    def __init__(self, metric: str = "cosine") -> None:
        if metric not in self.SUPPORTED_METRICS:
            raise IndexingError(
                f"Unknown metric '{metric}'. " f"Supported: {self.SUPPORTED_METRICS}"
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

    def _cosine_similarity(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
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

    def _l2_distance(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
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


class HNSWIndex:
    """Hierarchical Navigable Small World graph for approximate nearest
    neighbor search.

    Builds a multi-layer navigable graph where each node is a vector.
    Search begins at the top layer and greedily descends, then does a
    wider beam search at layer 0 to find the best candidates.

    This is the standard HNSW algorithm as described by Malkov & Yashunin
    (2016). It trades exact recall for dramatic speed improvements on
    large datasets.

    Args:
        metric: Distance metric. 'cosine' or 'l2'.
        m: Number of bi-directional links per node per layer.
            Higher = better recall, more memory. Default 16.
        ef_construction: Size of the dynamic candidate list during
            graph construction. Higher = better graph quality,
            slower build. Default 200.
        ef_search: Size of the candidate list during search.
            Higher = better recall, slower search. Default 50.
        seed: Random seed for level assignment reproducibility.
    """

    SUPPORTED_METRICS = ("cosine", "l2")

    def __init__(
        self,
        metric: str = "cosine",
        m: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        seed: int = 42,
    ) -> None:
        if metric not in self.SUPPORTED_METRICS:
            raise IndexingError(
                f"Unknown metric '{metric}'. " f"Supported: {self.SUPPORTED_METRICS}"
            )

        self._metric = metric
        self._m = m
        # layer 0 gets 2x the connections for better recall
        self._m0 = m * 2
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        # level multiplier from the paper: 1 / ln(M)
        self._ml = 1.0 / max(np.log(m), 1e-6)
        self._rng = np.random.RandomState(seed)

        # graph state
        self._entry_point: Optional[int] = None
        self._max_level: int = -1
        # node_id -> assigned level
        self._levels: dict[int, int] = {}
        # layer_idx -> {node_id: [neighbor_ids]}
        self._graphs: list[dict[int, list[int]]] = []

        # vector cache for distance computations
        self._vectors: Optional[np.ndarray] = None
        self._norm_vectors: Optional[np.ndarray] = None
        self._count: int = 0

        logger.info(
            f"HNSWIndex created: metric={metric}, M={m}, "
            f"ef_construction={ef_construction}, ef_search={ef_search}"
        )

    @property
    def metric(self) -> str:
        """The distance metric being used."""
        return self._metric

    def build(self, all_vectors: np.ndarray) -> None:
        """Build the HNSW graph from scratch.

        Clears any existing graph state and inserts all vectors
        one by one. The vectors array is cached internally for
        distance computations during search.

        Args:
            all_vectors: 2-D float32 array of shape (n, dimension).
        """
        self._reset()
        self._vectors = all_vectors.astype(np.float32, copy=False)
        self._update_norm_cache()

        n = len(all_vectors)
        for i in range(n):
            self._insert_node(i)

        logger.info(f"HNSW graph built with {n} nodes")

    def search(
        self,
        query_vector: np.ndarray,
        all_vectors: np.ndarray,
        top_k: int = 5,
        mask: Optional[np.ndarray] = None,
    ) -> list[tuple[int, float]]:
        """Find the top_k most similar vectors using graph traversal.

        Falls back to brute-force if the graph hasn't been built yet
        or is empty.

        Args:
            query_vector: The query embedding, shape (dimension,).
            all_vectors: The full vector array from storage (used to
                sync internal cache if needed).
            top_k: How many results to return.
            mask: Optional boolean array. True = include, False = skip.

        Returns:
            List of (index, score) tuples. For cosine, higher is better.
            For L2, lower is better.
        """
        if all_vectors is None or len(all_vectors) == 0:
            return []

        # sync vectors if they've changed
        if self._vectors is None or len(self._vectors) != len(all_vectors):
            self._sync_vectors(all_vectors)

        if self._entry_point is None or self._count == 0:
            return []

        # normalize query for cosine distance
        q_norm = float(np.linalg.norm(query_vector))
        q_normed = query_vector / (q_norm if q_norm > 0 else 1.0)

        # 1. Greedy descent through upper layers
        curr_ep = self._entry_point
        for level in range(self._max_level, 0, -1):
            curr_ep = self._greedy_search_layer(query_vector, q_normed, curr_ep, level)

        # 2. Search layer 0 with efSearch
        ef = max(self._ef_search, top_k * 2)
        candidates = self._search_layer(
            query_vector, q_normed, [curr_ep], ef=ef, layer=0
        )

        # 3. Filter by mask
        filtered = [(d, nid) for d, nid in candidates if (mask is None or mask[nid])]

        # If mask is very restrictive and graph search found fewer
        # than top_k, backfill with brute-force over eligible nodes
        if mask is not None and len(filtered) < top_k:
            active_indices = np.where(mask)[0]
            if len(active_indices) <= ef:
                exact_results = []
                for idx in active_indices:
                    d = self._calc_dist_query_node(query_vector, q_normed, int(idx))
                    exact_results.append((d, int(idx)))
                exact_results.sort(key=lambda x: x[0])
                filtered = exact_results

        # sort ascending by distance (closest first)
        filtered.sort(key=lambda x: x[0])
        k = min(top_k, len(filtered))
        top_candidates = filtered[:k]

        # convert distances to scores
        results = []
        for dist, idx in top_candidates:
            if self._metric == "cosine":
                # cosine distance -> cosine similarity
                score = float(1.0 - dist)
            else:
                score = float(dist)
            results.append((idx, score))

        return results

    # -- internal graph operations --

    def _reset(self) -> None:
        """Clear all graph state."""
        self._entry_point = None
        self._max_level = -1
        self._levels.clear()
        self._graphs.clear()
        self._vectors = None
        self._norm_vectors = None
        self._count = 0

    def _sync_vectors(self, all_vectors: np.ndarray) -> None:
        """Update the internal vector cache."""
        self._vectors = all_vectors.astype(np.float32, copy=False)
        self._update_norm_cache()

    def _update_norm_cache(self) -> None:
        """Pre-compute normalized vectors for cosine distance."""
        if self._metric == "cosine" and self._vectors is not None:
            norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self._norm_vectors = self._vectors / norms

    def _random_level(self) -> int:
        """Sample a random level for a new node.

        Uses the formula from the paper: floor(-ln(uniform) * mL)
        where mL = 1/ln(M). This gives an exponential distribution
        so most nodes are at level 0 and very few are at high levels.
        """
        r = self._rng.uniform(1e-9, 1.0)
        level = int(-np.log(r) * self._ml)
        return min(level, 16)  # cap to prevent pathological cases

    def _calc_dist_nodes(self, i: int, j: int) -> float:
        """Compute distance between two indexed nodes."""
        if self._metric == "cosine":
            # cosine distance = 1 - cosine_similarity
            dot = float(np.dot(self._norm_vectors[i], self._norm_vectors[j]))
            dot = max(-1.0, min(1.0, dot))
            return 1.0 - dot
        else:
            return float(np.linalg.norm(self._vectors[i] - self._vectors[j]))

    def _calc_dist_query_node(
        self,
        query: np.ndarray,
        q_normed: np.ndarray,
        node_id: int,
    ) -> float:
        """Compute distance between a query vector and an indexed node."""
        if self._metric == "cosine":
            dot = float(np.dot(self._norm_vectors[node_id], q_normed))
            dot = max(-1.0, min(1.0, dot))
            return 1.0 - dot
        else:
            return float(np.linalg.norm(self._vectors[node_id] - query))

    def _insert_node(self, node_id: int) -> None:
        """Insert a single node into the HNSW graph."""
        level = self._random_level()
        self._levels[node_id] = level

        # ensure graph layers exist up to this level
        while len(self._graphs) <= level:
            self._graphs.append({})

        for layer_idx in range(level + 1):
            self._graphs[layer_idx][node_id] = []

        if self._entry_point is None:
            self._entry_point = node_id
            self._max_level = level
            self._count += 1
            return

        curr_ep = self._entry_point
        top_level = self._max_level
        node_vec = self._vectors[node_id]
        q_norm = float(np.linalg.norm(node_vec))
        q_normed = node_vec / (q_norm if q_norm > 0 else 1.0)

        # 1. Greedy routing down to level + 1
        for layer_idx in range(top_level, level, -1):
            curr_ep = self._greedy_search_layer(node_vec, q_normed, curr_ep, layer_idx)

        # 2. Search and connect from min(top_level, level) down to 0
        ep_candidates = [curr_ep]
        for layer_idx in range(min(top_level, level), -1, -1):
            m_max = self._m0 if layer_idx == 0 else self._m
            w = self._search_layer(
                node_vec,
                q_normed,
                ep_candidates,
                ef=self._ef_construction,
                layer=layer_idx,
            )
            neighbors = self._select_neighbors(w, m_max)
            self._graphs[layer_idx][node_id] = list(neighbors)

            for n in neighbors:
                if node_id not in self._graphs[layer_idx][n]:
                    self._graphs[layer_idx][n].append(node_id)
                if len(self._graphs[layer_idx][n]) > m_max:
                    self._graphs[layer_idx][n] = self._shrink_connections(
                        n, m_max, layer=layer_idx
                    )

            ep_candidates = [n for _, n in w]

        if level > self._max_level:
            self._max_level = level
            self._entry_point = node_id

        self._count += 1

    def _greedy_search_layer(
        self,
        query: np.ndarray,
        q_normed: np.ndarray,
        entry_point: int,
        layer: int,
    ) -> int:
        """Greedy search for the closest node in a single layer."""
        curr = entry_point
        curr_dist = self._calc_dist_query_node(query, q_normed, curr)
        changed = True
        while changed:
            changed = False
            for neighbor in self._graphs[layer].get(curr, []):
                d = self._calc_dist_query_node(query, q_normed, neighbor)
                if d < curr_dist:
                    curr_dist = d
                    curr = neighbor
                    changed = True
        return curr

    def _search_layer(
        self,
        query: np.ndarray,
        q_normed: np.ndarray,
        entry_points: list[int],
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        """Beam search within a single layer.

        Returns a list of (distance, node_id) sorted by distance.
        """
        visited = set(entry_points)
        candidates: list[tuple[float, int]] = []  # min-heap
        w_nearest: list[tuple[float, int]] = []  # max-heap (-dist)

        for ep in entry_points:
            d = self._calc_dist_query_node(query, q_normed, ep)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(w_nearest, (-d, ep))

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            furthest_w_dist = -w_nearest[0][0]
            if c_dist > furthest_w_dist and len(w_nearest) >= ef:
                break

            for neighbor in self._graphs[layer].get(c_id, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    furthest_w_dist = -w_nearest[0][0]
                    n_dist = self._calc_dist_query_node(query, q_normed, neighbor)
                    if n_dist < furthest_w_dist or len(w_nearest) < ef:
                        heapq.heappush(candidates, (n_dist, neighbor))
                        heapq.heappush(w_nearest, (-n_dist, neighbor))
                        if len(w_nearest) > ef:
                            heapq.heappop(w_nearest)

        results = [(-neg_d, node) for neg_d, node in w_nearest]
        results.sort(key=lambda x: x[0])
        return results

    def _select_neighbors(
        self, candidates: list[tuple[float, int]], m_max: int
    ) -> list[int]:
        """Select the m_max closest neighbors from candidates."""
        sorted_cands = sorted(candidates, key=lambda x: x[0])
        return [node for _, node in sorted_cands[:m_max]]

    def _shrink_connections(self, node_id: int, m_max: int, layer: int) -> list[int]:
        """Prune a node's connections down to m_max."""
        neighbors = self._graphs[layer].get(node_id, [])
        dists = [(self._calc_dist_nodes(node_id, n), n) for n in neighbors]
        dists.sort(key=lambda x: x[0])
        return [n for _, n in dists[:m_max]]

    def __len__(self) -> int:
        return self._count

    def __repr__(self) -> str:
        return (
            f"HNSWIndex(metric='{self._metric}', nodes={self._count}, "
            f"max_level={self._max_level})"
        )
