## 1. Overview
The Indexing Engine is the mathematical core of SimpleV. Its responsibility is to take a query vector and efficiently find the most similar vectors stored in the database. To keep the project accessible and educational, our indexing strategy evolves in phases, starting with exact search methods before introducing approximate techniques for scale.

## 2. Phase 1: Flat Index (Exact Search)
For our initial releases (v0.1), SimpleV utilizes a Flat Index. This is an exact search method, meaning it compares the query vector against every single vector in the database to guarantee absolute accuracy in the results.

* **Mechanism:** It relies on brute-force mathematical comparisons using highly optimized array operations.
* **Performance:** While brute-force sounds computationally heavy, modern hardware can process exact nearest neighbor searches across tens of thousands of vectors in milliseconds. It is the perfect default for personal knowledge bases and small-scale prototyping.
* **Simplicity:** A flat index requires zero training, avoids complex graph building, and has no memory overhead beyond the raw vectors themselves.

## 3. Phase 2: Approximate Nearest Neighbor (ANN)
As user datasets grow beyond hundreds of thousands of vectors, the flat search approach becomes a bottleneck. To address this in version 0.4, we will introduce a Hierarchical Navigable Small World (HNSW) index.

* **Mechanism:** HNSW builds a multi-layered graph where vectors are connected based on their proximity. Searching becomes a process of navigating this graph from the top down, rather than scanning every individual record.
* **Trade-off:** This method trades a negligible fraction of accuracy for massive gains in search speed, allowing sub-millisecond query latencies on millions of vectors.
* **Educational Focus:** Our implementation will prioritize code readability, serving as a clean reference for developers wanting to understand how graph-based vector search functions under the hood.

## 4. Supported Distance Metrics
The engine is designed to support the standard metrics required by modern embedding models:
* **Cosine Similarity:** Measures the angle between two vectors. This is the default and is ideal for text embeddings where the direction of the vector matters more than its magnitude.
* **L2 Distance (Euclidean):** Measures the straight-line distance between points, which is necessary for certain specialized embedding types.

## 5. System Integration
The Indexing Engine is strictly decoupled from storage but works closely with the Query Engine. When a user applies metadata filters to a search, the Query Engine resolves those filters first and passes a boolean bitmask to the Indexing Engine. This ensures distance calculations are only performed on eligible documents, keeping the search pipeline highly optimized. 