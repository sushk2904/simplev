# SimpleV Pull Requests Tracker

> **Repository:** [sushk2904/simplev](https://github.com/sushk2904/simplev.git)  
> **Author:** Mohan SriRam ([@SriRamkunamsetty](https://github.com/SriRamkunamsetty))  
> **Email:** mohansriramkunamsetty@gmail.com  
> **Last Updated:** September 9, 2026  

---

## 1. Quick Shareable Format (Copy & Paste)

### PR #7 (Latest: Hybrid Search, In-Place CRUD, Rich Filtering)
```text
PR Number & Link: #7 - https://github.com/sushk2904/simplev/pull/7
Issue Number it solves: Fixes #6 (https://github.com/sushk2904/simplev/issues/6)
Brief description of your changes:
1. In-Place Updates & Upsert: Added Client.update(), Client.upsert(), StorageEngine.update(), and WAL update logging with crash recovery.
2. Primary-Key Retrieval & Pythonic Collections: Added Client.get(), db[doc_id], "doc" in db, len(db), and iter(db).
3. BM25 Hybrid Lexical-Semantic Search: Added zero-dependency Okapi BM25Index with Reciprocal Rank Fusion (RRF) in Client.hybrid_search().
4. Rich Metadata Filtering Operators: Added support for $eq, $ne, $gt, $gte, $lt, $lte, $in, and $nin.
5. Batch Search: Added Client.search_batch() for parallel query embedding.
6. HNSW Compaction Synchronization: Automatically rebuilds graph index on compaction/update to prevent index desynchronization.
7. CLI Subcommands: Added 'simplev get' and '--hybrid' flag to 'simplev search'.
```

### PR #5 (Phase 0.4 Roadmap: CLI, HNSW ANN, Benchmarks)
```text
PR Number & Link: #5 - https://github.com/sushk2904/simplev/pull/5
Issue Number it solves: Fixes #4 (https://github.com/sushk2904/simplev/issues/4)
Brief description of your changes: Implemented Phase 0.4 roadmap completing all remaining core milestones:
1. Command Line Interface (CLI): Built terminal commands (simplev init, ingest, search, info) conforming to CLI_SPEC.md.
2. Approximate Nearest Neighbor (ANN) Search: Implemented Hierarchical Navigable Small World (HNSWIndex) graph engine with multi-layer skip routing, beam search, Cosine/L2 distance metrics, and pre-filtering mask support.
3. Performance Benchmark Suite: Created reproducible benchmarks in examples/benchmarks.py satisfying all targets (>3,900 docs/s ingestion vs >500 target, 1.88ms latency vs <10ms target, 1.62MB storage vs <20MB target, 94% recall).
4. Durability & SDK Fixes: Enabled empty database persistence in FileManager, added Client.insert() alias and Client.info() diagnostic summary, added CONTRIBUTORS.md, and updated Info/TESTING_STRATEGY.md.
```

---

## 2. Pull Requests Summary Table

| # | PR Number | Title | Status | Date | Solves Issue | Full URL |
|---|-----------|-------|--------|------|--------------|----------|
| 1 | **#7** | [feat: in-place document updates, upsert, BM25 hybrid search, and rich metadata filtering](https://github.com/sushk2904/simplev/pull/7) | **OPEN** | 2026-09-09 | [Fixes #6](https://github.com/sushk2904/simplev/issues/6) | https://github.com/sushk2904/simplev/pull/7 |
| 2 | **#5** | [feat: implement Phase 0.4 roadmap (CLI, HNSW ANN search, and benchmark suite)](https://github.com/sushk2904/simplev/pull/5) | **OPEN** | 2026-09-09 | [Fixes #4](https://github.com/sushk2904/simplev/issues/4) | https://github.com/sushk2904/simplev/pull/5 |

---

## 3. Detailed Pull Request Records

### [PR #7](https://github.com/sushk2904/simplev/pull/7): feat: in-place document updates, upsert, BM25 hybrid search, and rich metadata filtering

- **PR Number & Link**: #7 - https://github.com/sushk2904/simplev/pull/7
- **Repository**: https://github.com/sushk2904/simplev
- **Author**: Mohan SriRam ([@SriRamkunamsetty](https://github.com/SriRamkunamsetty))
- **Email**: mohansriramkunamsetty@gmail.com
- **Issue Number it solves**: Fixes #6 (https://github.com/sushk2904/simplev/issues/6)
- **Status**: `OPEN`
- **Date Created**: 2026-09-09
- **Key Enhancements**:
  - **In-Place Document Updates & Upsert**: `Client.update()`, `Client.upsert()`, `StorageEngine.update()`, and `WALEntry.UPDATE` with full crash recovery.
  - **Primary-Key Retrieval & Pythonic Collections**: Added `Client.get()`, `db[doc_id]`, `"doc" in db`, `len(db)`, and `iter(db)`.
  - **Zero-Dependency Okapi BM25 Index & Hybrid Search**: Added `BM25Index` combined with dense cosine search using Reciprocal Rank Fusion (RRF) in `Client.hybrid_search()`.
  - **Rich Metadata Filtering Operators**: Added MongoDB-style operators `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, and `$nin`.
  - **Batch Search**: Added `Client.search_batch()`.
  - **HNSW Compaction Synchronization**: Rebuilds graph index upon compaction to guarantee index integrity.
  - **CLI Additions**: Added `simplev get` and `--hybrid` option to `simplev search`.
  - **Test Suite**: Added 39 new tests across `test_sparse.py`, `test_filtering.py`, `test_client_features.py`, `test_hybrid.py`, and `test_cli.py` (214 passing tests total).

---

### [PR #5](https://github.com/sushk2904/simplev/pull/5): feat: implement Phase 0.4 roadmap (CLI, HNSW ANN search, and benchmark suite)

- **PR Number & Link**: #5 - https://github.com/sushk2904/simplev/pull/5
- **Repository**: https://github.com/sushk2904/simplev
- **Author**: Mohan SriRam ([@SriRamkunamsetty](https://github.com/SriRamkunamsetty))
- **Email**: mohansriramkunamsetty@gmail.com
- **Issue Number it solves**: Fixes #4 (https://github.com/sushk2904/simplev/issues/4)
- **Status**: `OPEN`
- **Date Created**: 2026-09-09
- **Key Enhancements**:
  - **Phase 0.4: CLI**: Implemented `simplev/cli.py` with `argparse` providing `init`, `ingest`, `search`, and `info` commands matching `CLI_SPEC.md`, registered in `pyproject.toml`.
  - **Phase 0.4: ANN Search**: Implemented `HNSWIndex` multi-layer graph indexing engine supporting Cosine similarity and Euclidean ($L_2$) distance, greedy upper-layer routing, candidate beam search, and pre-filtering boolean masks.
  - **Phase 0.4: Benchmarks**: Created `examples/benchmarks.py` testing ingestion speed, query latency, storage footprint, and top-10 recall against targets.
  - **Durability & SDK Fixes**: Added empty database support in `FileManager`, `Client.insert()` alias, `Client.info()`, `CONTRIBUTORS.md`, and updated `Info/TESTING_STRATEGY.md`.
