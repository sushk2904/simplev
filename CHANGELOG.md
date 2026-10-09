# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-10-09

### Added
- **HNSW Indexing**: Approximate Nearest Neighbor (ANN) search implementing Hierarchical Navigable Small World graphs with Cosine and L2 metrics and tombstone filtering.
- **Sparse BM25 Indexing**: Zero-dependency Okapi BM25 keyword search using the Robertson-Spärck Jones formulation.
- **Hybrid Retrieval**: Search combining dense embeddings and BM25 sparse lexical scores via Reciprocal Rank Fusion (RRF).
- **Rich Metadata Filtering**: Pre-filtering query pipeline supporting comparison operators (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`).
- **Batch Search**: Query batching (`search_batch`) optimizing embedding model throughput.
- **In-Place Updates & Upserts**: `update()` and `upsert()` API on `Client` and `StorageEngine` with full Write-Ahead Log crash recovery replay.
- **SimpleV CLI**: Full command line interface with `init`, `ingest`, `search`, `info`, and `compact` commands with JSON piping for shell automation.
- **Automated Benchmarking Suite**: Formal reproducible benchmark in `examples/benchmarks.py` tracking ingestion speed, query latency, and disk footprint.
- **Comprehensive Test Suite**: Expanded test coverage to 226+ passing tests across all components.

## [0.3.0] - 2026-08-30

### Added
- **Document Parsers**: Standalone parsers for `.txt`, `.md`, `.csv`, `.pdf` (pymupdf), and `.docx` (python-docx) with `auto_parse()` format detection.
- **Text Chunking**: Sliding window text splitting with configurable chunk size, character overlap, and delimiter-aware boundaries (`chunk_text`, `chunk_document`).

## [0.2.0] - 2026-08-30

### Added
- **Binary Persistence**: Custom `.sv` binary format with 64-byte header, bit-packed tombstone bitmap, length-prefixed JSON metadata, and contiguous float32 vector block.
- **Write-Ahead Logging (WAL)**: Append-only JSONL log with fsync guarantees for crash recovery and atomic commit truncation.

## [0.1.0] - 2026-08-21

### Added
- **Core Storage Engine**: In-memory vector store with geometric array allocation and tombstone deletion masking.
- **Flat Index**: Exact nearest neighbor search supporting Cosine similarity and L2 Euclidean distance.
- **Query Engine**: Modular 6-step search pipeline (validate, embed, pre-filter, index search, hydrate, format).
- **Python Client SDK**: Clean facade (`Client`) exposing `add`, `insert`, `search`, `delete`, and `compact`.