# Roadmap: SimpleV

Our development roadmap is structured to deliver value incrementally. We are focused on building a solid foundation before expanding into advanced features and optimizations.

## v0.1: The Foundation
* Implement in-memory vector storage.
* Enable Flat L2 and Cosine similarity search.
* Establish the basic API functionality, including `insert`, `search`, and `delete` operations.

## v0.2: Persistence
* Introduce disk-backed storage using our custom `.sv` file format.
* Support metadata JSON storage alongside vectors, enabling basic filtering capabilities.

## v0.3: Ingestion
* Build automatic parsing tools for common document types, including PDF, DOCX, TXT, and Markdown.
* Develop chunking utilities to handle larger texts efficiently.

## v0.4: Performance
* Implement Hierarchical Navigable Small World (HNSW) indexing for faster, approximate searches.
* Create a Command Line Interface (CLI) for easier database management.
* Release a formal benchmarking suite to track our performance metrics.

## Future Milestones
* Introduce a Write-Ahead Log (WAL) for better data durability.
* Build a REST API for networked usage.
* Explore Rust and PyO3 bindings to optimize performance-critical modules.
* Add vector compression techniques to reduce storage footprints.