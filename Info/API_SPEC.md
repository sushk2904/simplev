This document outlines the primary Python interfaces for interacting with SimpleV. Our goal is to keep this API as intuitive and unobtrusive as possible, allowing developers to focus on building their applications rather than managing database connections.

## Core Client: `simplev.Client`

The `Client` class is the main entry point for all database operations.

### `__init__(db_path: str = ":memory:")`
Initializes the database connection. By default, it operates entirely in memory for quick prototyping. Providing a file path will enable disk-backed persistence.

### `insert(doc_id: str, text: str, metadata: dict = None) -> bool`
Automatically converts the provided text into a vector embedding and stores it. You may also attach optional dictionary metadata to assist with future filtering. Returns a boolean indicating success.

### `search(query: str, top_k: int = 5, filter: dict = None) -> List[dict]`
Accepts a natural language query, embeds it, and returns the top-K semantically similar documents from the database. The optional filter dictionary allows you to narrow down results based on stored metadata.

### `delete(doc_id: str) -> bool`
Performs a soft deletion of the specified document and its associated vector. Returns a boolean indicating success.

### `commit() -> None`
Manually flushes all pending in-memory operations to the underlying `.sv` disk file. This is particularly useful when Write-Ahead Logging (WAL) is enabled to ensure strict data durability.