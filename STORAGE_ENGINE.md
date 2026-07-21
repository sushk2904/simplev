## 1. Overview
The Storage Engine is the foundational data layer of SimpleV. Its primary objective is to bridge the gap between high-speed, in-memory mathematical operations and the safety of persistent disk storage. We have designed this engine to be lightweight and transparent, ensuring that user data remains safe across application restarts without requiring heavy background processes.

## 2. Core Responsibilities
* **State Management:** Maintaining the active working set of vectors and document metadata.
* **Disk Synchronization:** Writing in-memory structures to the local filesystem efficiently.
* **Data Durability:** Preventing data loss during unexpected crashes or application terminations.
* **Compaction:** Managing the cleanup of deleted records to maintain optimal performance and file size.

## 3. Core Mechanisms

### 3.1. In-Memory Operations
To achieve minimal latency during semantic searches, SimpleV operates primarily in memory. Active vectors are loaded into contiguous arrays, allowing the indexing and query engines to perform rapid mathematical operations. Document metadata is held in corresponding memory structures to allow for instant post-search hydration and filtering.

### 3.2. Disk Persistence (.sv files)
When the database state needs to be saved, the Storage Engine serializes the in-memory arrays and metadata dictionaries into our custom `.sv` binary file format. This file serves as the definitive source of truth. Upon initialization, the Storage Engine reads the `.sv` file to rebuild the necessary memory structures, bringing the database back to its exact previous state.

### 3.3. Write-Ahead Logging (WAL)
Relying solely on periodic disk flushes can lead to data loss if the application crashes unexpectedly. To mitigate this, the Storage Engine will utilize a Write-Ahead Log. Every write operation (insert or delete) is sequentially appended to a lightweight log file before the memory structures are updated. In the event of a crash, the engine will read the WAL upon the next startup and seamlessly replay any operations that were not yet committed to the main `.sv` file.

## 4. Deletion Strategy
Immediately resizing large contiguous arrays upon every deletion is computationally expensive. To maintain high throughput for write operations, the Storage Engine employs a soft-deletion strategy using tombstones. 

When a document is deleted, its index is marked as inactive via a bitmask. The query engine is aware of this bitmask and simply skips tombstoned records during searches. The actual removal of data from memory and disk occurs during a background compaction process, which is triggered manually via a commit or automatically during database initialization.