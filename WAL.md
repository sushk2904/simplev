# Write-Ahead Log (WAL)

## 1. Overview
Because SimpleV performs its primary mathematical operations in memory to ensure maximum speed, there is an inherent risk of data loss if the application crashes or loses power before the memory state is saved to the disk. To guarantee data durability without sacrificing write performance, SimpleV implements a Write-Ahead Log (WAL).

## 2. The Write Lifecycle
The WAL acts as a safety net, ensuring that every modification is recorded to disk before it is applied to the active memory state. 

1. **Operation Request:** A user calls an `insert()` or `delete()` operation via the client.
2. **Log Append:** Before any arrays or dictionaries in memory are touched, the operation and its associated data are appended to a lightweight `my_db.sv.wal` file. We use a simple, append-only JSON-lines (JSONL) format for this log to ensure write operations are nearly instantaneous.
3. **Memory Update:** Once the log confirms the write to disk, the Storage Engine updates the active in-memory structures.
4. **Synchronization:** When the user explicitly calls `commit()`, or when the WAL file reaches a predefined size threshold, the Storage Engine serializes the entire current memory state into the primary `.sv` file.
5. **Truncation:** After a successful save to the main `.sv` file, the `.wal` file is safely truncated (emptied) to begin logging fresh operations.

## 3. Crash Recovery Mechanism
The true value of the WAL is realized during database initialization. When a SimpleV client connects to an existing database path, it follows a strict boot sequence:

1. It loads the primary `.sv` file into memory.
2. It checks for the existence of a `.wal` file in the same directory.
3. If a non-empty `.wal` file is found, it means the database experienced an ungraceful shutdown before the last commit.
4. SimpleV sequentially reads the JSONL entries in the WAL and replays those missing inserts and deletes directly into memory.
5. It immediately forces a background commit to the main `.sv` file and clears the log, returning the database to a perfectly consistent state before accepting new queries.