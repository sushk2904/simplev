# SimpleV — Architecture Diagrams

> **Source of truth:** All diagrams below are derived exclusively from the specifications defined in
> [`ARCHITECTURE.md`](ARCHITECTURE.md), [`QUERY_ENGINE.md`](QUERY_ENGINE.md),
> [`STORAGE_ENGINE.md`](STORAGE_ENGINE.md), [`INDEXING.md`](INDEXING.md),
> [`FILE_FORMAT.md`](FILE_FORMAT.md), and [`WAL.md`](WAL.md).
> No features have been invented beyond what is documented.

---

## Table of Contents

1. [Detailed System Architecture](#1-detailed-system-architecture)
2. [Query Execution Pipeline](#2-query-execution-pipeline-6-step-search)
3. [Storage & WAL Lifecycle](#3-storage--wal-lifecycle)
4. [Crash Recovery Boot Sequence](#4-crash-recovery-boot-sequence)
5. [`.sv` Binary File Format Layout](#5-sv-binary-file-format-layout)
6. [Data Flow Contracts Summary](#6-data-flow-contracts-summary)

---

## 1. Detailed System Architecture

Expands on the high-level diagram in [`ARCHITECTURE.md`](ARCHITECTURE.md). Shows exact module boundaries, bidirectional data contracts between layers, and the two on-disk persistence artifacts (`.sv` and `.wal`).

**Key design decisions:**
- The **Query Engine** is the sole orchestrator — it is the only component that touches both the Indexing Engine and the Storage Engine.
- The **Indexing Engine** has a read-only, indirect relationship with the Storage Engine (it operates on vectors already loaded into memory by Storage).
- The **Embeddings Module** is a stateless utility; it caches models locally but holds no database state.
- Two distinct file types exist on disk: the canonical `.sv` binary and the append-only `.sv.wal` journal.

```mermaid
graph TB
    subgraph CLIENT ["Client Layer"]
        SDK["Python SDK<br/><i>simplev.Client</i>"]
        CLI["CLI Interface"]
    end

    subgraph QUERY ["Query Engine — Central Orchestrator"]
        VAL["Validation"]
        PIPE["Search Pipeline<br/>Coordination"]
        ERR["Error Handling<br/>& Normalization"]
    end

    subgraph EMBED ["Embeddings Module"]
        EMB["sentence-transformers<br/>wrapper"]
        CACHE["Local Model Cache"]
        EMB --- CACHE
    end

    subgraph INDEX ["Indexing Engine"]
        FLAT["Flat Index<br/><i>v0.1 — Exact NN</i>"]
        HNSW["HNSW Graph<br/><i>v0.4 — ANN</i>"]
        METRICS["Distance Metrics<br/>Cosine · L2"]
        FLAT --- METRICS
        HNSW --- METRICS
    end

    subgraph STORAGE ["Storage Engine"]
        MEM["In-Memory State<br/>Vector Arrays · Metadata Dicts<br/>Tombstone Bitmask"]
        WAL_MGR["WAL Manager<br/>Append · Replay · Truncate"]
        SER["Serializer<br/>.sv Read / Write"]
    end

    subgraph DISK ["Local File System"]
        SV_FILE[".sv File<br/><i>Header · Tombstones · Meta · Vectors</i>"]
        WAL_FILE[".sv.wal File<br/><i>Append-only JSONL</i>"]
    end

    SDK -- "insert / delete / search / commit" --> VAL
    CLI -- "CLI commands" --> VAL
    VAL --> PIPE
    PIPE -- "text → vector" --> EMB
    PIPE -- "query vector + bitmask" --> FLAT
    PIPE -- "query vector + bitmask" --> HNSW
    PIPE -- "pre-filter metadata<br/>hydrate results" --> MEM
    PIPE --> ERR

    MEM -- "WAL append before<br/>memory mutation" --> WAL_MGR
    WAL_MGR -- "append JSONL" --> WAL_FILE
    MEM -- "commit serialization" --> SER
    SER -- "full state write" --> SV_FILE
    SER -- "boot: load state" --> MEM
    WAL_MGR -- "boot: replay ops" --> MEM
    WAL_MGR -- "post-commit: truncate" --> WAL_FILE

    style CLIENT fill:#1e293b,stroke:#60a5fa,color:#e2e8f0
    style QUERY fill:#1e293b,stroke:#a78bfa,color:#e2e8f0
    style EMBED fill:#1e293b,stroke:#34d399,color:#e2e8f0
    style INDEX fill:#1e293b,stroke:#f59e0b,color:#e2e8f0
    style STORAGE fill:#1e293b,stroke:#f87171,color:#e2e8f0
    style DISK fill:#0f172a,stroke:#94a3b8,color:#cbd5e1
```

---

## 2. Query Execution Pipeline (6-Step Search)

Maps the strict, sequential pipeline defined in [`QUERY_ENGINE.md`](QUERY_ENGINE.md). Each step is annotated with the responsible module and the data type crossing the boundary.

**Key design decisions:**
- Pre-filtering uses **bitwise operations** on the metadata cache to produce a boolean bitmask — this keeps the Indexing Engine free of dictionary-lookup overhead.
- Hydration is a separate read path back into the Storage Engine; the Indexing Engine returns only `(doc_id, score)` tuples.
- The response is a standardized `List[dict]` — the only object the Client ever receives.

```mermaid
flowchart TD
    START(["search(query, top_k, filter)"])

    V["<b>① Validation</b><br/><i>Query Engine</i><br/>─────────────<br/>• Verify query string is non-empty<br/>• Validate top_k bounds<br/>• Parse & validate filter dict syntax"]

    E["<b>② Embedding</b><br/><i>Embeddings Module</i><br/>─────────────<br/>• Convert query text → dense float32 vector<br/>• Uses locally-cached sentence-transformer model"]

    PF["<b>③ Pre-Filtering</b><br/><i>Query Engine → Storage Engine</i><br/>─────────────<br/>• Evaluate metadata filter against in-memory metadata cache<br/>• Generate boolean bitmask of eligible doc IDs<br/>• Bitwise AND with active tombstone bitmask"]

    EX["<b>④ Execution</b><br/><i>Indexing Engine</i><br/>─────────────<br/>• Receive: query vector + bitmask<br/>• Flat Index: brute-force cosine/L2 over eligible vectors<br/>• Return: ranked list of (doc_id, similarity_score)"]

    HY["<b>⑤ Hydration</b><br/><i>Query Engine → Storage Engine</i><br/>─────────────<br/>• Request original text + metadata for returned doc IDs<br/>• Instant lookup from in-memory metadata dicts"]

    RF["<b>⑥ Response Formulation</b><br/><i>Query Engine</i><br/>─────────────<br/>• Pack doc_id, text, metadata, score into List of dict<br/>• Return standardized result to Client"]

    RESULT(["List[dict] → Client"])

    START --> V
    V -->|"valid request"| E
    V -->|"invalid"| ERR_OUT(["Graceful Error"])
    E -->|"query_vector: float32 array"| PF
    PF -->|"query_vector + bitmask"| EX
    EX -->|"List of doc_id, score"| HY
    HY -->|"hydrated records"| RF
    RF --> RESULT

    style START fill:#0f172a,stroke:#60a5fa,color:#e2e8f0
    style RESULT fill:#0f172a,stroke:#34d399,color:#e2e8f0
    style ERR_OUT fill:#0f172a,stroke:#f87171,color:#fca5a5
    style V fill:#1e293b,stroke:#a78bfa,color:#e2e8f0
    style E fill:#1e293b,stroke:#34d399,color:#e2e8f0
    style PF fill:#1e293b,stroke:#60a5fa,color:#e2e8f0
    style EX fill:#1e293b,stroke:#f59e0b,color:#e2e8f0
    style HY fill:#1e293b,stroke:#60a5fa,color:#e2e8f0
    style RF fill:#1e293b,stroke:#a78bfa,color:#e2e8f0
```

---

## 3. Storage & WAL Lifecycle

Sequence diagram covering the full write lifecycle as specified in [`WAL.md` §2](WAL.md) and [`STORAGE_ENGINE.md` §3](STORAGE_ENGINE.md). Covers `insert()`, `delete()`, and `commit()` — the three mutating operations from [`API_SPEC.md`](API_SPEC.md).

**Key design decisions:**
- **Write ordering is strict:** WAL append happens _before_ the in-memory mutation. This is the fundamental durability guarantee.
- `delete()` does not physically remove data — it sets a tombstone bit in the bitmask. Actual removal happens at compaction time during `commit()`.
- `commit()` serializes the _entire_ current memory state to `.sv`, then truncates the WAL. This is a full-state checkpoint, not an incremental patch.

```mermaid
sequenceDiagram
    participant C as Client SDK
    participant QE as Query Engine
    participant SE as Storage Engine
    participant MEM as In-Memory State
    participant WAL as .sv.wal File
    participant SV as .sv File

    note over C,SV: ── insert() Lifecycle ──

    C->>QE: insert(doc_id, text, metadata)
    QE->>QE: Validate input & embed text → vector
    QE->>SE: Store (doc_id, text, metadata, vector)
    SE->>WAL: 1. Append INSERT op as JSONL entry
    WAL-->>SE: Write confirmed (fsync)
    SE->>MEM: 2. Append vector to contiguous array
    SE->>MEM: 2. Add metadata to dict
    SE->>MEM: 2. Set tombstone bit = 1 (active)
    SE-->>QE: Success
    QE-->>C: return True

    note over C,SV: ── delete() Lifecycle ──

    C->>QE: delete(doc_id)
    QE->>SE: Mark doc_id as deleted
    SE->>WAL: 1. Append DELETE op as JSONL entry
    WAL-->>SE: Write confirmed (fsync)
    SE->>MEM: 2. Set tombstone bit = 0 (inactive)
    note right of MEM: Soft delete only —<br/>vector data remains<br/>until compaction
    SE-->>QE: Success
    QE-->>C: return True

    note over C,SV: ── commit() Lifecycle ──

    C->>QE: commit()
    QE->>SE: Trigger full state serialization
    SE->>SE: Run compaction (remove tombstoned records)
    SE->>SV: Serialize: Header → Tombstones → Metadata → Vectors
    SV-->>SE: Write complete
    SE->>WAL: Truncate .wal file (empty)
    SE-->>QE: Commit complete
    QE-->>C: return None
```

---

## 4. Crash Recovery Boot Sequence

State diagram showing the initialization path defined in [`WAL.md` §3](WAL.md). This is the sequence the Storage Engine follows every time `Client.__init__(db_path)` is called with a file-backed path.

**Key design decisions:**
- The system performs **deterministic replay:** WAL entries are applied sequentially in the exact order they were logged, preserving causal consistency.
- After replay, an **immediate forced commit** occurs — this collapses the WAL into the `.sv` file so the recovery path is never re-executed on the same entries.
- If no WAL exists or the WAL is empty, boot is a simple `.sv` load with no replay overhead.

```mermaid
stateDiagram-v2
    [*] --> LoadSVFile: Client.__init__(db_path)

    LoadSVFile: Load .sv File
    note right of LoadSVFile
        Deserialize Header → Tombstones
        → Metadata → Vectors into memory
    end note

    LoadSVFile --> CheckWAL

    CheckWAL: Check for .sv.wal file
    CheckWAL --> NoWAL: WAL missing or empty
    CheckWAL --> WALFound: Non-empty WAL exists

    NoWAL: No Recovery Needed
    NoWAL --> Ready

    WALFound: Ungraceful Shutdown Detected

    WALFound --> ReplayWAL

    ReplayWAL: Replay WAL Entries
    note right of ReplayWAL
        Read JSONL sequentially
        Apply each INSERT / DELETE
        to in-memory state
    end note

    ReplayWAL --> ForceCommit

    ForceCommit: Force Commit
    note right of ForceCommit
        Serialize full memory state → .sv
        Truncate .wal file
    end note

    ForceCommit --> Ready

    Ready: Database Ready
    note right of Ready
        Accept queries
        and write operations
    end note

    Ready --> [*]
```

---

## 5. `.sv` Binary File Format Layout

Visual representation of the sequential byte layout described in [`FILE_FORMAT.md`](FILE_FORMAT.md). The `.sv` file is structured sequentially into four distinct blocks, ensuring minimal read/write overhead.

**Key design decisions:**
- The header is **fixed at 64 bytes** — any tool can read the first 64 bytes to determine file validity, format version, vector dimensions, and record count without parsing the rest.
- The tombstone bitmap is **bit-packed** (1 bit per document), making it extremely compact: 1 million documents require only ~122 KB.
- The vector block is placed **last** by design — this enables append-only vector writes and allows NumPy `memmap` / `frombuffer` zero-copy loading.
- Metadata is serialized via MessagePack or JSON-lines, providing a pragmatic balance between parse speed and human debuggability.

```mermaid
block-beta
    columns 1

    block:HEADER:1
        columns 4
        H1["Magic Bytes<br/><code>SV01</code><br/>(4 bytes)"]
        H2["Endianness<br/>Marker<br/>(variable)"]
        H3["Dimension<br/>Size<br/>(e.g. 384)"]
        H4["Document<br/>Count<br/>(uint32)"]
    end

    space

    block:TOMBSTONE:1
        columns 1
        T1["Tombstone Bitmap<br/>─────────────────────<br/>1 bit per document<br/>1 = active · 0 = deleted<br/>─────────────────────<br/>Size: ⌈doc_count / 8⌉ bytes"]
    end

    space

    block:METADATA:1
        columns 1
        M1["Metadata Block<br/>─────────────────────<br/>Serialized via MessagePack or JSONL<br/>─────────────────────<br/>Per record: doc_id · raw text · user metadata dict<br/>Sequential read for hydration"]
    end

    space

    block:VECTORS:1
        columns 1
        V1["Vector Block<br/>─────────────────────<br/>Contiguous float32 binary blob<br/>─────────────────────<br/>Total size: doc_count × dimension × 4 bytes<br/>Direct NumPy load · append-friendly position"]
    end

    HEADER --> TOMBSTONE
    TOMBSTONE --> METADATA
    METADATA --> VECTORS

    style HEADER fill:#1e3a5f,stroke:#60a5fa,color:#e2e8f0
    style TOMBSTONE fill:#3b1f2b,stroke:#f87171,color:#fca5a5
    style METADATA fill:#1a3329,stroke:#34d399,color:#a7f3d0
    style VECTORS fill:#3b2f1a,stroke:#f59e0b,color:#fde68a
```

> **Note:** The 64-byte header is fixed-length regardless of database size. All variable-length sections (tombstones, metadata, vectors) scale linearly with document count. The vector block dominates file size: a 100K-document database at 384 dimensions occupies ~147 MB for vectors alone (`100,000 × 384 × 4 bytes`).

---

## 6. Data Flow Contracts Summary

| Boundary | Data Crossing | Format |
|---|---|---|
| Client → Query Engine | `insert(doc_id, text, metadata)` / `search(query, top_k, filter)` / `delete(doc_id)` / `commit()` | Python method calls |
| Query Engine → Embeddings | Raw text string | `str → float32[]` |
| Query Engine → Indexing Engine | Query vector + boolean bitmask | `(ndarray, ndarray)` |
| Indexing Engine → Query Engine | Ranked results | `List[(doc_id, score)]` |
| Query Engine ↔ Storage Engine | Pre-filter metadata lookup / Hydration request | `dict` / `List[doc_id]` |
| Storage Engine → WAL | Operation record | JSONL append |
| Storage Engine → `.sv` File | Full serialized state | Custom binary (Header · Tombstones · Meta · Vectors) |
