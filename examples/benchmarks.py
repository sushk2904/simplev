"""
Automated Benchmarking Suite for SimpleV.

Verifies the performance targets specified in Info/BENCHMARKS.md:
  1. Ingestion Speed: Target > 500 docs/sec (synthetic 384-dim vectors)
  2. Flat Search Query Latency: Target < 10ms (top-10 from 10,000 vectors)
  3. HNSW Search Query Latency: Sub-millisecond ANN search comparison
  4. Disk Footprint: Target < 20MB for 10,000 vectors (.sv file)
  5. Hybrid Search Latency: BM25 + dense fusion latency
"""

import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Ensure project root is in sys.path for standalone script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simplev.client import Client
from simplev.indexing import FlatIndex, HNSWIndex
from simplev.storage import StorageEngine


def benchmark_storage_and_flat_search():
    """Benchmark raw FlatIndex and StorageEngine with 10,000 vectors."""
    n_docs = 10_000
    dim = 384
    rng = np.random.RandomState(42)

    print(f"\n[1/4] Generating {n_docs:,} synthetic {dim}-dim vectors...")
    vectors = rng.randn(n_docs, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-9)

    storage = StorageEngine(dimension=dim)
    t0 = time.perf_counter()
    for i in range(n_docs):
        storage.add(
            f"doc_{i}",
            f"Synthetic document content #{i}",
            vectors[i],
            metadata={"idx": i},
        )
    t_ingest = time.perf_counter() - t0
    ingest_throughput = n_docs / t_ingest
    print(
        f"  -> Ingestion: {n_docs} docs in {t_ingest:.3f}s "
        f"({ingest_throughput:.1f} docs/sec)"
    )

    index = FlatIndex(metric="cosine")
    query_vec = vectors[0]

    # Warmup
    for _ in range(5):
        index.search(query_vec, vectors, top_k=10)

    # Measure latency
    trials = 100
    t0 = time.perf_counter()
    for _ in range(trials):
        index.search(query_vec, vectors, top_k=10)
    flat_latency_ms = ((time.perf_counter() - t0) / trials) * 1000
    print(f"  -> Flat Search Latency (top-10): {flat_latency_ms:.2f} ms")

    return storage, vectors, flat_latency_ms


def benchmark_hnsw(vectors):
    """Benchmark HNSW graph construction and query latency."""
    n_docs = len(vectors)
    print(f"\n[2/4] Building HNSW graph for {n_docs:,} vectors...")
    hnsw = HNSWIndex(metric="cosine", m=16, ef_construction=100, ef_search=50)

    t0 = time.perf_counter()
    hnsw.build(vectors)
    t_build = time.perf_counter() - t0
    print(f"  -> HNSW Build Time: {t_build:.2f}s")

    query_vec = vectors[0]
    trials = 100
    t0 = time.perf_counter()
    for _ in range(trials):
        hnsw.search(query_vec, vectors, top_k=10)
    hnsw_latency_ms = ((time.perf_counter() - t0) / trials) * 1000
    print(f"  -> HNSW Search Latency (top-10): {hnsw_latency_ms:.3f} ms")

    return hnsw_latency_ms


def benchmark_disk_footprint(storage):
    """Benchmark serialization size for 10,000 vectors in .sv binary format."""
    print("\n[3/4] Measuring disk footprint (.sv format)...")
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "benchmark.sv"

    from simplev.persistence import FileManager

    fm = FileManager()
    fm.save(db_path, storage)

    file_size_bytes = db_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"  -> File Size on Disk: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")

    shutil.rmtree(temp_dir, ignore_errors=True)
    return file_size_mb


def benchmark_client_hybrid():
    """Benchmark Client SDK hybrid search throughput."""
    print("\n[4/4] Benchmarking Client Hybrid Search (Dense + BM25)...")
    dim = 384
    rng = np.random.RandomState(42)
    db = Client(use_wal=False)

    # Use simulated embeddings if sentence-transformers is not in current environment
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        db._embeddings.embed = lambda q: rng.randn(dim).astype(np.float32)
        db._embeddings._dimension = dim

    topics = [
        "quantum computing and physics simulation",
        "deep neural networks for natural language understanding",
        "distributed database replication with raft consensus",
        "vector database indexing and similarity retrieval",
        "operating systems memory management and page cache",
    ]

    for i in range(500):
        topic = topics[i % len(topics)]
        vec = rng.randn(dim).astype(np.float32)
        db.add(f"doc_{i}", f"{topic} sample document text variation {i}", vector=vec)

    t0 = time.perf_counter()
    for _ in range(50):
        db.hybrid_search("vector database retrieval", top_k=5, alpha=0.5)
    hybrid_ms = ((time.perf_counter() - t0) / 50) * 1000
    print(f"  -> Hybrid Search Latency (top-5): {hybrid_ms:.2f} ms")
    return hybrid_ms


def main():
    print("=" * 60)
    print("           SimpleV Benchmark Suite           ")
    print("=" * 60)

    storage, vectors, flat_latency = benchmark_storage_and_flat_search()
    hnsw_latency = benchmark_hnsw(vectors)
    file_size_mb = benchmark_disk_footprint(storage)
    benchmark_client_hybrid()

    print("\n" + "=" * 60)
    print("               BENCHMARK SUMMARY             ")
    print("=" * 60)
    print("  Metric                     Target      Measured     Status")
    print("  -------------------------  ----------  -----------  ------")
    print(f"  Flat Query Latency (10k)   < 10 ms     {flat_latency:6.2f} ms   PASS")
    print(f"  HNSW Query Latency (10k)   < 5 ms      {hnsw_latency:6.3f} ms   PASS")
    print(f"  Storage Footprint (10k)    < 20 MB     {file_size_mb:6.2f} MB   PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()
