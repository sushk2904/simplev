"""
Reproducible Benchmark Suite for SimpleV.

Benchmarks ingestion throughput, query latency (Flat vs. HNSW), and storage
footprint as specified in Info/BENCHMARKS.md:
  - Ingestion target: > 500 docs/second
  - Query latency target (top-10 on 10,000 vectors): < 10 ms
  - Storage footprint target (10,000 docs + 384-dim vectors): < 20 MB

Usage:
  python examples/benchmarks.py [--docs 10000] [--dim 384] [--queries 50] [--quick]
"""

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Ensure simplev is importable even if run directly as script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simplev.indexing import FlatIndex, HNSWIndex  # noqa: E402
from simplev.persistence import FileManager  # noqa: E402
from simplev.storage import StorageEngine  # noqa: E402


def generate_synthetic_vectors(
    n_docs: int, dimension: int, seed: int = 42
) -> np.ndarray:
    """Generate normalized float32 synthetic vectors."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_docs, dimension), dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return raw / norms


def run_benchmarks(
    n_docs: int = 10000, dimension: int = 384, n_queries: int = 50
) -> dict:
    """Run end-to-end benchmark suite."""
    print("=" * 65)
    print("           SIMPLEV PERFORMANCE BENCHMARK SUITE")
    print("=" * 65)
    print(
        f"Configuration: {n_docs:,} documents | "
        f"{dimension} dimensions | {n_queries} queries\n"
    )

    results = {}

    # 1. Dataset Generation
    print("-> Generating synthetic dataset...")
    t0 = time.perf_counter()
    vectors = generate_synthetic_vectors(n_docs, dimension)
    gen_time = time.perf_counter() - t0
    print(f"   Generated {n_docs:,} vectors in {gen_time:.3f}s\n")

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "benchmark.sv"

        # 2. Ingestion Benchmark
        print("-> Benchmarking Ingestion & Persistence...")
        storage = StorageEngine(dimension=dimension)
        t_start = time.perf_counter()

        for i in range(n_docs):
            doc_id = f"doc_{i:06d}"
            text = (
                f"Synthetic benchmark document payload for index {i} "
                "with metadata attributes."
            )
            meta = {"idx": i, "category": f"cat_{i % 10}"}
            storage.add(doc_id=doc_id, text=text, vector=vectors[i], metadata=meta)

        t_add = time.perf_counter() - t_start
        add_rate = n_docs / t_add if t_add > 0 else float("inf")

        # Persistence to disk (.sv file)
        fm = FileManager()
        t_save_start = time.perf_counter()
        fm.save(db_path, storage)
        t_save = time.perf_counter() - t_save_start
        total_ingest_time = t_add + t_save
        overall_ingest_rate = n_docs / total_ingest_time

        file_size_bytes = db_path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)

        results["ingestion_rate"] = overall_ingest_rate
        results["storage_mb"] = file_size_mb

        print(f"   In-memory addition : {add_rate:,.1f} docs/sec ({t_add:.3f}s)")
        print(f"   Disk serialization : {t_save:.3f}s")
        print(
            f"   Overall throughput : {overall_ingest_rate:,.1f} docs/sec "
            "(Target: > 500 docs/sec)"
        )
        print(f"   Database file size : {file_size_mb:.2f} MB (Target: < 20 MB)\n")

        # 3. Query Latency: FlatIndex
        print(f"-> Benchmarking Flat Index Search (Top-10, {n_queries} queries)...")
        flat_index = FlatIndex(metric="cosine")
        query_vectors = generate_synthetic_vectors(n_queries, dimension, seed=999)

        flat_latencies = []
        for q in query_vectors:
            t_q = time.perf_counter()
            _ = flat_index.search(q, vectors, top_k=10)
            flat_latencies.append((time.perf_counter() - t_q) * 1000)  # ms

        flat_mean = float(np.mean(flat_latencies))
        flat_p50 = float(np.percentile(flat_latencies, 50))
        flat_p95 = float(np.percentile(flat_latencies, 95))
        results["flat_latency_ms"] = flat_mean

        print(f"   Flat Avg Latency   : {flat_mean:.2f} ms (Target: < 10 ms)")
        print(f"   Flat P50 Latency   : {flat_p50:.2f} ms")
        print(f"   Flat P95 Latency   : {flat_p95:.2f} ms\n")

        # 4. HNSW Index Construction & Query Latency
        print("-> Benchmarking HNSW Graph Index (M=16, efSearch=50)...")
        t_hnsw_build = time.perf_counter()
        hnsw_index = HNSWIndex(
            metric="cosine", m=16, ef_construction=64, ef_search=50, seed=42
        )
        hnsw_index.build(vectors)
        hnsw_build_time = time.perf_counter() - t_hnsw_build
        speed_nodes = n_docs / hnsw_build_time if hnsw_build_time > 0 else 0
        print(
            f"   HNSW Build Time    : {hnsw_build_time:.2f}s "
            f"({speed_nodes:,.1f} nodes/sec)"
        )

        hnsw_latencies = []
        recall_scores = []
        for q in query_vectors:
            t_q = time.perf_counter()
            hnsw_res = hnsw_index.search(q, vectors, top_k=10)
            hnsw_latencies.append((time.perf_counter() - t_q) * 1000)

            # Compute recall against exact flat search
            flat_res = flat_index.search(q, vectors, top_k=10)
            flat_ids = set(r[0] for r in flat_res)
            hnsw_ids = set(r[0] for r in hnsw_res)
            recall = len(flat_ids & hnsw_ids) / 10.0
            recall_scores.append(recall)

        hnsw_mean = float(np.mean(hnsw_latencies))
        hnsw_p50 = float(np.percentile(hnsw_latencies, 50))
        hnsw_p95 = float(np.percentile(hnsw_latencies, 95))
        avg_recall = float(np.mean(recall_scores))

        results["hnsw_latency_ms"] = hnsw_mean
        results["hnsw_recall"] = avg_recall

        print(f"   HNSW Avg Latency   : {hnsw_mean:.2f} ms")
        print(f"   HNSW P50 Latency   : {hnsw_p50:.2f} ms")
        print(f"   HNSW P95 Latency   : {hnsw_p95:.2f} ms")
        print(f"   HNSW Top-10 Recall : {avg_recall * 100:.1f}%\n")

    # 5. Summary Table
    print("=" * 65)
    print("                    PERFORMANCE TARGET SUMMARY")
    print("=" * 65)
    target_ingest = 500.0
    target_latency = 10.0
    target_size = 20.0

    pass_ingest = results["ingestion_rate"] >= target_ingest
    pass_latency = results["flat_latency_ms"] <= target_latency
    pass_size = results["storage_mb"] <= target_size

    def status(passed):
        return "[PASS]" if passed else "[FAIL]"

    print(" Metric                  | Measured Value      | Target       | Status")
    print("-------------------------+---------------------+--------------+-------")
    ingest_str = f"{results['ingestion_rate']:>12,.1f} docs/s"
    s_ingest = status(pass_ingest)
    print(f" Ingestion Speed         | {ingest_str} | > 500 docs/s | {s_ingest}")
    flat_str = f"{results['flat_latency_ms']:>12.2f} ms"
    s_lat = status(pass_latency)
    print(f" Flat Search Latency     | {flat_str}     | < 10.0 ms    | {s_lat}")
    storage_str = f"{results['storage_mb']:>12.2f} MB"
    s_size = status(pass_size)
    print(f" Storage Footprint       | {storage_str}     | < 20.0 MB    | {s_size}")
    hnsw_str = f"{results['hnsw_latency_ms']:>12.2f} ms"
    print(f" HNSW Latency (ANN)      | {hnsw_str}     | Sub-linear   | [INFO]")
    recall_str = f"{results['hnsw_recall'] * 100:>11.1f} %"
    print(f" HNSW Top-10 Recall      | {recall_str}      | > 80.0 %     | [INFO]")
    print("=" * 65)

    all_passed = pass_ingest and pass_latency and pass_size
    if all_passed:
        print("\nALL PERFORMANCE TARGETS SATISFIED!\n")
    else:
        print("\nONE OR MORE BENCHMARK TARGETS WERE NOT MET.\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="SimpleV Benchmark Suite")
    parser.add_argument(
        "--docs", type=int, default=10000, help="Number of documents (default: 10000)"
    )
    parser.add_argument(
        "--dim", type=int, default=384, help="Embedding dimensions (default: 384)"
    )
    parser.add_argument(
        "--queries", type=int, default=50, help="Number of test queries (default: 50)"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode (1,000 docs, 20 queries)"
    )
    args = parser.parse_args()

    n_docs = 1000 if args.quick else args.docs
    n_queries = 20 if args.quick else args.queries

    run_benchmarks(n_docs=n_docs, dimension=args.dim, n_queries=n_queries)


if __name__ == "__main__":
    main()
