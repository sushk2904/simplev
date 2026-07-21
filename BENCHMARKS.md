## 1. Overview
Performance transparency is crucial for a database project. While SimpleV does not aim to compete with enterprise distributed systems, we hold ourselves to strict performance standards for local environments. This document outlines our baseline hardware assumptions, testing datasets, and the performance targets we expect the database to hit reliably.

## 2. Testing Environment
To ensure our benchmarks reflect real-world developer setups, we establish the following baseline:
* **Hardware:** Standard consumer hardware, such as an Apple Silicon Mac (M1/M2) or an equivalent modern x86 CPU. 
* **Dataset:** 10,000 synthetically generated documents.
* **Embeddings:** 384-dimensional float32 vectors (the output size of our default model).

## 3. Performance Targets
Our goal is to maintain a snappy, unnoticeable latency footprint for local applications. We target the following metrics for our core index implementation:

* **Ingestion Speed:** Greater than 500 documents per second. This metric includes the overhead of passing text through the embedding model and persisting the resulting vectors to disk.
* **Query Latency (Flat Search):** Less than 10 milliseconds to retrieve the top 10 nearest neighbors from a pool of 10,000 vectors. 
* **Storage Footprint:** The total disk usage for 10,000 vectors, along with their associated metadata, should remain under 20 megabytes.

## 4. Reproducibility
We believe benchmarks should be transparent and easily reproducible. All benchmarking scripts are included directly within the repository. Developers can verify our performance claims on their own hardware by running the automated tests located in the `/examples/benchmarks.py` directory.