"""
SimpleV - The SQLite for Vector Search.

A local-first, zero-configuration vector database designed
for AI applications, RAG, and personal knowledge bases.
"""

__version__ = "0.1.0.dev1"

from simplev.client import Client
from simplev.indexing import FlatIndex, HNSWIndex
from simplev.query import QueryResult
from simplev.sparse import BM25Index

__all__ = [
    "BM25Index",
    "Client",
    "FlatIndex",
    "HNSWIndex",
    "QueryResult",
    "__version__",
]
