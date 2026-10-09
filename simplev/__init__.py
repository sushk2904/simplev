"""
SimpleV - The SQLite for Vector Search.

A local-first, zero-configuration vector database designed
for AI applications, RAG, and personal knowledge bases.
"""

__version__ = "0.1.0.dev1"

from simplev.chunking import chunk_document, chunk_text
from simplev.client import Client
from simplev.indexing import FlatIndex, HNSWIndex
from simplev.parsers import auto_parse
from simplev.query import QueryResult
from simplev.sparse import BM25Index
from simplev.storage import StorageEngine

__all__ = [
    "Client",
    "QueryResult",
    "FlatIndex",
    "HNSWIndex",
    "BM25Index",
    "StorageEngine",
    "chunk_text",
    "chunk_document",
    "auto_parse",
]
