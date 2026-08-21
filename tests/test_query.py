"""
Tests for the query engine.

These test the orchestration pipeline: validation, pre-filtering,
result hydration, and the glue between storage/indexing/embeddings.

We mock the embedding manager so tests don't need the actual
sentence-transformers model.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from simplev.embeddings import EmbeddingManager
from simplev.exceptions import QueryError
from simplev.indexing import FlatIndex
from simplev.query import QueryEngine, QueryResult
from simplev.storage import StorageEngine


# dimension we'll use for all test vectors
DIM = 4


def make_vector(dim: int = DIM) -> np.ndarray:
    return np.random.randn(dim).astype(np.float32)


def build_test_engine(docs=None, metric="cosine"):
    """Helper to set up a query engine with a mocked embeddings manager.

    The mock embeddings always return a fixed vector so we can
    control what the search sees.
    """
    storage = StorageEngine(dimension=DIM)
    embeddings = MagicMock(spec=EmbeddingManager)
    index = FlatIndex(metric=metric)

    # mock embed() to return a predictable vector
    query_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    embeddings.embed.return_value = query_vec
    embeddings.dimension = DIM

    if docs:
        for doc_id, text, vec, meta in docs:
            storage.add(doc_id, text, vec, metadata=meta)

    engine = QueryEngine(storage=storage, embeddings=embeddings, index=index)
    return engine, storage, embeddings


class TestQueryValidation:
    """Test that bad inputs are caught early."""

    def test_empty_query_raises(self):
        engine, _, _ = build_test_engine()
        with pytest.raises(QueryError, match="non-empty string"):
            engine.search("")

    def test_none_query_raises(self):
        engine, _, _ = build_test_engine()
        with pytest.raises(QueryError, match="non-empty string"):
            engine.search(None)

    def test_whitespace_query_raises(self):
        engine, _, _ = build_test_engine()
        with pytest.raises(QueryError, match="non-empty string"):
            engine.search("   ")

    def test_invalid_top_k_raises(self):
        engine, _, _ = build_test_engine()
        with pytest.raises(QueryError, match="top_k"):
            engine.search("hello", top_k=0)

    def test_negative_top_k_raises(self):
        engine, _, _ = build_test_engine()
        with pytest.raises(QueryError, match="top_k"):
            engine.search("hello", top_k=-3)


class TestQuerySearch:
    """Test the full search pipeline."""

    def test_search_empty_database(self):
        engine, _, _ = build_test_engine()
        results = engine.search("anything")
        assert results == []

    def test_search_returns_results(self):
        docs = [
            ("d1", "hello", np.array([1, 0, 0, 0], dtype=np.float32), None),
            ("d2", "world", np.array([0, 1, 0, 0], dtype=np.float32), None),
        ]
        engine, _, _ = build_test_engine(docs)

        results = engine.search("test query", top_k=2)
        assert len(results) == 2
        # d1 should be first since it points in the same direction as
        # our mock query vector [1,0,0,0]
        assert results[0].doc_id == "d1"

    def test_search_respects_top_k(self):
        docs = [
            ("d1", "a", make_vector(), None),
            ("d2", "b", make_vector(), None),
            ("d3", "c", make_vector(), None),
        ]
        engine, _, _ = build_test_engine(docs)
        results = engine.search("query", top_k=1)
        assert len(results) <= 1

    def test_search_skips_tombstoned(self):
        docs = [
            ("d1", "should appear", np.array([1, 0, 0, 0], dtype=np.float32), None),
            ("d2", "deleted", np.array([0.9, 0.1, 0, 0], dtype=np.float32), None),
        ]
        engine, storage, _ = build_test_engine(docs)
        storage.mark_deleted("d2")

        results = engine.search("test", top_k=5)
        doc_ids = [r.doc_id for r in results]
        assert "d2" not in doc_ids
        assert "d1" in doc_ids

    def test_result_has_correct_fields(self):
        meta = {"source": "unit_test", "page": 7}
        docs = [
            ("doc_x", "some text here", np.array([1, 0, 0, 0], dtype=np.float32), meta),
        ]
        engine, _, _ = build_test_engine(docs)

        results = engine.search("test")
        assert len(results) == 1
        r = results[0]
        assert r.doc_id == "doc_x"
        assert r.text == "some text here"
        assert r.metadata["source"] == "unit_test"
        assert r.metadata["page"] == 7
        assert isinstance(r.score, float)


class TestQueryFilters:
    """Test metadata pre-filtering."""

    @pytest.fixture
    def filtered_engine(self):
        docs = [
            ("d1", "cat", np.array([1, 0, 0, 0], dtype=np.float32),
             {"category": "animals"}),
            ("d2", "dog", np.array([0.9, 0.1, 0, 0], dtype=np.float32),
             {"category": "animals"}),
            ("d3", "car", np.array([0.8, 0.2, 0, 0], dtype=np.float32),
             {"category": "vehicles"}),
            ("d4", "bus", np.array([0.7, 0.3, 0, 0], dtype=np.float32),
             {"category": "vehicles", "size": "large"}),
        ]
        engine, storage, embeddings = build_test_engine(docs)
        return engine

    def test_filter_narrows_results(self, filtered_engine):
        results = filtered_engine.search(
            "test", top_k=10, filters={"category": "vehicles"}
        )
        doc_ids = [r.doc_id for r in results]
        assert "d1" not in doc_ids
        assert "d2" not in doc_ids
        assert "d3" in doc_ids
        assert "d4" in doc_ids

    def test_filter_multiple_keys(self, filtered_engine):
        results = filtered_engine.search(
            "test", top_k=10,
            filters={"category": "vehicles", "size": "large"}
        )
        assert len(results) == 1
        assert results[0].doc_id == "d4"

    def test_filter_no_matches(self, filtered_engine):
        results = filtered_engine.search(
            "test", filters={"category": "electronics"}
        )
        assert results == []

    def test_no_filter_returns_all(self, filtered_engine):
        results = filtered_engine.search("test", top_k=10)
        assert len(results) == 4


class TestQueryResult:
    """Test the QueryResult data class."""

    def test_to_dict(self):
        r = QueryResult(
            doc_id="abc", text="hello", score=0.95, metadata={"k": "v"}
        )
        d = r.to_dict()
        assert d["doc_id"] == "abc"
        assert d["text"] == "hello"
        assert d["score"] == 0.95
        assert d["metadata"] == {"k": "v"}

    def test_repr(self):
        r = QueryResult(doc_id="x", text="short", score=0.5, metadata={})
        s = repr(r)
        assert "x" in s
        assert "0.5" in s
