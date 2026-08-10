"""
Tests for the embeddings module.

These tests cover model loading, single and batch embedding,
error handling, and the lazy loading behavior.

Note: The first run will be slow because it downloads the model.
Subsequent runs use the cached version.
"""

import numpy as np
import pytest

from simplev.embeddings import EmbeddingManager, DEFAULT_MODEL_NAME
from simplev.exceptions import EmbeddingError


class TestEmbeddingManagerInit:
    """Test initialization and lazy loading."""

    def test_default_init(self):
        emb = EmbeddingManager()
        assert emb.model_name == DEFAULT_MODEL_NAME
        # model should NOT be loaded yet (lazy)
        assert not emb.is_loaded

    def test_custom_model_name(self):
        emb = EmbeddingManager(model_name="paraphrase-MiniLM-L3-v2")
        assert emb.model_name == "paraphrase-MiniLM-L3-v2"

    def test_lazy_loading(self):
        """Model only loads when we actually try to embed something."""
        emb = EmbeddingManager()
        assert emb._model is None

        # accessing dimension forces the load
        dim = emb.dimension
        assert dim > 0
        assert emb.is_loaded

    def test_repr_before_load(self):
        emb = EmbeddingManager()
        r = repr(emb)
        assert "not loaded" in r
        assert DEFAULT_MODEL_NAME in r

    def test_repr_after_load(self):
        emb = EmbeddingManager()
        _ = emb.dimension  # force load
        r = repr(emb)
        assert "loaded" in r
        assert "dim=" in r


class TestEmbedSingle:
    """Test single text embedding."""

    @pytest.fixture(autouse=True)
    def setup_manager(self):
        # reuse same model instance across tests in this class
        # to avoid reloading for every single test
        self.emb = EmbeddingManager()

    def test_basic_embed(self):
        vector = self.emb.embed("hello world")
        assert isinstance(vector, np.ndarray)
        assert vector.dtype == np.float32
        assert vector.ndim == 1
        assert len(vector) == self.emb.dimension

    def test_embed_returns_384_dims(self):
        """Default model should output 384-dimensional vectors."""
        vector = self.emb.embed("test sentence")
        assert vector.shape == (384,)

    def test_different_texts_give_different_vectors(self):
        v1 = self.emb.embed("the cat sat on the mat")
        v2 = self.emb.embed("quantum physics is fascinating")
        # these should definitely not be identical
        assert not np.array_equal(v1, v2)

    def test_similar_texts_have_high_similarity(self):
        v1 = self.emb.embed("I love dogs")
        v2 = self.emb.embed("I really like puppies")
        # cosine similarity should be reasonably high
        similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        assert similarity > 0.5  # conservative threshold

    def test_embed_empty_string_raises(self):
        with pytest.raises(EmbeddingError, match="empty"):
            self.emb.embed("")

    def test_embed_whitespace_only_raises(self):
        with pytest.raises(EmbeddingError, match="empty"):
            self.emb.embed("   \t\n  ")

    def test_embed_none_raises(self):
        with pytest.raises(EmbeddingError, match="non-string"):
            self.emb.embed(None)

    def test_embed_integer_raises(self):
        with pytest.raises(EmbeddingError, match="non-string"):
            self.emb.embed(42)

    def test_embed_long_text(self):
        """Model should handle reasonably long inputs without crashing."""
        long_text = "This is a sentence. " * 200
        vector = self.emb.embed(long_text)
        assert vector.shape == (384,)


class TestEmbedBatch:
    """Test batch embedding."""

    @pytest.fixture(autouse=True)
    def setup_manager(self):
        self.emb = EmbeddingManager()

    def test_batch_embed_basic(self):
        texts = ["hello", "world", "foo bar"]
        vectors = self.emb.embed_batch(texts)
        assert isinstance(vectors, np.ndarray)
        assert vectors.shape == (3, 384)
        assert vectors.dtype == np.float32

    def test_batch_embed_single_item(self):
        vectors = self.emb.embed_batch(["just one"])
        assert vectors.shape == (1, 384)

    def test_batch_matches_individual(self):
        """Batch encoding should give same results as individual encoding."""
        texts = ["alpha", "beta", "gamma"]
        batch_result = self.emb.embed_batch(texts)

        for i, text in enumerate(texts):
            individual = self.emb.embed(text)
            # they should be very close (might not be bit-identical
            # due to batching optimizations in the model)
            np.testing.assert_allclose(
                batch_result[i], individual, rtol=1e-4, atol=1e-5
            )

    def test_batch_embed_empty_list_raises(self):
        with pytest.raises(EmbeddingError, match="empty list"):
            self.emb.embed_batch([])

    def test_batch_embed_with_invalid_entry_raises(self):
        with pytest.raises(EmbeddingError, match="index 1"):
            self.emb.embed_batch(["valid", "", "also valid"])

    def test_batch_embed_with_none_raises(self):
        with pytest.raises(EmbeddingError, match="index 2"):
            self.emb.embed_batch(["ok", "fine", None])

    def test_batch_embed_with_string_raises(self):
        with pytest.raises(EmbeddingError, match="expects a list of strings"):
            self.emb.embed_batch("not a list, just a string")


class TestUnload:
    """Test model memory management."""

    def test_unload_frees_model(self):
        emb = EmbeddingManager()
        _ = emb.embed("force load")
        assert emb.is_loaded

        emb.unload()
        assert not emb.is_loaded

    def test_unload_preserves_dimension(self):
        emb = EmbeddingManager()
        dim = emb.dimension
        emb.unload()
        # dimension should still be cached even after unload
        assert emb._dimension == dim

    def test_embed_after_unload_reloads(self):
        """Should seamlessly reload if you embed after unloading."""
        emb = EmbeddingManager()
        v1 = emb.embed("test")
        emb.unload()
        v2 = emb.embed("test")
        np.testing.assert_array_equal(v1, v2)

    def test_unload_when_not_loaded(self):
        """Unloading when nothing is loaded should be fine."""
        emb = EmbeddingManager()
        emb.unload()  # should not raise