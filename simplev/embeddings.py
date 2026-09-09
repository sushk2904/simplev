"""
Embeddings module for SimpleV.

Handles converting raw text into dense vector representations
using sentence-transformers. The module manages model loading,
caching, and the actual encoding process.

We keep this as a thin wrapper on purpose -- the goal is to
hide the sentence-transformers API behind a clean interface
so the rest of the codebase never touches it directly.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

from simplev.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

# the default model is small enough to download quickly
# and produces 384-dim vectors which is a good balance
# between quality and memory usage for local applications
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingManager:
    """Manages the lifecycle of the embedding model.

    Loads a sentence-transformer model (downloading it if needed),
    caches it in memory, and provides methods to convert text
    into numpy vectors.

    Args:
        model_name: Name of the sentence-transformers model to use.
            Defaults to all-MiniLM-L6-v2 (384 dimensions, ~80MB).
        cache_dir: Where to store downloaded models on disk.
            If None, uses the sentence-transformers default cache.
        device: Device to run inference on. None means auto-detect
            (will use CUDA if available, otherwise CPU).

    Example:
        >>> emb = EmbeddingManager()
        >>> vector = emb.embed("hello world")
        >>> vector.shape
        (384,)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        cache_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = str(cache_dir) if cache_dir else None
        self._device = device
        self._model = None  # lazy loaded on first use
        self._dimension: Optional[int] = None

    def _load_model(self) -> None:
        """Load the sentence-transformer model into memory.

        This is called lazily on the first embed() call so we don't
        waste time loading a model if the user never actually embeds
        anything (e.g. they only do lookups by ID).

        Raises:
            EmbeddingError: If the model can't be loaded for any reason.
        """
        if self._model is not None:
            return

        try:
            # import here to keep it lazy -- no point importing
            # sentence_transformers at module level if we might not need it
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self._model_name}")

            kwargs = {}
            if self._cache_dir:
                kwargs["cache_folder"] = self._cache_dir
            if self._device:
                kwargs["device"] = self._device

            self._model = SentenceTransformer(self._model_name, **kwargs)
            self._dimension = self._model.get_embedding_dimension()

            logger.info(
                f"Model loaded successfully. Dimension: {self._dimension}"
            )

        except ImportError:
            raise EmbeddingError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )
        except Exception as e:
            raise EmbeddingError(
                f"Failed to load embedding model '{self._model_name}': {e}"
            ) from e

    @property
    def dimension(self) -> int:
        """Returns the vector dimension size of the loaded model.

        Forces model loading if it hasn't happened yet, since we
        need the model to know its output dimension.
        """
        self._load_model()
        assert self._dimension is not None
        return self._dimension

    @property
    def model_name(self) -> str:
        """Returns the name of the model being used."""
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded in memory."""
        return self._model is not None

    def embed(self, text: str) -> np.ndarray:
        """Convert a single text string into a dense vector.

        Args:
            text: The input text to embed. Should be non-empty.

        Returns:
            A 1-D numpy array of float32 values with shape (dimension,).

        Raises:
            EmbeddingError: If text is empty/invalid or encoding fails.
        """
        # basic sanity check before we bother the model
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError(
                "Cannot embed empty or non-string input. "
                f"Got type={type(text).__name__}, value={repr(text)[:100]}"
            )

        self._load_model()

        try:
            # sentence-transformers returns numpy arrays by default which
            # is exactly what we want. show_progress_bar=False because
            # single string encoding doesn't need a progress bar
            vector = self._model.encode(
                text,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # make sure we got back what we expected
            if not isinstance(vector, np.ndarray):
                vector = np.array(vector, dtype=np.float32)

            # ensure its float32 -- some models return float64
            # and we want consistency across the whole system
            vector = vector.astype(np.float32, copy=False)

            return vector

        except EmbeddingError:
            raise  # don't wrap our own errors
        except Exception as e:
            raise EmbeddingError(
                f"Encoding failed for text: {repr(text)[:80]}... - {e}"
            ) from e

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Convert multiple texts into vectors in one go.

        Batching is significantly faster than calling embed() in a loop
        because the model can process multiple inputs at once on the GPU
        or with optimized CPU batching.

        Args:
            texts: List of strings to embed. All must be non-empty.

        Returns:
            A 2-D numpy array of shape (len(texts), dimension) with
            dtype float32.

        Raises:
            EmbeddingError: If any text is invalid or encoding fails.
        """
        if not isinstance(texts, list):
            raise EmbeddingError(
                f"embed_batch expects a list of strings, got {type(texts).__name__}. "
                "Did you mean to call embed() instead?"
            )

        if not texts:
            raise EmbeddingError("Cannot embed an empty list of texts.")

        # validate everything upfront so we don't fail halfway through
        for i, t in enumerate(texts):
            if not isinstance(t, str) or not t.strip():
                raise EmbeddingError(
                    f"Invalid text at index {i}: "
                    f"type={type(t).__name__}, value={repr(t)[:100]}"
                )

        self._load_model()

        try:
            vectors = self._model.encode(
                texts,
                show_progress_bar=len(texts) > 100,  # only show for large batches
                convert_to_numpy=True,
                batch_size=32,
            )

            if not isinstance(vectors, np.ndarray):
                vectors = np.array(vectors, dtype=np.float32)

            vectors = vectors.astype(np.float32, copy=False)

            # sanity check the output shape
            expected_shape = (len(texts), self._dimension)
            if vectors.shape != expected_shape:
                raise EmbeddingError(
                    f"Unexpected output shape: got {vectors.shape}, "
                    f"expected {expected_shape}"
                )

            return vectors

        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                f"Batch encoding failed for {len(texts)} texts: {e}"
            ) from e

    def unload(self) -> None:
        """Free the model from memory.

        Useful if you need to reclaim RAM after you're done
        embedding things. The model will be reloaded automatically
        on the next embed() call.
        """
        if self._model is not None:
            logger.info(f"Unloading model: {self._model_name}")
            del self._model
            self._model = None
            # keep _dimension cached since it doesn't change

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        dim_str = f", dim={self._dimension}" if self._dimension else ""
        return f"EmbeddingManager(model='{self._model_name}', {status}{dim_str})"
