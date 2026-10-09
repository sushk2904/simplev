"""
Sparse keyword indexing using Okapi BM25.

Provides zero-dependency lexical search that can be used standalone
or blended with dense vector search for hybrid retrieval. The BM25
scoring uses the standard Robertson-Spärck Jones formulation.

This module is intentionally self-contained -- no numpy, no external
dependencies -- so it stays lightweight and doesn't interfere with
the vector search path.
"""

import math
import re
from typing import Optional


class BM25Index:
    """In-memory Okapi BM25 index for sparse keyword search.

    Builds an inverted index over tokenized documents and scores
    queries using the standard BM25 formula.

    Args:
        k1: Term frequency saturation parameter. Higher values
            give more weight to term frequency. Default 1.5.
        b: Document length normalization parameter. 0 means no
            normalization, 1 means full normalization. Default 0.75.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

        # doc_id -> token count
        self._doc_lens: dict[str, int] = {}
        self._total_tokens: int = 0

        # term -> {doc_id: count}
        self._inverted_index: dict[str, dict[str, int]] = {}

        # term -> number of docs containing term
        self._df: dict[str, int] = {}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Convert text into normalized lowercase word tokens.

        Uses a simple regex word boundary split. Good enough for
        English text and avoids pulling in nltk or spacy.
        """
        if not text:
            return []
        return re.findall(r"\w+", text.lower())

    @property
    def count(self) -> int:
        """Total number of active indexed documents."""
        return len(self._doc_lens)

    @property
    def doc_count(self) -> int:
        """Alias for count property."""
        return self.count

    @property
    def doc_lengths(self) -> dict[str, int]:
        """Dictionary of doc_id to token length."""
        return self._doc_lens

    def add_document(self, doc_id: str, text: str) -> None:
        """Index a single document by doc_id.

        If the document already exists, it is removed first and
        then re-indexed with the new text.

        Args:
            doc_id: Unique document identifier.
            text: Raw text to tokenize and index.
        """
        if doc_id in self._doc_lens:
            self.remove_document(doc_id)

        tokens = self.tokenize(text)
        doc_len = len(tokens)
        self._doc_lens[doc_id] = doc_len
        self._total_tokens += doc_len

        # count term frequencies within this document
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        for token, count in tf.items():
            if token not in self._inverted_index:
                self._inverted_index[token] = {}
            self._inverted_index[token][doc_id] = count
            self._df[token] = self._df.get(token, 0) + 1

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the sparse index.

        Args:
            doc_id: The document to remove.

        Returns:
            True if the document was found and removed, False otherwise.
        """
        if doc_id not in self._doc_lens:
            return False

        old_len = self._doc_lens.pop(doc_id)
        self._total_tokens = max(0, self._total_tokens - old_len)

        # clean up inverted index
        for token, postings in list(self._inverted_index.items()):
            if doc_id in postings:
                del postings[doc_id]
                self._df[token] = max(0, self._df.get(token, 1) - 1)
                if self._df[token] == 0:
                    self._df.pop(token, None)
                if not postings:
                    self._inverted_index.pop(token, None)

        return True

    def build(self, documents: list[tuple[str, str]]) -> None:
        """Batch build the index from a list of (doc_id, text) tuples.

        Clears any existing index state first.

        Args:
            documents: List of (doc_id, text) pairs to index.
        """
        self.clear()
        for doc_id, text in documents:
            self.add_document(doc_id, text)

    def clear(self) -> None:
        """Reset the index to empty state."""
        self._doc_lens.clear()
        self._total_tokens = 0
        self._inverted_index.clear()
        self._df.clear()

    def reset(self) -> None:
        """Alias for clear()."""
        self.clear()

    def search(
        self,
        query: str,
        top_k: int = 5,
        allowed_doc_ids: Optional[set[str]] = None,
    ) -> list[tuple[str, float]]:
        """Search for top_k documents matching query string.

        Scores each candidate document using the Okapi BM25 formula
        and returns results sorted by descending score.

        Args:
            query: Natural language query string.
            top_k: Maximum number of results to return.
            allowed_doc_ids: Optional set of doc_ids to restrict
                search to (used for metadata pre-filtering).

        Returns:
            List of (doc_id, bm25_score) tuples sorted descending
            by relevance score.
        """
        tokens = self.tokenize(query)
        if not tokens or self.count == 0:
            return []

        scores: dict[str, float] = {}
        n_docs = self.count
        avg_dl = self._total_tokens / max(n_docs, 1)

        for token in tokens:
            if token not in self._inverted_index:
                continue

            postings = self._inverted_index[token]
            df = self._df.get(token, 0)
            if df == 0:
                continue

            # Standard Robertson-Spärck Jones IDF
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

            for doc_id, tf in postings.items():
                if allowed_doc_ids is not None and doc_id not in allowed_doc_ids:
                    continue

                dl = self._doc_lens[doc_id]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * (dl / max(avg_dl, 1e-6))
                )
                score = idf * (numerator / max(denominator, 1e-6))
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def __repr__(self) -> str:
        return f"BM25Index(docs={self.count}, k1={self.k1}, b={self.b})"
