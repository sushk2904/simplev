"""
Text chunking utilities for SimpleV.

When you have a long document, you can't just embed the whole
thing as one vector -- embedding models have a token limit and
long texts lose semantic specificity anyway. So we split the
text into overlapping chunks, each of which gets its own vector.

The overlap ensures that information sitting right at a chunk
boundary doesn't get lost. If a sentence spans two chunks,
both chunks will contain part of it.
"""

import logging
from typing import Optional

from simplev.exceptions import SimpleVError


logger = logging.getLogger(__name__)


class ChunkingError(SimpleVError):
    """Something went wrong during text chunking."""
    pass


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separator: Optional[str] = None,
) -> list[str]:
    """Split text into overlapping chunks.

    By default, splits on character count. If a separator is
    provided (like '\\n' or '. '), it tries to break at those
    boundaries so chunks don't cut words in half.

    Args:
        text: The text to split.
        chunk_size: Target size for each chunk in characters.
        overlap: How many characters to overlap between chunks.
            Must be less than chunk_size.
        separator: Optional string to split on before chunking.
            Common choices: '\\n', '. ', '\\n\\n'

    Returns:
        List of text chunks. Empty list if text is empty.

    Raises:
        ChunkingError: If parameters are invalid.
    """
    if chunk_size <= 0:
        raise ChunkingError(f"chunk_size must be positive, got {chunk_size}")

    if overlap < 0:
        raise ChunkingError(f"overlap must be non-negative, got {overlap}")

    if overlap >= chunk_size:
        raise ChunkingError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    if not text or not text.strip():
        return []

    # if the text is short enough, just return it as one chunk
    if len(text) <= chunk_size:
        return [text]

    if separator:
        return _chunk_by_separator(text, chunk_size, overlap, separator)
    else:
        return _chunk_by_characters(text, chunk_size, overlap)


def _chunk_by_characters(
    text: str, chunk_size: int, overlap: int
) -> list[str]:
    """Simple character-based chunking with overlap."""
    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # don't add empty or whitespace-only chunks
        if chunk.strip():
            chunks.append(chunk)

        start += step

    return chunks


def _chunk_by_separator(
    text: str, chunk_size: int, overlap: int, separator: str
) -> list[str]:
    """Split by separator first, then combine into chunks.

    This produces cleaner chunks because we break at natural
    boundaries (sentences, paragraphs) rather than mid-word.
    """
    segments = text.split(separator)

    chunks = []
    current_chunk = ""
    overlap_buffer = ""

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # would adding this segment exceed the chunk size?
        test_chunk = (current_chunk + separator + segment).strip() if current_chunk else segment

        if len(test_chunk) <= chunk_size:
            current_chunk = test_chunk
        else:
            # save the current chunk if it has content
            if current_chunk.strip():
                chunks.append(current_chunk)

                # build the overlap from the tail end of the current chunk
                if overlap > 0:
                    overlap_buffer = current_chunk[-overlap:]
                else:
                    overlap_buffer = ""

            # start a new chunk with the overlap + this segment
            if overlap_buffer:
                current_chunk = overlap_buffer + separator + segment
            else:
                current_chunk = segment

            # if even a single segment exceeds chunk_size,
            # we have to include it anyway (can't split further
            # without the separator appearing inside it)

    # don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk)

    return chunks


def chunk_document(
    text: str,
    doc_id_prefix: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separator: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> list[dict]:
    """Chunk text and format it for add_many().

    Convenience function that takes a document, chunks it,
    and returns a list of dicts ready to pass straight into
    Client.add_many().

    Args:
        text: The full document text.
        doc_id_prefix: Base ID for the chunks. Each chunk gets
            an ID like '{prefix}_chunk_0', '{prefix}_chunk_1', etc.
        chunk_size: Target chunk size in characters.
        overlap: Overlap between chunks.
        separator: Optional split boundary.
        metadata: Optional metadata to attach to every chunk.
            Each chunk also gets 'chunk_index' and 'total_chunks'
            added automatically.

    Returns:
        List of dicts with 'doc_id', 'text', and 'metadata' keys.
    """
    chunks = chunk_text(text, chunk_size, overlap, separator)

    documents = []
    base_meta = metadata if metadata is not None else {}

    for i, chunk in enumerate(chunks):
        chunk_meta = {
            **base_meta,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "source_doc": doc_id_prefix,
        }

        documents.append({
            "doc_id": f"{doc_id_prefix}_chunk_{i}",
            "text": chunk,
            "metadata": chunk_meta,
        })

    return documents
