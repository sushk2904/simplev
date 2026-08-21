"""
Persistence layer for SimpleV (.sv file format).

Handles serializing and deserializing the in-memory storage
engine state to our custom binary format. The layout is:

  [ Header  - 64 bytes fixed ]
  [ Tombstone bitmap - ceil(n_docs / 8) bytes ]
  [ Metadata block - JSON blob ]
  [ Vector block - raw float32 bytes ]

The header contains magic bytes, version, dimension, and
document count so we can validate the file before loading.
The vector block is just a raw dump of the numpy array
which makes save/load extremely fast.

See FILE_FORMAT.md for the full specification.
"""

import json
import logging
import struct
from pathlib import Path
from typing import Union

import numpy as np

from simplev.exceptions import StorageError
from simplev.storage import StorageEngine


logger = logging.getLogger(__name__)

# file format constants
MAGIC = b"SV01"
FORMAT_VERSION = 1
HEADER_SIZE = 64
ENDIAN_MARKER = 0x04030201  # little-endian check value


class FileManager:
    """Reads and writes .sv files.

    This is stateless -- you just call save() or load() and
    it handles the binary packing/unpacking. Keeps things
    simple and testable.
    """

    def save(
        self, path: Union[str, Path], storage: StorageEngine
    ) -> None:
        """Serialize a StorageEngine to a .sv file.

        Args:
            path: Where to write the file.
            storage: The storage engine to serialize.

        Raises:
            StorageError: If serialization fails.
        """
        path = Path(path)

        vectors = storage.get_vectors()
        mask = storage.get_active_mask()
        all_meta = storage.get_all_metadata()
        count = storage.count
        dimension = storage.dimension

        if vectors is None or count == 0:
            raise StorageError("Cannot save an empty database.")

        try:
            with open(path, "wb") as f:
                # -- write the header --
                header = self._pack_header(dimension, count)
                f.write(header)

                # -- write tombstone bitmap --
                bitmap = self._pack_tombstones(mask, count)
                f.write(bitmap)

                # -- write metadata as json --
                meta_bytes = self._pack_metadata(all_meta, count)
                # write the length first so we know where metadata
                # ends and vectors begin when loading
                f.write(struct.pack("<I", len(meta_bytes)))
                f.write(meta_bytes)

                # -- write raw vector bytes --
                vec_bytes = vectors.astype(np.float32).tobytes()
                f.write(vec_bytes)

            logger.info(
                f"Saved {count} documents to {path} "
                f"({path.stat().st_size} bytes)"
            )

        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Failed to save database to '{path}': {e}"
            ) from e

    def load(
        self, path: Union[str, Path]
    ) -> StorageEngine:
        """Deserialize a .sv file into a StorageEngine.

        Args:
            path: Path to the .sv file.

        Returns:
            A fully populated StorageEngine instance.

        Raises:
            StorageError: If the file is invalid or corrupted.
        """
        path = Path(path)

        if not path.exists():
            raise StorageError(f"File not found: {path}")

        try:
            with open(path, "rb") as f:
                # -- read and validate header --
                header_bytes = f.read(HEADER_SIZE)
                if len(header_bytes) < HEADER_SIZE:
                    raise StorageError(
                        f"File too small to contain a valid header "
                        f"(got {len(header_bytes)} bytes, need {HEADER_SIZE})"
                    )
                dimension, count = self._unpack_header(header_bytes)

                # -- read tombstone bitmap --
                bitmap_size = (count + 7) // 8  # ceil division
                bitmap_bytes = f.read(bitmap_size)
                tombstones = self._unpack_tombstones(bitmap_bytes, count)

                # -- read metadata --
                meta_len_bytes = f.read(4)
                if len(meta_len_bytes) < 4:
                    raise StorageError("File truncated: missing metadata length")
                meta_len = struct.unpack("<I", meta_len_bytes)[0]

                meta_bytes = f.read(meta_len)
                if len(meta_bytes) < meta_len:
                    raise StorageError("File truncated: metadata block incomplete")
                metadata_list = self._unpack_metadata(meta_bytes)

                # -- read vector block --
                expected_vec_bytes = count * dimension * 4  # float32 = 4 bytes
                vec_bytes = f.read(expected_vec_bytes)
                if len(vec_bytes) < expected_vec_bytes:
                    raise StorageError(
                        f"File truncated: expected {expected_vec_bytes} bytes "
                        f"for vectors, got {len(vec_bytes)}"
                    )
                vectors = np.frombuffer(vec_bytes, dtype=np.float32)
                vectors = vectors.reshape(count, dimension)

            # rebuild the storage engine from what we loaded
            storage = self._rebuild_storage(
                dimension, count, vectors, tombstones, metadata_list
            )

            logger.info(
                f"Loaded {count} documents from {path} "
                f"(dim={dimension})"
            )
            return storage

        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                f"Failed to load database from '{path}': {e}"
            ) from e

    # -- header packing --

    def _pack_header(self, dimension: int, count: int) -> bytes:
        """Build the 64-byte file header.

        Layout:
          bytes 0-3:   magic "SV01"
          bytes 4-7:   endianness marker (0x04030201)
          bytes 8-9:   format version (uint16)
          bytes 10-11: reserved
          bytes 12-15: dimension (uint32)
          bytes 16-19: document count (uint32)
          bytes 20-63: reserved (zeros for future use)
        """
        header = bytearray(HEADER_SIZE)

        # magic bytes
        header[0:4] = MAGIC

        # endian marker so we can detect byte order issues
        struct.pack_into("<I", header, 4, ENDIAN_MARKER)

        # format version
        struct.pack_into("<H", header, 8, FORMAT_VERSION)

        # dimension and count
        struct.pack_into("<I", header, 12, dimension)
        struct.pack_into("<I", header, 16, count)

        # rest is zeros (reserved for future fields)
        return bytes(header)

    def _unpack_header(self, data: bytes) -> tuple[int, int]:
        """Parse the header and return (dimension, count)."""
        magic = data[0:4]
        if magic != MAGIC:
            raise StorageError(
                f"Invalid file: expected magic bytes {MAGIC!r}, "
                f"got {magic!r}"
            )

        endian = struct.unpack_from("<I", data, 4)[0]
        if endian != ENDIAN_MARKER:
            raise StorageError(
                "Endianness mismatch -- this file was created on "
                "a system with different byte order"
            )

        version = struct.unpack_from("<H", data, 8)[0]
        if version != FORMAT_VERSION:
            raise StorageError(
                f"Unsupported format version: {version} "
                f"(this build supports version {FORMAT_VERSION})"
            )

        dimension = struct.unpack_from("<I", data, 12)[0]
        count = struct.unpack_from("<I", data, 16)[0]

        if dimension == 0:
            raise StorageError("Invalid file: dimension is 0")
        if count == 0:
            raise StorageError("Invalid file: document count is 0")

        return dimension, count

    # -- tombstone bitmap --

    def _pack_tombstones(
        self, mask: np.ndarray, count: int
    ) -> bytes:
        """Pack the boolean tombstone array into a compact bitmap.

        Each bit represents one document. Bit = 1 means active,
        bit = 0 means tombstoned. We pack 8 documents per byte.
        """
        bitmap = bytearray((count + 7) // 8)

        for i in range(count):
            if mask[i]:
                byte_idx = i // 8
                bit_idx = i % 8
                bitmap[byte_idx] |= (1 << bit_idx)

        return bytes(bitmap)

    def _unpack_tombstones(
        self, data: bytes, count: int
    ) -> np.ndarray:
        """Unpack the bitmap back into a boolean array."""
        mask = np.zeros(count, dtype=bool)

        for i in range(count):
            byte_idx = i // 8
            bit_idx = i % 8
            if data[byte_idx] & (1 << bit_idx):
                mask[i] = True

        return mask

    # -- metadata --

    def _pack_metadata(
        self, all_meta: dict[int, dict], count: int
    ) -> bytes:
        """Serialize metadata as a JSON array.

        Each entry in the array corresponds to a document by index.
        We dump the whole thing as one JSON blob. For v0.1 this is
        fine -- if it becomes a bottleneck we can switch to msgpack
        or jsonlines later.
        """
        meta_list = []
        for idx in range(count):
            if idx in all_meta:
                meta_list.append(all_meta[idx])
            else:
                # shouldn't happen but be safe
                meta_list.append({
                    "doc_id": f"unknown_{idx}",
                    "text": "",
                    "metadata": {},
                })

        return json.dumps(meta_list, ensure_ascii=False).encode("utf-8")

    def _unpack_metadata(self, data: bytes) -> list[dict]:
        """Deserialize the JSON metadata blob."""
        try:
            return json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise StorageError(
                f"Corrupted metadata block: {e}"
            ) from e

    # -- rebuilding storage from loaded data --

    def _rebuild_storage(
        self,
        dimension: int,
        count: int,
        vectors: np.ndarray,
        tombstones: np.ndarray,
        metadata_list: list[dict],
    ) -> StorageEngine:
        """Reconstruct a StorageEngine from deserialized components.

        We bypass the normal add() method and set internal state
        directly because the data has already been validated when
        it was originally saved.
        """
        storage = StorageEngine(dimension=dimension)

        # set internal state directly -- this is the only place
        # we reach into the storage engine's internals, and it's
        # justified because we're restoring from a known-good file
        storage._vectors = vectors.copy()
        storage._tombstones = tombstones.copy()
        storage._count = count

        storage._metadata = {}
        storage._id_map = {}

        for idx, entry in enumerate(metadata_list):
            storage._metadata[idx] = entry
            storage._id_map[entry["doc_id"]] = idx

        return storage
