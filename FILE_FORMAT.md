## 1. Overview
To achieve optimal performance and maintain our zero-dependency philosophy, SimpleV relies on a custom binary file format denoted by the `.sv` extension. Rather than wrapping another database engine or relying on slow JSON serialization for large arrays, the `.sv` format is purpose-built to map directly to our internal memory structures, ensuring lightning-fast load and save times.

## 2. File Layout
A standard `.sv` file is structured sequentially into four distinct blocks. This linear design ensures that reading from and writing to the disk requires minimal computational overhead.

### 2.1. The Header (64 Bytes)
The file begins with a fixed-length 64-byte header containing crucial metadata about the database itself.
* **Magic Bytes:** Starts with `SV01` to confidently identify the file as a SimpleV database and establish the version format.
* **Endianness Marker:** Ensures compatibility if the file is moved between systems with different CPU architectures.
* **Dimension Size:** Stores the vector dimension size (e.g., 384 for our default models) so the system knows exactly how to slice the binary vector blob upon loading.
* **Document Count:** The total number of records currently stored in the file.

### 2.2. Tombstone Bitmap
Immediately following the header is a compact bit array used for managing deleted records. Each bit corresponds to a document in the database. A value of `1` indicates an active document, while a `0` signifies a soft-deleted (tombstoned) record. This allows the system to load deletion states instantly without parsing the entire file.

### 2.3. Metadata Block
This section contains the core data: document IDs, raw text, and user-provided dictionary metadata. To maintain a balance between speed and simplicity, this block is serialized using MessagePack or standard JSON-lines. This enables sequential reading and straightforward hydration of our internal Python dictionaries.

### 2.4. Vector Block
The final and largest portion of the file is a contiguous binary blob of raw `float32` vectors. By placing this at the end of the file, SimpleV can easily append new vectors without needing to rewrite the entire document structure. Furthermore, this contiguous layout allows mathematical libraries like NumPy to load the vectors directly into memory with near-zero processing overhead.