## 1. Overview
The Command Line Interface (CLI) provides a convenient way to manage and interact with SimpleV databases directly from your terminal. It is designed for developers who want to quickly inspect data, test queries, or ingest documents without needing to write a Python script. This makes SimpleV highly suitable for shell automation, quick debugging, and integrating into larger bash pipelines.

## 2. Command Reference

### 2.1. Initialization
`simplev init <db_name>`
Creates a new, empty SimpleV database file (`.sv`) at the specified path. This is useful for pre-allocating a database or establishing the storage structure before a background process begins writing to it.

### 2.2. Document Ingestion
`simplev ingest <path/to/docs> --db <db_name>`
Reads a target file or a directory of documents, automatically chunks the text into manageable pieces, generates embeddings using the default model, and inserts them into the specified database. This allows for rapid population of a knowledge base directly from the filesystem.

### 2.3. Semantic Search
`simplev search "query text" --top 5 --db <db_name>`
Executes a semantic search against the specified database directly from the command line. The results are returned in a clean, standard JSON-formatted output. This structured output makes it incredibly easy to pipe the search results into other command-line utilities like `jq` for further filtering or processing.

### 2.4. Database Information
`simplev info <db_name>`
Displays a quick diagnostic summary of the database's internal state. It outputs critical metadata, including the total number of documents (vector count), the dimension size of the stored embeddings, the current file size on disk, and the status of the Write-Ahead Log.