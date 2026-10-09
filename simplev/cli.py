"""
Command Line Interface (CLI) for SimpleV.

Allows developers and shell pipelines to initialize databases,
ingest documents, execute semantic/hybrid searches, and inspect
database health directly from the terminal.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from simplev.chunking import chunk_document
from simplev.client import Client
from simplev.parsers import auto_parse
from simplev.persistence import FileManager

logger = logging.getLogger("simplev.cli")


def _cmd_init(args: argparse.Namespace) -> int:
    """Initialize a new empty SimpleV database file."""
    db_path = Path(args.db_path)
    if db_path.exists() and not args.force:
        print(f"Error: Database file already exists: {db_path}", file=sys.stderr)
        return 1

    try:
        fm = FileManager()
        fm.create_empty(db_path, dimension=args.dimension)
        print(
            f"Initialized empty SimpleV database at '{db_path}' (dim={args.dimension})"
        )
        return 0
    except Exception as e:
        print(f"Error initializing database: {e}", file=sys.stderr)
        return 1


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Ingest a file or directory of documents into the database."""
    source_path = Path(args.path)
    db_path = Path(args.db)

    if not source_path.exists():
        print(f"Error: Source path does not exist: {source_path}", file=sys.stderr)
        return 1

    target_files = []
    supported_extensions = {".txt", ".md", ".markdown", ".csv", ".pdf", ".docx"}

    if source_path.is_file():
        if source_path.suffix.lower() not in supported_extensions:
            print(
                f"Error: Unsupported file format '{source_path.suffix}'. "
                f"Supported: {sorted(list(supported_extensions))}",
                file=sys.stderr,
            )
            return 1
        target_files.append(source_path)
    else:
        for p in source_path.rglob("*"):
            if p.is_file() and p.suffix.lower() in supported_extensions:
                target_files.append(p)

    if not target_files:
        print(f"No parseable documents found at '{source_path}'.")
        return 0

    try:
        client = Client(path=db_path)
    except Exception as e:
        print(f"Error opening database '{db_path}': {e}", file=sys.stderr)
        return 1

    total_chunks = 0
    total_files = 0

    for file_path in target_files:
        try:
            text = auto_parse(file_path)
            if not text.strip():
                continue

            chunks = chunk_document(
                text=text,
                doc_id_prefix=file_path.stem,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                metadata={"file_name": file_path.name, "source_path": str(file_path)},
            )
            if chunks:
                client.add_many(chunks)
                total_chunks += len(chunks)
                total_files += 1
        except Exception as e:
            print(f"Warning: Failed to ingest '{file_path}': {e}", file=sys.stderr)

    client.commit()
    print(
        f"Successfully ingested {total_chunks} chunks from "
        f"{total_files} file(s) into '{db_path}'."
    )
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    """Execute semantic or hybrid search and print JSON results."""
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database file not found: {db_path}", file=sys.stderr)
        return 1

    try:
        client = Client(path=db_path)
        top_k = args.top

        if getattr(args, "hybrid", False):
            results = client.hybrid_search(args.query, top_k=top_k, alpha=args.alpha)
        else:
            results = client.search(args.query, top_k=top_k)

        output = [
            {
                "doc_id": r.doc_id,
                "text": r.text,
                "score": float(r.score),
                "metadata": r.metadata,
            }
            for r in results
        ]
        print(json.dumps(output, indent=2 if args.pretty else None))
        return 0
    except Exception as e:
        print(f"Error during search: {e}", file=sys.stderr)
        return 1


def _cmd_info(args: argparse.Namespace) -> int:
    """Display diagnostic database summary."""
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"Error: Database file not found: {db_path}", file=sys.stderr)
        return 1

    try:
        client = Client(path=db_path)
        info_data = client.info()

        if getattr(args, "json", False):
            print(json.dumps(info_data, indent=2))
        else:
            print("=== SimpleV Database Diagnostics ===")
            print(f"Database Path:   {info_data['path']}")
            print(f"Active Docs:     {info_data['active_count']}")
            print(f"Total Count:     {info_data['count']}")
            print(f"Dimension:       {info_data['dimension']}")
            print(f"Model Name:      {info_data['model_name']}")
            print(f"Distance Metric: {info_data['metric']}")
            print(f"Index Type:      {info_data['index_type']}")
            print(f"File Size:       {info_data['file_size_bytes']} bytes")
            wal_status = (
                f"Active ({info_data['wal_size_bytes']} bytes)"
                if info_data["wal_exists"]
                else "Clean / Truncated"
            )
            print(f"WAL Status:      {wal_status}")
        return 0
    except Exception as e:
        print(f"Error inspecting database: {e}", file=sys.stderr)
        return 1


def _cmd_compact(args: argparse.Namespace) -> int:
    """Compact database to permanently purge soft-deleted documents."""
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: Database file not found: {db_path}", file=sys.stderr)
        return 1

    try:
        client = Client(path=db_path)
        removed = client.compact()
        if removed > 0:
            client.commit()
        print(
            f"Compaction complete: purged {removed} deleted records from '{db_path}'."
        )
        return 0
    except Exception as e:
        print(f"Error during compaction: {e}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build and configure the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="simplev",
        description="SimpleV: The SQLite for Vector Search (CLI)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # init
    p_init = subparsers.add_parser("init", help="Create a new empty database")
    p_init.add_argument("db_path", help="Path to create .sv database file")
    p_init.add_argument(
        "--dimension",
        "--dim",
        type=int,
        default=384,
        help="Embedding dimension (default: 384)",
    )
    p_init.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing database if present",
    )
    p_init.set_defaults(func=_cmd_init)

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest documents into database")
    p_ingest.add_argument("path", help="Target document file or directory")
    p_ingest.add_argument("--db", "-d", required=True, help="Target database path")
    p_ingest.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Chunk character length (default: 500)",
    )
    p_ingest.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Overlap characters between chunks (default: 50)",
    )
    p_ingest.set_defaults(func=_cmd_ingest)

    # search
    p_search = subparsers.add_parser("search", help="Execute semantic search query")
    p_search.add_argument("query", help="Query string")
    p_search.add_argument("--db", "-d", required=True, help="Target database path")
    p_search.add_argument(
        "--top", "-k", type=int, default=5, help="Number of results (default: 5)"
    )
    p_search.add_argument(
        "--hybrid", action="store_true", help="Enable hybrid BM25 + dense search"
    )
    p_search.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Hybrid dense weight (0.0 to 1.0, default: 0.5)",
    )
    p_search.add_argument(
        "--pretty", action="store_true", help="Pretty-print JSON output"
    )
    p_search.set_defaults(func=_cmd_search)

    # info
    p_info = subparsers.add_parser("info", help="Inspect database diagnostics")
    p_info.add_argument("db_path", help="Target database path")
    p_info.add_argument(
        "--json", action="store_true", help="Output information in JSON format"
    )
    p_info.set_defaults(func=_cmd_info)

    # compact
    p_compact = subparsers.add_parser(
        "compact", help="Compact and remove soft deletions"
    )
    p_compact.add_argument("--db", "-d", required=True, help="Target database path")
    p_compact.set_defaults(func=_cmd_compact)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
