"""
Command Line Interface (CLI) for SimpleV.

Provides terminal commands to initialize, ingest, search, and inspect
SimpleV databases without writing Python scripts.

See CLI_SPEC.md for full details.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from simplev.chunking import chunk_document
from simplev.client import Client
from simplev.parsers import auto_parse
from simplev.persistence import FileManager


def _ensure_sv_extension(path: Path) -> Path:
    """Ensure database path ends with .sv."""
    if path.suffix != ".sv":
        return path.with_suffix(".sv")
    return path


def cmd_init(args: argparse.Namespace) -> int:
    """Create a new empty SimpleV database."""
    db_path = _ensure_sv_extension(Path(args.db_name))
    if db_path.exists() and not args.force:
        print(
            f"Error: Database file '{db_path}' already exists. "
            "Use --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    try:
        fm = FileManager()
        fm.create_empty(db_path, dimension=args.dimension)
        print(
            f"Successfully initialized empty SimpleV database: {db_path} "
            f"(dimension={args.dimension})"
        )
        return 0
    except Exception as e:
        print(f"Error initializing database: {e}", file=sys.stderr)
        return 1


def cmd_ingest(args: argparse.Namespace) -> int:
    """Ingest document(s) from a file or directory into the database."""
    source_path = Path(args.path)
    if not source_path.exists():
        print(f"Error: Path '{source_path}' does not exist.", file=sys.stderr)
        return 1

    db_path = _ensure_sv_extension(Path(args.db))
    supported_extensions = {".txt", ".md", ".markdown", ".csv", ".pdf", ".docx"}

    if source_path.is_file():
        files_to_process = [source_path]
    elif source_path.is_dir():
        files_to_process = [
            p for p in source_path.rglob("*")
            if p.is_file() and p.suffix.lower() in supported_extensions
        ]
    else:
        print(
            f"Error: '{source_path}' is neither a regular file nor a directory.",
            file=sys.stderr,
        )
        return 1

    if not files_to_process:
        print(
            f"No supported document files found in '{source_path}'.",
            file=sys.stderr,
        )
        return 1

    try:
        db = Client(path=db_path)
    except Exception as e:
        print(f"Error opening database '{db_path}': {e}", file=sys.stderr)
        return 1

    all_chunks = []
    files_processed = 0

    for file_path in files_to_process:
        try:
            text = auto_parse(file_path)
            if not text.strip():
                continue

            chunks = chunk_document(
                text=text,
                doc_id_prefix=file_path.stem,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                separator=args.separator,
                metadata={"source_file": str(file_path.name)},
            )
            all_chunks.extend(chunks)
            files_processed += 1
        except Exception as e:
            print(f"Warning: Failed to parse '{file_path}': {e}", file=sys.stderr)

    if not all_chunks:
        print(
            "No content could be extracted from the specified documents.",
            file=sys.stderr,
        )
        return 1

    try:
        added_ids = db.add_many(all_chunks)
        db.commit()
        print(
            f"Successfully ingested {files_processed} file(s) "
            f"({len(added_ids)} chunks) into {db_path}."
        )
        return 0
    except Exception as e:
        print(f"Error adding documents to database: {e}", file=sys.stderr)
        return 1


def cmd_search(args: argparse.Namespace) -> int:
    """Search the database semantically."""
    db_path = _ensure_sv_extension(Path(args.db))
    if not db_path.exists():
        print(f"Error: Database file '{db_path}' not found.", file=sys.stderr)
        return 1

    # Parse metadata filters if provided (key=value)
    filters = None
    if args.filter:
        filters = {}
        for f in args.filter:
            if "=" in f:
                k, v = f.split("=", 1)
                filters[k.strip()] = v.strip()
            else:
                print(
                    f"Warning: Ignoring malformed filter '{f}' (expected key=value).",
                    file=sys.stderr,
                )

    try:
        index_type = getattr(args, "index_type", "flat")
        db = Client(path=db_path, index_type=index_type)
        if getattr(args, "hybrid", False):
            results = db.hybrid_search(
                query=args.query,
                top_k=args.top,
                alpha=getattr(args, "alpha", 0.5),
                filters=filters,
            )
        else:
            results = db.search(
                query=args.query, top_k=args.top, filters=filters
            )

        if args.format == "json":
            output = [r.to_dict() for r in results]
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            if not results:
                print("No matching documents found.")
            else:
                print(f"Top {len(results)} results for query: '{args.query}'\n")
                for i, r in enumerate(results, 1):
                    print(f"{i}. [{r.score:.4f}] {r.doc_id}")
                    preview = r.text.replace("\n", " ")[:120]
                    print(f"   {preview}...")
                    if r.metadata:
                        print(f"   Metadata: {r.metadata}")
                    print()
        return 0
    except Exception as e:
        print(f"Error during search: {e}", file=sys.stderr)
        return 1


def cmd_get(args: argparse.Namespace) -> int:
    """Retrieve a document record by doc_id."""
    db_path = _ensure_sv_extension(Path(args.db))
    if not db_path.exists():
        print(f"Error: Database file '{db_path}' not found.", file=sys.stderr)
        return 1

    try:
        db = Client(path=db_path)
        record = db.get(args.doc_id)
        if record is None:
            print(
                f"Error: Document '{args.doc_id}' not found.", file=sys.stderr
            )
            return 1

        if args.json:
            print(json.dumps(record, indent=2, ensure_ascii=False))
        else:
            print(f"Document ID: {record['doc_id']}")
            print(f"Text       : {record['text']}")
            if record.get("metadata"):
                print(f"Metadata   : {record['metadata']}")
        return 0
    except Exception as e:
        print(f"Error retrieving document: {e}", file=sys.stderr)
        return 1


def cmd_info(args: argparse.Namespace) -> int:
    """Display diagnostic database summary."""
    db_path = _ensure_sv_extension(Path(args.db_name))
    if not db_path.exists():
        print(f"Error: Database file '{db_path}' not found.", file=sys.stderr)
        return 1

    try:
        db = Client(path=db_path)
        info = db.info()

        if args.json:
            print(json.dumps(info, indent=2))
        else:
            print(f"SimpleV Database Diagnostic: {db_path}")
            print(f"  Active Documents : {info['active_count']}")
            print(f"  Total Documents  : {info['count']}")
            print(f"  Vector Dimension : {info['dimension']}")
            print(f"  Distance Metric  : {info['metric']}")
            print(f"  Index Type       : {info['index_type']}")
            print(f"  Model Name       : {info['model_name']}")
            if info["wal_exists"] and info["wal_size_bytes"] > 0:
                wal_status = f"Active ({info['wal_size_bytes']} bytes)"
            else:
                wal_status = "Clean / Empty"
            print(f"  Write-Ahead Log  : {wal_status}")
        return 0
    except Exception as e:
        print(f"Error reading database info: {e}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        prog="simplev",
        description="SimpleV - The SQLite for Vector Search (Command Line Interface)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # init
    p_init = subparsers.add_parser(
        "init", help="Create a new empty SimpleV database"
    )
    p_init.add_argument(
        "db_name", help="Path to database file (e.g., knowledge_base.sv)"
    )
    p_init.add_argument(
        "--dimension",
        type=int,
        default=384,
        help="Embedding dimension size (default: 384)",
    )
    p_init.add_argument(
        "--force", action="store_true", help="Overwrite existing file if present"
    )

    # ingest
    p_ingest = subparsers.add_parser(
        "ingest", help="Chunk, embed, and ingest documents"
    )
    p_ingest.add_argument(
        "path", help="Path to a document file or directory of documents"
    )
    p_ingest.add_argument(
        "--db", required=True, help="Path to target SimpleV database"
    )
    p_ingest.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Target chunk size in characters (default: 500)",
    )
    p_ingest.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Chunk overlap in characters (default: 50)",
    )
    p_ingest.add_argument(
        "--separator",
        type=str,
        default=None,
        help="Separator boundary to split on (e.g., '\\n')",
    )

    # search
    p_search = subparsers.add_parser(
        "search", help="Perform semantic search on a database"
    )
    p_search.add_argument("query", help="Natural language query string")
    p_search.add_argument(
        "--db", required=True, help="Path to SimpleV database"
    )
    p_search.add_argument(
        "--top",
        "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    p_search.add_argument(
        "--filter",
        action="append",
        help="Filter criteria in KEY=VALUE format (can be specified multiple times)",
    )
    p_search.add_argument(
        "--format",
        choices=["json", "table"],
        default="json",
        help="Output format (default: json)",
    )
    p_search.add_argument(
        "--index-type",
        choices=["flat", "hnsw"],
        default="flat",
        help="Index type to use for search (default: flat)",
    )
    p_search.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid dense + sparse (BM25) search",
    )
    p_search.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Hybrid balance weight: 1.0 = dense only, 0.0 = BM25 only (default: 0.5)",
    )

    # get
    p_get = subparsers.add_parser(
        "get", help="Retrieve a document record by doc_id"
    )
    p_get.add_argument("doc_id", help="Document ID to retrieve")
    p_get.add_argument(
        "--db", required=True, help="Path to SimpleV database"
    )
    p_get.add_argument(
        "--json", action="store_true", help="Output document record as JSON"
    )

    # info
    p_info = subparsers.add_parser(
        "info", help="Display diagnostic summary of a database"
    )
    p_info.add_argument("db_name", help="Path to SimpleV database")
    p_info.add_argument(
        "--json", action="store_true", help="Output diagnostic info as JSON"
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "init": cmd_init,
        "ingest": cmd_ingest,
        "search": cmd_search,
        "get": cmd_get,
        "info": cmd_info,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
