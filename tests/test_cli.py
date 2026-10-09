"""
Tests for SimpleV Command Line Interface (CLI).

Covers init, ingest, search, info, and compact commands along with
argument validation, error handling, and JSON output formats.
"""

import json
from unittest.mock import patch

import numpy as np
import pytest

from simplev.cli import main
from simplev.client import Client


@pytest.fixture
def mock_emb():
    with patch("simplev.client.EmbeddingManager") as MockEmb:
        inst = MockEmb.return_value
        inst.dimension = 4
        inst.embed.side_effect = lambda t: np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        inst.embed_batch.side_effect = lambda texts: np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1)
        )
        yield inst


class TestCLIInit:
    """Test 'simplev init' command."""

    def test_init_creates_database_file(self, tmp_path):
        db_path = tmp_path / "test.sv"
        ret = main(["init", str(db_path)])
        assert ret == 0
        assert db_path.exists()

    def test_init_existing_fails_without_force(self, tmp_path):
        db_path = tmp_path / "test.sv"
        db_path.touch()
        ret = main(["init", str(db_path)])
        assert ret == 1

    def test_init_existing_with_force(self, tmp_path):
        db_path = tmp_path / "test.sv"
        db_path.touch()
        ret = main(["init", str(db_path), "--force"])
        assert ret == 0
        assert db_path.exists()


class TestCLIIngest:
    """Test 'simplev ingest' command."""

    def test_ingest_single_file(self, tmp_path, mock_emb):
        db_path = tmp_path / "data.sv"
        main(["init", str(db_path), "--dimension", "4"])

        doc_file = tmp_path / "sample.txt"
        doc_file.write_text(
            "Hello SimpleV! This is a test document for CLI ingestion.",
            encoding="utf-8",
        )

        ret = main(["ingest", str(doc_file), "--db", str(db_path)])
        assert ret == 0

        client = Client(path=db_path)
        assert client.count >= 1

    def test_ingest_directory(self, tmp_path, mock_emb):
        db_path = tmp_path / "data.sv"
        main(["init", str(db_path), "--dimension", "4"])

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "doc1.txt").write_text("Content of first file.", encoding="utf-8")
        (docs_dir / "doc2.md").write_text(
            "Content of second markdown file.", encoding="utf-8"
        )

        ret = main(["ingest", str(docs_dir), "--db", str(db_path)])
        assert ret == 0

        client = Client(path=db_path)
        assert client.count >= 2


class TestCLISearch:
    """Test 'simplev search' command."""

    def test_search_output_json(self, tmp_path, mock_emb, capsys):
        db_path = tmp_path / "search.sv"
        client = Client(path=db_path)
        client.add("d1", "Search engine vector database")
        client.commit()

        ret = main(["search", "database", "--db", str(db_path), "--top", "1"])
        assert ret == 0

        captured = capsys.readouterr()
        results = json.loads(captured.out)
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["doc_id"] == "d1"


class TestCLIInfo:
    """Test 'simplev info' command."""

    def test_info_text(self, tmp_path, mock_emb, capsys):
        db_path = tmp_path / "info.sv"
        client = Client(path=db_path)
        client.add("d1", "Test doc")
        client.commit()

        ret = main(["info", str(db_path)])
        assert ret == 0
        captured = capsys.readouterr()
        assert "SimpleV Database Diagnostics" in captured.out
        assert "Active Docs:     1" in captured.out

    def test_info_json(self, tmp_path, mock_emb, capsys):
        db_path = tmp_path / "info.sv"
        client = Client(path=db_path)
        client.add("d1", "Test doc")
        client.commit()

        ret = main(["info", str(db_path), "--json"])
        assert ret == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["active_count"] == 1


class TestCLICompact:
    """Test 'simplev compact' command."""

    def test_compact_purges_deletions(self, tmp_path, mock_emb):
        db_path = tmp_path / "compact.sv"
        client = Client(path=db_path)
        client.add("d1", "Doc 1")
        client.add("d2", "Doc 2")
        client.delete("d1")
        client.commit()

        ret = main(["compact", "--db", str(db_path)])
        assert ret == 0

        client2 = Client(path=db_path)
        assert client2.count == 1
