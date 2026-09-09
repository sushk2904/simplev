"""
Tests for the SimpleV Command Line Interface (CLI).

Tests init, ingest, search, and info commands and options.
"""

import json
from unittest.mock import patch

import numpy as np
import pytest

from simplev.cli import main


@pytest.fixture
def mock_embeddings():
    """Mock EmbeddingManager so CLI tests run quickly and offline."""
    with patch("simplev.client.EmbeddingManager") as MockEmb:
        instance = MockEmb.return_value
        instance.dimension = 4
        vec4 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        instance.embed.side_effect = lambda text: vec4
        instance.embed_batch.side_effect = (
            lambda texts: np.stack([vec4 for _ in texts])
        )
        yield instance


class TestCLIInit:
    """Test 'simplev init' command."""

    def test_init_creates_database(self, tmp_path):
        db_file = tmp_path / "test_init.sv"
        exit_code = main(["init", str(db_file), "--dimension", "128"])
        assert exit_code == 0
        assert db_file.exists()
        assert db_file.stat().st_size > 0

    def test_init_existing_without_force_fails(self, tmp_path):
        db_file = tmp_path / "exists.sv"
        db_file.touch()
        exit_code = main(["init", str(db_file)])
        assert exit_code == 1

    def test_init_existing_with_force_succeeds(self, tmp_path):
        db_file = tmp_path / "exists.sv"
        db_file.touch()
        exit_code = main(["init", str(db_file), "--force", "--dimension", "64"])
        assert exit_code == 0
        assert db_file.exists()


class TestCLIInfo:
    """Test 'simplev info' command."""

    def test_info_human_readable(self, tmp_path, mock_embeddings, capsys):
        db_file = tmp_path / "info_test.sv"
        main(["init", str(db_file), "--dimension", "4"])

        exit_code = main(["info", str(db_file)])
        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "SimpleV Database Diagnostic" in captured
        assert "Active Documents" in captured

    def test_info_json_output(self, tmp_path, mock_embeddings, capsys):
        db_file = tmp_path / "info_json.sv"
        main(["init", str(db_file), "--dimension", "4"])
        capsys.readouterr()  # clear init output

        exit_code = main(["info", str(db_file), "--json"])
        assert exit_code == 0
        captured = capsys.readouterr().out
        data = json.loads(captured)
        assert data["count"] == 0
        assert data["dimension"] == 4
        assert "index_type" in data

    def test_info_missing_database_fails(self, tmp_path):
        exit_code = main(["info", str(tmp_path / "nonexistent.sv")])
        assert exit_code == 1


class TestCLIIngestAndSearch:
    """Test 'simplev ingest' and 'simplev search' commands end-to-end."""

    def test_ingest_and_search_json(self, tmp_path, mock_embeddings, capsys):
        db_file = tmp_path / "kb.sv"
        doc_file = tmp_path / "sample.txt"
        doc_file.write_text(
            "Artificial Intelligence is evolving rapidly across the world.",
            encoding="utf-8",
        )

        # Ingest document
        ingest_code = main(
            ["ingest", str(doc_file), "--db", str(db_file), "--chunk-size", "100"]
        )
        assert ingest_code == 0
        assert db_file.exists()
        capsys.readouterr()  # clear ingest output

        # Search database
        search_code = main(
            ["search", "AI technology", "--db", str(db_file), "--top", "2"]
        )
        assert search_code == 0
        captured = capsys.readouterr().out
        results = json.loads(captured)
        assert len(results) >= 1
        assert "doc_id" in results[0]
        assert "score" in results[0]
        assert "AI" in results[0]["text"] or "Artificial" in results[0]["text"]

    def test_search_table_format(self, tmp_path, mock_embeddings, capsys):
        db_file = tmp_path / "kb_table.sv"
        doc_file = tmp_path / "sample.md"
        doc_file.write_text("# Title\n\nVector databases are fast.", encoding="utf-8")

        main(["ingest", str(doc_file), "--db", str(db_file)])
        search_code = main(
            ["search", "vector db", "--db", str(db_file), "--format", "table"]
        )
        assert search_code == 0
        out = capsys.readouterr().out
        assert "Top" in out
        assert "Vector databases are fast" in out

    def test_ingest_missing_file_fails(self, tmp_path):
        exit_code = main(
            ["ingest", str(tmp_path / "missing.txt"), "--db", str(tmp_path / "db.sv")]
        )
        assert exit_code == 1

    def test_search_missing_db_fails(self, tmp_path):
        exit_code = main(["search", "query", "--db", str(tmp_path / "missing.sv")])
        assert exit_code == 1
