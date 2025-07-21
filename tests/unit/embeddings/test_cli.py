"""Unit tests for the embeddings CLI functionality."""

import sys
from unittest.mock import patch

import pytest

from app.embeddings.__main__ import main, parse_arguments


class TestCLIArgumentParsing:
    """Test CLI argument parsing functionality."""

    def test_parse_arguments_basic(self):
        """Test basic argument parsing."""
        sys.argv = ["app.embeddings", "https://github.com/test/repo"]
        args = parse_arguments()

        assert args.repo_url == "https://github.com/test/repo"
        assert args.chunk_size == 1024
        assert args.chunk_overlap == 128
        assert args.batch_size == 64
        assert args.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert args.file_glob is None
        assert args.collection_name is None
        assert args.persist_dir is None
        assert not args.verbose

    def test_parse_arguments_with_options(self):
        """Test argument parsing with all options."""
        sys.argv = [
            "app.embeddings",
            "https://github.com/test/repo",
            "--chunk-size",
            "512",
            "--chunk-overlap",
            "64",
            "--batch-size",
            "32",
            "--embedding-model",
            "test-model",
            "--file-glob",
            "*.md",
            "docs/**/*.md",
            "--collection-name",
            "test-collection",
            "--persist-dir",
            "./data",
            "--verbose",
        ]
        args = parse_arguments()

        assert args.repo_url == "https://github.com/test/repo"
        assert args.chunk_size == 512
        assert args.chunk_overlap == 64
        assert args.batch_size == 32
        assert args.embedding_model == "test-model"
        assert args.file_glob == ["*.md", "docs/**/*.md"]
        assert args.collection_name == "test-collection"
        assert args.persist_dir == "./data"
        assert args.verbose

    def test_parse_arguments_help(self):
        """Test help argument."""
        sys.argv = ["app.embeddings", "--help"]

        with pytest.raises(SystemExit) as exc_info:
            parse_arguments()

        assert exc_info.value.code == 0


class TestCLIMainFunction:
    """Test CLI main function functionality."""

    def test_main_ingestion_failure(self):
        """Test CLI execution when ingestion fails."""
        with patch(
            "app.embeddings.ingest.ingest_repository",
            side_effect=Exception("Test error"),
        ):
            with patch.object(
                sys, "argv", ["app.embeddings", "https://github.com/test/repo"]
            ):
                result = main()

        assert result == 1

    def test_main_keyboard_interrupt(self):
        """Test CLI execution with keyboard interrupt."""
        with patch(
            "app.embeddings.ingest.ingest_repository", side_effect=KeyboardInterrupt()
        ):
            with patch.object(
                sys, "argv", ["app.embeddings", "https://github.com/test/repo"]
            ):
                result = main()

        assert result == 1
