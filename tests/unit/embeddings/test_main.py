"""Unit tests for embeddings CLI module."""

import argparse
import logging
import tempfile
from unittest.mock import Mock, patch

import pytest

from app.embeddings.__main__ import (
    main,
    parse_arguments,
    setup_logging,
    validate_arguments,
)


class TestParseArguments:
    """Test command line argument parsing."""

    def test_parse_arguments_basic(self):
        """Test basic argument parsing."""
        with patch("sys.argv", ["script", "https://github.com/octocat/Hello-World"]):
            args = parse_arguments()

            assert args.repo_url == "https://github.com/octocat/Hello-World"
            assert args.file_pattern == ["README*", "docs/**/*.md"]
            assert args.chunk_size == 1024
            assert args.chunk_overlap == 128
            assert args.batch_size == 64
            assert args.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
            assert args.collection_name is None
            assert args.persist_directory is None
            assert args.verbose is False
            assert args.dry_run is False

    def test_parse_arguments_with_options(self):
        """Test argument parsing with custom options."""
        with patch(
            "sys.argv",
            [
                "script",
                "https://github.com/octocat/Hello-World",
                "--chunk-size",
                "512",
                "--chunk-overlap",
                "64",
                "--batch-size",
                "32",
                "--embedding-model",
                "test-model",
                "--collection-name",
                "test-collection",
                "--persist-directory",
                "/tmp/test",
                "--verbose",
                "--dry-run",
            ],
        ):
            args = parse_arguments()

            assert args.repo_url == "https://github.com/octocat/Hello-World"
            assert args.chunk_size == 512
            assert args.chunk_overlap == 64
            assert args.batch_size == 32
            assert args.embedding_model == "test-model"
            assert args.collection_name == "test-collection"
            assert args.persist_directory == "/tmp/test"
            assert args.verbose is True
            assert args.dry_run is True

    def test_parse_arguments_custom_file_patterns(self):
        """Test argument parsing with custom file patterns."""
        with patch(
            "sys.argv",
            [
                "script",
                "https://github.com/octocat/Hello-World",
                "--file-pattern",
                "*.md",
                "docs/**/*.rst",
            ],
        ):
            args = parse_arguments()

            assert args.file_pattern == ["*.md", "docs/**/*.rst"]


class TestValidateArguments:
    """Test argument validation."""

    def test_validate_arguments_valid(self):
        """Test validation with valid arguments."""
        args = argparse.Namespace(
            chunk_size=1024, chunk_overlap=128, batch_size=64, persist_directory=None
        )

        # Should not raise any exceptions
        validate_arguments(args)

    def test_validate_arguments_invalid_chunk_size(self):
        """Test validation with invalid chunk size."""
        args = argparse.Namespace(
            chunk_size=0, chunk_overlap=128, batch_size=64, persist_directory=None
        )

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            validate_arguments(args)

    def test_validate_arguments_invalid_chunk_overlap(self):
        """Test validation with invalid chunk overlap."""
        args = argparse.Namespace(
            chunk_size=1024, chunk_overlap=-1, batch_size=64, persist_directory=None
        )

        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            validate_arguments(args)

    def test_validate_arguments_overlap_greater_than_size(self):
        """Test validation when overlap is greater than chunk size."""
        args = argparse.Namespace(
            chunk_size=100, chunk_overlap=150, batch_size=64, persist_directory=None
        )

        with pytest.raises(
            ValueError, match="chunk_overlap must be less than chunk_size"
        ):
            validate_arguments(args)

    def test_validate_arguments_invalid_batch_size(self):
        """Test validation with invalid batch size."""
        args = argparse.Namespace(
            chunk_size=1024, chunk_overlap=128, batch_size=0, persist_directory=None
        )

        with pytest.raises(ValueError, match="batch_size must be positive"):
            validate_arguments(args)

    def test_validate_arguments_invalid_persist_directory(self):
        """Test validation with invalid persist directory."""
        with tempfile.NamedTemporaryFile() as temp_file:
            args = argparse.Namespace(
                chunk_size=1024,
                chunk_overlap=128,
                batch_size=64,
                persist_directory=temp_file.name,
            )

            with pytest.raises(
                ValueError, match="persist_directory must be a directory"
            ):
                validate_arguments(args)


class TestSetupLogging:
    """Test logging setup."""

    def test_setup_logging_verbose(self):
        """Test logging setup with verbose mode."""
        setup_logging(True)

        # Check that logging level is set to DEBUG
        assert logging.getLogger().level <= logging.DEBUG

    def test_setup_logging_normal(self):
        """Test logging setup with normal mode."""
        setup_logging(False)

        # Check that logging level is set to INFO
        assert logging.getLogger().level == logging.INFO


class TestMain:
    """Test main CLI function."""

    @patch("app.embeddings.__main__.parse_arguments")
    @patch("app.embeddings.__main__.setup_logging")
    @patch("app.embeddings.__main__.validate_arguments")
    @patch("app.embeddings.__main__.validate_repo_url")
    @patch("app.embeddings.__main__.ingest_repository")
    def test_main_success(
        self,
        mock_ingest,
        mock_validate_url,
        mock_validate_args,
        mock_setup_logging,
        mock_parse_args,
    ):
        """Test successful main execution."""
        # Mock arguments
        args = Mock()
        args.repo_url = "https://github.com/octocat/Hello-World"
        args.verbose = False
        args.dry_run = False
        args.file_pattern = ["README*", "docs/**/*.md"]
        args.chunk_size = 1024
        args.chunk_overlap = 128
        args.batch_size = 64
        args.embedding_model = "test-model"
        args.collection_name = None
        args.persist_directory = None

        mock_parse_args.return_value = args
        mock_validate_url.return_value = "https://github.com/octocat/Hello-World"

        # Mock vectorstore
        mock_vectorstore = Mock()
        mock_vectorstore._collection.name = "test-collection"
        mock_vectorstore._collection.count.return_value = 10
        mock_ingest.return_value = mock_vectorstore

        with patch("sys.argv", ["script", "https://github.com/octocat/Hello-World"]):
            result = main()

            assert result == 0
            mock_parse_args.assert_called_once()
            mock_setup_logging.assert_called_once_with(False)
            mock_validate_args.assert_called_once_with(args)
            mock_validate_url.assert_called_once_with(
                "https://github.com/octocat/Hello-World"
            )
            mock_ingest.assert_called_once()

    @patch("app.embeddings.__main__.parse_arguments")
    @patch("app.embeddings.__main__.setup_logging")
    @patch("app.embeddings.__main__.validate_arguments")
    @patch("app.embeddings.__main__.validate_repo_url")
    def test_main_dry_run(
        self, mock_validate_url, mock_validate_args, mock_setup_logging, mock_parse_args
    ):
        """Test main execution in dry run mode."""
        # Mock arguments
        args = Mock()
        args.repo_url = "https://github.com/octocat/Hello-World"
        args.verbose = False
        args.dry_run = True
        args.file_pattern = ["README*", "docs/**/*.md"]
        args.chunk_size = 1024
        args.chunk_overlap = 128
        args.batch_size = 64
        args.embedding_model = "test-model"
        args.collection_name = None
        args.persist_directory = None

        mock_parse_args.return_value = args
        mock_validate_url.return_value = "https://github.com/octocat/Hello-World"

        with patch("sys.argv", ["script", "https://github.com/octocat/Hello-World"]):
            result = main()

            assert result == 0
            mock_parse_args.assert_called_once()
            mock_setup_logging.assert_called_once_with(False)
            mock_validate_args.assert_called_once_with(args)
            mock_validate_url.assert_called_once_with(
                "https://github.com/octocat/Hello-World"
            )

    @patch("app.embeddings.__main__.parse_arguments")
    @patch("app.embeddings.__main__.setup_logging")
    @patch("app.embeddings.__main__.validate_arguments")
    @patch("app.embeddings.__main__.validate_repo_url")
    def test_main_invalid_url(
        self, mock_validate_url, mock_validate_args, mock_setup_logging, mock_parse_args
    ):
        """Test main execution with invalid URL."""
        # Mock arguments
        args = Mock()
        args.repo_url = "https://github.com/octocat/Hello-World"
        args.verbose = False
        args.dry_run = False

        mock_parse_args.return_value = args
        mock_validate_url.side_effect = ValueError("Invalid URL")

        with patch("sys.argv", ["script", "https://github.com/octocat/Hello-World"]):
            result = main()

            assert result == 1

    @patch("app.embeddings.__main__.parse_arguments")
    def test_main_keyboard_interrupt(self, mock_parse_args):
        """Test main execution with keyboard interrupt."""
        mock_parse_args.side_effect = KeyboardInterrupt()

        with patch("sys.argv", ["script", "https://github.com/octocat/Hello-World"]):
            result = main()

            assert result == 1

    @patch("app.embeddings.__main__.parse_arguments")
    @patch("app.embeddings.__main__.setup_logging")
    @patch("app.embeddings.__main__.validate_arguments")
    @patch("app.embeddings.__main__.validate_repo_url")
    def test_main_unexpected_error(
        self, mock_validate_url, mock_validate_args, mock_setup_logging, mock_parse_args
    ):
        """Test main execution with unexpected error."""
        # Mock arguments first, then make ingest_repository raise an exception
        args = Mock()
        args.repo_url = "https://github.com/octocat/Hello-World"
        args.verbose = False
        args.dry_run = False
        args.file_pattern = ["README*", "docs/**/*.md"]
        args.chunk_size = 1024
        args.chunk_overlap = 128
        args.batch_size = 64
        args.embedding_model = "test-model"
        args.collection_name = None
        args.persist_directory = None

        mock_parse_args.return_value = args

        with patch("app.embeddings.__main__.ingest_repository") as mock_ingest:
            mock_ingest.side_effect = Exception("Unexpected error")

            with patch(
                "sys.argv", ["script", "https://github.com/octocat/Hello-World"]
            ):
                result = main()

                assert result == 1


if __name__ == "__main__":
    pytest.main([__file__])
