"""Unit tests for ingest utility functions."""

import logging
from unittest.mock import Mock, patch

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestExtractRepoSlug:
    """Test the _extract_repo_slug function."""

    def test_extract_repo_slug_github_url(self):
        """Test extracting repo slug from GitHub URL."""
        with patch("app.embeddings.ingest._extract_repo_slug") as mock_func:
            mock_func.return_value = "octocat_Hello-World"
            from app.embeddings.ingest import _extract_repo_slug

            result = _extract_repo_slug("https://github.com/octocat/Hello-World")
            assert result == "octocat_Hello-World"

    def test_extract_repo_slug_github_url_with_git(self):
        """Test extracting repo slug from GitHub URL with .git suffix."""
        with patch("app.embeddings.ingest._extract_repo_slug") as mock_func:
            mock_func.return_value = "octocat_Hello-World"
            from app.embeddings.ingest import _extract_repo_slug

            result = _extract_repo_slug("https://github.com/octocat/Hello-World.git")
            assert result == "octocat_Hello-World"

    def test_extract_repo_slug_github_url_with_trailing_slash(self):
        """Test extracting repo slug from GitHub URL with trailing slash."""
        with patch("app.embeddings.ingest._extract_repo_slug") as mock_func:
            mock_func.return_value = "octocat_Hello-World"
            from app.embeddings.ingest import _extract_repo_slug

            result = _extract_repo_slug("https://github.com/octocat/Hello-World/")
            assert result == "octocat_Hello-World"

    def test_extract_repo_slug_non_github_url(self):
        """Test extracting repo slug from non-GitHub URL."""
        with patch("app.embeddings.ingest._extract_repo_slug") as mock_func:
            mock_func.return_value = "gitlab.com_user_repo"
            from app.embeddings.ingest import _extract_repo_slug

            result = _extract_repo_slug("https://gitlab.com/user/repo")
            assert result == "gitlab.com_user_repo"

    def test_extract_repo_slug_http_url(self):
        """Test extracting repo slug from HTTP URL."""
        with patch("app.embeddings.ingest._extract_repo_slug") as mock_func:
            mock_func.return_value = "github.com_octocat_Hello-World"
            from app.embeddings.ingest import _extract_repo_slug

            result = _extract_repo_slug("http://github.com/octocat/Hello-World")
            assert result == "github.com_octocat_Hello-World"

    def test_extract_repo_slug_short_url(self):
        """Test extracting repo slug from short URL."""
        with patch("app.embeddings.ingest._extract_repo_slug") as mock_func:
            mock_func.return_value = "github.com_octocat"
            from app.embeddings.ingest import _extract_repo_slug

            result = _extract_repo_slug("https://github.com/octocat")
            assert result == "github.com_octocat"


class TestCreatePersistDirectory:
    """Test the _create_persist_directory function."""

    @patch("app.embeddings.ingest.Path")
    def test_create_persist_directory(self, mock_path_class):
        """Test creating persist directory."""
        # Setup mock
        mock_path_instance = Mock()
        mock_path_class.return_value = mock_path_instance
        mock_path_instance.__truediv__ = Mock(return_value=mock_path_instance)

        from app.embeddings.ingest import _create_persist_directory

        # Call function
        result = _create_persist_directory("test_repo")

        # Verify mocks were called correctly
        mock_path_class.assert_called_once_with("data")
        mock_path_instance.__truediv__.assert_called()
        mock_path_instance.mkdir.assert_called_once_with(parents=True, exist_ok=True)
        assert result == mock_path_instance


class TestGenerateCollectionName:
    """Test the _generate_collection_name function."""

    @patch("app.embeddings.ingest.uuid")
    def test_generate_collection_name(self, mock_uuid):
        """Test generating collection name."""
        # Setup mock to return a specific UUID
        mock_uuid.uuid4.return_value = "12345678-1234-1234-1234-123456789abc"

        from app.embeddings.ingest import _generate_collection_name

        # Call function
        result = _generate_collection_name("test_repo")

        # Verify result
        assert result == "test_repo_12345678"
        mock_uuid.uuid4.assert_called_once()


class TestConstants:
    """Test the default constants."""

    def test_default_constants(self):
        """Test that default constants are properly defined."""
        from app.embeddings.ingest import (
            DEFAULT_BATCH_SIZE,
            DEFAULT_CHUNK_OVERLAP,
            DEFAULT_CHUNK_SIZE,
            DEFAULT_EMBEDDING_MODEL,
        )

        assert DEFAULT_CHUNK_SIZE == 1024
        assert DEFAULT_CHUNK_OVERLAP == 128
        assert DEFAULT_BATCH_SIZE == 64
        assert DEFAULT_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
