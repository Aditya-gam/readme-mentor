"""Unit tests for ingest utility functions."""

from pathlib import Path
from unittest.mock import Mock, patch

from app.embeddings.ingest import (
    _create_persist_directory,
    _extract_repo_slug,
    _generate_collection_name,
)


class TestExtractRepoSlug:
    """Test the _extract_repo_slug function."""

    def test_extract_repo_slug_github_url(self):
        """Test extracting repo slug from GitHub URL."""
        url = "https://github.com/octocat/Hello-World"
        result = _extract_repo_slug(url)
        assert result == "octocat_Hello-World"

    def test_extract_repo_slug_github_url_with_git(self):
        """Test extracting repo slug from GitHub URL with .git suffix."""
        url = "https://github.com/octocat/Hello-World.git"
        result = _extract_repo_slug(url)
        assert result == "octocat_Hello-World"

    def test_extract_repo_slug_github_url_with_trailing_slash(self):
        """Test extracting repo slug from GitHub URL with trailing slash."""
        url = "https://github.com/octocat/Hello-World/"
        result = _extract_repo_slug(url)
        assert result == "octocat_Hello-World"

    def test_extract_repo_slug_non_github_url(self):
        """Test extracting repo slug from non-GitHub URL."""
        url = "https://gitlab.com/user/repo"
        result = _extract_repo_slug(url)
        assert result == "gitlab.com_user_repo"

    def test_extract_repo_slug_http_url(self):
        """Test extracting repo slug from HTTP URL."""
        url = "http://github.com/octocat/Hello-World"
        result = _extract_repo_slug(url)
        assert result == "octocat_Hello-World"

    def test_extract_repo_slug_short_url(self):
        """Test extracting repo slug from short URL."""
        url = "github.com/octocat/Hello-World"
        result = _extract_repo_slug(url)
        # The function should extract owner and repo name from URLs containing github.com
        assert result == "octocat_Hello-World"


class TestCreatePersistDirectory:
    """Test the _create_persist_directory function."""

    def test_create_persist_directory(self):
        """Test creating persist directory."""
        # Use a temporary directory for testing
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the data directory to be our temp directory
            with patch("app.embeddings.ingest.Path") as mock_path_class:
                mock_path_instance = Mock()
                mock_path_class.return_value = mock_path_instance
                mock_path_instance.__truediv__ = Mock(
                    return_value=Path(temp_dir) / "test_repo"
                )

                # Call function
                result = _create_persist_directory("test_repo")

                # Verify mocks were called correctly
                mock_path_class.assert_called_once_with("data")
                mock_path_instance.__truediv__.assert_called()
                # The function returns the path with /chroma suffix
                assert result == Path(temp_dir) / "test_repo" / "chroma"


class TestGenerateCollectionName:
    """Test the _generate_collection_name function."""

    def test_generate_collection_name(self):
        """Test generating collection name."""
        # Call function
        result = _generate_collection_name("test_repo")

        # Verify result format - should be "test_repo_" followed by 8 hex characters
        assert result.startswith("test_repo_")
        assert len(result) == len("test_repo_") + 8
        # Check that the suffix is hexadecimal
        suffix = result[len("test_repo_") :]
        assert all(c in "0123456789abcdef" for c in suffix)
