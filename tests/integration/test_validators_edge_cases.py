"""Edge case tests for URL validator functionality."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from app.utils.validators import (
    InvalidRepoURLError,
    _contains_invalid_path_segments,
    _is_valid_github_repo_url,
    _normalize_github_url,
    validate_repo_url,
)


class TestValidatorsEdgeCases:
    """Test edge cases for URL validator functions."""

    def test_normalize_github_url_invalid_protocol(self):
        """Test _normalize_github_url with invalid protocol."""
        with pytest.raises(ValueError):
            _normalize_github_url("ftp://github.com/user/repo")

    def test_normalize_github_url_invalid_domain(self):
        """Test _normalize_github_url with invalid domain."""
        with pytest.raises(ValueError):
            _normalize_github_url("https://gitlab.com/user/repo")

    def test_is_valid_github_repo_url_edge_cases(self):
        """Test _is_valid_github_repo_url with edge cases."""
        # Test valid URLs
        assert _is_valid_github_repo_url("https://github.com/user/repo") is True
        assert (
            _is_valid_github_repo_url("https://github.com/user-name/repo_name") is True
        )
        assert _is_valid_github_repo_url("https://github.com/user/repo-name") is True

        # Test invalid URLs
        assert _is_valid_github_repo_url("https://github.com/user") is False
        assert _is_valid_github_repo_url("https://github.com/user/repo/extra") is False
        assert (
            _is_valid_github_repo_url("https://github.com/user/repo?param=value")
            is False
        )
        assert (
            _is_valid_github_repo_url("https://github.com/user/repo#fragment") is False
        )

        # Test edge cases
        assert _is_valid_github_repo_url("https://github.com/-user/repo") is False
        assert _is_valid_github_repo_url("https://github.com/user-/repo") is False
        assert _is_valid_github_repo_url("https://github.com/user/-repo") is False
        assert _is_valid_github_repo_url("https://github.com/user/repo-") is False

    def test_contains_invalid_path_segments_edge_cases(self):
        """Test _contains_invalid_path_segments with edge cases."""
        # Test valid paths
        assert _contains_invalid_path_segments("user/repo") is False
        assert _contains_invalid_path_segments("user-name/repo_name") is False

        # Test invalid paths
        assert _contains_invalid_path_segments("user/repo/tree/main") is True
        assert _contains_invalid_path_segments("user/repo/blob/main/file.md") is True
        assert _contains_invalid_path_segments("user/repo/pull/123") is True
        assert _contains_invalid_path_segments("user/repo/issues/456") is True
        assert _contains_invalid_path_segments("user/repo/wiki") is False
        assert _contains_invalid_path_segments("user/repo/settings") is False
        assert _contains_invalid_path_segments("user/repo/actions") is False
        assert _contains_invalid_path_segments("user/repo/projects") is False
        assert _contains_invalid_path_segments("user/repo/security") is False
        assert _contains_invalid_path_segments("user/repo/network") is False
        assert _contains_invalid_path_segments("user/repo/pulse") is False
        assert _contains_invalid_path_segments("user/repo/graphs") is False
        assert _contains_invalid_path_segments("user/repo/releases") is False
        assert _contains_invalid_path_segments("user/repo/tags") is False
        assert _contains_invalid_path_segments("user/repo/branches") is False
        assert _contains_invalid_path_segments("user/repo/compare") is False
        assert _contains_invalid_path_segments("user/repo/search") is True
        assert _contains_invalid_path_segments("user/repo/archive") is False
        assert _contains_invalid_path_segments("user/repo/downloads") is False
        assert _contains_invalid_path_segments("user/repo/forks") is False
        assert _contains_invalid_path_segments("user/repo/stargazers") is False
        assert _contains_invalid_path_segments("user/repo/watchers") is False
        assert _contains_invalid_path_segments("user/repo/contributors") is False
        assert _contains_invalid_path_segments("user/repo/commits") is False

        # Test edge cases
        assert (
            _contains_invalid_path_segments("user/repo/") is False
        )  # Trailing slash should be handled by normalization
        assert _contains_invalid_path_segments("user/repo/extra/segment") is False

    def test_validate_repo_url_edge_cases(self):
        """Test validate_repo_url with edge cases."""
        # Test valid URLs
        assert (
            validate_repo_url("https://github.com/user/repo")
            == "https://github.com/user/repo"
        )
        assert (
            validate_repo_url("http://github.com/user/repo")
            == "https://github.com/user/repo"
        )
        assert (
            validate_repo_url("https://GITHUB.COM/user/repo")
            == "https://github.com/user/repo"
        )
        assert (
            validate_repo_url("https://github.com/user/repo/")
            == "https://github.com/user/repo"
        )

        # Test invalid URLs
        with pytest.raises(ValueError):
            validate_repo_url("")

        with pytest.raises(ValueError):
            validate_repo_url("   ")

        with pytest.raises(InvalidRepoURLError):
            validate_repo_url("https://gitlab.com/user/repo")

        with pytest.raises(InvalidRepoURLError):
            validate_repo_url("https://github.com/user/repo/tree/main")

    def test_save_file_content_error_logging(self, caplog):
        """Test error logging in _save_file_content."""
        from app.github.loader import _save_file_content

        result = _save_file_content(b"test", Path("/invalid/path/file.txt"))

        # Check that function returned False on error
        assert result is False

        # Check that error was logged
        assert "Failed to save file" in caplog.text

    def test_download_file_content_error_logging(self, caplog):
        """Test error logging in _download_file_content."""
        from app.github.loader import _download_file_content

        with patch("app.github.loader.httpx.Client") as mock_client:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = Exception("Download failed")
            mock_client.return_value.__enter__.return_value.get.return_value = (
                mock_response
            )

            result = _download_file_content(
                "https://github.com/octocat/Hello-World", "README.md", "main"
            )

            assert result is None
            assert "Failed to download file" in caplog.text

    def test_get_files_recursively_error_logging(self, caplog):
        """Test error logging in _get_files_recursively."""
        from app.github.loader import _get_files_recursively

        mock_repo = Mock()
        mock_repo.get_contents.side_effect = Exception("API Error")

        result = _get_files_recursively(mock_repo, "docs", "main")

        assert result == []
        assert "Failed to get contents" in caplog.text

    def test_validate_repo_url_empty_and_whitespace(self):
        """Test validate_repo_url with empty and whitespace inputs."""
        # Test empty string
        with pytest.raises(ValueError, match="URL cannot be empty or None"):
            validate_repo_url("")

        # Test whitespace only
        with pytest.raises(
            ValueError, match="URL cannot be empty after trimming whitespace"
        ):
            validate_repo_url("   ")

        # Test None
        with pytest.raises(ValueError, match="URL cannot be empty or None"):
            validate_repo_url(None)

    def test_is_valid_github_repo_url_extra_segments(self):
        """Test _is_valid_github_repo_url with extra path segments."""
        # Test URLs with extra segments
        assert _is_valid_github_repo_url("https://github.com/user/repo/extra") is False
        assert (
            _is_valid_github_repo_url("https://github.com/user/repo/segment1/segment2")
            is False
        )

        # Test URLs with query parameters
        assert (
            _is_valid_github_repo_url("https://github.com/user/repo?param=value")
            is False
        )
        assert _is_valid_github_repo_url("https://github.com/user/repo?") is False

        # Test URLs with fragments
        assert (
            _is_valid_github_repo_url("https://github.com/user/repo#fragment") is False
        )
        assert _is_valid_github_repo_url("https://github.com/user/repo#") is False
