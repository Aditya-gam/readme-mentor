"""URL parsing and utility function tests for GitHub loader."""

import pytest

from app.github.loader import (
    _create_repo_slug,
    _extract_repo_info,
    _get_default_branch,
    _is_markdown_file,
    _matches_readme_pattern,
)


class TestURLParsing:
    """Test URL parsing and utility functions."""

    def test_extract_repo_info_valid_urls(self):
        """Test extracting repository info from valid GitHub URLs."""
        test_cases = [
            ("https://github.com/octocat/Hello-World", ("octocat", "Hello-World")),
            ("https://github.com/octocat/Hello-World.git", ("octocat", "Hello-World")),
            ("https://github.com/octocat/Hello-World/", ("octocat", "Hello-World")),
            (
                "https://github.com/octocat/Hello-World/tree/main",
                ("octocat", "Hello-World"),
            ),
            ("git@github.com:octocat/Hello-World.git", ("octocat", "Hello-World")),
        ]

        for url, expected in test_cases:
            result = _extract_repo_info(url)
            assert result == expected, f"Failed for URL: {url}"

    def test_extract_repo_info_invalid_url(self):
        """Test extracting repository info from invalid GitHub URLs."""
        invalid_urls = [
            "https://gitlab.com/octocat/Hello-World",
            "https://github.com/octocat",
            "https://github.com/",
            "not-a-url",
        ]

        for url in invalid_urls:
            with pytest.raises(ValueError):
                _extract_repo_info(url)

    def test_create_repo_slug(self):
        """Test creating repository slug."""
        assert _create_repo_slug("octocat", "Hello-World") == "octocat_Hello-World"
        assert _create_repo_slug("user", "repo-name") == "user_repo-name"

    def test_is_markdown_file(self):
        """Test markdown file detection."""
        assert _is_markdown_file("README.md") is True
        assert _is_markdown_file("documentation.markdown") is True
        assert _is_markdown_file("guide.mdx") is True
        assert _is_markdown_file("script.py") is False
        assert _is_markdown_file("README.txt") is False

    def test_matches_readme_pattern(self):
        """Test README pattern matching."""
        assert _matches_readme_pattern("README.md") is True
        assert _matches_readme_pattern("readme.markdown") is True
        assert _matches_readme_pattern("README.mdx") is True
        assert _matches_readme_pattern("README.txt") is False
        assert _matches_readme_pattern("documentation.md") is False

    def test_get_default_branch_success(self):
        """Test successful default branch retrieval."""
        from unittest.mock import Mock

        mock_repo = Mock()
        mock_repo.default_branch = "main"

        result = _get_default_branch(mock_repo)
        assert result == "main"

    def test_get_default_branch_exception(self):
        """Test default branch retrieval when exception occurs."""
        from unittest.mock import Mock, PropertyMock

        mock_repo = Mock()
        # Mock the property to raise an exception when accessed
        type(mock_repo).default_branch = PropertyMock(
            side_effect=Exception("API Error")
        )

        with pytest.raises(Exception, match="API Error"):
            _get_default_branch(mock_repo)
