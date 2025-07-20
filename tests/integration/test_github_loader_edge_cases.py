"""Edge case tests for GitHub loader functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from app.github.loader import (
    _download_file_content,
    _get_files_recursively,
    _is_markdown_file,
    _matches_readme_pattern,
    _save_file_content,
    fetch_repository_files,
)


class TestGitHubLoaderEdgeCases:
    """Test edge cases for GitHub loader functions."""

    def test_is_markdown_file_edge_cases(self):
        """Test _is_markdown_file with edge cases."""
        # Test with empty string
        assert _is_markdown_file("") is False

        # Test with None (should raise TypeError)
        with pytest.raises(TypeError):
            _is_markdown_file(None)

        # Test with uppercase extensions
        assert _is_markdown_file("README.MD") is True
        assert _is_markdown_file("doc.MARKDOWN") is True
        assert _is_markdown_file("guide.MDX") is True

    def test_matches_readme_pattern_edge_cases(self):
        """Test _matches_readme_pattern with edge cases."""
        # Test with empty string
        assert _matches_readme_pattern("") is False

        # Test with None (should raise TypeError)
        with pytest.raises(TypeError):
            _matches_readme_pattern(None)

        # Test with different README variations
        assert _matches_readme_pattern("readme.md") is True
        assert _matches_readme_pattern("README.MD") is True
        assert _matches_readme_pattern("ReadMe.markdown") is True
        assert _matches_readme_pattern("readme.mdx") is True

        # Test non-README files
        assert _matches_readme_pattern("documentation.md") is False
        assert _matches_readme_pattern("README.txt") is False

    def test_get_files_recursively_exception(self):
        """Test _get_files_recursively when API raises exception."""
        mock_repo = Mock()
        mock_repo.get_contents.side_effect = Exception("API Error")

        result = _get_files_recursively(mock_repo, "docs", "main")
        assert result == []

    def test_download_file_content_exception(self):
        """Test _download_file_content when download fails."""
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

    def test_save_file_content_exception(self):
        """Test _save_file_content when save fails."""
        # Test with invalid path that can't be created
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            mock_mkdir.side_effect = Exception("Permission denied")

            result = _save_file_content(b"test content", Path("/invalid/path/file.txt"))
            assert result is False

    def test_save_file_content_write_exception(self):
        """Test _save_file_content when file write fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a directory with the same name as the file to cause write error
            file_path = Path(temp_dir) / "test.txt"
            file_path.mkdir()  # Make it a directory instead of a file

            result = _save_file_content(b"test content", file_path)
            assert result is False

    def test_fetch_repository_files_invalid_repo(self):
        """Test fetch_repository_files with invalid repository."""
        with pytest.raises(Exception):
            fetch_repository_files("https://github.com/nonexistent/repo")

    def test_fetch_repository_files_custom_patterns(self):
        """Test fetch_repository_files with custom file patterns."""
        with patch("app.github.loader.Github") as mock_github_class:
            mock_github = Mock()
            mock_github_class.return_value = mock_github

            mock_repo = Mock()
            mock_github.get_repo.return_value = mock_repo

            # Mock repository structure
            mock_readme = Mock()
            mock_readme.name = "README.md"
            mock_readme.path = "README.md"
            mock_readme.type = "file"
            mock_readme.size = 1000

            mock_docs_file = Mock()
            mock_docs_file.name = "guide.md"
            mock_docs_file.path = "docs/guide.md"
            mock_docs_file.type = "file"
            mock_docs_file.size = 500

            mock_repo.get_contents.side_effect = [
                [mock_readme],  # Root contents
                [mock_docs_file],  # Docs contents
            ]

            # Mock file download
            with patch("app.github.loader._download_file_content") as mock_download:
                mock_download.return_value = b"# Test content"

                result = fetch_repository_files(
                    "https://github.com/test/repo", file_glob=("*.md",)
                )

                assert len(result) > 0
                assert any("README.md" in f for f in result)
                assert any("guide.md" in f for f in result)

    def test_fetch_repository_files_large_readme_skip(self):
        """Test fetch_repository_files skips large README files."""
        with patch("app.github.loader.Github") as mock_github_class:
            mock_github = Mock()
            mock_github_class.return_value = mock_github

            mock_repo = Mock()
            mock_github.get_repo.return_value = mock_repo

            # Mock large README file
            mock_readme = Mock()
            mock_readme.name = "README.md"
            mock_readme.path = "README.md"
            mock_readme.type = "file"
            mock_readme.size = 2 * 1024 * 1024  # 2MB, exceeds limit

            mock_repo.get_contents.return_value = [mock_readme]

            result = fetch_repository_files("https://github.com/test/repo")
            assert result == []

    def test_fetch_repository_files_docs_processing_exception(self):
        """Test fetch_repository_files when docs processing fails."""
        with patch("app.github.loader.Github") as mock_github_class:
            mock_github = Mock()
            mock_github_class.return_value = mock_github

            mock_repo = Mock()
            mock_github.get_repo.return_value = mock_repo

            # Mock README file
            mock_readme = Mock()
            mock_readme.name = "README.md"
            mock_readme.path = "README.md"
            mock_readme.type = "file"
            mock_readme.size = 1000

            # Mock docs directory that raises exception
            mock_docs = Mock()
            mock_docs.name = "docs"
            mock_docs.path = "docs"
            mock_docs.type = "dir"

            mock_repo.get_contents.side_effect = [
                [mock_readme, mock_docs],  # Root contents
                Exception("Docs processing failed"),  # Docs contents
            ]

            # Mock file download
            with patch("app.github.loader._download_file_content") as mock_download:
                mock_download.return_value = b"# Test content"

                result = fetch_repository_files("https://github.com/test/repo")

                # Should still process README even if docs fails
                assert len(result) > 0
                assert any("README.md" in f for f in result)

    def test_fetch_repository_files_readme_fallback(self):
        """Test fetch_repository_files fallback to README when docs fail."""
        with patch("app.github.loader.Github") as mock_github_class:
            mock_github = Mock()
            mock_github_class.return_value = mock_github

            mock_repo = Mock()
            mock_github.get_repo.return_value = mock_repo

            # Mock README file
            mock_readme = Mock()
            mock_readme.name = "README.md"
            mock_readme.path = "README.md"
            mock_readme.type = "file"
            mock_readme.size = 1000

            # Mock docs directory that raises exception
            mock_docs = Mock()
            mock_docs.name = "docs"
            mock_docs.path = "docs"
            mock_docs.type = "dir"

            mock_repo.get_contents.side_effect = [
                [mock_readme, mock_docs],  # Root contents
                Exception("Docs processing failed"),  # Docs contents
            ]

            # Mock file download
            with patch("app.github.loader._download_file_content") as mock_download:
                mock_download.return_value = b"# Test content"

                result = fetch_repository_files("https://github.com/test/repo")

                # Should still process README even if docs fails
                assert len(result) > 0
                assert any("README.md" in f for f in result)

    def test_fetch_repository_files_docs_recursive_processing(self):
        """Test fetch_repository_files recursive docs processing."""
        with patch("app.github.loader.Github") as mock_github_class:
            mock_github = Mock()
            mock_github_class.return_value = mock_github

            mock_repo = Mock()
            mock_github.get_repo.return_value = mock_repo

            # Mock README file
            mock_readme = Mock()
            mock_readme.name = "README.md"
            mock_readme.path = "README.md"
            mock_readme.type = "file"
            mock_readme.size = 1000

            # Mock docs directory with subdirectories
            mock_docs = Mock()
            mock_docs.name = "docs"
            mock_docs.path = "docs"
            mock_docs.type = "dir"

            mock_subdir = Mock()
            mock_subdir.name = "api"
            mock_subdir.path = "docs/api"
            mock_subdir.type = "dir"

            mock_docs_file = Mock()
            mock_docs_file.name = "guide.md"
            mock_docs_file.path = "docs/guide.md"
            mock_docs_file.type = "file"
            mock_docs_file.size = 500

            mock_api_file = Mock()
            mock_api_file.name = "reference.md"
            mock_api_file.path = "docs/api/reference.md"
            mock_api_file.type = "file"
            mock_api_file.size = 300

            def mock_get_contents(path, ref=None):
                if path == "":
                    return [mock_readme, mock_docs]
                elif path == "docs":
                    return [mock_docs_file, mock_subdir]
                elif path == "docs/api":
                    return [mock_api_file]
                else:
                    return []

            mock_repo.get_contents.side_effect = mock_get_contents

            # Mock file download
            with patch("app.github.loader._download_file_content") as mock_download:
                mock_download.return_value = b"# Test content"

                result = fetch_repository_files("https://github.com/test/repo")

                assert len(result) >= 3
                assert any("README.md" in f for f in result)
                assert any("guide.md" in f for f in result)
                assert any("reference.md" in f for f in result)

    def test_fetch_repository_files_docs_large_file_skip(self):
        """Test fetch_repository_files skips large files in docs."""
        with patch("app.github.loader.Github") as mock_github_class:
            mock_github = Mock()
            mock_github_class.return_value = mock_github

            mock_repo = Mock()
            mock_github.get_repo.return_value = mock_repo

            # Mock large docs file
            mock_large_file = Mock()
            mock_large_file.name = "large.md"
            mock_large_file.path = "docs/large.md"
            mock_large_file.type = "file"
            mock_large_file.size = 2 * 1024 * 1024  # 2MB, exceeds limit

            mock_repo.get_contents.return_value = [mock_large_file]

            result = fetch_repository_files("https://github.com/test/repo")
            assert result == []

    def test_fetch_repository_files_readme_too_large(self):
        """Test fetch_repository_files when README is too large."""
        with patch("app.github.loader.Github") as mock_github_class:
            mock_github = Mock()
            mock_github_class.return_value = mock_github

            mock_repo = Mock()
            mock_github.get_repo.return_value = mock_repo

            # Mock large README file
            mock_readme = Mock()
            mock_readme.name = "README.md"
            mock_readme.path = "README.md"
            mock_readme.type = "file"
            mock_readme.size = 2 * 1024 * 1024  # 2MB, exceeds limit

            # Mock docs directory
            mock_docs = Mock()
            mock_docs.name = "docs"
            mock_docs.path = "docs"
            mock_docs.type = "dir"

            mock_docs_file = Mock()
            mock_docs_file.name = "guide.md"
            mock_docs_file.path = "docs/guide.md"
            mock_docs_file.type = "file"
            mock_docs_file.size = 500

            mock_repo.get_contents.side_effect = [
                [mock_readme, mock_docs],  # Root contents
                [mock_docs_file],  # Docs contents
            ]

            # Mock file download
            with patch("app.github.loader._download_file_content") as mock_download:
                mock_download.return_value = b"# Test content"

                result = fetch_repository_files("https://github.com/test/repo")

                # Should skip README but process docs
                assert len(result) > 0
                assert not any("README.md" in f for f in result)
                assert any("guide.md" in f for f in result)

    def test_fetch_repository_files_readme_fallback_exception(self):
        """Test fetch_repository_files when README fallback also fails."""
        with patch("app.github.loader.Github") as mock_github_class:
            mock_github = Mock()
            mock_github_class.return_value = mock_github

            mock_repo = Mock()
            mock_github.get_repo.return_value = mock_repo

            # Mock README file that will fail download
            mock_readme = Mock()
            mock_readme.name = "README.md"
            mock_readme.path = "README.md"
            mock_readme.type = "file"
            mock_readme.size = 1000

            # Mock docs directory that raises exception
            mock_docs = Mock()
            mock_docs.name = "docs"
            mock_docs.path = "docs"
            mock_docs.type = "dir"

            mock_repo.get_contents.side_effect = [
                [mock_readme, mock_docs],  # Root contents
                Exception("Docs processing failed"),  # Docs contents
            ]

            # Mock file download to fail
            with patch("app.github.loader._download_file_content") as mock_download:
                mock_download.return_value = None  # Download fails

                result = fetch_repository_files("https://github.com/test/repo")

                # Should return empty list when both README and docs fail
                assert result == []
