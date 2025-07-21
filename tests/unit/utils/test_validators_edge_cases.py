"""Edge case tests for URL validator functionality."""

from unittest.mock import patch

import pytest

from app.utils.validators import (
    InvalidRepoURLError,
    _contains_invalid_path_segments,
    _is_valid_github_repo_url,
    _normalize_github_url,
    validate_repo_url,
)


class TestNormalizeGitHubURLEdgeCases:
    """Test edge cases for URL normalization."""

    def test_normalize_github_url_ftp_protocol(self):
        """Test _normalize_github_url with invalid protocol."""
        with pytest.raises(ValueError):
            _normalize_github_url("ftp://github.com/user/repo")

    def test_normalize_github_url_gitlab_domain(self):
        """Test _normalize_github_url with invalid domain."""
        with pytest.raises(ValueError):
            _normalize_github_url("https://gitlab.com/user/repo")

    def test_normalize_github_url_http_protocol(self):
        """Test _normalize_github_url with HTTP protocol."""
        result = _normalize_github_url("http://github.com/user/repo")
        assert result == "https://github.com/user/repo"

    def test_normalize_github_url_uppercase_domain(self):
        """Test _normalize_github_url with uppercase domain."""
        result = _normalize_github_url("https://GITHUB.COM/user/repo")
        assert result == "https://github.com/user/repo"

    def test_normalize_github_url_trailing_slash(self):
        """Test _normalize_github_url with trailing slash."""
        result = _normalize_github_url("https://github.com/user/repo/")
        assert result == "https://github.com/user/repo"

    def test_normalize_github_url_multiple_trailing_slashes(self):
        """Test _normalize_github_url with multiple trailing slashes."""
        result = _normalize_github_url("https://github.com/user/repo//")
        assert result == "https://github.com/user/repo/"


class TestIsValidGitHubRepoURLEdgeCases:
    """Test edge cases for GitHub repo URL validation."""

    def test_is_valid_github_repo_url_owner_starting_with_hyphen(self):
        """Test _is_valid_github_repo_url with owner starting with hyphen."""
        assert _is_valid_github_repo_url("https://github.com/-user/repo") is False

    def test_is_valid_github_repo_url_owner_ending_with_hyphen(self):
        """Test _is_valid_github_repo_url with owner ending with hyphen."""
        assert _is_valid_github_repo_url("https://github.com/user-/repo") is False

    def test_is_valid_github_repo_url_repo_starting_with_hyphen(self):
        """Test _is_valid_github_repo_url with repo starting with hyphen."""
        assert _is_valid_github_repo_url("https://github.com/user/-repo") is False

    def test_is_valid_github_repo_url_repo_ending_with_hyphen(self):
        """Test _is_valid_github_repo_url with repo ending with hyphen."""
        assert _is_valid_github_repo_url("https://github.com/user/repo-") is False

    def test_is_valid_github_repo_url_owner_starting_with_dot(self):
        """Test _is_valid_github_repo_url with owner starting with dot."""
        assert _is_valid_github_repo_url("https://github.com/.user/repo") is False

    def test_is_valid_github_repo_url_owner_ending_with_dot(self):
        """Test _is_valid_github_repo_url with owner ending with dot."""
        assert _is_valid_github_repo_url("https://github.com/user./repo") is False

    def test_is_valid_github_repo_url_repo_starting_with_dot(self):
        """Test _is_valid_github_repo_url with repo starting with dot."""
        assert _is_valid_github_repo_url("https://github.com/user/.repo") is False

    def test_is_valid_github_repo_url_repo_ending_with_dot(self):
        """Test _is_valid_github_repo_url with repo ending with dot."""
        assert _is_valid_github_repo_url("https://github.com/user/repo.") is False

    def test_is_valid_github_repo_url_owner_starting_with_underscore(self):
        """Test _is_valid_github_repo_url with owner starting with underscore."""
        assert _is_valid_github_repo_url("https://github.com/_user/repo") is False

    def test_is_valid_github_repo_url_owner_ending_with_underscore(self):
        """Test _is_valid_github_repo_url with owner ending with underscore."""
        assert _is_valid_github_repo_url("https://github.com/user_/repo") is False

    def test_is_valid_github_repo_url_repo_starting_with_underscore(self):
        """Test _is_valid_github_repo_url with repo starting with underscore."""
        assert _is_valid_github_repo_url("https://github.com/user/_repo") is False

    def test_is_valid_github_repo_url_repo_ending_with_underscore(self):
        """Test _is_valid_github_repo_url with repo ending with underscore."""
        assert _is_valid_github_repo_url("https://github.com/user/repo_") is False

    def test_is_valid_github_repo_url_valid_characters(self):
        """Test _is_valid_github_repo_url with valid characters."""
        # Test valid URLs
        assert _is_valid_github_repo_url("https://github.com/user/repo") is True
        assert (
            _is_valid_github_repo_url("https://github.com/user-name/repo_name") is True
        )
        assert _is_valid_github_repo_url("https://github.com/user/repo-name") is True
        assert _is_valid_github_repo_url("https://github.com/user/repo.name") is True

    def test_is_valid_github_repo_url_extra_path_segments(self):
        """Test _is_valid_github_repo_url with extra path segments."""
        assert _is_valid_github_repo_url("https://github.com/user/repo/extra") is False
        assert (
            _is_valid_github_repo_url("https://github.com/user/repo/segment1/segment2")
            is False
        )

    def test_is_valid_github_repo_url_query_parameters(self):
        """Test _is_valid_github_repo_url with query parameters."""
        assert (
            _is_valid_github_repo_url("https://github.com/user/repo?param=value")
            is False
        )
        assert _is_valid_github_repo_url("https://github.com/user/repo?") is False

    def test_is_valid_github_repo_url_fragments(self):
        """Test _is_valid_github_repo_url with fragments."""
        assert (
            _is_valid_github_repo_url("https://github.com/user/repo#fragment") is False
        )
        assert _is_valid_github_repo_url("https://github.com/user/repo#") is False


class TestContainsInvalidPathSegmentsEdgeCases:
    """Test edge cases for invalid path segment detection."""

    def test_contains_invalid_path_segments_valid_paths(self):
        """Test _contains_invalid_path_segments with valid paths."""
        assert _contains_invalid_path_segments("user/repo") is False
        assert _contains_invalid_path_segments("user-name/repo_name") is False

    def test_contains_invalid_path_segments_tree_path(self):
        """Test _contains_invalid_path_segments with tree path."""
        assert _contains_invalid_path_segments("user/repo/tree/main") is True

    def test_contains_invalid_path_segments_blob_path(self):
        """Test _contains_invalid_path_segments with blob path."""
        assert _contains_invalid_path_segments("user/repo/blob/main/file.md") is True

    def test_contains_invalid_path_segments_commit_path(self):
        """Test _contains_invalid_path_segments with commit path."""
        assert _contains_invalid_path_segments("user/repo/commit/abc123") is True

    def test_contains_invalid_path_segments_pull_path(self):
        """Test _contains_invalid_path_segments with pull path."""
        assert _contains_invalid_path_segments("user/repo/pull/123") is True

    def test_contains_invalid_path_segments_issues_path(self):
        """Test _contains_invalid_path_segments with issues path."""
        assert _contains_invalid_path_segments("user/repo/issues/456") is True

    def test_contains_invalid_path_segments_wiki_path(self):
        """Test _contains_invalid_path_segments with wiki path."""
        assert _contains_invalid_path_segments("user/repo/wiki") is False

    def test_contains_invalid_path_segments_settings_path(self):
        """Test _contains_invalid_path_segments with settings path."""
        assert _contains_invalid_path_segments("user/repo/settings") is False

    def test_contains_invalid_path_segments_actions_path(self):
        """Test _contains_invalid_path_segments with actions path."""
        assert _contains_invalid_path_segments("user/repo/actions") is False

    def test_contains_invalid_path_segments_projects_path(self):
        """Test _contains_invalid_path_segments with projects path."""
        assert _contains_invalid_path_segments("user/repo/projects") is False

    def test_contains_invalid_path_segments_security_path(self):
        """Test _contains_invalid_path_segments with security path."""
        assert _contains_invalid_path_segments("user/repo/security") is False

    def test_contains_invalid_path_segments_network_path(self):
        """Test _contains_invalid_path_segments with network path."""
        assert _contains_invalid_path_segments("user/repo/network") is False

    def test_contains_invalid_path_segments_pulse_path(self):
        """Test _contains_invalid_path_segments with pulse path."""
        assert _contains_invalid_path_segments("user/repo/pulse") is False

    def test_contains_invalid_path_segments_graphs_path(self):
        """Test _contains_invalid_path_segments with graphs path."""
        assert _contains_invalid_path_segments("user/repo/graphs") is False

    def test_contains_invalid_path_segments_releases_path(self):
        """Test _contains_invalid_path_segments with releases path."""
        assert _contains_invalid_path_segments("user/repo/releases") is False

    def test_contains_invalid_path_segments_tags_path(self):
        """Test _contains_invalid_path_segments with tags path."""
        assert _contains_invalid_path_segments("user/repo/tags") is False

    def test_contains_invalid_path_segments_branches_path(self):
        """Test _contains_invalid_path_segments with branches path."""
        assert _contains_invalid_path_segments("user/repo/branches") is False

    def test_contains_invalid_path_segments_compare_path(self):
        """Test _contains_invalid_path_segments with compare path."""
        assert _contains_invalid_path_segments("user/repo/compare") is False

    def test_contains_invalid_path_segments_search_path(self):
        """Test _contains_invalid_path_segments with search path."""
        assert _contains_invalid_path_segments("user/repo/search") is True

    def test_contains_invalid_path_segments_archive_path(self):
        """Test _contains_invalid_path_segments with archive path."""
        assert _contains_invalid_path_segments("user/repo/archive") is False

    def test_contains_invalid_path_segments_downloads_path(self):
        """Test _contains_invalid_path_segments with downloads path."""
        assert _contains_invalid_path_segments("user/repo/downloads") is False

    def test_contains_invalid_path_segments_forks_path(self):
        """Test _contains_invalid_path_segments with forks path."""
        assert _contains_invalid_path_segments("user/repo/forks") is False

    def test_contains_invalid_path_segments_stargazers_path(self):
        """Test _contains_invalid_path_segments with stargazers path."""
        assert _contains_invalid_path_segments("user/repo/stargazers") is False

    def test_contains_invalid_path_segments_watchers_path(self):
        """Test _contains_invalid_path_segments with watchers path."""
        assert _contains_invalid_path_segments("user/repo/watchers") is False

    def test_contains_invalid_path_segments_contributors_path(self):
        """Test _contains_invalid_path_segments with contributors path."""
        assert _contains_invalid_path_segments("user/repo/contributors") is False

    def test_contains_invalid_path_segments_commits_path(self):
        """Test _contains_invalid_path_segments with commits path."""
        assert _contains_invalid_path_segments("user/repo/commits") is False

    def test_contains_invalid_path_segments_trailing_slash(self):
        """Test _contains_invalid_path_segments with trailing slash."""
        assert _contains_invalid_path_segments("user/repo/") is False

    def test_contains_invalid_path_segments_extra_segments(self):
        """Test _contains_invalid_path_segments with extra segments."""
        assert _contains_invalid_path_segments("user/repo/extra/segment") is False

    def test_contains_invalid_path_segments_case_insensitive(self):
        """Test _contains_invalid_path_segments with case insensitive matching."""
        assert _contains_invalid_path_segments("user/repo/TREE/main") is True
        assert _contains_invalid_path_segments("user/repo/BLOB/main/file.md") is True
        assert _contains_invalid_path_segments("user/repo/PULL/123") is True


class TestValidateRepoURLEdgeCases:
    """Test edge cases for URL validation."""

    def test_validate_repo_url_empty_string(self):
        """Test validate_repo_url with empty string."""
        with pytest.raises(ValueError, match="URL cannot be empty or None"):
            validate_repo_url("")

    def test_validate_repo_url_none(self):
        """Test validate_repo_url with None."""
        with pytest.raises(ValueError, match="URL cannot be empty or None"):
            validate_repo_url(None)

    def test_validate_repo_url_whitespace_only(self):
        """Test validate_repo_url with whitespace only."""
        with pytest.raises(
            ValueError, match="URL cannot be empty after trimming whitespace"
        ):
            validate_repo_url("   ")

    def test_validate_repo_url_whitespace_around(self):
        """Test validate_repo_url with whitespace around URL."""
        result = validate_repo_url("  https://github.com/user/repo  ")
        assert result == "https://github.com/user/repo"

    def test_validate_repo_url_http_protocol(self):
        """Test validate_repo_url with HTTP protocol."""
        result = validate_repo_url("http://github.com/user/repo")
        assert result == "https://github.com/user/repo"

    def test_validate_repo_url_uppercase_domain(self):
        """Test validate_repo_url with uppercase domain."""
        result = validate_repo_url("https://GITHUB.COM/user/repo")
        assert result == "https://github.com/user/repo"

    def test_validate_repo_url_trailing_slash(self):
        """Test validate_repo_url with trailing slash."""
        result = validate_repo_url("https://github.com/user/repo/")
        assert result == "https://github.com/user/repo"

    def test_validate_repo_url_gitlab_domain(self):
        """Test validate_repo_url with GitLab domain."""
        with pytest.raises(
            InvalidRepoURLError, match="URL must be a GitHub repository URL"
        ):
            validate_repo_url("https://gitlab.com/user/repo")

    def test_validate_repo_url_tree_path(self):
        """Test validate_repo_url with tree path."""
        with pytest.raises(
            InvalidRepoURLError,
            match="URL must match pattern: https://github.com/<owner>/<repo>",
        ):
            validate_repo_url("https://github.com/user/repo/tree/main")

    def test_validate_repo_url_blob_path(self):
        """Test validate_repo_url with blob path."""
        with pytest.raises(
            InvalidRepoURLError,
            match="URL must match pattern: https://github.com/<owner>/<repo>",
        ):
            validate_repo_url("https://github.com/user/repo/blob/main/file.md")

    def test_validate_repo_url_query_parameters(self):
        """Test validate_repo_url with query parameters."""
        with pytest.raises(
            InvalidRepoURLError,
            match="URL must match pattern: https://github.com/<owner>/<repo>",
        ):
            validate_repo_url("https://github.com/user/repo?param=value")

    def test_validate_repo_url_fragments(self):
        """Test validate_repo_url with fragments."""
        with pytest.raises(
            InvalidRepoURLError,
            match="URL must match pattern: https://github.com/<owner>/<repo>",
        ):
            validate_repo_url("https://github.com/user/repo#fragment")

    def test_validate_repo_url_owner_starting_with_hyphen(self):
        """Test validate_repo_url with owner starting with hyphen."""
        with pytest.raises(
            InvalidRepoURLError,
            match="URL must match pattern: https://github.com/<owner>/<repo>",
        ):
            validate_repo_url("https://github.com/-user/repo")

    def test_validate_repo_url_owner_ending_with_hyphen(self):
        """Test validate_repo_url with owner ending with hyphen."""
        with pytest.raises(
            InvalidRepoURLError,
            match="URL must match pattern: https://github.com/<owner>/<repo>",
        ):
            validate_repo_url("https://github.com/user-/repo")

    def test_validate_repo_url_repo_starting_with_hyphen(self):
        """Test validate_repo_url with repo starting with hyphen."""
        with pytest.raises(
            InvalidRepoURLError,
            match="URL must match pattern: https://github.com/<owner>/<repo>",
        ):
            validate_repo_url("https://github.com/user/-repo")

    def test_validate_repo_url_repo_ending_with_hyphen(self):
        """Test validate_repo_url with repo ending with hyphen."""
        with pytest.raises(
            InvalidRepoURLError,
            match="URL must match pattern: https://github.com/<owner>/<repo>",
        ):
            validate_repo_url("https://github.com/user/repo-")

    def test_validate_repo_url_owner_starting_with_dot(self):
        """Test validate_repo_url with owner starting with dot."""
        with pytest.raises(
            InvalidRepoURLError,
            match="URL must match pattern: https://github.com/<owner>/<repo>",
        ):
            validate_repo_url("https://github.com/.user/repo")

    def test_validate_repo_url_owner_ending_with_dot(self):
        """Test validate_repo_url with owner ending with dot."""
        with pytest.raises(
            InvalidRepoURLError,
            match="URL must match pattern: https://github.com/<owner>/<repo>",
        ):
            validate_repo_url("https://github.com/user./repo")

    def test_validate_repo_url_repo_starting_with_dot(self):
        """Test validate_repo_url with repo starting with dot."""
        with pytest.raises(
            InvalidRepoURLError,
            match="URL must match pattern: https://github.com/<owner>/<repo>",
        ):
            validate_repo_url("https://github.com/user/.repo")

    def test_validate_repo_url_repo_ending_with_dot(self):
        """Test validate_repo_url with repo ending with dot."""
        with pytest.raises(
            InvalidRepoURLError,
            match="URL must match pattern: https://github.com/<owner>/<repo>",
        ):
            validate_repo_url("https://github.com/user/repo.")

    def test_validate_repo_url_owner_starting_with_underscore(self):
        """Test validate_repo_url with owner starting with underscore."""
        with pytest.raises(
            InvalidRepoURLError,
            match="URL must match pattern: https://github.com/<owner>/<repo>",
        ):
            validate_repo_url("https://github.com/_user/repo")

    def test_validate_repo_url_owner_ending_with_underscore(self):
        """Test validate_repo_url with owner ending with underscore."""
        with pytest.raises(
            InvalidRepoURLError,
            match="URL must match pattern: https://github.com/<owner>/<repo>",
        ):
            validate_repo_url("https://github.com/user_/repo")

    def test_validate_repo_url_repo_starting_with_underscore(self):
        """Test validate_repo_url with repo starting with underscore."""
        with pytest.raises(
            InvalidRepoURLError,
            match="URL must match pattern: https://github.com/<owner>/<repo>",
        ):
            validate_repo_url("https://github.com/user/_repo")

    def test_validate_repo_url_repo_ending_with_underscore(self):
        """Test validate_repo_url with repo ending with underscore."""
        with pytest.raises(
            InvalidRepoURLError,
            match="URL must match pattern: https://github.com/<owner>/<repo>",
        ):
            validate_repo_url("https://github.com/user/repo_")

    def test_validate_repo_url_valid_characters(self):
        """Test validate_repo_url with valid characters."""
        # Test valid URLs
        assert (
            validate_repo_url("https://github.com/user/repo")
            == "https://github.com/user/repo"
        )
        assert (
            validate_repo_url("https://github.com/user-name/repo_name")
            == "https://github.com/user-name/repo_name"
        )
        assert (
            validate_repo_url("https://github.com/user/repo-name")
            == "https://github.com/user/repo-name"
        )
        assert (
            validate_repo_url("https://github.com/user/repo.name")
            == "https://github.com/user/repo.name"
        )

    def test_validate_repo_url_normalization_error(self):
        """Test validate_repo_url when normalization fails."""
        with patch("app.utils.validators._normalize_github_url") as mock_normalize:
            mock_normalize.side_effect = ValueError(
                "URL must be a GitHub repository URL"
            )

            with pytest.raises(
                InvalidRepoURLError, match="URL must be a GitHub repository URL"
            ):
                validate_repo_url("invalid-url")


class TestInvalidRepoURLErrorEdgeCases:
    """Test edge cases for InvalidRepoURLError exception."""

    def test_invalid_repo_url_error_initialization(self):
        """Test InvalidRepoURLError initialization."""
        error = InvalidRepoURLError("Invalid URL", "https://example.com")
        assert str(error) == "Invalid URL"
        assert error.url == "https://example.com"
        assert error.message == "Invalid URL"

    def test_invalid_repo_url_error_inheritance(self):
        """Test InvalidRepoURLError inheritance."""
        error = InvalidRepoURLError("Invalid URL", "https://example.com")
        assert isinstance(error, ValueError)

    def test_invalid_repo_url_error_without_url(self):
        """Test InvalidRepoURLError without URL."""
        error = InvalidRepoURLError("Invalid URL", "")
        assert str(error) == "Invalid URL"
        assert error.url == ""

    def test_invalid_repo_url_error_with_none_url(self):
        """Test InvalidRepoURLError with None URL."""
        error = InvalidRepoURLError("Invalid URL", None)
        assert str(error) == "Invalid URL"
        assert error.url is None
