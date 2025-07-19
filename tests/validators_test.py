"""Unit tests for URL validation utilities."""

import pytest

from app.utils.validators import InvalidRepoURLError, validate_repo_url


class TestValidateRepoURL:
    """Test cases for the validate_repo_url function."""

    def test_valid_standard_url(self) -> None:
        """Test that a standard valid GitHub URL is accepted."""
        url = "https://github.com/octocat/Hello-World"
        result = validate_repo_url(url)
        assert result == url

    def test_valid_url_with_trailing_slash(self) -> None:
        """Test that a URL with trailing slash is normalized and accepted."""
        url = "https://github.com/octocat/Hello-World/"
        result = validate_repo_url(url)
        assert result == "https://github.com/octocat/Hello-World"

    def test_valid_url_with_uppercase_domain(self) -> None:
        """Test that a URL with uppercase domain is normalized and accepted."""
        url = "https://GITHUB.COM/octocat/Hello-World"
        result = validate_repo_url(url)
        assert result == "https://github.com/octocat/Hello-World"

    def test_valid_url_with_http_protocol(self) -> None:
        """Test that HTTP URLs are converted to HTTPS and accepted."""
        url = "http://github.com/octocat/Hello-World"
        result = validate_repo_url(url)
        assert result == "https://github.com/octocat/Hello-World"

    def test_valid_url_with_whitespace(self) -> None:
        """Test that URLs with leading/trailing whitespace are trimmed."""
        url = "  https://github.com/octocat/Hello-World  "
        result = validate_repo_url(url)
        assert result == "https://github.com/octocat/Hello-World"

    def test_valid_url_with_hyphens(self) -> None:
        """Test that URLs with hyphens in owner/repo names are accepted."""
        url = "https://github.com/my-org/my-repo"
        result = validate_repo_url(url)
        assert result == url

    def test_valid_url_with_underscores(self) -> None:
        """Test that URLs with underscores in owner/repo names are accepted."""
        url = "https://github.com/my_org/my_repo"
        result = validate_repo_url(url)
        assert result == url

    def test_valid_url_with_dots(self) -> None:
        """Test that URLs with dots in owner/repo names are accepted."""
        url = "https://github.com/my.org/my.repo"
        result = validate_repo_url(url)
        assert result == url

    def test_valid_url_with_numbers(self) -> None:
        """Test that URLs with numbers in owner/repo names are accepted."""
        url = "https://github.com/user123/repo456"
        result = validate_repo_url(url)
        assert result == url

    def test_empty_url_raises_value_error(self) -> None:
        """Test that empty URLs raise ValueError."""
        with pytest.raises(ValueError, match="URL cannot be empty or None"):
            validate_repo_url("")

    def test_none_url_raises_value_error(self) -> None:
        """Test that None URLs raise ValueError."""
        with pytest.raises(ValueError, match="URL cannot be empty or None"):
            validate_repo_url(None)  # type: ignore

    def test_whitespace_only_url_raises_value_error(self) -> None:
        """Test that URLs with only whitespace raise ValueError."""
        with pytest.raises(ValueError, match="URL cannot be empty after trimming whitespace"):
            validate_repo_url("   ")

    def test_non_github_domain_raises_invalid_repo_url_error(self) -> None:
        """Test that non-GitHub domains raise InvalidRepoURLError."""
        url = "https://gitlab.com/user/repo"
        with pytest.raises(InvalidRepoURLError, match="URL must be a GitHub repository URL"):
            validate_repo_url(url)

    def test_url_with_branch_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with branch paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/tree/main"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_blob_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with blob paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/blob/main/README.md"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_commit_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with commit paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/commit/abc123"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_pull_request_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with pull request paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/pull/123"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_issues_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with issues paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/issues/456"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_wiki_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with wiki paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/wiki"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_settings_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with settings paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/settings"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_actions_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with actions paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/actions"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_projects_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with projects paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/projects"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_security_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with security paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/security"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_network_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with network paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/network"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_pulse_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with pulse paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/pulse"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_graphs_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with graphs paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/graphs"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_releases_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with releases paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/releases"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_tags_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with tags paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/tags"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_branches_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with branches paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/branches"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_compare_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with compare paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/compare"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_search_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with search paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/search"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_archive_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with archive paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/archive"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_downloads_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with downloads paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/downloads"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_forks_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with forks paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/forks"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_stargazers_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with stargazers paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/stargazers"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_watchers_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with watchers paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/watchers"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_contributors_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with contributors paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/contributors"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_commits_path_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with commits paths raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/commits"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_query_parameters_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with query parameters raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World?ref=main"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_fragment_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with fragments raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World#readme"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_extra_path_segments_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with extra path segments raise InvalidRepoURLError."""
        url = "https://github.com/octocat/Hello-World/extra/segment"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_owner_starting_with_hyphen_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with owner starting with hyphen raise InvalidRepoURLError."""
        url = "https://github.com/-invalid/repo"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_owner_ending_with_hyphen_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with owner ending with hyphen raise InvalidRepoURLError."""
        url = "https://github.com/invalid-/repo"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_repo_starting_with_hyphen_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with repo starting with hyphen raise InvalidRepoURLError."""
        url = "https://github.com/owner/-invalid"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_repo_ending_with_hyphen_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with repo ending with hyphen raise InvalidRepoURLError."""
        url = "https://github.com/owner/invalid-"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_owner_starting_with_dot_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with owner starting with dot raise InvalidRepoURLError."""
        url = "https://github.com/.invalid/repo"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_owner_ending_with_dot_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with owner ending with dot raise InvalidRepoURLError."""
        url = "https://github.com/invalid./repo"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_repo_starting_with_dot_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with repo starting with dot raise InvalidRepoURLError."""
        url = "https://github.com/owner/.invalid"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_repo_ending_with_dot_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with repo ending with dot raise InvalidRepoURLError."""
        url = "https://github.com/owner/invalid."
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_owner_starting_with_underscore_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with owner starting with underscore raise InvalidRepoURLError."""
        url = "https://github.com/_invalid/repo"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_owner_ending_with_underscore_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with owner ending with underscore raise InvalidRepoURLError."""
        url = "https://github.com/invalid_/repo"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_repo_starting_with_underscore_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with repo starting with underscore raise InvalidRepoURLError."""
        url = "https://github.com/owner/_invalid"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)

    def test_url_with_repo_ending_with_underscore_raises_invalid_repo_url_error(self) -> None:
        """Test that URLs with repo ending with underscore raise InvalidRepoURLError."""
        url = "https://github.com/owner/invalid_"
        with pytest.raises(InvalidRepoURLError, match="URL must match pattern"):
            validate_repo_url(url)


class TestInvalidRepoURLError:
    """Test cases for the InvalidRepoURLError exception."""

    def test_exception_initialization(self) -> None:
        """Test that the exception is properly initialized."""
        message = "Invalid URL format"
        url = "https://invalid-url.com"
        error = InvalidRepoURLError(message, url)

        assert str(error) == message
        assert error.message == message
        assert error.url == url

    def test_exception_inheritance(self) -> None:
        """Test that the exception inherits from ValueError."""
        error = InvalidRepoURLError("test", "test-url")
        assert isinstance(error, ValueError)
