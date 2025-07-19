"""URL validation utilities for readme-mentor application."""

import re


class InvalidRepoURLError(ValueError):
    """Custom exception for invalid repository URLs."""

    def __init__(self, message: str, url: str) -> None:
        """Initialize the exception with a message and the invalid URL.

        Args:
            message: Error description
            url: The invalid URL that caused the error
        """
        super().__init__(message)
        self.url = url
        self.message = message


def validate_repo_url(url: str) -> str:
    """Validate and normalize a GitHub repository URL.

    This function validates that the URL matches the pattern
    `https://github.com/<owner>/<repo>` and rejects URLs containing
    branch paths, subdirectories, or query parameters.

    Args:
        url: The repository URL to validate

    Returns:
        The normalized repository URL (trimmed, lowercase domain)

    Raises:
        InvalidRepoURLError: If the URL format is invalid or contains
            unsupported path segments or query parameters
        ValueError: If the URL is empty or None

    Examples:
        >>> validate_repo_url("https://github.com/octocat/Hello-World")
        'https://github.com/octocat/Hello-World'
        >>> validate_repo_url("https://GITHUB.COM/octocat/Hello-World/")
        'https://github.com/octocat/Hello-World'
    """
    if not url:
        raise ValueError("URL cannot be empty or None")

    # Trim whitespace
    url = url.strip()

    if not url:
        raise ValueError("URL cannot be empty after trimming whitespace")

    # Parse the URL to extract components
    try:
        normalized_url = _normalize_github_url(url)
    except ValueError as e:
        raise InvalidRepoURLError(str(e), url) from e

    # Validate the normalized URL format
    if not _is_valid_github_repo_url(normalized_url):
        raise InvalidRepoURLError(
            "URL must match pattern: https://github.com/<owner>/<repo>", url
        )

    return normalized_url


def _normalize_github_url(url: str) -> str:
    """Normalize a GitHub URL by converting domain to lowercase and removing trailing slash.

    Args:
        url: The URL to normalize

    Returns:
        The normalized URL

    Raises:
        ValueError: If the URL is not a valid GitHub URL
    """
    # Convert domain to lowercase
    if url.lower().startswith("https://github.com/"):
        normalized = "https://github.com/" + url[19:]
    elif url.lower().startswith("http://github.com/"):
        normalized = "https://github.com/" + url[18:]
    else:
        raise ValueError("URL must be a GitHub repository URL")

    # Remove trailing slash
    if normalized.endswith("/"):
        normalized = normalized[:-1]

    return normalized


def _is_valid_github_repo_url(url: str) -> bool:
    """Check if the URL matches the valid GitHub repository pattern.

    Args:
        url: The URL to validate

    Returns:
        True if the URL is valid, False otherwise
    """
    # Pattern: https://github.com/<owner>/<repo>
    # Owner and repo names can contain alphanumeric, hyphens, underscores, dots
    # but cannot start or end with hyphens, dots, or underscores
    pattern = r"^https://github\.com/[a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?/[a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?$"

    if not re.match(pattern, url):
        return False

    # Additional checks for invalid path segments
    if _contains_invalid_path_segments(url):
        return False

    return True


def _contains_invalid_path_segments(url: str) -> bool:
    """Check if the URL contains invalid path segments like branches, trees, etc.

    Args:
        url: The URL to check

    Returns:
        True if invalid segments are found, False otherwise
    """
    # List of invalid path segments that indicate branches, trees, etc.
    invalid_segments = [
        "/tree/",
        "/blob/",
        "/commit/",
        "/pull/",
        "/issues/",
        "/wiki/",
        "/settings/",
        "/actions/",
        "/projects/",
        "/security/",
        "/network/",
        "/pulse/",
        "/graphs/",
        "/releases/",
        "/tags/",
        "/branches/",
        "/compare/",
        "/search",
        "/archive/",
        "/downloads/",
        "/forks/",
        "/stargazers/",
        "/watchers/",
        "/contributors/",
        "/commits/",
        "/branches/",
        "/tags/",
    ]

    url_lower = url.lower()
    for segment in invalid_segments:
        if segment in url_lower:
            return True

    # Check for query parameters
    if "?" in url:
        return True

    # Check for fragments
    if "#" in url:
        return True

    # Check for more than 2 path segments (domain/owner/repo is valid)
    path_parts = url.split("/")
    if len(path_parts) > 5:  # https://github.com/owner/repo = 5 parts
        return True

    return False
