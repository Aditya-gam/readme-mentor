"""GitHub repository content loader for readme-mentor."""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
from github import Github
from github.Repository import Repository

from ..config import settings

# Configure logging
logger = logging.getLogger(__name__)

# File size limit in bytes (1MB)
MAX_FILE_SIZE = 1024 * 1024

# Supported markdown extensions
MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdx"}


def _get_default_branch(repo: Repository) -> str:
    """
    Get the default branch of a repository.

    Args:
        repo: GitHub repository object

    Returns:
        Default branch name (e.g., 'main', 'master', etc.)

    Raises:
        Exception: If unable to determine default branch
    """
    try:
        return repo.default_branch
    except Exception as e:
        logger.error(f"Failed to get default branch: {e}")
        raise


def _extract_repo_info(repo_url: str) -> Tuple[str, str]:
    """
    Extract owner and repository name from GitHub URL.

    Args:
        repo_url: GitHub repository URL

    Returns:
        Tuple of (owner, repo_name)

    Raises:
        ValueError: If URL format is invalid
    """
    # Handle different GitHub URL formats
    patterns = [
        r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?/?$",
        r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?/.*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, repo_url)
        if match:
            owner, repo_name = match.groups()
            return owner, repo_name.rstrip("/")

    raise ValueError(f"Invalid GitHub repository URL: {repo_url}")


def _create_repo_slug(owner: str, repo_name: str) -> str:
    """
    Create a repository slug for local storage.

    Args:
        owner: Repository owner
        repo_name: Repository name

    Returns:
        Repository slug (e.g., 'octocat_Hello-World')
    """
    return f"{owner}_{repo_name}"


def _is_markdown_file(filename: str) -> bool:
    """
    Check if a file is a markdown file based on extension.

    Args:
        filename: Name of the file

    Returns:
        True if file has markdown extension
    """
    return Path(filename).suffix.lower() in MARKDOWN_EXTENSIONS


def _matches_readme_pattern(filename: str) -> bool:
    """
    Check if a file matches README pattern.

    Args:
        filename: Name of the file

    Returns:
        True if file starts with 'README' and has markdown extension
    """
    name = Path(filename).stem.lower()
    return name.startswith("readme") and _is_markdown_file(filename)


def _get_files_recursively(
    repo: Repository,
    path: str,
    default_branch: str
) -> List[Tuple[str, int]]:
    """
    Recursively get all files in a directory path.

    Args:
        repo: GitHub repository object
        path: Directory path to search
        default_branch: Default branch name

    Returns:
        List of tuples (file_path, file_size)
    """
    files = []

    try:
        contents = repo.get_contents(path, ref=default_branch)

        for content in contents:
            if content.type == "dir":
                # Recursively search subdirectories
                sub_files = _get_files_recursively(
                    repo, content.path, default_branch
                )
                files.extend(sub_files)
            elif content.type == "file":
                files.append((content.path, content.size))

    except Exception as e:
        logger.warning(f"Failed to get contents for path '{path}': {e}")

    return files


def _download_file_content(
    repo_url: str,
    file_path: str,
    default_branch: str
) -> Optional[bytes]:
    """
    Download file content from GitHub raw URL.

    Args:
        repo_url: Repository URL
        file_path: Path to the file in the repository
        default_branch: Default branch name

    Returns:
        File content as bytes, or None if download fails
    """
    try:
        # Construct raw GitHub URL
        owner, repo_name = _extract_repo_info(repo_url)
        raw_url = (
            f"https://raw.githubusercontent.com/{owner}/{repo_name}"
            f"/{default_branch}/{file_path}"
        )

        # Download file content
        with httpx.Client(timeout=30.0) as client:
            response = client.get(raw_url)
            response.raise_for_status()
            return response.content

    except Exception as e:
        logger.error(f"Failed to download file '{file_path}': {e}")
        return None


def _save_file_content(
    content: bytes,
    local_path: Path
) -> bool:
    """
    Save file content to local path.

    Args:
        content: File content as bytes
        local_path: Local path to save the file

    Returns:
        True if save was successful
    """
    try:
        # Create parent directories if they don't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file content
        with open(local_path, "wb") as f:
            f.write(content)

        logger.info(f"Saved file: {local_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save file '{local_path}': {e}")
        return False


def fetch_repository_files(
    repo_url: str,
    file_glob: Tuple[str, ...] = ("README*", "docs/**/*.md")
) -> List[str]:
    """
    Fetch repository files matching specified glob patterns.

    Args:
        repo_url: GitHub repository URL
        file_glob: Tuple of glob patterns to match files

    Returns:
        List of successfully saved file paths

    Raises:
        ValueError: If repository URL is invalid
        Exception: If GitHub API operations fail
    """
    # Extract repository information
    owner, repo_name = _extract_repo_info(repo_url)
    repo_slug = _create_repo_slug(owner, repo_name)

    logger.info(f"Fetching files from repository: {owner}/{repo_name}")

    # Initialize GitHub client
    github_token = settings.GITHUB_TOKEN
    if github_token:
        github_client = Github(github_token)
        logger.info("Using GitHub token for API access")
    else:
        github_client = Github()
        logger.warning(
            "No GitHub token provided. Using anonymous access with rate limits."
        )

    try:
        # Get repository object
        repo = github_client.get_repo(f"{owner}/{repo_name}")

        # Get default branch
        default_branch = _get_default_branch(repo)
        logger.info(f"Using default branch: {default_branch}")

        # Create local data directory
        data_dir = Path("data") / repo_slug / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []

        # Process each glob pattern
        for pattern in file_glob:
            logger.info(f"Processing pattern: {pattern}")

            if pattern == "README*":
                # Handle README files in repository root
                try:
                    # Try to get README using GitHub API
                    readme = repo.get_readme()
                    if readme and readme.size <= MAX_FILE_SIZE:
                        content = _download_file_content(
                            repo_url, readme.path, default_branch
                        )
                        if content:
                            local_path = data_dir / readme.path
                            if _save_file_content(content, local_path):
                                saved_files.append(str(local_path))
                    else:
                        logger.warning(
                            f"README file too large ({readme.size} bytes) or not found"
                        )
                except Exception as e:
                    logger.warning(f"Failed to get README: {e}")

                    # Fallback: search for README files in root
                    try:
                        root_contents = repo.get_contents(
                            "", ref=default_branch)
                        for content in root_contents:
                            if (content.type == "file" and
                                _matches_readme_pattern(content.name) and
                                    content.size <= MAX_FILE_SIZE):

                                file_content = _download_file_content(
                                    repo_url, content.path, default_branch
                                )
                                if file_content:
                                    local_path = data_dir / content.path
                                    if _save_file_content(file_content, local_path):
                                        saved_files.append(str(local_path))
                    except Exception as e2:
                        logger.warning(
                            f"Failed to search for README files: {e2}")

            elif pattern == "docs/**/*.md":
                # Handle documentation files in docs directory
                try:
                    # Get all files in docs directory recursively
                    docs_files = _get_files_recursively(
                        repo, "docs", default_branch)

                    for file_path, file_size in docs_files:
                        if (_is_markdown_file(file_path) and
                                file_size <= MAX_FILE_SIZE):

                            content = _download_file_content(
                                repo_url, file_path, default_branch
                            )
                            if content:
                                local_path = data_dir / file_path
                                if _save_file_content(content, local_path):
                                    saved_files.append(str(local_path))
                        elif file_size > MAX_FILE_SIZE:
                            logger.info(
                                f"Skipping large file: {file_path} ({file_size} bytes)"
                            )

                except Exception as e:
                    logger.warning(f"Failed to process docs directory: {e}")

        logger.info(f"Successfully saved {len(saved_files)} files")
        return saved_files

    except Exception as e:
        logger.error(f"Failed to fetch repository files: {e}")
        raise
    finally:
        # Close GitHub client
        github_client.close()
