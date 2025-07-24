"""Suggestion generator for user-facing error system.

This module provides intelligent suggestion generation for different error types,
including command examples and helpful resources.
"""

import platform
from typing import Any, Dict, List, Optional

from ..models import ErrorCategory, ErrorCode, ErrorSuggestion


class SuggestionGenerator:
    """Generates actionable suggestions for different error types."""

    def __init__(self):
        """Initialize the suggestion generator."""
        self._suggestion_templates = self._load_suggestion_templates()
        self._command_templates = self._load_command_templates()
        self._help_urls = self._load_help_urls()

    def generate_suggestions(
        self, error_code: ErrorCode, context: Optional[Dict[str, Any]] = None
    ) -> List[ErrorSuggestion]:
        """Generate suggestions for a specific error code.

        Args:
            error_code: The error code to generate suggestions for
            context: Additional context information

        Returns:
            List of actionable suggestions
        """
        suggestions = []

        # Get base suggestions for the error code
        if error_code in self._suggestion_templates:
            for template in self._suggestion_templates[error_code]:
                suggestion = self._create_suggestion_from_template(template, context)
                if suggestion:
                    suggestions.append(suggestion)

        # Add category-specific suggestions
        category = self._get_category_from_error_code(error_code)
        if category in self._suggestion_templates:
            for template in self._suggestion_templates[category]:
                suggestion = self._create_suggestion_from_template(template, context)
                if suggestion and suggestion not in suggestions:
                    suggestions.append(suggestion)

        return suggestions

    def _create_suggestion_from_template(
        self, template: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Optional[ErrorSuggestion]:
        """Create a suggestion from a template with context substitution.

        Args:
            template: Template dictionary
            context: Context for variable substitution

        Returns:
            ErrorSuggestion instance or None if template is invalid
        """
        try:
            title = self._substitute_variables(template.get("title", ""), context)
            description = self._substitute_variables(
                template.get("description", ""), context
            )
            command = self._substitute_variables(template.get("command", ""), context)
            url = template.get("url", "")

            return ErrorSuggestion(
                title=title,
                description=description,
                command=command if command else None,
                url=url if url else None,
            )
        except Exception:
            return None

    def _substitute_variables(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Substitute variables in text with context values.

        Args:
            text: Text with variable placeholders
            context: Context dictionary

        Returns:
            Text with substituted variables
        """
        if not context:
            return text

        for key, value in context.items():
            placeholder = f"{{{key}}}"
            if placeholder in text:
                text = text.replace(placeholder, str(value))

        return text

    def _get_category_from_error_code(self, error_code: ErrorCode) -> ErrorCategory:
        """Get the category from an error code.

        Args:
            error_code: The error code

        Returns:
            Error category
        """
        code_prefix = error_code.value.split("_")[0]
        category_map = {
            "CONFIG": ErrorCategory.CONFIGURATION,
            "NETWORK": ErrorCategory.NETWORK,
            "PERM": ErrorCategory.PERMISSION,
            "VALID": ErrorCategory.VALIDATION,
            "RESOURCE": ErrorCategory.RESOURCE,
            "SYSTEM": ErrorCategory.SYSTEM,
        }
        return category_map.get(code_prefix, ErrorCategory.UNKNOWN)

    def _load_suggestion_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load suggestion templates for different error codes and categories.

        Returns:
            Dictionary mapping error codes/categories to suggestion templates
        """
        return {
            # Configuration errors
            ErrorCode.MISSING_API_KEY: [
                {
                    "title": "Set API Key",
                    "description": "Set the required API key as an environment variable",
                    "command": "export OPENAI_API_KEY='your-api-key-here'",
                    "url": "https://platform.openai.com/api-keys",
                },
                {
                    "title": "Use .env File",
                    "description": "Create a .env file in your project root with your API key",
                    "command": "echo 'OPENAI_API_KEY=your-api-key-here' > .env",
                    "url": "",
                },
            ],
            ErrorCode.INVALID_CONFIG_FILE: [
                {
                    "title": "Check Config File",
                    "description": "Verify your configuration file format and syntax",
                    "command": "cat {config_path}",
                    "url": "",
                },
                {
                    "title": "Recreate Config",
                    "description": "Create a new configuration file with default settings",
                    "command": "readme-mentor init --config",
                    "url": "",
                },
            ],
            ErrorCode.MISSING_REQUIRED_ENV: [
                {
                    "title": "Set Environment Variables",
                    "description": "Set the missing environment variable: {env_var}",
                    "command": "export {env_var}='your-value-here'",
                    "url": "",
                },
            ],
            # Network errors
            ErrorCode.CONNECTION_TIMEOUT: [
                {
                    "title": "Check Internet Connection",
                    "description": "Verify your internet connection is working",
                    "command": "ping {url}",
                    "url": "",
                },
                {
                    "title": "Try Again Later",
                    "description": "The service might be temporarily unavailable. Wait a few minutes and try again",
                    "command": "",
                    "url": "",
                },
                {
                    "title": "Check Firewall",
                    "description": "Ensure your firewall allows connections to {url}",
                    "command": "",
                    "url": "",
                },
            ],
            ErrorCode.RATE_LIMIT_EXCEEDED: [
                {
                    "title": "Wait and Retry",
                    "description": "Rate limit exceeded. Wait {retry_after} seconds before trying again",
                    "command": "sleep {retry_after}",
                    "url": "",
                },
                {
                    "title": "Use Authentication",
                    "description": "Authenticate to get higher rate limits",
                    "command": "export GITHUB_TOKEN='your-token-here'",
                    "url": "https://github.com/settings/tokens",
                },
            ],
            ErrorCode.DNS_RESOLUTION_FAILED: [
                {
                    "title": "Check URL",
                    "description": "Verify the URL is correct and accessible",
                    "command": "nslookup {url}",
                    "url": "",
                },
                {
                    "title": "Try Different DNS",
                    "description": "Use a different DNS server to resolve the domain",
                    "command": "nslookup {url} 8.8.8.8",
                    "url": "",
                },
            ],
            # Permission errors
            ErrorCode.ACCESS_DENIED: [
                {
                    "title": "Check Permissions",
                    "description": "Verify you have the required permissions for {resource}",
                    "command": "ls -la {resource}",
                    "url": "",
                },
                {
                    "title": "Use Elevated Privileges",
                    "description": "Try running the command with elevated privileges",
                    "command": "sudo readme-mentor {command}",
                    "url": "",
                },
            ],
            ErrorCode.TOKEN_EXPIRED: [
                {
                    "title": "Refresh Token",
                    "description": "Your authentication token has expired. Generate a new one",
                    "command": "export GITHUB_TOKEN='your-new-token-here'",
                    "url": "https://github.com/settings/tokens",
                },
            ],
            ErrorCode.REPOSITORY_PRIVATE: [
                {
                    "title": "Authenticate",
                    "description": "This is a private repository. Provide authentication credentials",
                    "command": "export GITHUB_TOKEN='your-token-here'",
                    "url": "https://github.com/settings/tokens",
                },
                {
                    "title": "Request Access",
                    "description": "Contact the repository owner to request access",
                    "command": "",
                    "url": "",
                },
            ],
            # Validation errors
            ErrorCode.INVALID_REPO_URL: [
                {
                    "title": "Check URL Format",
                    "description": "Ensure the repository URL follows the correct format: https://github.com/owner/repo",
                    "command": "",
                    "url": "",
                },
                {
                    "title": "Verify Repository Exists",
                    "description": "Check if the repository exists and is accessible",
                    "command": "curl -I {url}",
                    "url": "",
                },
            ],
            ErrorCode.INVALID_FILE_PATTERN: [
                {
                    "title": "Check Pattern Syntax",
                    "description": "Verify the file pattern syntax is correct",
                    "command": "find . -name '{pattern}'",
                    "url": "",
                },
                {
                    "title": "Use Simple Pattern",
                    "description": "Try using a simpler file pattern",
                    "command": "readme-mentor ingest {url} --files '*.md'",
                    "url": "",
                },
            ],
            ErrorCode.MISSING_REQUIRED_FIELD: [
                {
                    "title": "Provide Required Field",
                    "description": "The field '{field_name}' is required. Please provide a value",
                    "command": "",
                    "url": "",
                },
            ],
            # Resource errors
            ErrorCode.REPOSITORY_NOT_FOUND: [
                {
                    "title": "Check Repository URL",
                    "description": "Verify the repository URL is correct and the repository exists",
                    "command": "curl -I {url}",
                    "url": "",
                },
                {
                    "title": "Check Repository Visibility",
                    "description": "Ensure the repository is public or you have access to it",
                    "command": "",
                    "url": "",
                },
            ],
            ErrorCode.FILE_NOT_FOUND: [
                {
                    "title": "Check File Path",
                    "description": "Verify the file path is correct: {resource_path}",
                    "command": "ls -la {resource_path}",
                    "url": "",
                },
                {
                    "title": "Create Directory",
                    "description": "Create the required directory structure",
                    "command": "mkdir -p {resource_path}",
                    "url": "",
                },
            ],
            ErrorCode.VECTOR_STORE_NOT_FOUND: [
                {
                    "title": "Ingest Repository First",
                    "description": "The repository needs to be ingested before querying",
                    "command": "readme-mentor ingest {repo_url}",
                    "url": "",
                },
            ],
            # System errors
            ErrorCode.MEMORY_ERROR: [
                {
                    "title": "Free Memory",
                    "description": "Close other applications to free up memory",
                    "command": "free -h",
                    "url": "",
                },
                {
                    "title": "Use Smaller Chunks",
                    "description": "Try ingesting with smaller chunk sizes",
                    "command": "readme-mentor ingest {url} --fast",
                    "url": "",
                },
            ],
            ErrorCode.DISK_SPACE_ERROR: [
                {
                    "title": "Check Disk Space",
                    "description": "Verify you have sufficient disk space",
                    "command": "df -h",
                    "url": "",
                },
                {
                    "title": "Clean Up Space",
                    "description": "Remove unnecessary files to free up disk space",
                    "command": "rm -rf cache/",
                    "url": "",
                },
            ],
            # Category-level suggestions
            ErrorCategory.CONFIGURATION: [
                {
                    "title": "Check Documentation",
                    "description": "Review the configuration documentation for setup instructions",
                    "command": "",
                    "url": "https://github.com/Aditya-gam/readme-mentor#configuration",
                },
            ],
            ErrorCategory.NETWORK: [
                {
                    "title": "Check Network Status",
                    "description": "Verify your network connection and try again",
                    "command": "ping 8.8.8.8",
                    "url": "",
                },
            ],
            ErrorCategory.PERMISSION: [
                {
                    "title": "Check File Permissions",
                    "description": "Verify file and directory permissions",
                    "command": "ls -la",
                    "url": "",
                },
            ],
            ErrorCategory.VALIDATION: [
                {
                    "title": "Review Input Format",
                    "description": "Check the input format requirements in the documentation",
                    "command": "",
                    "url": "https://github.com/Aditya-gam/readme-mentor#usage",
                },
            ],
            ErrorCategory.RESOURCE: [
                {
                    "title": "Verify Resource Exists",
                    "description": "Check if the required resource exists and is accessible",
                    "command": "",
                    "url": "",
                },
            ],
            ErrorCategory.SYSTEM: [
                {
                    "title": "Check System Resources",
                    "description": "Verify system resources (memory, disk space, etc.)",
                    "command": "top && df -h",
                    "url": "",
                },
            ],
        }

    def _load_command_templates(self) -> Dict[str, str]:
        """Load command templates for different platforms.

        Returns:
            Dictionary of command templates
        """
        system = platform.system().lower()

        templates = {
            "check_permissions": {
                "linux": "ls -la {path}",
                "darwin": "ls -la {path}",
                "windows": "dir {path}",
            },
            "create_directory": {
                "linux": "mkdir -p {path}",
                "darwin": "mkdir -p {path}",
                "windows": "mkdir {path}",
            },
            "check_disk_space": {
                "linux": "df -h",
                "darwin": "df -h",
                "windows": "wmic logicaldisk get size,freespace,caption",
            },
            "check_memory": {
                "linux": "free -h",
                "darwin": "vm_stat",
                "windows": "wmic computersystem get TotalPhysicalMemory",
            },
        }

        return {
            key: value.get(system, value.get("linux", ""))
            for key, value in templates.items()
        }

    def _load_help_urls(self) -> Dict[str, str]:
        """Load help URLs for different topics.

        Returns:
            Dictionary of help URLs
        """
        return {
            "configuration": "https://github.com/Aditya-gam/readme-mentor#configuration",
            "authentication": "https://github.com/Aditya-gam/readme-mentor#authentication",
            "usage": "https://github.com/Aditya-gam/readme-mentor#usage",
            "troubleshooting": "https://github.com/Aditya-gam/readme-mentor#troubleshooting",
            "api_keys": "https://platform.openai.com/api-keys",
            "github_tokens": "https://github.com/settings/tokens",
        }
