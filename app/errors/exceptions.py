"""Custom exceptions for the readme-mentor error system.

This module defines custom exception classes that provide structured error information
and integrate with the user-facing error handling system.
"""

from datetime import datetime
from typing import Any, List, Optional

from ..models import (
    ErrorCategory,
    ErrorCode,
    ErrorContext,
    ErrorSeverity,
    ErrorSuggestion,
    UserFacingError,
)


class ReadmeMentorError(Exception):
    """Base exception class for readme-mentor application errors.

    This exception provides structured error information and integrates with
    the user-facing error handling system.
    """

    def __init__(
        self,
        error_code: ErrorCode,
        title: str,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        technical_details: Optional[str] = None,
        suggestions: Optional[List[ErrorSuggestion]] = None,
        context: Optional[ErrorContext] = None,
        retry_after: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize the exception with structured error information.

        Args:
            error_code: Standardized error code
            title: Human-readable error title
            message: Clear, non-technical error message
            category: Error category
            severity: Error severity level
            technical_details: Technical details for debugging
            suggestions: List of actionable suggestions
            context: Error context information
            retry_after: Seconds to wait before retry
            **kwargs: Additional exception arguments
        """
        super().__init__(message, **kwargs)

        self.error_code = error_code
        self.title = title
        self.message = message
        self.category = category
        self.severity = severity
        self.technical_details = technical_details
        self.suggestions = suggestions or []
        self.retry_after = retry_after

        # Create context if not provided
        if context is None:
            self.context = ErrorContext(
                operation="unknown",
                component="unknown",
                timestamp=datetime.now().isoformat(),
            )
        else:
            self.context = context

    def to_user_facing_error(self) -> UserFacingError:
        """Convert exception to UserFacingError model."""
        return UserFacingError(
            error_code=self.error_code,
            category=self.category,
            severity=self.severity,
            title=self.title,
            message=self.message,
            technical_details=self.technical_details,
            suggestions=self.suggestions,
            context=self.context,
            retry_after=self.retry_after,
        )

    def __str__(self) -> str:
        """Return formatted error message."""
        return f"{self.title}: {self.message}"


class ConfigurationError(ReadmeMentorError):
    """Exception for configuration-related errors."""

    def __init__(
        self,
        error_code: ErrorCode,
        title: str,
        message: str,
        setting_name: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize configuration error.

        Args:
            error_code: Standardized error code
            title: Human-readable error title
            message: Clear, non-technical error message
            setting_name: Name of the problematic setting
            **kwargs: Additional exception arguments
        """
        super().__init__(
            error_code=error_code,
            title=title,
            message=message,
            category=ErrorCategory.CONFIGURATION,
            **kwargs,
        )
        self.setting_name = setting_name


class NetworkError(ReadmeMentorError):
    """Exception for network-related errors."""

    def __init__(
        self,
        error_code: ErrorCode,
        title: str,
        message: str,
        url: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs: Any,
    ):
        """Initialize network error.

        Args:
            error_code: Standardized error code
            title: Human-readable error title
            message: Clear, non-technical error message
            url: URL that caused the error
            status_code: HTTP status code if applicable
            **kwargs: Additional exception arguments
        """
        super().__init__(
            error_code=error_code,
            title=title,
            message=message,
            category=ErrorCategory.NETWORK,
            **kwargs,
        )
        self.url = url
        self.status_code = status_code


class PermissionError(ReadmeMentorError):
    """Exception for permission-related errors."""

    def __init__(
        self,
        error_code: ErrorCode,
        title: str,
        message: str,
        resource: Optional[str] = None,
        required_permissions: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """Initialize permission error.

        Args:
            error_code: Standardized error code
            title: Human-readable error title
            message: Clear, non-technical error message
            resource: Resource that requires permissions
            required_permissions: List of required permissions
            **kwargs: Additional exception arguments
        """
        super().__init__(
            error_code=error_code,
            title=title,
            message=message,
            category=ErrorCategory.PERMISSION,
            **kwargs,
        )
        self.resource = resource
        self.required_permissions = required_permissions or []


class ValidationError(ReadmeMentorError):
    """Exception for validation-related errors."""

    def __init__(
        self,
        error_code: ErrorCode,
        title: str,
        message: str,
        field_name: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        expected_format: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize validation error.

        Args:
            error_code: Standardized error code
            title: Human-readable error title
            message: Clear, non-technical error message
            field_name: Name of the field that failed validation
            invalid_value: Value that failed validation
            expected_format: Expected format or constraints
            **kwargs: Additional exception arguments
        """
        super().__init__(
            error_code=error_code,
            title=title,
            message=message,
            category=ErrorCategory.VALIDATION,
            **kwargs,
        )
        self.field_name = field_name
        self.invalid_value = invalid_value
        self.expected_format = expected_format


class ResourceError(ReadmeMentorError):
    """Exception for resource-related errors."""

    def __init__(
        self,
        error_code: ErrorCode,
        title: str,
        message: str,
        resource_path: Optional[str] = None,
        resource_type: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize resource error.

        Args:
            error_code: Standardized error code
            title: Human-readable error title
            message: Clear, non-technical error message
            resource_path: Path to the missing resource
            resource_type: Type of resource (file, directory, etc.)
            **kwargs: Additional exception arguments
        """
        super().__init__(
            error_code=error_code,
            title=title,
            message=message,
            category=ErrorCategory.RESOURCE,
            **kwargs,
        )
        self.resource_path = resource_path
        self.resource_type = resource_type


class SystemError(ReadmeMentorError):
    """Exception for system-related errors."""

    def __init__(
        self,
        error_code: ErrorCode,
        title: str,
        message: str,
        system_component: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize system error.

        Args:
            error_code: Standardized error code
            title: Human-readable error title
            message: Clear, non-technical error message
            system_component: System component that failed
            **kwargs: Additional exception arguments
        """
        super().__init__(
            error_code=error_code,
            title=title,
            message=message,
            category=ErrorCategory.SYSTEM,
            **kwargs,
        )
        self.system_component = system_component
