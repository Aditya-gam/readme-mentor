"""Error handlers for the user-facing error system.

This module provides specialized error handlers for different error categories
that can analyze exceptions and convert them to user-facing errors.
"""

import re
from datetime import datetime
from typing import Any, Dict, Optional

from ..models import (
    ErrorCategory,
    ErrorCode,
    ErrorContext,
    ErrorSeverity,
    UserFacingError,
)
from .exceptions import (
    ConfigurationError,
    NetworkError,
    PermissionError,
    ResourceError,
    SystemError,
    ValidationError,
)
from .suggestions import SuggestionGenerator


class BaseErrorHandler:
    """Base class for error handlers."""

    def __init__(self):
        """Initialize the error handler."""
        self.suggestion_generator = SuggestionGenerator()

    def can_handle(self, exception: Exception) -> bool:
        """Check if this handler can handle the given exception.

        Args:
            exception: The exception to check

        Returns:
            True if the handler can handle this exception
        """
        raise NotImplementedError

    def handle(
        self, exception: Exception, context: Optional[Dict[str, Any]] = None
    ) -> UserFacingError:
        """Handle an exception and convert it to a user-facing error.

        Args:
            exception: The exception to handle
            context: Additional context information

        Returns:
            UserFacingError instance
        """
        raise NotImplementedError

    def _create_context(
        self, operation: str, component: str, **kwargs: Any
    ) -> ErrorContext:
        """Create error context with common information.

        Args:
            operation: Operation that failed
            component: Component where error occurred
            **kwargs: Additional context information

        Returns:
            ErrorContext instance
        """
        return ErrorContext(
            operation=operation,
            component=component,
            timestamp=datetime.now().isoformat(),
            user_input=kwargs.get("user_input"),
            environment=kwargs.get("environment"),
        )


class ConfigurationErrorHandler(BaseErrorHandler):
    """Handler for configuration-related errors."""

    def can_handle(self, exception: Exception) -> bool:
        """Check if this handler can handle configuration errors."""
        if isinstance(exception, ConfigurationError):
            return True

        error_message = str(exception).lower()
        config_keywords = [
            "config",
            "configuration",
            "setting",
            "environment",
            "env",
            "api_key",
            "token",
            "credential",
            "authentication",
        ]
        return any(keyword in error_message for keyword in config_keywords)

    def handle(
        self, exception: Exception, context: Optional[Dict[str, Any]] = None
    ) -> UserFacingError:
        """Handle configuration errors."""
        if isinstance(exception, ConfigurationError):
            return exception.to_user_facing_error()

        error_message = str(exception).lower()

        # Determine specific error type
        if "api_key" in error_message or "openai" in error_message:
            error_code = ErrorCode.MISSING_API_KEY
            title = "Missing API Key"
            message = (
                "OpenAI API key is required but not found. Please set your API key."
            )
        elif "github_token" in error_message or "github" in error_message:
            error_code = ErrorCode.MISSING_API_KEY
            title = "Missing GitHub Token"
            message = (
                "GitHub token is required but not found. Please set your GitHub token."
            )
        elif "config" in error_message and "file" in error_message:
            error_code = ErrorCode.INVALID_CONFIG_FILE
            title = "Invalid Configuration File"
            message = "The configuration file format is invalid or corrupted."
        elif "env" in error_message or "environment" in error_message:
            error_code = ErrorCode.MISSING_REQUIRED_ENV
            title = "Missing Environment Variable"
            message = "A required environment variable is not set."
        else:
            error_code = ErrorCode.INVALID_SETTING_VALUE
            title = "Configuration Error"
            message = "There is an issue with your configuration settings."

        # Generate suggestions
        suggestions = self.suggestion_generator.generate_suggestions(
            error_code, context
        )

        # Create error context
        error_context = self._create_context(
            operation=context.get("operation", "configuration"),
            component=context.get("component", "config"),
            user_input=context.get("user_input"),
            environment=context.get("environment"),
        )

        return UserFacingError(
            error_code=error_code,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            title=title,
            message=message,
            technical_details=str(exception),
            suggestions=suggestions,
            context=error_context,
        )


class NetworkErrorHandler(BaseErrorHandler):
    """Handler for network-related errors."""

    def can_handle(self, exception: Exception) -> bool:
        """Check if this handler can handle network errors."""
        if isinstance(exception, NetworkError):
            return True

        error_message = str(exception).lower()
        network_keywords = [
            "connection",
            "timeout",
            "network",
            "dns",
            "ssl",
            "certificate",
            "rate limit",
            "too many requests",
            "http",
            "https",
            "url",
        ]
        return any(keyword in error_message for keyword in network_keywords)

    def handle(
        self, exception: Exception, context: Optional[Dict[str, Any]] = None
    ) -> UserFacingError:
        """Handle network errors."""
        if isinstance(exception, NetworkError):
            return exception.to_user_facing_error()

        error_message = str(exception).lower()

        # Determine specific error type
        if "timeout" in error_message:
            error_code = ErrorCode.CONNECTION_TIMEOUT
            title = "Connection Timeout"
            message = "The connection to the server timed out. Please check your internet connection."
        elif "rate limit" in error_message or "too many requests" in error_message:
            error_code = ErrorCode.RATE_LIMIT_EXCEEDED
            title = "Rate Limit Exceeded"
            message = (
                "Too many requests have been made. Please wait before trying again."
            )
        elif "dns" in error_message or "name resolution" in error_message:
            error_code = ErrorCode.DNS_RESOLUTION_FAILED
            title = "DNS Resolution Failed"
            message = "Could not resolve the domain name. Please check the URL."
        elif "ssl" in error_message or "certificate" in error_message:
            error_code = ErrorCode.SSL_CERTIFICATE_ERROR
            title = "SSL Certificate Error"
            message = "There is an issue with the SSL certificate."
        else:
            error_code = ErrorCode.CONNECTION_TIMEOUT
            title = "Network Error"
            message = "A network error occurred while connecting to the server."

        # Extract URL from context or error message
        url = context.get("url") if context else None
        if not url:
            # Try to extract URL from error message
            url_match = re.search(r"https?://[^\s]+", str(exception))
            if url_match:
                url = url_match.group(0)

        # Generate suggestions with URL context
        suggestion_context = context or {}
        if url:
            suggestion_context["url"] = url
        suggestions = self.suggestion_generator.generate_suggestions(
            error_code, suggestion_context
        )

        # Create error context
        error_context = self._create_context(
            operation=context.get("operation", "network_request"),
            component=context.get("component", "network"),
            user_input=context.get("user_input"),
            environment=context.get("environment"),
        )

        return UserFacingError(
            error_code=error_code,
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            title=title,
            message=message,
            technical_details=str(exception),
            suggestions=suggestions,
            context=error_context,
        )


class PermissionErrorHandler(BaseErrorHandler):
    """Handler for permission-related errors."""

    def can_handle(self, exception: Exception) -> bool:
        """Check if this handler can handle permission errors."""
        if isinstance(exception, PermissionError):
            return True

        error_message = str(exception).lower()
        permission_keywords = [
            "permission",
            "access denied",
            "forbidden",
            "unauthorized",
            "token",
            "authentication",
            "private",
            "insufficient",
        ]
        return any(keyword in error_message for keyword in permission_keywords)

    def handle(
        self, exception: Exception, context: Optional[Dict[str, Any]] = None
    ) -> UserFacingError:
        """Handle permission errors."""
        if isinstance(exception, PermissionError):
            return exception.to_user_facing_error()

        error_message = str(exception).lower()

        # Determine specific error type
        if "private" in error_message and "repository" in error_message:
            error_code = ErrorCode.REPOSITORY_PRIVATE
            title = "Private Repository"
            message = "This repository is private and requires authentication."
        elif "token" in error_message and (
            "expired" in error_message or "invalid" in error_message
        ):
            error_code = ErrorCode.TOKEN_EXPIRED
            title = "Token Expired or Invalid"
            message = "Your authentication token has expired or is invalid."
        elif "access denied" in error_message or "forbidden" in error_message:
            error_code = ErrorCode.ACCESS_DENIED
            title = "Access Denied"
            message = "You don't have permission to access this resource."
        else:
            error_code = ErrorCode.INSUFFICIENT_PERMISSIONS
            title = "Insufficient Permissions"
            message = "You don't have the required permissions for this operation."

        # Generate suggestions
        suggestions = self.suggestion_generator.generate_suggestions(
            error_code, context
        )

        # Create error context
        error_context = self._create_context(
            operation=context.get("operation", "permission_check"),
            component=context.get("component", "auth"),
            user_input=context.get("user_input"),
            environment=context.get("environment"),
        )

        return UserFacingError(
            error_code=error_code,
            category=ErrorCategory.PERMISSION,
            severity=ErrorSeverity.HIGH,
            title=title,
            message=message,
            technical_details=str(exception),
            suggestions=suggestions,
            context=error_context,
        )


class ValidationErrorHandler(BaseErrorHandler):
    """Handler for validation-related errors."""

    def can_handle(self, exception: Exception) -> bool:
        """Check if this handler can handle validation errors."""
        if isinstance(exception, ValidationError):
            return True

        error_message = str(exception).lower()
        validation_keywords = [
            "validation",
            "invalid",
            "malformed",
            "format",
            "required",
            "missing",
            "field",
            "value",
            "pattern",
        ]
        return any(keyword in error_message for keyword in validation_keywords)

    def handle(
        self, exception: Exception, context: Optional[Dict[str, Any]] = None
    ) -> UserFacingError:
        """Handle validation errors."""
        if isinstance(exception, ValidationError):
            return exception.to_user_facing_error()

        error_message = str(exception).lower()

        # Determine specific error type
        if "url" in error_message and (
            "invalid" in error_message or "malformed" in error_message
        ):
            error_code = ErrorCode.INVALID_REPO_URL
            title = "Invalid Repository URL"
            message = "The repository URL format is invalid. Please use the format: https://github.com/owner/repo"
        elif "pattern" in error_message or "file" in error_message:
            error_code = ErrorCode.INVALID_FILE_PATTERN
            title = "Invalid File Pattern"
            message = "The file pattern format is invalid."
        elif "required" in error_message or "missing" in error_message:
            error_code = ErrorCode.MISSING_REQUIRED_FIELD
            title = "Missing Required Field"
            message = "A required field is missing from your input."
        else:
            error_code = ErrorCode.INVALID_INPUT_FORMAT
            title = "Invalid Input Format"
            message = "The input format is invalid or doesn't meet the requirements."

        # Generate suggestions
        suggestions = self.suggestion_generator.generate_suggestions(
            error_code, context
        )

        # Create error context
        error_context = self._create_context(
            operation=context.get("operation", "validation"),
            component=context.get("component", "input"),
            user_input=context.get("user_input"),
            environment=context.get("environment"),
        )

        return UserFacingError(
            error_code=error_code,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            title=title,
            message=message,
            technical_details=str(exception),
            suggestions=suggestions,
            context=error_context,
        )


class ResourceErrorHandler(BaseErrorHandler):
    """Handler for resource-related errors."""

    def can_handle(self, exception: Exception) -> bool:
        """Check if this handler can handle resource errors."""
        if isinstance(exception, ResourceError):
            return True

        error_message = str(exception).lower()
        resource_keywords = [
            "not found",
            "404",
            "file not found",
            "directory",
            "path",
            "repository",
            "vector store",
            "model",
            "resource",
        ]
        return any(keyword in error_message for keyword in resource_keywords)

    def handle(
        self, exception: Exception, context: Optional[Dict[str, Any]] = None
    ) -> UserFacingError:
        """Handle resource errors."""
        if isinstance(exception, ResourceError):
            return exception.to_user_facing_error()

        error_message = str(exception).lower()

        # Determine specific error type
        if "repository" in error_message and "not found" in error_message:
            error_code = ErrorCode.REPOSITORY_NOT_FOUND
            title = "Repository Not Found"
            message = "The specified repository could not be found."
        elif "file" in error_message and "not found" in error_message:
            error_code = ErrorCode.FILE_NOT_FOUND
            title = "File Not Found"
            message = "The specified file could not be found."
        elif "vector store" in error_message:
            error_code = ErrorCode.VECTOR_STORE_NOT_FOUND
            title = "Vector Store Not Found"
            message = "The vector store for this repository has not been created yet."
        elif "model" in error_message and "not available" in error_message:
            error_code = ErrorCode.MODEL_NOT_AVAILABLE
            title = "Model Not Available"
            message = "The required model is not available."
        else:
            error_code = ErrorCode.REPOSITORY_NOT_FOUND
            title = "Resource Not Found"
            message = "The specified resource could not be found."

        # Generate suggestions
        suggestions = self.suggestion_generator.generate_suggestions(
            error_code, context
        )

        # Create error context
        error_context = self._create_context(
            operation=context.get("operation", "resource_access"),
            component=context.get("component", "storage"),
            user_input=context.get("user_input"),
            environment=context.get("environment"),
        )

        return UserFacingError(
            error_code=error_code,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.MEDIUM,
            title=title,
            message=message,
            technical_details=str(exception),
            suggestions=suggestions,
            context=error_context,
        )


class SystemErrorHandler(BaseErrorHandler):
    """Handler for system-related errors."""

    def can_handle(self, exception: Exception) -> bool:
        """Check if this handler can handle system errors."""
        if isinstance(exception, SystemError):
            return True

        error_message = str(exception).lower()
        system_keywords = [
            "memory",
            "disk",
            "space",
            "timeout",
            "process",
            "system",
            "resource",
            "out of",
            "insufficient",
        ]
        return any(keyword in error_message for keyword in system_keywords)

    def handle(
        self, exception: Exception, context: Optional[Dict[str, Any]] = None
    ) -> UserFacingError:
        """Handle system errors."""
        if isinstance(exception, SystemError):
            return exception.to_user_facing_error()

        error_message = str(exception).lower()

        # Determine specific error type
        if "memory" in error_message and (
            "out of" in error_message or "insufficient" in error_message
        ):
            error_code = ErrorCode.MEMORY_ERROR
            title = "Insufficient Memory"
            message = "There is not enough memory available to complete the operation."
        elif "disk" in error_message and (
            "space" in error_message or "full" in error_message
        ):
            error_code = ErrorCode.DISK_SPACE_ERROR
            title = "Insufficient Disk Space"
            message = "There is not enough disk space available."
        elif "timeout" in error_message:
            error_code = ErrorCode.PROCESS_TIMEOUT
            title = "Process Timeout"
            message = "The operation timed out due to system constraints."
        else:
            error_code = ErrorCode.UNEXPECTED_ERROR
            title = "System Error"
            message = "An unexpected system error occurred."

        # Generate suggestions
        suggestions = self.suggestion_generator.generate_suggestions(
            error_code, context
        )

        # Create error context
        error_context = self._create_context(
            operation=context.get("operation", "system_operation"),
            component=context.get("component", "system"),
            user_input=context.get("user_input"),
            environment=context.get("environment"),
        )

        return UserFacingError(
            error_code=error_code,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            title=title,
            message=message,
            technical_details=str(exception),
            suggestions=suggestions,
            context=error_context,
        )


class ErrorHandlerRegistry:
    """Registry for error handlers."""

    def __init__(self):
        """Initialize the error handler registry."""
        self.handlers = [
            ConfigurationErrorHandler(),
            NetworkErrorHandler(),
            PermissionErrorHandler(),
            ValidationErrorHandler(),
            ResourceErrorHandler(),
            SystemErrorHandler(),
        ]

    def get_handler(self, exception: Exception) -> Optional[BaseErrorHandler]:
        """Get the appropriate handler for an exception.

        Args:
            exception: The exception to handle

        Returns:
            Appropriate error handler or None if no handler found
        """
        for handler in self.handlers:
            if handler.can_handle(exception):
                return handler
        return None

    def handle_exception(
        self, exception: Exception, context: Optional[Dict[str, Any]] = None
    ) -> UserFacingError:
        """Handle an exception using the appropriate handler.

        Args:
            exception: The exception to handle
            context: Additional context information

        Returns:
            UserFacingError instance
        """
        handler = self.get_handler(exception)
        if handler:
            return handler.handle(exception, context)

        # Fallback to generic error
        return UserFacingError(
            error_code=ErrorCode.UNEXPECTED_ERROR,
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            title="Unexpected Error",
            message="An unexpected error occurred. Please try again or contact support.",
            technical_details=str(exception),
            suggestions=[],
            context=self._create_fallback_context(context),
        )

    def _create_fallback_context(
        self, context: Optional[Dict[str, Any]] = None
    ) -> ErrorContext:
        """Create fallback error context."""
        return ErrorContext(
            operation=context.get("operation", "unknown") if context else "unknown",
            component=context.get("component", "unknown") if context else "unknown",
            timestamp=datetime.now().isoformat(),
            user_input=context.get("user_input") if context else None,
            environment=context.get("environment") if context else None,
        )
