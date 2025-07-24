"""User-facing error system for readme-mentor.

This module provides a comprehensive error handling system that categorizes errors,
generates actionable suggestions, and presents clear, non-technical messages to users.
"""

from .exceptions import (
    ConfigurationError,
    NetworkError,
    PermissionError,
    ReadmeMentorError,
    ResourceError,
    SystemError,
    ValidationError,
)
from .handlers import (
    ConfigurationErrorHandler,
    ErrorHandlerRegistry,
    NetworkErrorHandler,
    PermissionErrorHandler,
    ResourceErrorHandler,
    SystemErrorHandler,
    ValidationErrorHandler,
)
from .manager import (
    ErrorContext,
    ErrorManager,
    create_user_facing_error,
    get_error_manager,
    handle_exception,
)
from .suggestions import SuggestionGenerator

__all__ = [
    # Error handlers
    "ConfigurationErrorHandler",
    "NetworkErrorHandler",
    "PermissionErrorHandler",
    "ValidationErrorHandler",
    "ResourceErrorHandler",
    "SystemErrorHandler",
    "ErrorHandlerRegistry",
    # Error management
    "ErrorManager",
    "ErrorContext",
    "get_error_manager",
    "handle_exception",
    "create_user_facing_error",
    "SuggestionGenerator",
    # Custom exceptions
    "ReadmeMentorError",
    "ConfigurationError",
    "NetworkError",
    "PermissionError",
    "ValidationError",
    "ResourceError",
    "SystemError",
]
