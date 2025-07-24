"""User-facing error system for readme-mentor.

This module provides a comprehensive error handling system that categorizes errors,
generates actionable suggestions, and presents clear, non-technical messages to users.
"""

from ..models import (
    DeveloperErrorCategory,
    DeveloperErrorCode,
    DeveloperErrorSeverity,
)
from .exceptions import (
    CodeExecutionError,
    ConfigurationError,
    DependencyError,
    # Developer Error System Exceptions
    DeveloperError,
    IntegrationError,
    MemoryManagementError,
    NetworkError,
    PerformanceError,
    PermissionError,
    ReadmeMentorError,
    ResourceError,
    SystemError,
    ValidationError,
)
from .handlers import (
    CodeExecutionErrorHandler,
    ConfigurationErrorHandler,
    DependencyErrorHandler,
    # Developer Error System Handlers
    DeveloperErrorHandler,
    ErrorHandlerRegistry,
    IntegrationErrorHandler,
    MemoryManagementErrorHandler,
    NetworkErrorHandler,
    PerformanceErrorHandler,
    PermissionErrorHandler,
    ResourceErrorHandler,
    SystemErrorHandler,
    ValidationErrorHandler,
)
from .manager import (
    # Developer Error System Manager
    DeveloperErrorManager,
    ErrorContext,
    ErrorManager,
    create_developer_error,
    create_user_facing_error,
    get_developer_error_manager,
    get_error_manager,
    handle_developer_exception,
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
    # Developer Error System
    "DeveloperError",
    "CodeExecutionError",
    "MemoryManagementError",
    "PerformanceError",
    "IntegrationError",
    "DependencyError",
    "DeveloperErrorHandler",
    "CodeExecutionErrorHandler",
    "MemoryManagementErrorHandler",
    "PerformanceErrorHandler",
    "IntegrationErrorHandler",
    "DependencyErrorHandler",
    "DeveloperErrorManager",
    "get_developer_error_manager",
    "handle_developer_exception",
    "create_developer_error",
]
