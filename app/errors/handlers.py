"""Error handlers for the readme-mentor error system.

This module provides specialized error handlers that process different types of
errors and generate appropriate user-facing error messages and suggestions.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ..models import (
    DeveloperError,
    DeveloperErrorCategory,
    DeveloperErrorCode,
    DeveloperErrorContext,
    DeveloperErrorSeverity,
    ErrorCategory,
    ErrorCode,
    ErrorContext,
    ErrorSeverity,
    StackFrame,
    UserFacingError,
)
from .exceptions import (
    CodeExecutionError,
    ConfigurationError,
    DependencyError,
    IntegrationError,
    MemoryManagementError,
    NetworkError,
    PerformanceError,
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


class ErrorHandler:
    """Base class for error handlers."""

    def __init__(self):
        """Initialize the error handler."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def can_handle(self, error: Union[Exception, UserFacingError]) -> bool:
        """Check if this handler can handle the given error.

        Args:
            error: Error to check

        Returns:
            True if this handler can handle the error
        """
        raise NotImplementedError

    def handle(self, error: Union[Exception, UserFacingError]) -> UserFacingError:
        """Handle the error and return a user-facing error.

        Args:
            error: Error to handle

        Returns:
            UserFacingError instance
        """
        raise NotImplementedError

    def _extract_error_details(self, error: Exception) -> Dict[str, Any]:
        """Extract relevant details from an exception.

        Args:
            error: Exception to extract details from

        Returns:
            Dictionary of error details
        """
        details = {
            "type": type(error).__name__,
            "message": str(error),
        }

        # Extract additional details based on error type
        if hasattr(error, "args") and error.args:
            details["args"] = error.args

        return details


class DeveloperErrorHandler:
    """Base class for developer error handlers."""

    def __init__(self):
        """Initialize the developer error handler."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def can_handle(self, error: Union[Exception, DeveloperError]) -> bool:
        """Check if this handler can handle the given developer error.

        Args:
            error: Developer error to check

        Returns:
            True if this handler can handle the error
        """
        raise NotImplementedError

    def handle(self, error: Union[Exception, DeveloperError]) -> DeveloperError:
        """Handle the developer error and return a structured developer error.

        Args:
            error: Developer error to handle

        Returns:
            DeveloperError instance
        """
        raise NotImplementedError

    def _analyze_stack_trace(self, stack_trace: str) -> Dict[str, Any]:
        """Analyze stack trace for patterns and insights.

        Args:
            stack_trace: Stack trace string

        Returns:
            Dictionary with analysis results
        """
        analysis = {
            "frame_count": 0,
            "application_frames": [],
            "library_frames": [],
            "error_location": None,
            "suspicious_patterns": [],
        }

        lines = stack_trace.split("\n")
        frame_count = 0

        for line in lines:
            if "File " in line and "line " in line:
                frame_count += 1

                # Extract file and line information
                file_match = re.search(r'File "([^"]+)"', line)
                line_match = re.search(r"line (\d+)", line)

                if file_match and line_match:
                    filename = file_match.group(1)
                    line_number = int(line_match.group(1))

                    frame_info = {
                        "filename": filename,
                        "line_number": line_number,
                        "is_application": self._is_application_file(filename),
                    }

                    if frame_info["is_application"]:
                        analysis["application_frames"].append(frame_info)
                    else:
                        analysis["library_frames"].append(frame_info)

                    # Identify error location (first application frame)
                    if not analysis["error_location"] and frame_info["is_application"]:
                        analysis["error_location"] = frame_info

        analysis["frame_count"] = frame_count

        # Detect suspicious patterns
        analysis["suspicious_patterns"] = self._detect_suspicious_patterns(stack_trace)

        return analysis

    def _is_application_file(self, filename: str) -> bool:
        """Check if a file is part of the application code.

        Args:
            filename: File path to check

        Returns:
            True if it's an application file
        """
        # Check if it's in the app directory
        if "app/" in filename or "readme-mentor" in filename:
            return True

        # Check if it's not a standard library or third-party library
        stdlib_paths = [
            "/usr/lib/python",
            "/usr/local/lib/python",
            "site-packages",
            "dist-packages",
        ]

        return not any(path in filename for path in stdlib_paths)

    def _detect_suspicious_patterns(self, stack_trace: str) -> List[str]:
        """Detect suspicious patterns in stack trace.

        Args:
            stack_trace: Stack trace string

        Returns:
            List of detected patterns
        """
        patterns = []

        # Check for common error patterns
        if "RecursionError" in stack_trace:
            patterns.append("recursion_error")
        if "MemoryError" in stack_trace:
            patterns.append("memory_error")
        if "TimeoutError" in stack_trace:
            patterns.append("timeout_error")
        if "ImportError" in stack_trace:
            patterns.append("import_error")
        if "AttributeError" in stack_trace:
            patterns.append("attribute_error")
        if "TypeError" in stack_trace:
            patterns.append("type_error")
        if "ValueError" in stack_trace:
            patterns.append("value_error")
        if "KeyError" in stack_trace:
            patterns.append("key_error")
        if "IndexError" in stack_trace:
            patterns.append("index_error")

        return patterns

    def _extract_context_variables(self, error: Exception) -> Dict[str, Any]:
        """Extract context variables from error.

        Args:
            error: Exception to extract context from

        Returns:
            Dictionary of context variables
        """
        context = {}

        # Extract attributes from error object
        for attr in dir(error):
            if not attr.startswith("_") and not callable(getattr(error, attr)):
                try:
                    value = getattr(error, attr)
                    context[attr] = str(value)[:200]  # Limit length
                except Exception:
                    context[attr] = "<unserializable>"

        return context


class CodeExecutionErrorHandler(DeveloperErrorHandler):
    """Handler for code execution errors."""

    def can_handle(self, error: Union[Exception, DeveloperError]) -> bool:
        """Check if this handler can handle code execution errors."""
        if isinstance(error, CodeExecutionError):
            return True

        error_type = type(error).__name__.lower()
        return any(
            pattern in error_type
            for pattern in [
                "typeerror",
                "valueerror",
                "attributeerror",
                "keyerror",
                "indexerror",
                "nameerror",
                "syntaxerror",
                "indentationerror",
                "zerodivisionerror",
            ]
        )

    def handle(self, error: Union[Exception, DeveloperError]) -> DeveloperError:
        """Handle code execution error."""
        if isinstance(error, CodeExecutionError):
            return error.to_developer_error()

        # Analyze the error
        stack_trace = self._get_stack_trace(error)
        analysis = self._analyze_stack_trace(stack_trace)
        context_vars = self._extract_context_variables(error)

        # Determine error code based on error type
        error_code = self._determine_error_code(error)

        # Create developer error context
        context = DeveloperErrorContext(
            operation="code_execution",
            component="application",
            function_name=self._extract_function_name(error),
            timestamp=datetime.now().isoformat(),
            stack_trace=self._parse_stack_frames(error),
            local_variables=context_vars,
            debug_info={
                "stack_analysis": analysis,
                "error_type": type(error).__name__,
                "error_args": getattr(error, "args", []),
            },
        )

        return DeveloperError(
            error_code=error_code,
            category=DeveloperErrorCategory.CODE_EXECUTION,
            severity=DeveloperErrorSeverity.ERROR,
            title=f"Code Execution Error: {type(error).__name__}",
            message=str(error),
            exception_type=type(error).__name__,
            exception_message=str(error),
            stack_trace=stack_trace,
            stack_frames=self._parse_stack_frames(error),
            context=context,
            created_at=datetime.now().isoformat(),
        )

    def _determine_error_code(self, error: Exception) -> DeveloperErrorCode:
        """Determine error code based on exception type."""
        error_type = type(error).__name__.lower()

        if "type" in error_type:
            return DeveloperErrorCode.INVALID_ARGUMENT_TYPE
        elif "value" in error_type:
            return DeveloperErrorCode.FUNCTION_CALL_FAILED
        elif "attribute" in error_type:
            return DeveloperErrorCode.FUNCTION_CALL_FAILED
        elif "key" in error_type or "index" in error_type:
            return DeveloperErrorCode.FUNCTION_CALL_FAILED
        elif "name" in error_type:
            return DeveloperErrorCode.MISSING_REQUIRED_PARAMETER
        else:
            return DeveloperErrorCode.FUNCTION_CALL_FAILED

    def _get_stack_trace(self, error: Exception) -> str:
        """Get stack trace from exception."""
        import traceback

        return "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )

    def _extract_function_name(self, error: Exception) -> str:
        """Extract function name from error."""
        if error.__traceback__:
            return error.__traceback__.tb_frame.f_code.co_name
        return "unknown"

    def _parse_stack_frames(self, error: Exception) -> List[StackFrame]:
        """Parse stack trace into structured frames."""
        frames = []

        if error.__traceback__:
            tb = error.__traceback__
            while tb:
                frame = tb.tb_frame

                # Get frame information
                filename = frame.f_code.co_filename
                line_number = tb.tb_lineno
                function_name = frame.f_code.co_name

                # Get code context
                try:
                    with open(filename, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        start_line = max(0, line_number - 3)
                        end_line = min(len(lines), line_number + 2)
                        code_context = "".join(lines[start_line:end_line])
                except Exception:
                    code_context = None

                # Get local variables (sanitized)
                local_vars = {}
                try:
                    for name, value in frame.f_locals.items():
                        if not name.startswith("_"):
                            try:
                                local_vars[name] = str(value)[:200]  # Limit length
                            except Exception:
                                local_vars[name] = "<unserializable>"
                except Exception:
                    pass

                frames.append(
                    StackFrame(
                        filename=filename,
                        line_number=line_number,
                        function_name=function_name,
                        code_context=code_context,
                        local_variables=local_vars if local_vars else None,
                    )
                )

                tb = tb.tb_next

        return frames


class MemoryManagementErrorHandler(DeveloperErrorHandler):
    """Handler for memory management errors."""

    def can_handle(self, error: Union[Exception, DeveloperError]) -> bool:
        """Check if this handler can handle memory management errors."""
        if isinstance(error, MemoryManagementError):
            return True

        error_type = type(error).__name__.lower()
        return any(
            pattern in error_type
            for pattern in ["memoryerror", "recursionerror", "overflowerror"]
        )

    def handle(self, error: Union[Exception, DeveloperError]) -> DeveloperError:
        """Handle memory management error."""
        if isinstance(error, MemoryManagementError):
            return error.to_developer_error()

        # Get memory usage information
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        memory_usage = {
            "rss": memory_info.rss,
            "vms": memory_info.vms,
            "percent": process.memory_percent(),
            "available": psutil.virtual_memory().available,
            "total": psutil.virtual_memory().total,
        }

        # Create developer error context
        context = DeveloperErrorContext(
            operation="memory_management",
            component="system",
            function_name=self._extract_function_name(error),
            timestamp=datetime.now().isoformat(),
            memory_usage=memory_usage,
            debug_info={
                "error_type": type(error).__name__,
                "memory_threshold_exceeded": memory_usage["percent"] > 80,
            },
        )

        return DeveloperError(
            error_code=DeveloperErrorCode.MEMORY_LEAK_DETECTED,
            category=DeveloperErrorCategory.MEMORY_MANAGEMENT,
            severity=DeveloperErrorSeverity.CRITICAL,
            title=f"Memory Management Error: {type(error).__name__}",
            message=str(error),
            exception_type=type(error).__name__,
            exception_message=str(error),
            stack_trace=self._get_stack_trace(error),
            stack_frames=self._parse_stack_frames(error),
            context=context,
            created_at=datetime.now().isoformat(),
        )

    def _extract_function_name(self, error: Exception) -> str:
        """Extract function name from error."""
        if error.__traceback__:
            return error.__traceback__.tb_frame.f_code.co_name
        return "unknown"

    def _get_stack_trace(self, error: Exception) -> str:
        """Get stack trace from exception."""
        import traceback

        return "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )

    def _parse_stack_frames(self, error: Exception) -> List[StackFrame]:
        """Parse stack trace into structured frames."""
        frames = []

        if error.__traceback__:
            tb = error.__traceback__
            while tb:
                frame = tb.tb_frame

                frames.append(
                    StackFrame(
                        filename=frame.f_code.co_filename,
                        line_number=tb.tb_lineno,
                        function_name=frame.f_code.co_name,
                        code_context=None,
                        local_variables=None,
                    )
                )

                tb = tb.tb_next

        return frames


class PerformanceErrorHandler(DeveloperErrorHandler):
    """Handler for performance-related errors."""

    def can_handle(self, error: Union[Exception, DeveloperError]) -> bool:
        """Check if this handler can handle performance errors."""
        if isinstance(error, PerformanceError):
            return True

        error_type = type(error).__name__.lower()
        return any(
            pattern in error_type for pattern in ["timeouterror", "resourcewarning"]
        )

    def handle(self, error: Union[Exception, DeveloperError]) -> DeveloperError:
        """Handle performance error."""
        if isinstance(error, PerformanceError):
            return error.to_developer_error()

        # Get system resource information
        import psutil

        cpu_usage = {
            "percent": psutil.cpu_percent(interval=0.1),
            "count": psutil.cpu_count(),
        }

        memory_usage = {
            "percent": psutil.virtual_memory().percent,
            "available": psutil.virtual_memory().available,
        }

        # Create developer error context
        context = DeveloperErrorContext(
            operation="performance_monitoring",
            component="system",
            function_name=self._extract_function_name(error),
            timestamp=datetime.now().isoformat(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            debug_info={
                "error_type": type(error).__name__,
                "high_cpu_usage": cpu_usage["percent"] > 80,
                "high_memory_usage": memory_usage["percent"] > 80,
            },
        )

        return DeveloperError(
            error_code=DeveloperErrorCode.TIMEOUT_EXCEEDED,
            category=DeveloperErrorCategory.PERFORMANCE,
            severity=DeveloperErrorSeverity.WARNING,
            title=f"Performance Error: {type(error).__name__}",
            message=str(error),
            exception_type=type(error).__name__,
            exception_message=str(error),
            stack_trace=self._get_stack_trace(error),
            stack_frames=self._parse_stack_frames(error),
            context=context,
            created_at=datetime.now().isoformat(),
        )

    def _extract_function_name(self, error: Exception) -> str:
        """Extract function name from error."""
        if error.__traceback__:
            return error.__traceback__.tb_frame.f_code.co_name
        return "unknown"

    def _get_stack_trace(self, error: Exception) -> str:
        """Get stack trace from exception."""
        import traceback

        return "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )

    def _parse_stack_frames(self, error: Exception) -> List[StackFrame]:
        """Parse stack trace into structured frames."""
        frames = []

        if error.__traceback__:
            tb = error.__traceback__
            while tb:
                frame = tb.tb_frame

                frames.append(
                    StackFrame(
                        filename=frame.f_code.co_filename,
                        line_number=tb.tb_lineno,
                        function_name=frame.f_code.co_name,
                        code_context=None,
                        local_variables=None,
                    )
                )

                tb = tb.tb_next

        return frames


class IntegrationErrorHandler(DeveloperErrorHandler):
    """Handler for integration errors."""

    def can_handle(self, error: Union[Exception, DeveloperError]) -> bool:
        """Check if this handler can handle integration errors."""
        if isinstance(error, IntegrationError):
            return True

        error_type = type(error).__name__.lower()
        return any(
            pattern in error_type
            for pattern in ["connectionerror", "timeouterror", "httperror", "urlerror"]
        )

    def handle(self, error: Union[Exception, DeveloperError]) -> DeveloperError:
        """Handle integration error."""
        if isinstance(error, IntegrationError):
            return error.to_developer_error()

        # Extract integration-specific information
        service_name = getattr(error, "service_name", "unknown")
        endpoint = getattr(error, "endpoint", "unknown")
        response_data = getattr(error, "response_data", {})

        # Create developer error context
        context = DeveloperErrorContext(
            operation="external_integration",
            component="network",
            function_name=self._extract_function_name(error),
            timestamp=datetime.now().isoformat(),
            debug_info={
                "error_type": type(error).__name__,
                "service_name": service_name,
                "endpoint": endpoint,
                "response_data": response_data,
            },
        )

        return DeveloperError(
            error_code=DeveloperErrorCode.API_CALL_FAILED,
            category=DeveloperErrorCategory.INTEGRATION,
            severity=DeveloperErrorSeverity.ERROR,
            title=f"Integration Error: {type(error).__name__}",
            message=str(error),
            exception_type=type(error).__name__,
            exception_message=str(error),
            stack_trace=self._get_stack_trace(error),
            stack_frames=self._parse_stack_frames(error),
            context=context,
            created_at=datetime.now().isoformat(),
        )

    def _extract_function_name(self, error: Exception) -> str:
        """Extract function name from error."""
        if error.__traceback__:
            return error.__traceback__.tb_frame.f_code.co_name
        return "unknown"

    def _get_stack_trace(self, error: Exception) -> str:
        """Get stack trace from exception."""
        import traceback

        return "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )

    def _parse_stack_frames(self, error: Exception) -> List[StackFrame]:
        """Parse stack trace into structured frames."""
        frames = []

        if error.__traceback__:
            tb = error.__traceback__
            while tb:
                frame = tb.tb_frame

                frames.append(
                    StackFrame(
                        filename=frame.f_code.co_filename,
                        line_number=tb.tb_lineno,
                        function_name=frame.f_code.co_name,
                        code_context=None,
                        local_variables=None,
                    )
                )

                tb = tb.tb_next

        return frames


class DependencyErrorHandler(DeveloperErrorHandler):
    """Handler for dependency-related errors."""

    def can_handle(self, error: Union[Exception, DeveloperError]) -> bool:
        """Check if this handler can handle dependency errors."""
        if isinstance(error, DependencyError):
            return True

        error_type = type(error).__name__.lower()
        return any(
            pattern in error_type
            for pattern in ["importerror", "modulenotfounderror", "attributeerror"]
        )

    def handle(self, error: Union[Exception, DeveloperError]) -> DeveloperError:
        """Handle dependency error."""
        if isinstance(error, DependencyError):
            return error.to_developer_error()

        # Extract dependency information
        dependency_name = getattr(error, "dependency_name", "unknown")
        required_version = getattr(error, "required_version", "unknown")
        installed_version = getattr(error, "installed_version", "unknown")

        # Create developer error context
        context = DeveloperErrorContext(
            operation="dependency_management",
            component="system",
            function_name=self._extract_function_name(error),
            timestamp=datetime.now().isoformat(),
            debug_info={
                "error_type": type(error).__name__,
                "dependency_name": dependency_name,
                "required_version": required_version,
                "installed_version": installed_version,
            },
        )

        return DeveloperError(
            error_code=DeveloperErrorCode.IMPORT_ERROR,
            category=DeveloperErrorCategory.DEPENDENCY,
            severity=DeveloperErrorSeverity.ERROR,
            title=f"Dependency Error: {type(error).__name__}",
            message=str(error),
            exception_type=type(error).__name__,
            exception_message=str(error),
            stack_trace=self._get_stack_trace(error),
            stack_frames=self._parse_stack_frames(error),
            context=context,
            created_at=datetime.now().isoformat(),
        )

    def _extract_function_name(self, error: Exception) -> str:
        """Extract function name from error."""
        if error.__traceback__:
            return error.__traceback__.tb_frame.f_code.co_name
        return "unknown"

    def _get_stack_trace(self, error: Exception) -> str:
        """Get stack trace from exception."""
        import traceback

        return "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )

    def _parse_stack_frames(self, error: Exception) -> List[StackFrame]:
        """Parse stack trace into structured frames."""
        frames = []

        if error.__traceback__:
            tb = error.__traceback__
            while tb:
                frame = tb.tb_frame

                frames.append(
                    StackFrame(
                        filename=frame.f_code.co_filename,
                        line_number=tb.tb_lineno,
                        function_name=frame.f_code.co_name,
                        code_context=None,
                        local_variables=None,
                    )
                )

                tb = tb.tb_next

        return frames
