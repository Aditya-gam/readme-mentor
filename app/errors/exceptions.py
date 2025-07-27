"""Custom exceptions for the readme-mentor error system.

This module defines custom exception classes that provide structured error information
and integrate with the user-facing error handling system.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

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
    ErrorSuggestion,
    StackFrame,
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


# Developer Error System Exceptions


class DeveloperError(Exception):
    """Base exception class for developer errors with technical details.

    This exception provides comprehensive technical information for debugging
    and integrates with the developer error handling system.
    """

    def __init__(
        self,
        error_code: DeveloperErrorCode,
        title: str,
        message: str,
        category: DeveloperErrorCategory,
        severity: DeveloperErrorSeverity = DeveloperErrorSeverity.ERROR,
        exception_type: Optional[str] = None,
        exception_message: Optional[str] = None,
        stack_trace: Optional[str] = None,
        context: Optional[DeveloperErrorContext] = None,
        debug_info: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize the developer exception with technical error information.

        Args:
            error_code: Developer error code
            title: Technical error title
            message: Technical error message
            category: Error category
            severity: Error severity level
            exception_type: Type of the original exception
            exception_message: Original exception message
            stack_trace: Full stack trace
            context: Developer error context
            debug_info: Additional debugging information
            **kwargs: Additional exception arguments
        """
        super().__init__(message, **kwargs)

        self.error_code = error_code
        self.title = title
        self.message = message
        self.category = category
        self.severity = severity
        self.exception_type = exception_type or self.__class__.__name__
        self.exception_message = exception_message or message
        self.stack_trace = stack_trace or self._capture_stack_trace()
        self.debug_info = debug_info or {}

        # Create context if not provided
        if context is None:
            self.context = self._create_default_context()
        else:
            self.context = context

        # Parse stack trace into frames
        self.stack_frames = self._parse_stack_trace()

    def _capture_stack_trace(self) -> str:
        """Capture the current stack trace."""
        import traceback

        return "".join(traceback.format_exception(type(self), self, self.__traceback__))

    def _create_default_context(self) -> DeveloperErrorContext:
        """Create default developer error context."""
        import platform
        import sys

        import psutil

        # Get current frame information
        current_frame = sys._getframe(1) if hasattr(sys, "_getframe") else None

        context_data = {
            "operation": "unknown",
            "component": "unknown",
            "function_name": current_frame.f_code.co_name
            if current_frame
            else "unknown",
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "platform_info": platform.platform(),
        }

        # Add memory usage if available
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            context_data["memory_usage"] = {
                "rss": memory_info.rss,
                "vms": memory_info.vms,
                "percent": process.memory_percent(),
            }
        except Exception:
            pass

        # Add CPU usage if available
        try:
            context_data["cpu_usage"] = {
                "percent": psutil.cpu_percent(interval=0.1),
                "count": psutil.cpu_count(),
            }
        except Exception:
            pass

        return DeveloperErrorContext(**context_data)

    def _parse_stack_trace(self) -> List[StackFrame]:
        """Parse stack trace into structured frames."""

        frames = []

        if self.__traceback__:
            tb = self.__traceback__
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
                            # Convert to string representation to avoid circular references
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

    def to_developer_error(self) -> DeveloperError:
        """Convert exception to DeveloperError model."""
        return DeveloperError(
            error_code=self.error_code,
            category=self.category,
            severity=self.severity,
            title=self.title,
            message=self.message,
            exception_type=self.exception_type,
            exception_message=self.exception_message,
            stack_trace=self.stack_trace,
            stack_frames=self.stack_frames,
            context=self.context,
            debug_info=self.debug_info,
            created_at=datetime.now().isoformat(),
        )

    def __str__(self) -> str:
        """Return formatted error message with technical details."""
        return f"{self.title}: {self.message} (Code: {self.error_code})"


class CodeExecutionError(DeveloperError):
    """Exception for code execution errors."""

    def __init__(
        self,
        error_code: DeveloperErrorCode,
        title: str,
        message: str,
        function_name: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize code execution error.

        Args:
            error_code: Developer error code
            title: Technical error title
            message: Technical error message
            function_name: Name of the function that failed
            arguments: Function arguments that caused the error
            **kwargs: Additional exception arguments
        """
        super().__init__(
            error_code=error_code,
            title=title,
            message=message,
            category=DeveloperErrorCategory.CODE_EXECUTION,
            **kwargs,
        )
        self.function_name = function_name
        self.arguments = arguments

        # Update context with function information
        if function_name:
            self.context.function_name = function_name
        if arguments:
            self.context.function_arguments = arguments


class MemoryManagementError(DeveloperError):
    """Exception for memory management errors."""

    def __init__(
        self,
        error_code: DeveloperErrorCode,
        title: str,
        message: str,
        memory_usage: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize memory management error.

        Args:
            error_code: Developer error code
            title: Technical error title
            message: Technical error message
            memory_usage: Memory usage information
            **kwargs: Additional exception arguments
        """
        super().__init__(
            error_code=error_code,
            title=title,
            message=message,
            category=DeveloperErrorCategory.MEMORY_MANAGEMENT,
            **kwargs,
        )
        self.memory_usage = memory_usage

        # Update context with memory information
        if memory_usage:
            self.context.memory_usage = memory_usage


class PerformanceError(DeveloperError):
    """Exception for performance-related errors."""

    def __init__(
        self,
        error_code: DeveloperErrorCode,
        title: str,
        message: str,
        execution_time: Optional[float] = None,
        resource_usage: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize performance error.

        Args:
            error_code: Developer error code
            title: Technical error title
            message: Technical error message
            execution_time: Execution time in seconds
            resource_usage: Resource usage information
            **kwargs: Additional exception arguments
        """
        super().__init__(
            error_code=error_code,
            title=title,
            message=message,
            category=DeveloperErrorCategory.PERFORMANCE,
            **kwargs,
        )
        self.execution_time = execution_time
        self.resource_usage = resource_usage

        # Update debug info with performance data
        if execution_time:
            self.debug_info["execution_time"] = execution_time
        if resource_usage:
            self.debug_info["resource_usage"] = resource_usage


class IntegrationError(DeveloperError):
    """Exception for integration errors."""

    def __init__(
        self,
        error_code: DeveloperErrorCode,
        title: str,
        message: str,
        service_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        response_data: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize integration error.

        Args:
            error_code: Developer error code
            title: Technical error title
            message: Technical error message
            service_name: Name of the external service
            endpoint: API endpoint that failed
            response_data: Response data from the service
            **kwargs: Additional exception arguments
        """
        super().__init__(
            error_code=error_code,
            title=title,
            message=message,
            category=DeveloperErrorCategory.INTEGRATION,
            **kwargs,
        )
        self.service_name = service_name
        self.endpoint = endpoint
        self.response_data = response_data

        # Update debug info with integration data
        if service_name:
            self.debug_info["service_name"] = service_name
        if endpoint:
            self.debug_info["endpoint"] = endpoint
        if response_data:
            self.debug_info["response_data"] = response_data


class DependencyError(DeveloperError):
    """Exception for dependency-related errors."""

    def __init__(
        self,
        error_code: DeveloperErrorCode,
        title: str,
        message: str,
        dependency_name: Optional[str] = None,
        required_version: Optional[str] = None,
        installed_version: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize dependency error.

        Args:
            error_code: Developer error code
            title: Technical error title
            message: Technical error message
            dependency_name: Name of the dependency
            required_version: Required version
            installed_version: Installed version
            **kwargs: Additional exception arguments
        """
        super().__init__(
            error_code=error_code,
            title=title,
            message=message,
            category=DeveloperErrorCategory.DEPENDENCY,
            **kwargs,
        )
        self.dependency_name = dependency_name
        self.required_version = required_version
        self.installed_version = installed_version

        # Update debug info with dependency data
        if dependency_name:
            self.debug_info["dependency_name"] = dependency_name
        if required_version:
            self.debug_info["required_version"] = required_version
        if installed_version:
            self.debug_info["installed_version"] = installed_version
