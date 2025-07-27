"""Error manager for the user-facing error system.

This module provides a centralized error management system that handles error
tracking, context management, and error reporting.
"""

import logging
import os
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ..models import (
    DeveloperError,
    DeveloperErrorCategory,
    DeveloperErrorCode,
    DeveloperErrorContext,
    DeveloperErrorReport,
    DeveloperErrorSeverity,
    ErrorCategory,
    ErrorCode,
    ErrorContext,
    ErrorReport,
    ErrorSeverity,
    StackFrame,
    UserFacingError,
)
from .handlers import ErrorHandlerRegistry


class ErrorManager:
    """Centralized error management system."""

    def __init__(self):
        """Initialize the error manager."""
        self.handler_registry = ErrorHandlerRegistry()
        self.error_reports: List[ErrorReport] = []
        self.current_session_id: Optional[str] = None
        self.logger = logging.getLogger(__name__)

        # Configure logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup error logging."""
        # Create error log directory if it doesn't exist
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Configure file handler for errors
        error_log_file = os.path.join(log_dir, "errors.log")
        file_handler = logging.FileHandler(error_log_file)
        file_handler.setLevel(logging.ERROR)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(file_handler)

    def start_session(self, session_id: Optional[str] = None) -> None:
        """Start a new error tracking session.

        Args:
            session_id: Optional session identifier
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_session_id = session_id
        self.logger.info(f"Started error tracking session: {session_id}")

    def end_session(self) -> ErrorReport:
        """End the current session and return error report.

        Returns:
            ErrorReport for the session
        """
        if not self.current_session_id:
            return ErrorReport(errors=[], summary={})

        # Create summary statistics
        summary = self._create_session_summary()

        # Create error report
        report = ErrorReport(
            errors=[],  # Will be populated by individual error tracking
            summary=summary,
            session_id=self.current_session_id,
        )

        self.error_reports.append(report)
        self.logger.info(f"Ended error tracking session: {self.current_session_id}")

        return report

    def _create_session_summary(self) -> Dict[str, Any]:
        """Create summary statistics for the current session.

        Returns:
            Dictionary with summary statistics
        """
        # This would be populated with actual error statistics
        # For now, return basic structure
        return {
            "total_errors": 0,
            "error_categories": {},
            "severity_distribution": {},
            "session_duration": 0,
        }

    def handle_exception(
        self,
        exception: Exception,
        context: Optional[Union[ErrorContext, Dict[str, Any]]] = None,
        operation: Optional[str] = None,
        component: Optional[str] = None,
    ) -> UserFacingError:
        """Handle an exception and convert it to a user-facing error.

        Args:
            exception: The exception to handle
            context: Error context information
            operation: Operation that failed (if not in context)
            component: Component where error occurred (if not in context)

        Returns:
            UserFacingError instance
        """
        # Convert context to proper format
        if isinstance(context, ErrorContext):
            context_dict = context.to_dict()
        elif isinstance(context, dict):
            context_dict = context
        else:
            context_dict = {}

        # Add operation and component if provided
        if operation:
            context_dict["operation"] = operation
        if component:
            context_dict["component"] = component

        # Handle the exception
        user_error = self.handler_registry.handle_exception(exception, context_dict)

        # Log the error
        self._log_error(user_error, exception)

        return user_error

    def _log_error(
        self, user_error: UserFacingError, original_exception: Exception
    ) -> None:
        """Log error information.

        Args:
            user_error: User-facing error
            original_exception: Original exception
        """
        # Log to file
        self.logger.error(
            f"Error {user_error.error_code}: {user_error.title} - {user_error.message}",
            extra={
                "error_code": user_error.error_code,
                "category": user_error.category,
                "severity": user_error.severity,
                "technical_details": user_error.technical_details,
                "context": user_error.context.dict() if user_error.context else None,
            },
        )

        # Log full traceback for debugging
        if user_error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.error(f"Full traceback: {traceback.format_exc()}")

    @contextmanager
    def error_context(self, operation: str, component: str, **kwargs: Any):
        """Context manager for error tracking.

        Args:
            operation: Operation being performed
            component: Component where operation is happening
            **kwargs: Additional context information

        Yields:
            ErrorContext instance
        """
        context = ErrorContext(operation, component, **kwargs)
        try:
            yield context
        except Exception as e:
            # Handle the exception and re-raise as user-facing error
            user_error = self.handle_exception(e, context)
            raise user_error from e

    def create_user_facing_error(
        self,
        error_code: ErrorCode,
        title: str,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        technical_details: Optional[str] = None,
        context: Optional[Union[ErrorContext, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> UserFacingError:
        """Create a user-facing error manually.

        Args:
            error_code: Standardized error code
            title: Human-readable error title
            message: Clear, non-technical error message
            category: Error category
            severity: Error severity level
            technical_details: Technical details for debugging
            context: Error context information
            **kwargs: Additional error information

        Returns:
            UserFacingError instance
        """
        # Convert context to proper format
        if isinstance(context, ErrorContext):
            context_dict = context.to_dict()
        elif isinstance(context, dict):
            context_dict = context
        else:
            context_dict = {}

        # Create error context if not provided
        if not context_dict:
            context_dict = {
                "operation": kwargs.get("operation", "manual"),
                "component": kwargs.get("component", "unknown"),
                "timestamp": datetime.now().isoformat(),
            }

        # Create user-facing error
        user_error = UserFacingError(
            error_code=error_code,
            category=category,
            severity=severity,
            title=title,
            message=message,
            technical_details=technical_details,
            context=ErrorContext(**context_dict),
            **kwargs,
        )

        # Log the error
        self._log_error(user_error, Exception(message))

        return user_error

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics across all sessions.

        Returns:
            Dictionary with error statistics
        """
        total_errors = 0
        category_counts = {}
        severity_counts = {}
        error_code_counts = {}

        for report in self.error_reports:
            for error in report.errors:
                total_errors += 1

                # Count by category
                category = error.category
                category_counts[category] = category_counts.get(category, 0) + 1

                # Count by severity
                severity = error.severity
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

                # Count by error code
                error_code = error.error_code
                error_code_counts[error_code] = error_code_counts.get(error_code, 0) + 1

        return {
            "total_errors": total_errors,
            "category_distribution": category_counts,
            "severity_distribution": severity_counts,
            "error_code_distribution": error_code_counts,
            "sessions_analyzed": len(self.error_reports),
        }

    def clear_error_history(self) -> None:
        """Clear all error history."""
        self.error_reports.clear()
        self.logger.info("Cleared error history")

    def export_error_report(self, session_id: Optional[str] = None) -> ErrorReport:
        """Export error report for a specific session or all sessions.

        Args:
            session_id: Session ID to export, or None for all sessions

        Returns:
            ErrorReport instance
        """
        if session_id:
            # Export specific session
            for report in self.error_reports:
                if report.session_id == session_id:
                    return report
            return ErrorReport(errors=[], summary={})
        else:
            # Export all sessions combined
            all_errors = []
            for report in self.error_reports:
                all_errors.extend(report.errors)

            return ErrorReport(
                errors=all_errors,
                summary=self.get_error_statistics(),
            )

    def validate_configuration(self) -> List[UserFacingError]:
        """Validate application configuration and return any errors.

        Returns:
            List of configuration errors found
        """
        errors = []

        # Check for required environment variables
        required_env_vars = ["OPENAI_API_KEY"]
        for env_var in required_env_vars:
            if not os.getenv(env_var):
                error = self.create_user_facing_error(
                    error_code=ErrorCode.MISSING_API_KEY,
                    title="Missing API Key",
                    message=f"Required environment variable {env_var} is not set",
                    category=ErrorCategory.CONFIGURATION,
                    severity=ErrorSeverity.HIGH,
                    operation="configuration_validation",
                    component="config",
                )
                errors.append(error)

        # Check for required directories
        required_dirs = ["data", "cache"]
        for dir_name in required_dirs:
            dir_path = os.path.join(os.getcwd(), dir_name)
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                except PermissionError:
                    error = self.create_user_facing_error(
                        error_code=ErrorCode.ACCESS_DENIED,
                        title="Permission Denied",
                        message=f"Cannot create required directory: {dir_path}",
                        category=ErrorCategory.PERMISSION,
                        severity=ErrorSeverity.HIGH,
                        operation="configuration_validation",
                        component="filesystem",
                    )
                    errors.append(error)

        return errors


# Global error manager instance
_error_manager: Optional[ErrorManager] = None


def get_error_manager() -> ErrorManager:
    """Get the global error manager instance.

    Returns:
        ErrorManager instance
    """
    global _error_manager
    if _error_manager is None:
        _error_manager = ErrorManager()
    return _error_manager


def handle_exception(
    exception: Exception,
    context: Optional[Union[ErrorContext, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> UserFacingError:
    """Convenience function to handle exceptions using the global error manager.

    Args:
        exception: The exception to handle
        context: Error context information
        **kwargs: Additional arguments

    Returns:
        UserFacingError instance
    """
    return get_error_manager().handle_exception(exception, context, **kwargs)


def create_user_facing_error(
    error_code: ErrorCode,
    title: str,
    message: str,
    category: ErrorCategory,
    **kwargs: Any,
) -> UserFacingError:
    """Convenience function to create user-facing errors.

    Args:
        error_code: Standardized error code
        title: Human-readable error title
        message: Clear, non-technical error message
        category: Error category
        **kwargs: Additional arguments

    Returns:
        UserFacingError instance
    """
    return get_error_manager().create_user_facing_error(
        error_code, title, message, category, **kwargs
    )


# Developer Error System Manager


class DeveloperErrorManager:
    """Centralized developer error management system.

    This manager handles developer errors with comprehensive technical details,
    structured logging, and debugging information for developers.
    """

    def __init__(self):
        """Initialize the developer error manager."""
        self.error_reports: List[DeveloperErrorReport] = []
        self.current_session_id: Optional[str] = None
        self.developer_id: Optional[str] = None
        self.environment: str = "development"
        self.logger = logging.getLogger(f"{__name__}.developer")

        # Configure developer error logging
        self._setup_developer_logging()

    def _setup_developer_logging(self) -> None:
        """Setup developer error logging with structured output."""
        # Create developer log directory if it doesn't exist
        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Configure file handler for developer errors
        dev_error_log_file = os.path.join(log_dir, "developer_errors.log")
        file_handler = logging.FileHandler(dev_error_log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create structured formatter for developer errors
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add handler to logger
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)

    def start_session(
        self,
        session_id: Optional[str] = None,
        developer_id: Optional[str] = None,
        environment: Optional[str] = None,
    ) -> None:
        """Start a new developer error tracking session.

        Args:
            session_id: Optional session identifier
            developer_id: Optional developer identifier
            environment: Optional environment (dev, staging, prod)
        """
        if session_id is None:
            session_id = f"dev_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        self.current_session_id = session_id
        self.developer_id = developer_id
        if environment:
            self.environment = environment

        self.logger.info(
            f"Started developer error tracking session: {session_id} "
            f"(Developer: {developer_id}, Environment: {self.environment})"
        )

    def end_session(self) -> DeveloperErrorReport:
        """End the current developer error tracking session.

        Returns:
            DeveloperErrorReport with session summary
        """
        if not self.current_session_id:
            raise ValueError("No active session to end")

        # Create session summary
        summary = self._create_developer_session_summary()

        report = DeveloperErrorReport(
            errors=[],  # Will be populated from error tracking
            summary=summary,
            session_id=self.current_session_id,
            developer_id=self.developer_id,
            environment=self.environment,
            created_at=datetime.now().isoformat(),
        )

        self.logger.info(
            f"Ended developer error tracking session: {self.current_session_id} "
            f"(Total errors: {summary.get('total_errors', 0)})"
        )

        # Reset session
        self.current_session_id = None
        self.developer_id = None

        return report

    def _create_developer_session_summary(self) -> Dict[str, Any]:
        """Create summary statistics for developer error session."""
        return {
            "session_id": self.current_session_id,
            "developer_id": self.developer_id,
            "environment": self.environment,
            "start_time": datetime.now().isoformat(),
            "total_errors": 0,  # Will be updated when errors are tracked
            "error_categories": {},
            "error_severities": {},
            "most_common_errors": [],
        }

    def handle_developer_exception(
        self,
        exception: Exception,
        context: Optional[Union[DeveloperErrorContext, Dict[str, Any]]] = None,
        operation: Optional[str] = None,
        component: Optional[str] = None,
        function_name: Optional[str] = None,
    ) -> DeveloperError:
        """Handle a developer exception with comprehensive technical details.

        Args:
            exception: The exception to handle
            context: Developer error context
            operation: Operation that failed
            component: Component where error occurred
            function_name: Function where error occurred

        Returns:
            DeveloperError instance with technical details
        """
        # Create developer error context if not provided
        if context is None:
            context = self._create_developer_context(
                operation, component, function_name
            )
        elif isinstance(context, dict):
            context = self._create_developer_context(
                operation, component, function_name, **context
            )

        # Determine error category and code based on exception type
        error_category, error_code = self._classify_developer_error(exception)

        # Create developer error
        developer_error = DeveloperError(
            error_code=error_code,
            category=error_category,
            severity=DeveloperErrorSeverity.ERROR,
            title=f"Developer Error: {type(exception).__name__}",
            message=str(exception),
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            stack_trace=self._capture_stack_trace(exception),
            stack_frames=self._parse_stack_trace(exception),
            context=context,
            error_id=str(uuid.uuid4()),
            created_at=datetime.now().isoformat(),
        )

        # Log the developer error
        self._log_developer_error(developer_error, exception)

        return developer_error

    def _create_developer_context(
        self,
        operation: Optional[str] = None,
        component: Optional[str] = None,
        function_name: Optional[str] = None,
        **kwargs: Any,
    ) -> DeveloperErrorContext:
        """Create developer error context with system information."""
        import platform
        import sys

        import psutil

        # Get current frame information
        current_frame = sys._getframe(2) if hasattr(sys, "_getframe") else None

        context_data = {
            "operation": operation or "unknown",
            "component": component or "unknown",
            "function_name": function_name
            or (current_frame.f_code.co_name if current_frame else "unknown"),
            "timestamp": datetime.now().isoformat(),
            "session_id": self.current_session_id,
            "python_version": sys.version,
            "platform_info": platform.platform(),
        }

        # Add system information
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

        try:
            context_data["cpu_usage"] = {
                "percent": psutil.cpu_percent(interval=0.1),
                "count": psutil.cpu_count(),
            }
        except Exception:
            pass

        # Add additional context from kwargs
        context_data.update(kwargs)

        return DeveloperErrorContext(**context_data)

    def _classify_developer_error(
        self, exception: Exception
    ) -> tuple[DeveloperErrorCategory, DeveloperErrorCode]:
        """Classify developer error based on exception type."""
        exception_type = type(exception).__name__.lower()

        # Map exception types to categories and codes
        if "memory" in exception_type or "memory" in str(exception).lower():
            return (
                DeveloperErrorCategory.MEMORY_MANAGEMENT,
                DeveloperErrorCode.MEMORY_LEAK_DETECTED,
            )
        elif "timeout" in exception_type or "timeout" in str(exception).lower():
            return (
                DeveloperErrorCategory.PERFORMANCE,
                DeveloperErrorCode.TIMEOUT_EXCEEDED,
            )
        elif "connection" in exception_type or "network" in str(exception).lower():
            return DeveloperErrorCategory.NETWORK, DeveloperErrorCode.CONNECTION_REFUSED
        elif "import" in exception_type or "module" in str(exception).lower():
            return DeveloperErrorCategory.DEPENDENCY, DeveloperErrorCode.IMPORT_ERROR
        elif "config" in exception_type or "configuration" in str(exception).lower():
            return (
                DeveloperErrorCategory.CONFIGURATION,
                DeveloperErrorCode.INVALID_CONFIG_VALUE,
            )
        elif "auth" in exception_type or "permission" in str(exception).lower():
            return (
                DeveloperErrorCategory.SECURITY,
                DeveloperErrorCode.AUTHENTICATION_FAILED,
            )
        else:
            return (
                DeveloperErrorCategory.CODE_EXECUTION,
                DeveloperErrorCode.FUNCTION_CALL_FAILED,
            )

    def _capture_stack_trace(self, exception: Exception) -> str:
        """Capture stack trace from exception."""
        return "".join(
            traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
        )

    def _parse_stack_trace(self, exception: Exception) -> List[StackFrame]:
        """Parse stack trace into structured frames."""
        frames = []

        if exception.__traceback__:
            tb = exception.__traceback__
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

    def _log_developer_error(
        self, developer_error: DeveloperError, original_exception: Exception
    ) -> None:
        """Log developer error with structured information."""
        log_data = {
            "error_id": developer_error.error_id,
            "error_code": developer_error.error_code,
            "category": developer_error.category,
            "severity": developer_error.severity,
            "title": developer_error.title,
            "message": developer_error.message,
            "exception_type": developer_error.exception_type,
            "stack_trace": developer_error.stack_trace,
            "context": developer_error.context.dict(),
            "debug_info": developer_error.debug_info,
            "session_id": self.current_session_id,
            "developer_id": self.developer_id,
            "environment": self.environment,
        }

        # Log with appropriate level based on severity
        if developer_error.severity == DeveloperErrorSeverity.CRITICAL:
            self.logger.critical(f"Developer Error: {log_data}")
        elif developer_error.severity == DeveloperErrorSeverity.ERROR:
            self.logger.error(f"Developer Error: {log_data}")
        elif developer_error.severity == DeveloperErrorSeverity.WARNING:
            self.logger.warning(f"Developer Error: {log_data}")
        elif developer_error.severity == DeveloperErrorSeverity.INFO:
            self.logger.info(f"Developer Error: {log_data}")
        else:
            self.logger.debug(f"Developer Error: {log_data}")

    @contextmanager
    def developer_error_context(
        self,
        operation: str,
        component: str,
        function_name: Optional[str] = None,
        **kwargs: Any,
    ):
        """Context manager for developer error tracking.

        Args:
            operation: Operation being performed
            component: Component where operation is happening
            function_name: Function being executed
            **kwargs: Additional context information
        """
        context = self._create_developer_context(
            operation, component, function_name, **kwargs
        )

        try:
            yield context
        except Exception as e:
            # Handle the exception with developer error system
            developer_error = self.handle_developer_exception(e, context)
            raise developer_error from e

    def create_developer_error(
        self,
        error_code: DeveloperErrorCode,
        title: str,
        message: str,
        category: DeveloperErrorCategory,
        severity: DeveloperErrorSeverity = DeveloperErrorSeverity.ERROR,
        context: Optional[Union[DeveloperErrorContext, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> DeveloperError:
        """Create a developer error with technical details.

        Args:
            error_code: Developer error code
            title: Technical error title
            message: Technical error message
            category: Error category
            severity: Error severity level
            context: Developer error context
            **kwargs: Additional error parameters

        Returns:
            DeveloperError instance
        """
        # Create context if not provided
        if context is None:
            context = self._create_developer_context()
        elif isinstance(context, dict):
            context = self._create_developer_context(**context)

        developer_error = DeveloperError(
            error_code=error_code,
            category=category,
            severity=severity,
            title=title,
            message=message,
            exception_type="DeveloperError",
            exception_message=message,
            stack_trace="",
            context=context,
            error_id=str(uuid.uuid4()),
            created_at=datetime.now().isoformat(),
            **kwargs,
        )

        # Log the developer error
        self._log_developer_error(developer_error, Exception(message))

        return developer_error

    def get_developer_error_statistics(self) -> Dict[str, Any]:
        """Get developer error statistics."""
        return {
            "total_errors": len(self.error_reports),
            "session_id": self.current_session_id,
            "developer_id": self.developer_id,
            "environment": self.environment,
            "error_categories": {},
            "error_severities": {},
        }

    def clear_developer_error_history(self) -> None:
        """Clear developer error history."""
        self.error_reports.clear()
        self.logger.info("Cleared developer error history")

    def export_developer_error_report(
        self, session_id: Optional[str] = None, format: str = "json"
    ) -> Union[DeveloperErrorReport, str]:
        """Export developer error report.

        Args:
            session_id: Session identifier (uses current if not provided)
            format: Export format (json, text)

        Returns:
            DeveloperErrorReport or formatted string
        """
        if session_id is None:
            session_id = self.current_session_id

        if not session_id:
            raise ValueError("No session ID provided and no active session")

        # Create report
        report = DeveloperErrorReport(
            errors=[],  # Will be populated from error tracking
            summary=self._create_developer_session_summary(),
            session_id=session_id,
            developer_id=self.developer_id,
            environment=self.environment,
            created_at=datetime.now().isoformat(),
        )

        if format.lower() == "json":
            return report
        else:
            return self._format_developer_error_report(report)

    def _format_developer_error_report(self, report: DeveloperErrorReport) -> str:
        """Format developer error report as text."""
        lines = [
            "Developer Error Report",
            "=====================",
            f"Session ID: {report.session_id}",
            f"Developer ID: {report.developer_id}",
            f"Environment: {report.environment}",
            f"Created: {report.created_at}",
            "",
            "Summary:",
            "--------",
        ]

        for key, value in report.summary.items():
            lines.append(f"  {key}: {value}")

        return "\n".join(lines)


# Global developer error manager instance
_developer_error_manager: Optional[DeveloperErrorManager] = None


def get_developer_error_manager() -> DeveloperErrorManager:
    """Get the global developer error manager instance."""
    global _developer_error_manager
    if _developer_error_manager is None:
        _developer_error_manager = DeveloperErrorManager()
    return _developer_error_manager


def handle_developer_exception(
    exception: Exception,
    context: Optional[Union[DeveloperErrorContext, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> DeveloperError:
    """Handle a developer exception with the global developer error manager.

    Args:
        exception: The exception to handle
        context: Developer error context
        **kwargs: Additional parameters

    Returns:
        DeveloperError instance
    """
    return get_developer_error_manager().handle_developer_exception(
        exception, context, **kwargs
    )


def create_developer_error(
    error_code: DeveloperErrorCode,
    title: str,
    message: str,
    category: DeveloperErrorCategory,
    **kwargs: Any,
) -> DeveloperError:
    """Create a developer error with the global developer error manager.

    Args:
        error_code: Developer error code
        title: Technical error title
        message: Technical error message
        category: Error category
        **kwargs: Additional error parameters

    Returns:
        DeveloperError instance
    """
    return get_developer_error_manager().create_developer_error(
        error_code=error_code,
        title=title,
        message=message,
        category=category,
        **kwargs,
    )
