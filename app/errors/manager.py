"""Error manager for the user-facing error system.

This module provides a centralized error management system that handles error
tracking, context management, and error reporting.
"""

import logging
import os
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from ..models import (
    ErrorCategory,
    ErrorCode,
    ErrorContext,
    ErrorReport,
    ErrorSeverity,
    UserFacingError,
)
from .handlers import ErrorHandlerRegistry


class ErrorContext:
    """Context manager for error tracking."""

    def __init__(self, operation: str, component: str, **kwargs: Any):
        """Initialize error context.

        Args:
            operation: Current operation being performed
            component: Component where the operation is happening
            **kwargs: Additional context information
        """
        self.operation = operation
        self.component = component
        self.timestamp = datetime.now().isoformat()
        self.user_input = kwargs.get("user_input")
        self.environment = kwargs.get("environment")
        self.metadata = kwargs.get("metadata", {})

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "operation": self.operation,
            "component": self.component,
            "timestamp": self.timestamp,
            "user_input": self.user_input,
            "environment": self.environment,
            "metadata": self.metadata,
        }


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
            raise user_error

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
