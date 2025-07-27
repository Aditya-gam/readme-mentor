"""Custom log handlers for the logging system.

This module provides custom handlers that integrate with the logging system
to provide structured logging, performance metrics tracking, and integration
with the user output system.
"""

import logging
import sys
from typing import Any, Dict, Optional

from .config import LoggingConfig
from .formatters import LogFormatter


class CustomHandler(logging.Handler):
    """Custom handler that integrates with the logging system.

    This handler provides structured logging with performance metrics
    and integrates with the user output system for consistent formatting.
    """

    def __init__(self, config: LoggingConfig, stream=None):
        """Initialize custom handler.

        Args:
            config: Logging configuration
            stream: Output stream (defaults to sys.stderr)
        """
        super().__init__()
        self.config = config
        self.formatter = LogFormatter(config)
        self.stream = stream or sys.stderr

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record.

        Args:
            record: Log record to emit
        """
        try:
            # Format the record
            formatted = self.formatter.format_log_record(record)

            # Write to stream
            self.stream.write(formatted + "\n")
            self.stream.flush()
        except Exception:
            # Fallback to basic logging if formatting fails
            self.handleError(record)


class PerformanceMetricsHandler(logging.Handler):
    """Handler for tracking performance metrics in log records.

    This handler extracts performance metrics from log records and
    makes them available for display in user output.
    """

    def __init__(self, config: LoggingConfig):
        """Initialize performance metrics handler.

        Args:
            config: Logging configuration
        """
        super().__init__()
        self.config = config
        self.metrics: Dict[str, Any] = {}

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record and extract performance metrics.

        Args:
            record: Log record to emit
        """
        # Extract performance metrics if present
        if hasattr(record, "performance_metrics"):
            self.metrics.update(record.performance_metrics)

        # Extract individual metrics from record attributes
        for attr in [
            "wall_time",
            "token_count",
            "tool_calls",
            "memory_usage",
            "cpu_usage",
        ]:
            if hasattr(record, attr):
                self.metrics[attr] = getattr(record, attr)

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        return self.metrics.copy()

    def clear_metrics(self) -> None:
        """Clear collected performance metrics."""
        self.metrics.clear()


class StructuredLogRecord(logging.LogRecord):
    """Extended log record with support for structured data.

    This class extends the standard LogRecord to support additional
    fields for structured logging and performance metrics.
    """

    def __init__(
        self,
        name: str,
        level: int,
        pathname: str,
        lineno: int,
        msg: str,
        args: tuple,
        exc_info: Optional[Any] = None,
        func: Optional[str] = None,
        sinfo: Optional[str] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, Any]] = None,
    ):
        """Initialize structured log record.

        Args:
            name: Logger name
            level: Log level
            pathname: Path to source file
            lineno: Line number in source file
            msg: Log message
            args: Message arguments
            exc_info: Exception info
            func: Function name
            sinfo: Stack info
            extra_fields: Additional structured fields
            performance_metrics: Performance metrics
        """
        super().__init__(
            name, level, pathname, lineno, msg, args, exc_info, func, sinfo
        )
        self.extra_fields = extra_fields or {}
        self.performance_metrics = performance_metrics or {}


class StructuredLogger(logging.Logger):
    """Logger with support for structured logging and performance metrics.

    This logger extends the standard Logger to provide methods for
    structured logging and performance metrics tracking.
    """

    def __init__(self, name: str, level: int = logging.NOTSET):
        """Initialize structured logger.

        Args:
            name: Logger name
            level: Log level
        """
        super().__init__(name, level)

    def log_with_metrics(
        self,
        level: int,
        msg: str,
        performance_metrics: Optional[Dict[str, Any]] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
        *args: Any,
    ) -> None:
        """Log a message with performance metrics and extra fields.

        Args:
            level: Log level
            msg: Log message
            performance_metrics: Performance metrics to include
            extra_fields: Additional structured fields
            *args: Message arguments
            **kwargs: Additional keyword arguments
        """
        if self.isEnabledFor(level):
            # Create structured record
            record = StructuredLogRecord(
                name=self.name,
                level=level,
                pathname="",
                lineno=0,
                msg=msg,
                args=args,
                extra_fields=extra_fields,
                performance_metrics=performance_metrics,
            )

            # Set source location if possible
            import inspect

            try:
                frame = inspect.currentframe().f_back
                if frame:
                    record.pathname = frame.f_code.co_filename
                    record.lineno = frame.f_lineno
                    record.func = frame.f_code.co_name
            except Exception:
                pass

            self.handle(record)

    def debug_with_metrics(
        self,
        msg: str,
        performance_metrics: Optional[Dict[str, Any]] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Log debug message with metrics."""
        self.log_with_metrics(
            logging.DEBUG, msg, performance_metrics, extra_fields, *args, **kwargs
        )

    def info_with_metrics(
        self,
        msg: str,
        performance_metrics: Optional[Dict[str, Any]] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Log info message with metrics."""
        self.log_with_metrics(
            logging.INFO, msg, performance_metrics, extra_fields, *args, **kwargs
        )

    def warning_with_metrics(
        self,
        msg: str,
        performance_metrics: Optional[Dict[str, Any]] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Log warning message with metrics."""
        self.log_with_metrics(
            logging.WARNING, msg, performance_metrics, extra_fields, *args, **kwargs
        )

    def error_with_metrics(
        self,
        msg: str,
        performance_metrics: Optional[Dict[str, Any]] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Log error message with metrics."""
        self.log_with_metrics(
            logging.ERROR, msg, performance_metrics, extra_fields, *args, **kwargs
        )

    def critical_with_metrics(
        self,
        msg: str,
        performance_metrics: Optional[Dict[str, Any]] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Log critical message with metrics."""
        self.log_with_metrics(
            logging.CRITICAL, msg, performance_metrics, extra_fields, *args, **kwargs
        )


def setup_structured_logging(config: LoggingConfig) -> StructuredLogger:
    """Set up structured logging with custom handlers.

    Args:
        config: Logging configuration

    Returns:
        Configured structured logger
    """
    # Create structured logger
    logger = StructuredLogger("readme-mentor")

    # Set log level
    logger.setLevel(getattr(logging, config.log_level.value))

    # Add custom handler
    handler = CustomHandler(config)
    logger.addHandler(handler)

    # Add performance metrics handler if enabled
    if config.show_performance_metrics:
        metrics_handler = PerformanceMetricsHandler(config)
        logger.addHandler(metrics_handler)

    return logger


def get_structured_logger(name: str = "readme-mentor") -> logging.Logger:
    """Get a structured logger instance.

    Args:
        name: Logger name

    Returns:
        Structured logger instance
    """
    # Check if logger already exists
    existing_logger = logging.getLogger(name)

    if isinstance(existing_logger, StructuredLogger):
        return existing_logger

    # Create new structured logger
    logger = StructuredLogger(name)

    # Replace existing logger
    logging.Logger.manager.loggerDict[name] = logger

    return logger
