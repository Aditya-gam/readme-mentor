"""Output channel implementations for the logging system.

This module provides UserOutput and DeveloperLogger classes that handle
user-facing output and technical logging respectively, with support for
Rich-based interfaces, verbosity levels, and performance metrics.
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.status import Status
from rich.table import Table

from .config import LoggingConfig
from .formatters import get_formatter
from .handlers import get_structured_logger


class UserOutput:
    """Rich-based user interface with progress bars, status indicators, and structured output.

    This class provides a user-friendly interface with support for different
    verbosity levels, progress tracking, and performance metrics display.
    """

    def __init__(self, config: LoggingConfig):
        """Initialize user output with configuration.

        Args:
            config: Logging configuration
        """
        self.config = config
        self.formatter = get_formatter(config)
        self.console = Console(
            file=self.formatter.console.file,
            color_system="auto" if config.should_use_color() else None,
            highlight=False,
        )
        self._current_progress: Optional[Progress] = None
        self._current_status: Optional[Status] = None
        self._performance_metrics: Dict[str, Any] = {}

    def info(self, message: str, emoji: Optional[str] = None, **kwargs: Any) -> None:
        """Display an info message.

        Args:
            message: Message to display
            emoji: Optional emoji prefix
            **kwargs: Additional formatting options
        """
        if not self.config.is_quiet():
            formatted = self.formatter.format_message(
                message, "info", emoji=emoji, **kwargs
            )
            self.console.print(formatted)

    def success(self, message: str, emoji: Optional[str] = None, **kwargs: Any) -> None:
        """Display a success message.

        Args:
            message: Message to display
            emoji: Optional emoji prefix
            **kwargs: Additional formatting options
        """
        if not self.config.is_quiet():
            formatted = self.formatter.format_message(
                message, "success", emoji=emoji, **kwargs
            )
            self.console.print(formatted)

    def warning(self, message: str, emoji: Optional[str] = None, **kwargs: Any) -> None:
        """Display a warning message.

        Args:
            message: Message to display
            emoji: Optional emoji prefix
            **kwargs: Additional formatting options
        """
        if not self.config.is_quiet():
            formatted = self.formatter.format_message(
                message, "warning", emoji=emoji, **kwargs
            )
            self.console.print(formatted)

    def error(
        self,
        message: str,
        error: Optional[Exception] = None,
        context: Optional[str] = None,
        emoji: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Display an error message with optional exception details.

        Args:
            message: Error message
            error: Optional exception object
            context: Additional context information
            emoji: Optional emoji prefix
            **kwargs: Additional formatting options
        """
        if not self.config.is_quiet():
            if error:
                formatted = self.formatter.format_error(error, context)
            else:
                formatted = self.formatter.format_message(
                    message, "error", emoji=emoji, **kwargs
                )
            self.console.print(formatted)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Display a debug message (only in debug mode).

        Args:
            message: Debug message
            **kwargs: Additional formatting options
        """
        if self.config.is_debug():
            formatted = self.formatter.format_message(
                message, "info", emoji="🔍", **kwargs
            )
            self.console.print(formatted)

    def verbose(self, message: str, **kwargs: Any) -> None:
        """Display a verbose message (only in verbose or debug mode).

        Args:
            message: Verbose message
            **kwargs: Additional formatting options
        """
        if self.config.is_verbose():
            formatted = self.formatter.format_message(
                message, "info", emoji="📝", **kwargs
            )
            self.console.print(formatted)

    def print_performance_metrics(
        self, metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """Display performance metrics.

        Args:
            metrics: Performance metrics to display (uses internal metrics if None)
        """
        if not self.config.show_performance_metrics:
            return

        display_metrics = metrics or self._performance_metrics
        if not display_metrics:
            return

        formatted = self.formatter.format_performance_metrics(display_metrics)
        if formatted:
            self.console.print(formatted)

    def add_performance_metric(self, key: str, value: Any) -> None:
        """Add a performance metric.

        Args:
            key: Metric key
            value: Metric value
        """
        if self.config.show_performance_metrics:
            self._performance_metrics[key] = value

    def clear_performance_metrics(self) -> None:
        """Clear all performance metrics."""
        self._performance_metrics.clear()

    @contextmanager
    def progress_bar(self, total: int, description: str = "", **kwargs: Any):
        """Context manager for progress bar.

        Args:
            total: Total number of items
            description: Progress bar description
            **kwargs: Additional progress bar options

        Yields:
            Progress bar instance
        """
        if not self.config.show_progress_bars or self.config.is_quiet():
            # Return a dummy progress bar that does nothing
            yield DummyProgressBar()
            return

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
            **kwargs,
        )

        try:
            with progress:
                self._current_progress = progress
                yield progress
        finally:
            self._current_progress = None

    @contextmanager
    def status(self, status: str, spinner: str = "dots"):
        """Context manager for status indicator.

        Args:
            status: Status message
            spinner: Spinner style

        Yields:
            Status instance
        """
        if not self.config.show_status_indicators or self.config.is_quiet():
            # Just print the status message
            self.info(status)
            yield DummyStatus()
            return

        with self.console.status(status, spinner=spinner) as status_obj:
            self._current_status = status_obj
            try:
                yield status_obj
            finally:
                self._current_status = None

    def _create_rich_table(
        self, data: List[Dict[str, Any]], title: str, show_header: bool, **kwargs
    ) -> Table:
        """Create a Rich table from data."""
        table = Table(title=title, show_header=show_header, **kwargs)

        # Add columns based on first row
        if data:
            for key in data[0].keys():
                table.add_column(str(key), style="cyan")

            # Add rows
            for row in data:
                table.add_row(*[str(value) for value in row.values()])

        return table

    def _print_plain_table(
        self, data: List[Dict[str, Any]], title: str, show_header: bool
    ) -> None:
        """Print table in plain text format."""
        if title:
            self.info(title)

        if data and show_header:
            headers = list(data[0].keys())
            self.info(" | ".join(headers))
            self.info("-" * len(" | ".join(headers)))

        for row in data:
            self.info(" | ".join(str(value) for value in row.values()))

    def print_table(
        self,
        data: List[Dict[str, Any]],
        title: str = "",
        show_header: bool = True,
        **kwargs: Any,
    ) -> None:
        """Print data as a table.

        Args:
            data: List of dictionaries representing table rows
            title: Table title
            show_header: Whether to show column headers
            **kwargs: Additional table options
        """
        if not data:
            return

        if self.config.output_format.value == "rich":
            table = self._create_rich_table(data, title, show_header, **kwargs)
            self.console.print(table)
        else:
            self._print_plain_table(data, title, show_header)

    def print_panel(self, content: str, title: str = "", **kwargs: Any) -> None:
        """Print content in a panel.

        Args:
            content: Panel content
            title: Panel title
            **kwargs: Additional panel options
        """
        if self.config.output_format.value == "rich":
            panel = Panel(content, title=title, **kwargs)
            self.console.print(panel)
        else:
            # Fallback to plain text
            if title:
                self.info(f"=== {title} ===")
            self.info(content)

    def print_separator(self, char: str = "=", length: int = 80) -> None:
        """Print a separator line.

        Args:
            char: Character to use for separator
            length: Length of separator line
        """
        if not self.config.is_quiet():
            self.console.print(char * length)

    def print_newline(self) -> None:
        """Print a newline."""
        if not self.config.is_quiet():
            self.console.print()


class DummyProgressBar:
    """Dummy progress bar that does nothing (for quiet mode)."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Dummy implementation - no cleanup needed for quiet mode
        pass

    def add_task(self, description: str, total: int = 100):
        return DummyTask(description, total)

    def update(self, task_id, advance: int = 1, **kwargs):
        # Dummy implementation - no progress updates in quiet mode
        pass


class DummyTask:
    """Dummy task for progress bar."""

    def __init__(self, description: str, total: int):
        self.description = description
        self.total = total
        self.completed = 0
        self.progress = 0
        self.task = None
        self.task_id = None


class DummyStatus:
    """Dummy status that does nothing (for quiet mode)."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Dummy implementation - no cleanup needed for quiet mode
        pass

    def update(self, status: str):
        # Dummy implementation - no cleanup needed for quiet mode
        pass


class DeveloperLogger:
    """Technical logging with structured format and performance metrics.

    This class provides structured logging capabilities for developers,
    with support for debug information, stack traces, and performance metrics.
    """

    def __init__(self, config: LoggingConfig):
        """Initialize developer logger with configuration.

        Args:
            config: Logging configuration
        """
        self.config = config
        self.logger = get_structured_logger("readme-mentor")
        self._performance_metrics: Dict[str, Any] = {}
        self._start_times: Dict[str, float] = {}

    def debug(
        self, message: str, extra_fields: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Log a debug message.

        Args:
            message: Debug message
            extra_fields: Additional structured fields
            **kwargs: Additional keyword arguments
        """
        if self.config.log_level.value == "DEBUG":
            self.logger.debug_with_metrics(
                message,
                performance_metrics=self._performance_metrics,
                extra_fields=extra_fields,
                **kwargs,
            )

    def info(
        self, message: str, extra_fields: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Log an info message.

        Args:
            message: Info message
            extra_fields: Additional structured fields
            **kwargs: Additional keyword arguments
        """
        self.logger.info_with_metrics(
            message,
            performance_metrics=self._performance_metrics,
            extra_fields=extra_fields,
            **kwargs,
        )

    def warning(
        self, message: str, extra_fields: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Log a warning message.

        Args:
            message: Warning message
            extra_fields: Additional structured fields
            **kwargs: Additional keyword arguments
        """
        self.logger.warning_with_metrics(
            message,
            performance_metrics=self._performance_metrics,
            extra_fields=extra_fields,
            **kwargs,
        )

    def error(
        self,
        message: str,
        error: Optional[Exception] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Log an error message with optional exception details.

        Args:
            message: Error message
            error: Optional exception object
            extra_fields: Additional structured fields
            **kwargs: Additional keyword arguments
        """
        if error:
            extra_fields = extra_fields or {}
            extra_fields.update(
                {
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )

            if self.config.show_stack_traces:
                import traceback

                extra_fields["stack_trace"] = traceback.format_exc()

        self.logger.error_with_metrics(
            message,
            performance_metrics=self._performance_metrics,
            extra_fields=extra_fields,
            **kwargs,
        )

    def critical(
        self, message: str, extra_fields: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Log a critical message.

        Args:
            message: Critical message
            extra_fields: Additional structured fields
            **kwargs: Additional keyword arguments
        """
        self.logger.critical_with_metrics(
            message,
            performance_metrics=self._performance_metrics,
            extra_fields=extra_fields,
            **kwargs,
        )

    def add_performance_metric(self, key: str, value: Any) -> None:
        """Add a performance metric.

        Args:
            key: Metric key
            value: Metric value
        """
        if self.config.show_performance_metrics:
            self._performance_metrics[key] = value

    def start_timer(self, name: str) -> None:
        """Start a timer for performance measurement.

        Args:
            name: Timer name
        """
        if self.config.track_wall_time:
            self._start_times[name] = time.time()

    def stop_timer(self, name: str) -> float:
        """Stop a timer and return elapsed time.

        Args:
            name: Timer name

        Returns:
            Elapsed time in seconds
        """
        if name not in self._start_times:
            return 0.0

        elapsed = time.time() - self._start_times[name]
        del self._start_times[name]

        if self.config.track_wall_time:
            self.add_performance_metric(f"{name}_time", elapsed)

        return elapsed

    def clear_performance_metrics(self) -> None:
        """Clear all performance metrics."""
        self._performance_metrics.clear()
        self._start_times.clear()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        return self._performance_metrics.copy()

    @contextmanager
    def timer(self, name: str):
        """Context manager for timing operations.

        Args:
            name: Timer name

        Yields:
            Timer context
        """
        self.start_timer(name)
        try:
            yield
        finally:
            self.stop_timer(name)

    def log_function_call(
        self,
        func_name: str,
        args: Optional[Dict[str, Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a function call with parameters.

        Args:
            func_name: Function name
            args: Function arguments
            kwargs: Function keyword arguments
        """
        if self.config.is_debug():
            extra_fields = {"function": func_name}

            if args:
                extra_fields["args"] = args
            if kwargs:
                extra_fields["kwargs"] = kwargs

            self.debug(f"Calling function: {func_name}", extra_fields=extra_fields)

    def log_function_result(
        self, func_name: str, result: Any, execution_time: Optional[float] = None
    ) -> None:
        """Log a function result.

        Args:
            func_name: Function name
            result: Function result
            execution_time: Execution time in seconds
        """
        if self.config.is_debug():
            extra_fields = {"function": func_name}

            if execution_time is not None:
                extra_fields["execution_time"] = execution_time
                self.add_performance_metric(f"{func_name}_time", execution_time)

            self.debug(f"Function {func_name} completed", extra_fields=extra_fields)
