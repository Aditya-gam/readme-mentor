"""Output formatting logic for logging system.

This module provides formatters for different output formats including
Rich-based user interfaces, plain text, and structured JSON output.
"""

import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .config import LoggingConfig
from .enums import OutputFormat


class OutputFormatter:
    """Base class for output formatting."""

    def __init__(self, config: LoggingConfig):
        """Initialize formatter with configuration.

        Args:
            config: Logging configuration
        """
        self.config = config
        self.console = Console(file=sys.stdout)

    def format_message(self, message: str, level: str = "info", **kwargs: Any) -> str:
        """Format a message for output.

        Args:
            message: Message to format
            level: Message level (info, warning, error, success)
            **kwargs: Additional formatting options

        Returns:
            Formatted message string
        """
        raise NotImplementedError

    def format_error(self, error: Exception, context: Optional[str] = None) -> str:
        """Format an error message with context.

        Args:
            error: Exception to format
            context: Additional context information

        Returns:
            Formatted error message
        """
        raise NotImplementedError

    def format_performance_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format performance metrics for display.

        Args:
            metrics: Dictionary of performance metrics

        Returns:
            Formatted metrics string
        """
        raise NotImplementedError


class RichFormatter(OutputFormatter):
    """Rich-based formatter for beautiful user interfaces."""

    def __init__(self, config: LoggingConfig):
        """Initialize Rich formatter.

        Args:
            config: Logging configuration
        """
        super().__init__(config)
        self.console = Console(
            file=sys.stdout,
            color_system="auto" if config.should_use_color() else None,
            highlight=False,
        )

    def format_message(self, message: str, level: str = "info", **kwargs: Any) -> str:
        """Format a message using Rich styling.

        Args:
            message: Message to format
            level: Message level (info, warning, error, success)
            **kwargs: Additional formatting options

        Returns:
            Formatted message string
        """
        # Create styled text based on level
        if level == "error":
            text = Text(message, style="bold red")
        elif level == "warning":
            text = Text(message, style="bold yellow")
        elif level == "success":
            text = Text(message, style="bold green")
        elif level == "info":
            text = Text(message, style="blue")
        else:
            text = Text(message)

        # Add emoji prefix if provided
        emoji = kwargs.get("emoji")
        if emoji:
            text = Text(f"{emoji} ", style="white") + text

        return str(text)

    def format_error(self, error: Exception, context: Optional[str] = None) -> str:
        """Format an error with Rich styling and context.

        Args:
            error: Exception to format
            context: Additional context information

        Returns:
            Formatted error message
        """
        # Create error panel
        error_text = Text(f"❌ {type(error).__name__}: {str(error)}", style="bold red")

        if context:
            context_text = Text(f"\nContext: {context}", style="dim")
            error_text += context_text

        if self.config.show_stack_traces:
            import traceback

            stack_trace = Text("\n\nStack Trace:", style="bold")
            stack_trace += Text(f"\n{''.join(traceback.format_exc())}", style="dim red")
            error_text += stack_trace

        # Create panel with error
        panel = Panel(
            error_text,
            title="Error",
            border_style="red",
            padding=(1, 2),
        )

        return str(panel)

    def format_performance_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format performance metrics as a Rich table.

        Args:
            metrics: Dictionary of performance metrics

        Returns:
            Formatted metrics string
        """
        if not self.config.show_performance_metrics:
            return ""

        table = Table(
            title="Performance Metrics", show_header=True, header_style="bold magenta"
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="dim")

        # Add metrics to table
        for metric, value in metrics.items():
            if isinstance(value, dict):
                # Handle nested metrics
                for sub_metric, sub_value in value.items():
                    unit = self._get_metric_unit(sub_metric)
                    table.add_row(f"{metric}.{sub_metric}", str(sub_value), unit)
            else:
                unit = self._get_metric_unit(metric)
                table.add_row(metric, str(value), unit)

        return str(table)

    def _get_metric_unit(self, metric: str) -> str:
        """Get the unit for a metric.

        Args:
            metric: Metric name

        Returns:
            Unit string
        """
        units = {
            "wall_time": "seconds",
            "token_count": "tokens",
            "tool_calls": "calls",
            "memory_usage": "MB",
            "cpu_usage": "%",
        }
        return units.get(metric, "")

    def create_progress_bar(self, total: int, description: str = "") -> Any:
        """Create a Rich progress bar.

        Args:
            total: Total number of items
            description: Progress bar description

        Returns:
            Rich progress bar instance
        """
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
        )

        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        )

    def create_status_indicator(self, status: str, style: str = "blue") -> str:
        """Create a status indicator.

        Args:
            status: Status text
            style: Rich style for the status

        Returns:
            Formatted status string
        """
        return f"[{style}]{status}[/{style}]"


class PlainFormatter(OutputFormatter):
    """Plain text formatter for simple output."""

    def format_message(self, message: str, level: str = "info", **kwargs: Any) -> str:
        """Format a message as plain text.

        Args:
            message: Message to format
            level: Message level (info, warning, error, success)
            **kwargs: Additional formatting options

        Returns:
            Formatted message string
        """
        # Add level prefix
        level_prefix = {
            "error": "ERROR: ",
            "warning": "WARNING: ",
            "success": "SUCCESS: ",
            "info": "INFO: ",
        }.get(level, "")

        # Add emoji if provided
        emoji = kwargs.get("emoji", "")
        if emoji:
            emoji += " "

        return f"{emoji}{level_prefix}{message}"

    def format_error(self, error: Exception, context: Optional[str] = None) -> str:
        """Format an error as plain text.

        Args:
            error: Exception to format
            context: Additional context information

        Returns:
            Formatted error message
        """
        lines = [f"ERROR: {type(error).__name__}: {str(error)}"]

        if context:
            lines.append(f"Context: {context}")

        if self.config.show_stack_traces:
            import traceback

            lines.append("Stack Trace:")
            lines.append(traceback.format_exc())

        return "\n".join(lines)

    def format_performance_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format performance metrics as plain text.

        Args:
            metrics: Dictionary of performance metrics

        Returns:
            Formatted metrics string
        """
        if not self.config.show_performance_metrics:
            return ""

        lines = ["Performance Metrics:"]

        for metric, value in metrics.items():
            if isinstance(value, dict):
                # Handle nested metrics
                for sub_metric, sub_value in value.items():
                    unit = self._get_metric_unit(sub_metric)
                    lines.append(f"  {metric}.{sub_metric}: {sub_value} {unit}")
            else:
                unit = self._get_metric_unit(metric)
                lines.append(f"  {metric}: {value} {unit}")

        return "\n".join(lines)

    def _get_metric_unit(self, metric: str) -> str:
        """Get the unit for a metric.

        Args:
            metric: Metric name

        Returns:
            Unit string
        """
        units = {
            "wall_time": "seconds",
            "token_count": "tokens",
            "tool_calls": "calls",
            "memory_usage": "MB",
            "cpu_usage": "%",
        }
        return units.get(metric, "")


class JSONFormatter(OutputFormatter):
    """JSON formatter for structured output."""

    def format_message(self, message: str, level: str = "info", **kwargs: Any) -> str:
        """Format a message as JSON.

        Args:
            message: Message to format
            level: Message level (info, warning, error, success)
            **kwargs: Additional formatting options

        Returns:
            JSON formatted message
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "level": level.upper(),
            "message": message,
            **kwargs,
        }

        return json.dumps(data, indent=2)

    def format_error(self, error: Exception, context: Optional[str] = None) -> str:
        """Format an error as JSON.

        Args:
            error: Exception to format
            context: Additional context information

        Returns:
            JSON formatted error
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "level": "ERROR",
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        if context:
            data["context"] = context

        if self.config.show_stack_traces:
            import traceback

            data["stack_trace"] = traceback.format_exc()

        return json.dumps(data, indent=2)

    def format_performance_metrics(self, metrics: Dict[str, Any]) -> str:
        """Format performance metrics as JSON.

        Args:
            metrics: Dictionary of performance metrics

        Returns:
            JSON formatted metrics
        """
        if not self.config.show_performance_metrics:
            return ""

        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "performance_metrics",
            "metrics": metrics,
        }

        return json.dumps(data, indent=2)


class LogFormatter:
    """Formatter for developer logging."""

    def __init__(self, config: LoggingConfig):
        """Initialize log formatter.

        Args:
            config: Logging configuration
        """
        self.config = config

    def format_log_record(self, record: Any) -> str:
        """Format a log record for output.

        Args:
            record: Log record to format

        Returns:
            Formatted log message
        """
        timestamp = datetime.fromtimestamp(record.created).isoformat()

        # Basic format
        formatted = (
            f"{timestamp} [{record.levelname}] {record.name}: {record.getMessage()}"
        )

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            for key, value in record.extra_fields.items():
                formatted += f" | {key}={value}"

        # Add performance metrics if present
        if hasattr(record, "performance_metrics"):
            metrics_str = self._format_metrics_for_log(record.performance_metrics)
            formatted += f" | {metrics_str}"

        return formatted

    def _format_metrics_for_log(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for log output.

        Args:
            metrics: Performance metrics

        Returns:
            Formatted metrics string
        """
        if not self.config.show_performance_metrics:
            return ""

        parts = []
        for key, value in metrics.items():
            unit = self._get_metric_unit(key)
            parts.append(f"{key}={value}{unit}")

        return "metrics={" + ", ".join(parts) + "}"

    def _get_metric_unit(self, metric: str) -> str:
        """Get the unit for a metric.

        Args:
            metric: Metric name

        Returns:
            Unit string
        """
        units = {
            "wall_time": "s",
            "token_count": "t",
            "tool_calls": "c",
            "memory_usage": "MB",
            "cpu_usage": "%",
        }
        return units.get(metric, "")


def get_formatter(config: LoggingConfig) -> OutputFormatter:
    """Get the appropriate formatter based on configuration.

    Args:
        config: Logging configuration

    Returns:
        OutputFormatter instance
    """
    if config.output_format == OutputFormat.RICH:
        return RichFormatter(config)
    elif config.output_format == OutputFormat.JSON:
        return JSONFormatter(config)
    else:  # PLAIN
        return PlainFormatter(config)
