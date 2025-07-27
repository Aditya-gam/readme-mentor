"""Enhanced output manager for README-Mentor.

This module provides a unified interface for all output formatting operations
with support for Rich, Plain, and JSON formats as specified in Phase 2 requirements.
"""

import json
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.spinner import Spinner
from rich.status import Status
from rich.table import Table

from ..logging import UserOutput
from ..logging.config import LoggingConfig
from ..logging.enums import OutputFormat
from .formatters import (
    ErrorFormatter,
    IngestionFormatter,
    PerformanceFormatter,
    QAFormatter,
)


class OutputManager:
    """Enhanced output manager with unified interface for all formatting operations.

    This class provides a comprehensive interface for handling all output formatting
    requirements specified in Phase 2, including:
    - Rich format with progress indicators, status messages, structured displays, and interactive elements
    - Plain text format with simple output, structured layout, error formatting, and performance display
    - JSON format with machine-readable data, metadata inclusion, error structure, and progress tracking
    """

    def __init__(self, config: LoggingConfig):
        """Initialize output manager with configuration.

        Args:
            config: Logging configuration
        """
        self.config = config
        self.console = Console(
            file=sys.stdout,
            color_system="auto" if config.should_use_color() else None,
            highlight=False,
        )

        # Initialize specialized formatters
        self.ingestion = IngestionFormatter(UserOutput(config))
        self.qa = QAFormatter(UserOutput(config))
        self.performance = PerformanceFormatter(UserOutput(config))
        self.error = ErrorFormatter(UserOutput(config))

        # Performance tracking
        self._operation_start_times: Dict[str, float] = {}
        self._performance_metrics: Dict[str, Any] = {}

    def start_operation(self, operation_name: str) -> None:
        """Start tracking an operation with enhanced status indicators.

        Args:
            operation_name: Name of the operation to track
        """
        self._operation_start_times[operation_name] = datetime.now().timestamp()

        if self.config.output_format == OutputFormat.RICH:
            self.console.print(
                f"[bold blue]🔄 Starting:[/bold blue] {operation_name}", style="blue"
            )
        elif self.config.output_format == OutputFormat.JSON:
            start_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "operation_start",
                "operation": operation_name,
                "status": "started",
            }
            self.console.print(json.dumps(start_data))
        else:
            # Plain format with structured layout
            self.console.print(f"🔄 Starting: {operation_name}")

    def end_operation(
        self, operation_name: str, additional_metrics: Optional[Dict[str, Any]] = None
    ) -> float:
        """End tracking an operation and display completion status.

        Args:
            operation_name: Name of the operation to end
            additional_metrics: Additional metrics to display

        Returns:
            Duration of the operation in seconds
        """
        if operation_name not in self._operation_start_times:
            return 0.0

        start_time = self._operation_start_times[operation_name]
        duration = datetime.now().timestamp() - start_time

        if self.config.output_format == OutputFormat.RICH:
            self.console.print(
                f"[bold green]✅ Completed:[/bold green] {operation_name} "
                f"[yellow]({duration:.2f}s)[/yellow]",
                style="green",
            )

            if additional_metrics:
                metrics_table = Table(
                    title=f"Performance Metrics - {operation_name}",
                    show_header=True,
                    header_style="bold magenta",
                )
                metrics_table.add_column("Metric", style="cyan")
                metrics_table.add_column("Value", style="green")
                metrics_table.add_column("Unit", style="dim")

                for key, value in additional_metrics.items():
                    unit = self._get_metric_unit(key)
                    formatted_key = key.replace("_", " ").title()
                    metrics_table.add_row(formatted_key, str(value), unit)

                self.console.print(metrics_table)

        elif self.config.output_format == OutputFormat.JSON:
            completion_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "operation_complete",
                "operation": operation_name,
                "duration_seconds": round(duration, 2),
                "status": "completed",
            }

            if additional_metrics:
                completion_data["metrics"] = additional_metrics

            self.console.print(json.dumps(completion_data))
        else:
            # Plain format with structured layout
            self.console.print(f"✅ Completed: {operation_name} ({duration:.2f}s)")

            if additional_metrics:
                self.console.print("📊 Additional metrics:")
                for key, value in additional_metrics.items():
                    unit = self._get_metric_unit(key)
                    formatted_key = key.replace("_", " ").title()
                    self.console.print(f"  {formatted_key}: {value} {unit}")

        del self._operation_start_times[operation_name]
        return duration

    @contextmanager
    def progress_bar(self, total: int, description: str = "Processing"):
        """Context manager for progress bars with enhanced indicators.

        Args:
            total: Total number of items to process
            description: Description for the progress bar

        Yields:
            Progress instance for updating
        """
        if self.config.output_format == OutputFormat.RICH:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                expand=True,
            )
            progress.add_task(description, total=total)

            with progress:
                yield progress
        else:
            # Dummy progress for non-rich formats
            yield DummyProgress()

    @contextmanager
    def status_spinner(self, status: str, spinner: str = "dots"):
        """Context manager for status spinners with enhanced display.

        Args:
            status: Status message to display
            spinner: Spinner type to use

        Yields:
            Status instance for updating
        """
        if self.config.output_format == OutputFormat.RICH:
            with Status(status, spinner=spinner, console=self.console) as status_obj:
                yield status_obj
        else:
            # Dummy status for non-rich formats
            yield DummyStatus()

    @contextmanager
    def live_spinner(self, description: str, spinner: str = "dots"):
        """Context manager for live spinners with enhanced display.

        Args:
            description: Description for the spinner
            spinner: Spinner type to use

        Yields:
            Live instance for updating
        """
        if self.config.output_format == OutputFormat.RICH:
            spinner_obj = Spinner(spinner, text=description)
            with Live(spinner_obj, console=self.console, refresh_per_second=10):
                yield spinner_obj
        else:
            # Dummy live for non-rich formats
            yield DummyLive()

    def print_table(
        self, data: List[Dict[str, Any]], title: str = "", show_header: bool = True
    ) -> None:
        """Print data in a structured table format.

        Args:
            data: List of dictionaries containing table data
            title: Table title
            show_header: Whether to show table header
        """
        if not data:
            return

        if self.config.output_format == OutputFormat.RICH:
            # Rich table with enhanced styling
            table = Table(
                title=title,
                show_header=show_header,
                header_style="bold magenta",
                expand=True,
            )

            # Add columns based on data
            if data:
                for key in data[0].keys():
                    table.add_column(
                        key.replace("_", " ").title(), style="cyan", no_wrap=True
                    )

                # Add rows
                for row in data:
                    table.add_row(*[str(value) for value in row.values()])

            self.console.print(table)

        elif self.config.output_format == OutputFormat.JSON:
            # JSON format with metadata
            table_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "table_data",
                "title": title,
                "data": data,
                "row_count": len(data),
                "column_count": len(data[0]) if data else 0,
            }
            self.console.print(json.dumps(table_data))
        else:
            # Plain format with structured layout
            if title:
                self.console.print(f"\n{title}")
                self.console.print("=" * len(title))

            if data and show_header:
                headers = list(data[0].keys())
                header_line = " | ".join(
                    header.replace("_", " ").title() for header in headers
                )
                self.console.print(header_line)
                self.console.print("-" * len(header_line))

            for row in data:
                row_line = " | ".join(str(value) for value in row.values())
                self.console.print(row_line)

    def print_panel(self, content: str, title: str = "", style: str = "blue") -> None:
        """Print content in a panel format.

        Args:
            content: Content to display
            title: Panel title
            style: Panel style
        """
        if self.config.output_format == OutputFormat.RICH:
            panel = Panel(content, title=title, border_style=style, padding=(1, 2))
            self.console.print(panel)
        elif self.config.output_format == OutputFormat.JSON:
            panel_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "panel",
                "title": title,
                "content": content,
                "style": style,
            }
            self.console.print(json.dumps(panel_data))
        else:
            # Plain format with structured layout
            if title:
                self.console.print(f"\n{title}")
                self.console.print("-" * len(title))
            self.console.print(content)

    def print_separator(self, char: str = "=", length: int = 80) -> None:
        """Print a separator line.

        Args:
            char: Character to use for separator
            length: Length of separator line
        """
        if self.config.output_format == OutputFormat.RICH:
            self.console.print(Rule(char=char, style="dim"))
        elif self.config.output_format == OutputFormat.JSON:
            separator_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "separator",
                "character": char,
                "length": length,
            }
            self.console.print(json.dumps(separator_data))
        else:
            # Plain format
            self.console.print(char * length)

    def print_success(self, message: str, emoji: str = "✅") -> None:
        """Print a success message with enhanced formatting.

        Args:
            message: Success message
            emoji: Emoji to display
        """
        if self.config.output_format == OutputFormat.RICH:
            self.console.print(f"[bold green]{emoji} {message}[/bold green]")
        elif self.config.output_format == OutputFormat.JSON:
            success_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "success_message",
                "message": message,
                "emoji": emoji,
                "status": "success",
            }
            self.console.print(json.dumps(success_data))
        else:
            # Plain format with structured layout
            self.console.print(f"{emoji} {message}")

    def print_warning(self, message: str, emoji: str = "⚠️") -> None:
        """Print a warning message with enhanced formatting.

        Args:
            message: Warning message
            emoji: Emoji to display
        """
        if self.config.output_format == OutputFormat.RICH:
            self.console.print(f"[bold yellow]{emoji} {message}[/bold yellow]")
        elif self.config.output_format == OutputFormat.JSON:
            warning_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "warning_message",
                "message": message,
                "emoji": emoji,
                "status": "warning",
            }
            self.console.print(json.dumps(warning_data))
        else:
            # Plain format with structured layout
            self.console.print(f"{emoji} {message}")

    def print_error(
        self, message: str, error: Optional[Exception] = None, emoji: str = "❌"
    ) -> None:
        """Print an error message with enhanced formatting.

        Args:
            message: Error message
            error: Optional exception
            emoji: Emoji to display
        """
        if self.config.output_format == OutputFormat.RICH:
            error_text = f"[bold red]{emoji} {message}[/bold red]"
            if error:
                error_text += f"\n[red]Error: {str(error)}[/red]"
            self.console.print(error_text)
        elif self.config.output_format == OutputFormat.JSON:
            error_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "error_message",
                "message": message,
                "emoji": emoji,
                "status": "error",
            }
            if error:
                error_data["error_type"] = type(error).__name__
                error_data["error_message"] = str(error)
            self.console.print(json.dumps(error_data))
        else:
            # Plain format with structured layout
            self.console.print(f"{emoji} {message}")
            if error:
                self.console.print(f"Error: {str(error)}")

    def print_info(self, message: str, emoji: str = "ℹ️") -> None:
        """Print an info message with enhanced formatting.

        Args:
            message: Info message
            emoji: Emoji to display
        """
        if self.config.output_format == OutputFormat.RICH:
            self.console.print(f"[bold blue]{emoji} {message}[/bold blue]")
        elif self.config.output_format == OutputFormat.JSON:
            info_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "info_message",
                "message": message,
                "emoji": emoji,
                "status": "info",
            }
            self.console.print(json.dumps(info_data))
        else:
            # Plain format with structured layout
            self.console.print(f"{emoji} {message}")

    def add_performance_metric(self, key: str, value: Any) -> None:
        """Add a performance metric for tracking.

        Args:
            key: Metric key
            value: Metric value
        """
        self._performance_metrics[key] = value

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get all collected performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        return self._performance_metrics.copy()

    def clear_performance_metrics(self) -> None:
        """Clear all performance metrics."""
        self._performance_metrics.clear()

    def print_performance_summary(self) -> None:
        """Print a comprehensive performance summary."""
        if not self._performance_metrics:
            return

        if self.config.output_format == OutputFormat.RICH:
            # Rich performance summary table
            summary_table = Table(
                title="Performance Summary",
                show_header=True,
                header_style="bold magenta",
                expand=True,
            )
            summary_table.add_column("Metric", style="cyan", no_wrap=True)
            summary_table.add_column("Value", style="green")
            summary_table.add_column("Unit", style="dim")
            summary_table.add_column("Category", style="yellow")

            for metric, value in self._performance_metrics.items():
                if isinstance(value, dict):
                    for sub_metric, sub_value in value.items():
                        unit = self._get_metric_unit(sub_metric)
                        category = metric.replace("_", " ").title()
                        summary_table.add_row(
                            f"{metric}.{sub_metric}", str(sub_value), unit, category
                        )
                else:
                    unit = self._get_metric_unit(metric)
                    category = "General"
                    summary_table.add_row(metric, str(value), unit, category)

            self.console.print(summary_table)

        elif self.config.output_format == OutputFormat.JSON:
            # JSON format with comprehensive metadata
            summary_data = {
                "timestamp": datetime.now().isoformat(),
                "type": "performance_summary",
                "metrics": self._performance_metrics,
                "metadata": {
                    "summary_type": "comprehensive",
                    "generated_at": datetime.now().isoformat(),
                },
            }
            self.console.print(json.dumps(summary_data))
        else:
            # Plain format with structured layout
            self.console.print("📊 Performance Summary:")

            for metric, value in self._performance_metrics.items():
                if isinstance(value, dict):
                    self.console.print(f"  {metric}:")
                    for sub_metric, sub_value in value.items():
                        unit = self._get_metric_unit(sub_metric)
                        self.console.print(f"    {sub_metric}: {sub_value} {unit}")
                else:
                    unit = self._get_metric_unit(metric)
                    self.console.print(f"  {metric}: {value} {unit}")

    def _get_metric_unit(self, metric: str) -> str:
        """Get the unit for a metric."""
        units = {
            "wall_time": "seconds",
            "token_count": "tokens",
            "tool_calls": "calls",
            "memory_usage": "MB",
            "cpu_usage": "%",
            "duration": "seconds",
            "latency": "ms",
            "throughput": "items/sec",
            "accuracy": "%",
            "precision": "%",
            "recall": "%",
            "f1_score": "%",
            "files_processed": "files",
            "chunks_created": "chunks",
            "processing_rate": "files/sec",
            "embedding_rate": "chunks/sec",
        }
        return units.get(metric, "")


class DummyProgress:
    """Dummy progress bar for non-rich formats."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def update(self, task_id, **kwargs):
        pass


class DummyStatus:
    """Dummy status for non-rich formats."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def update(self, status: str):
        pass


class DummyLive:
    """Dummy live for non-rich formats."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def update(self, renderable):
        pass
