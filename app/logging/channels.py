"""Output channel implementations for the logging system.

This module provides UserOutput and DeveloperLogger classes that handle
user-facing output and technical logging respectively, with support for
Rich-based interfaces, verbosity levels, and performance metrics.
"""

import time
from contextlib import contextmanager
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

from .config import LoggingConfig
from .enums import OutputFormat
from .formatters import get_formatter
from .handlers import CustomHandler, get_structured_logger


class UserOutput:
    """Rich-based user interface with progress bars, status indicators, and structured output.

    This class provides a user-friendly interface with support for different
    verbosity levels, progress tracking, and performance metrics display.

    Phase 3 implementation supports 4 verbosity levels (0-3):
    - QUIET (0): Only critical errors and final results
    - NORMAL (1): Default level with success/failure status and basic metrics
    - VERBOSE (2): Detailed operation steps and extended metrics
    - DEBUG (3): All available information including raw data and internal state
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
        self._operation_start_times: Dict[str, float] = {}
        self._operation_steps: Dict[str, List[str]] = {}
        self._operation_metrics: Dict[str, Dict[str, Any]] = {}
        self._token_counts: Dict[str, int] = {}
        self._tool_call_counts: Dict[str, int] = {}

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
        # Always show errors, even in quiet mode
        if error and self.config.show_error_details:
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

    def step(
        self, message: str, operation: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Display an operation step (only in verbose or debug mode).

        Args:
            message: Step message
            operation: Operation name for tracking
            **kwargs: Additional formatting options
        """
        if self.config.show_operation_steps:
            formatted = self.formatter.format_message(
                message, "info", emoji="⚡", **kwargs
            )
            self.console.print(formatted)

            # Track step for operation if provided
            if operation:
                if operation not in self._operation_steps:
                    self._operation_steps[operation] = []
                self._operation_steps[operation].append(message)

    def detail(self, message: str, **kwargs: Any) -> None:
        """Display detailed information (only in verbose or debug mode).

        Args:
            message: Detailed message
            **kwargs: Additional formatting options
        """
        if self.config.user_output_level.should_show_details():
            formatted = self.formatter.format_message(
                message, "info", emoji="🔍", **kwargs
            )
            self.console.print(formatted)

    def raw(self, data: Any, label: str = "Raw Data", **kwargs: Any) -> None:
        """Display raw data (only in debug mode).

        Args:
            data: Raw data to display
            label: Label for the data
            **kwargs: Additional formatting options
        """
        if self.config.show_raw_data:
            if self.config.output_format == OutputFormat.RICH:
                self.console.print(
                    Panel(str(data), title=f"🔬 {label}", border_style="dim", **kwargs)
                )
            else:
                self.info(f"🔬 {label}:")
                self.info(str(data))

    def config_detail(self, key: str, value: Any, **kwargs: Any) -> None:
        """Display configuration detail (only in verbose or debug mode).

        Args:
            key: Configuration key
            value: Configuration value
            **kwargs: Additional formatting options
        """
        if self.config.show_configuration_details:
            self.detail(f"Config: {key} = {value}", **kwargs)

    def internal_state(self, state: Dict[str, Any], **kwargs: Any) -> None:
        """Display internal state information (only in debug mode).

        Args:
            state: Internal state dictionary
            **kwargs: Additional formatting options
        """
        if self.config.show_internal_state:
            if self.config.output_format == OutputFormat.RICH:
                state_table = Table(title="🔧 Internal State", show_header=True)
                state_table.add_column("Component", style="cyan")
                state_table.add_column("State", style="yellow")

                for component, component_state in state.items():
                    state_table.add_row(component, str(component_state))

                self.console.print(state_table)
            else:
                self.info("🔧 Internal State:")
                for component, component_state in state.items():
                    self.info(f"  {component}: {component_state}")

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

        # Filter metrics based on verbosity level
        if self.config.show_raw_metrics:
            # Show all metrics in debug mode
            filtered_metrics = display_metrics
        elif self.config.show_detailed_metrics:
            # Show detailed metrics in verbose mode
            filtered_metrics = {
                k: v
                for k, v in display_metrics.items()
                if not k.startswith("raw_") and not k.startswith("internal_")
            }
        else:
            # Show basic metrics in normal mode
            filtered_metrics = {
                k: v
                for k, v in display_metrics.items()
                if k
                in [
                    "ingestion_duration",
                    "qa_session_duration",
                    "total_files",
                    "total_chunks",
                ]
            }

        if filtered_metrics:
            formatted = self.formatter.format_performance_metrics(filtered_metrics)
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

    def start_operation_timer(self, operation_name: str) -> None:
        """Start timing an operation.

        Args:
            operation_name: Name of the operation being timed
        """
        self._operation_start_times[operation_name] = time.time()
        if operation_name not in self._operation_steps:
            self._operation_steps[operation_name] = []
        if operation_name not in self._operation_metrics:
            self._operation_metrics[operation_name] = {}

    def end_operation_timer(self, operation_name: str) -> float:
        """End timing an operation and return duration.

        Args:
            operation_name: Name of the operation being timed

        Returns:
            Duration in seconds
        """
        if operation_name in self._operation_start_times:
            duration = time.time() - self._operation_start_times[operation_name]
            self.add_performance_metric(f"{operation_name}_duration", duration)

            # Add operation summary if verbose
            if self.config.show_operation_steps:
                steps = self._operation_steps.get(operation_name, [])
                self.step(
                    f"Operation '{operation_name}' completed in {duration:.2f}s with {len(steps)} steps"
                )

            del self._operation_start_times[operation_name]
            return duration
        return 0.0

    def add_token_count(self, operation: str, count: int) -> None:
        """Add token count for an operation.

        Args:
            operation: Operation name
            count: Token count
        """
        if self.config.track_token_counts:
            self._token_counts[operation] = count
            if self.config.show_detailed_metrics:
                self.detail(f"Token count for {operation}: {count}")

    def add_tool_call(self, tool_name: str) -> None:
        """Track a tool call.

        Args:
            tool_name: Name of the tool
        """
        if self.config.track_tool_calls:
            self._tool_call_counts[tool_name] = (
                self._tool_call_counts.get(tool_name, 0) + 1
            )
            if self.config.show_detailed_metrics:
                self.detail(
                    f"Tool call: {tool_name} (total: {self._tool_call_counts[tool_name]})"
                )

    def get_operation_summary(self, operation_name: str) -> Dict[str, Any]:
        """Get a summary of an operation.

        Args:
            operation_name: Name of the operation

        Returns:
            Operation summary dictionary
        """
        summary = {
            "duration": self._performance_metrics.get(f"{operation_name}_duration", 0),
            "steps": self._operation_steps.get(operation_name, []),
            "token_count": self._token_counts.get(operation_name, 0),
        }

        # Add operation-specific metrics
        if operation_name in self._operation_metrics:
            summary.update(self._operation_metrics[operation_name])

        return summary

    def print_operation_summary(self, operation_name: str) -> None:
        """Print a detailed summary of an operation.

        Args:
            operation_name: Name of the operation
        """
        if not self.config.show_detailed_metrics:
            return

        summary = self.get_operation_summary(operation_name)
        formatted = self.formatter.format_operation_summary(operation_name, summary)

        if formatted:
            self.console.print(formatted)

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

        # Enhanced progress bar with time tracking
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
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
    def detailed_progress_bar(
        self, total: int, description: str = "", operation: str = "", **kwargs: Any
    ):
        """Context manager for detailed progress bar with operation tracking.

        Args:
            total: Total number of items
            description: Progress bar description
            operation: Operation name for tracking
            **kwargs: Additional progress bar options

        Yields:
            Progress bar instance
        """
        if not self.config.show_detailed_progress or self.config.is_quiet():
            # Return a dummy progress bar that does nothing
            yield DummyProgressBar()
            return

        # Start operation timer if provided
        if operation:
            self.start_operation_timer(operation)

        # Enhanced progress bar with detailed tracking
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            **kwargs,
        )

        try:
            with progress:
                self._current_progress = progress
                yield progress
        finally:
            self._current_progress = None
            # End operation timer if provided
            if operation:
                self.end_operation_timer(operation)

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

    @contextmanager
    def live_spinner(self, description: str, spinner: str = "dots"):
        """Context manager for live spinner display.

        Args:
            description: Description text
            spinner: Spinner style

        Yields:
            Live display instance
        """
        if not self.config.show_spinners or self.config.is_quiet():
            self.info(description)
            yield DummyLive()
            return

        spinner_obj = Spinner(spinner, text=description)
        with Live(spinner_obj, console=self.console, refresh_per_second=10):
            yield spinner_obj

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
            if self.config.output_format.value == "rich":
                self.console.print(Rule(char=char, style="dim"))
            else:
                self.console.print(char * length)

    def print_newline(self) -> None:
        """Print a newline."""
        if not self.config.is_quiet():
            self.console.print()

    def _print_rich_qa_session(
        self,
        question: str,
        answer: str,
        citations: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Print Q&A session in rich format."""
        self.console.print(
            Panel(
                f"[bold blue]❓ Question:[/bold blue]\n{question}",
                title="Question",
                border_style="blue",
            )
        )

        self.console.print(
            Panel(
                f"[bold green]🤖 Answer:[/bold green]\n{answer}",
                title="Answer",
                border_style="green",
            )
        )

        if citations:
            self._print_rich_citations(citations)

        if metadata:
            self._print_rich_metadata(metadata)

    def _print_rich_citations(self, citations: List[Dict]) -> None:
        """Print citations in rich table format."""
        citation_table = Table(
            title="📖 Sources", show_header=True, header_style="bold magenta"
        )
        citation_table.add_column("File", style="cyan")
        citation_table.add_column("Lines", style="yellow")
        citation_table.add_column("Content", style="white")

        for citation in citations:
            file_path = citation.get("file", "Unknown")
            start_line = citation.get("start_line", "?")
            end_line = citation.get("end_line", "?")
            content = citation.get("content", "")
            if len(content) > 100:
                content = content[:100] + "..."

            citation_table.add_row(file_path, f"{start_line}-{end_line}", content)

        self.console.print(citation_table)

    def _print_rich_metadata(self, metadata: Dict[str, Any]) -> None:
        """Print metadata in rich table format."""
        meta_table = Table(
            title="📊 Metadata", show_header=True, header_style="bold cyan"
        )
        meta_table.add_column("Metric", style="cyan")
        meta_table.add_column("Value", style="green")

        for key, value in metadata.items():
            meta_table.add_row(key.replace("_", " ").title(), str(value))

        self.console.print(meta_table)

    def _print_plain_qa_session(
        self,
        question: str,
        answer: str,
        citations: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Print Q&A session in plain text format."""
        self.info("❓ Question:")
        self.info(question)
        self.info("")
        self.info("🤖 Answer:")
        self.info(answer)

        if citations:
            self._print_plain_citations(citations)

        if metadata:
            self._print_plain_metadata(metadata)

    def _print_plain_citations(self, citations: List[Dict]) -> None:
        """Print citations in plain text format."""
        self.info("")
        self.info("📖 Sources:")
        for i, citation in enumerate(citations, 1):
            file_path = citation.get("file", "Unknown")
            start_line = citation.get("start_line", "?")
            end_line = citation.get("end_line", "?")
            self.info(f"  {i}. {file_path} (lines {start_line}-{end_line})")

    def _print_plain_metadata(self, metadata: Dict[str, Any]) -> None:
        """Print metadata in plain text format."""
        self.info("")
        self.info("📊 Metadata:")
        for key, value in metadata.items():
            self.info(f"  {key.replace('_', ' ').title()}: {value}")

    def print_qa_session(
        self,
        question: str,
        answer: str,
        citations: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Print a Q&A session with enhanced formatting.

        Args:
            question: User question
            answer: AI answer
            citations: Optional list of citations
            metadata: Optional metadata (latency, etc.)
        """
        if self.config.output_format == OutputFormat.RICH:
            self._print_rich_qa_session(question, answer, citations, metadata)
        else:
            self._print_plain_qa_session(question, answer, citations, metadata)

    def print_ingestion_summary(
        self,
        repo_url: str,
        total_files: int,
        total_chunks: int,
        duration: float,
        collection_name: str,
        persist_directory: Optional[str] = None,
    ) -> None:
        """Print a comprehensive ingestion summary.

        Args:
            repo_url: Repository URL that was ingested
            total_files: Number of files processed
            total_chunks: Number of chunks created
            duration: Ingestion duration in seconds
            collection_name: ChromaDB collection name
            persist_directory: Optional persistence directory
        """
        if self.config.output_format == OutputFormat.RICH:
            # Create a summary panel
            summary_content = f"""
[bold cyan]Repository:[/bold cyan] {repo_url}
[bold green]Files Processed:[/bold green] {total_files}
[bold yellow]Chunks Created:[/bold yellow] {total_chunks}
[bold blue]Collection:[/bold blue] {collection_name}
[bold magenta]Duration:[/bold magenta] {duration:.2f}s
"""
            if persist_directory:
                summary_content += (
                    f"[bold green]Persisted to:[/bold green] {persist_directory}"
                )

            self.console.print(
                Panel(
                    summary_content,
                    title="✅ Ingestion Summary",
                    border_style="green",
                    padding=(1, 2),
                )
            )
        else:
            # Plain text summary
            self.info("✅ Ingestion Summary:")
            self.info(f"  Repository: {repo_url}")
            self.info(f"  Files Processed: {total_files}")
            self.info(f"  Chunks Created: {total_chunks}")
            self.info(f"  Collection: {collection_name}")
            self.info(f"  Duration: {duration:.2f}s")
            if persist_directory:
                self.info(f"  Persisted to: {persist_directory}")

    def print_error_summary(
        self, error: Exception, context: str, suggestions: Optional[List[str]] = None
    ) -> None:
        """Print a comprehensive error summary.

        Args:
            error: The exception that occurred
            context: Context where the error occurred
            suggestions: Optional list of suggestions to fix the error
        """
        if self.config.output_format == OutputFormat.RICH:
            error_content = f"""
[bold red]Error Type:[/bold red] {type(error).__name__}
[bold red]Error Message:[/bold red] {str(error)}
[bold yellow]Context:[/bold yellow] {context}
"""
            if suggestions:
                error_content += "\n[bold cyan]Suggestions:[/bold cyan]\n"
                for suggestion in suggestions:
                    error_content += f"• {suggestion}\n"

            self.console.print(
                Panel(
                    error_content,
                    title="❌ Error Summary",
                    border_style="red",
                    padding=(1, 2),
                )
            )
        else:
            # Plain text error summary
            self.info("❌ Error Summary:")
            self.info(f"  Error Type: {type(error).__name__}")
            self.info(f"  Error Message: {str(error)}")
            self.info(f"  Context: {context}")
            if suggestions:
                self.info("  Suggestions:")
                for suggestion in suggestions:
                    self.info(f"    • {suggestion}")


class DummyProgressBar:
    """Dummy progress bar that does nothing (for quiet mode)."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Dummy implementation - no cleanup needed for quiet mode
        pass

    def add_task(self, description: str, total: int = 100):
        return DummyTask()

    def update(self, task_id, **kwargs):
        # Temporary implementation
        pass


class DummyTask:
    """Dummy task for progress bar."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Temporary implementation
        pass


class DummyStatus:
    """Dummy status that does nothing (for quiet mode)."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Temporary implementation
        pass

    def update(self, status: str):
        # Temporary implementation
        pass


class DummyLive:
    """Dummy live display that does nothing (for quiet mode)."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Temporary implementation
        pass

    def update(self, renderable):
        # Temporary implementation
        pass


class DeveloperLogger:
    """Developer-focused logger with structured output and performance tracking.

    This class provides detailed logging for developers with support for
    structured data, performance metrics, and integration with the user
    output system.
    """

    def __init__(self, config: LoggingConfig):
        """Initialize developer logger with configuration.

        Args:
            config: Logging configuration
        """
        self.config = config
        self.logger = get_structured_logger("readme-mentor")
        self.handler = CustomHandler(config)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message.

        Args:
            message: Debug message
            **kwargs: Additional context
        """
        if self.config.log_level.value <= 10:  # DEBUG
            self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message.

        Args:
            message: Info message
            **kwargs: Additional context
        """
        if self.config.log_level.value <= 20:  # INFO
            self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message.

        Args:
            message: Warning message
            **kwargs: Additional context
        """
        if self.config.log_level.value <= 30:  # WARNING
            self.logger.warning(message, extra=kwargs)

    def error(self, message: str, exc_info: bool = True, **kwargs: Any) -> None:
        """Log an error message.

        Args:
            message: Error message
            exc_info: Whether to include exception info
            **kwargs: Additional context
        """
        if self.config.log_level.value <= 40:  # ERROR
            self.logger.error(message, exc_info=exc_info, extra=kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log an exception with traceback.

        Args:
            message: Exception message
            **kwargs: Additional context
        """
        if self.config.log_level.value <= 40:  # ERROR
            self.logger.exception(message, extra=kwargs)

    def critical(self, message: str, exc_info: bool = True, **kwargs: Any) -> None:
        """Log a critical message.

        Args:
            message: Critical message
            exc_info: Whether to include exception info
            **kwargs: Additional context
        """
        if self.config.log_level.value <= 50:  # CRITICAL
            self.logger.critical(message, exc_info=exc_info, extra=kwargs)

    def log_performance(self, operation: str, duration: float, **kwargs: Any) -> None:
        """Log performance metrics.

        Args:
            operation: Operation name
            duration: Duration in seconds
            **kwargs: Additional metrics
        """
        if self.config.track_wall_time:
            self.info(
                f"Performance: {operation} completed in {duration:.3f}s",
                operation=operation,
                duration=duration,
                **kwargs,
            )

    def log_tool_call(self, tool_name: str, **kwargs: Any) -> None:
        """Log a tool call.

        Args:
            tool_name: Name of the tool
            **kwargs: Additional context
        """
        if self.config.track_tool_calls:
            self.debug(f"Tool call: {tool_name}", tool_name=tool_name, **kwargs)

    def log_token_count(self, count: int, **kwargs: Any) -> None:
        """Log token count.

        Args:
            count: Number of tokens
            **kwargs: Additional context
        """
        if self.config.track_token_counts:
            self.debug(f"Token count: {count}", token_count=count, **kwargs)

    def log_operation_step(self, operation: str, step: str, **kwargs: Any) -> None:
        """Log an operation step.

        Args:
            operation: Operation name
            step: Step description
            **kwargs: Additional context
        """
        if self.config.show_operation_steps:
            self.debug(
                f"Operation step: {operation} - {step}",
                operation=operation,
                step=step,
                **kwargs,
            )

    def log_internal_state(
        self, component: str, state: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Log internal state information.

        Args:
            component: Component name
            state: State dictionary
            **kwargs: Additional context
        """
        if self.config.show_internal_state:
            self.debug(
                f"Internal state: {component}",
                component=component,
                state=state,
                **kwargs,
            )

    def log_configuration(self, key: str, value: Any, **kwargs: Any) -> None:
        """Log configuration details.

        Args:
            key: Configuration key
            value: Configuration value
            **kwargs: Additional context
        """
        if self.config.show_configuration_details:
            self.debug(
                f"Configuration: {key} = {value}",
                config_key=key,
                config_value=value,
                **kwargs,
            )
