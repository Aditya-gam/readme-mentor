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

# Error suggestion constants
NETWORK_ERROR_SUGGESTIONS = [
    "Check your internet connection",
    "Verify the repository URL is accessible",
    "Try again in a few moments",
]

NOT_FOUND_ERROR_SUGGESTIONS = [
    "Verify the repository URL is correct",
    "Check if the repository is public",
    "Ensure you have access to the repository",
]

RATE_LIMIT_ERROR_SUGGESTIONS = [
    "Wait a few minutes before trying again",
    "Check your GitHub API rate limits",
    "Consider using a GitHub token for higher limits",
]

AUTH_ERROR_SUGGESTIONS = [
    "Check your GitHub credentials",
    "Verify your GitHub token is valid",
    "Ensure you have access to the repository",
]

VALIDATION_ERROR_SUGGESTIONS = [
    "Check the input format and requirements",
    "Verify all required fields are provided",
    "Ensure the data meets validation rules",
]

PERMISSION_ERROR_SUGGESTIONS = [
    "Check file and directory permissions",
    "Ensure you have write access to the target directory",
    "Try running with elevated privileges if needed",
]

# Error message pattern constants
ERROR_PATTERNS = {
    "network": ["network", "connection"],
    "not_found": ["not found", "404"],
    "rate_limit": ["rate limit"],
    "auth": ["authentication", "unauthorized"],
    "validation": ["validation"],
    "permission": ["permission", "access"],
}


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

    def format_qa_session(
        self,
        question: str,
        answer: str,
        citations: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format a Q&A session.

        Args:
            question: User question
            answer: AI answer
            citations: Optional list of citations
            metadata: Optional metadata

        Returns:
            Formatted Q&A session string
        """
        raise NotImplementedError

    def format_progress_update(
        self, operation: str, current: int, total: int, description: str = ""
    ) -> str:
        """Format a progress update.

        Args:
            operation: Operation name
            current: Current progress
            total: Total items
            description: Optional description

        Returns:
            Formatted progress update string
        """
        raise NotImplementedError

    def format_ingestion_summary(
        self,
        repo_url: str,
        total_files: int,
        total_chunks: int,
        duration: float,
        collection_name: str,
        persist_directory: Optional[str] = None,
    ) -> str:
        """Format an ingestion summary.

        Args:
            repo_url: Repository URL that was ingested
            total_files: Number of files processed
            total_chunks: Number of chunks created
            duration: Ingestion duration in seconds
            collection_name: ChromaDB collection name
            persist_directory: Optional persistence directory

        Returns:
            Formatted ingestion summary string
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
        """Format an error using Rich styling.

        Args:
            error: Exception to format
            context: Additional context information

        Returns:
            Formatted error message
        """
        # Create error panel
        error_text = f"[bold red]Error:[/bold red] {type(error).__name__}: {str(error)}"

        if context:
            error_text += f"\n[dim]Context:[/dim] {context}"

        if self.config.show_stack_traces:
            import traceback

            error_text += f"\n[dim]Stack Trace:[/dim]\n{traceback.format_exc()}"

        # Create actionable suggestions
        if self.config.show_actionable_suggestions:
            suggestions = self._get_error_suggestions(error)
            if suggestions:
                error_text += "\n[bold yellow]Suggestions:[/bold yellow]\n"
                for suggestion in suggestions:
                    error_text += f"• {suggestion}\n"

        return error_text

    def _get_error_suggestions(self, error: Exception) -> list:
        """Get actionable suggestions for an error.

        Args:
            error: Exception to get suggestions for

        Returns:
            List of suggestion strings
        """
        suggestions = []
        _ = type(error).__name__
        error_message = str(error).lower()

        if any(pattern in error_message for pattern in ERROR_PATTERNS["network"]):
            suggestions.extend(NETWORK_ERROR_SUGGESTIONS)
        elif any(pattern in error_message for pattern in ERROR_PATTERNS["not_found"]):
            suggestions.extend(NOT_FOUND_ERROR_SUGGESTIONS)
        elif any(pattern in error_message for pattern in ERROR_PATTERNS["rate_limit"]):
            suggestions.extend(RATE_LIMIT_ERROR_SUGGESTIONS)
        elif any(pattern in error_message for pattern in ERROR_PATTERNS["auth"]):
            suggestions.extend(AUTH_ERROR_SUGGESTIONS)
        elif any(pattern in error_message for pattern in ERROR_PATTERNS["validation"]):
            suggestions.extend(VALIDATION_ERROR_SUGGESTIONS)
        elif any(pattern in error_message for pattern in ERROR_PATTERNS["permission"]):
            suggestions.extend(PERMISSION_ERROR_SUGGESTIONS)

        return suggestions

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

    def format_qa_session(
        self,
        question: str,
        answer: str,
        citations: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format a Q&A session using Rich styling.

        Args:
            question: User question
            answer: AI answer
            citations: Optional list of citations
            metadata: Optional metadata

        Returns:
            Formatted Q&A session string
        """
        # Create question panel
        question_panel = Panel(
            f"[bold blue]❓ Question:[/bold blue]\n{question}",
            title="Question",
            border_style="blue",
        )

        # Create answer panel
        answer_panel = Panel(
            f"[bold green]🤖 Answer:[/bold green]\n{answer}",
            title="Answer",
            border_style="green",
        )

        result = str(question_panel) + "\n" + str(answer_panel)

        # Add citations table if available
        if citations:
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
                content = (
                    citation.get("content", "")[:100] + "..."
                    if len(citation.get("content", "")) > 100
                    else citation.get("content", "")
                )

                citation_table.add_row(file_path, f"{start_line}-{end_line}", content)

            result += "\n" + str(citation_table)

        # Add metadata table if available
        if metadata:
            meta_table = Table(
                title="📊 Metadata", show_header=True, header_style="bold cyan"
            )
            meta_table.add_column("Metric", style="cyan")
            meta_table.add_column("Value", style="green")

            for key, value in metadata.items():
                meta_table.add_row(key.replace("_", " ").title(), str(value))

            result += "\n" + str(meta_table)

        return result

    def format_progress_update(
        self, operation: str, current: int, total: int, description: str = ""
    ) -> str:
        """Format a progress update using Rich styling.

        Args:
            operation: Operation name
            current: Current progress
            total: Total items
            description: Optional description

        Returns:
            Formatted progress update string
        """
        percentage = (current / total * 100) if total > 0 else 0
        progress_bar = "█" * int(percentage / 5) + "░" * (20 - int(percentage / 5))

        text = f"[bold blue]{operation}[/bold blue]"
        if description:
            text += f": {description}"

        text += f"\n[{progress_bar}] {current}/{total} ({percentage:.1f}%)"

        return text

    def format_ingestion_summary(
        self,
        repo_url: str,
        total_files: int,
        total_chunks: int,
        duration: float,
        collection_name: str,
        persist_directory: Optional[str] = None,
    ) -> str:
        """Format an ingestion summary using Rich styling.

        Args:
            repo_url: Repository URL that was ingested
            total_files: Number of files processed
            total_chunks: Number of chunks created
            duration: Ingestion duration in seconds
            collection_name: ChromaDB collection name
            persist_directory: Optional persistence directory

        Returns:
            Formatted ingestion summary string
        """
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

        panel = Panel(
            summary_content,
            title="✅ Ingestion Summary",
            border_style="green",
            padding=(1, 2),
        )
        return str(panel)

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
            "duration": "seconds",
            "latency": "ms",
        }
        return units.get(metric, "")

    def create_progress_bar(self, description: str = "") -> Any:
        """Create a Rich progress bar.

        Args:
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
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
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

        # Add actionable suggestions
        if self.config.show_actionable_suggestions:
            suggestions = self._get_error_suggestions(error)
            if suggestions:
                lines.append("Suggestions:")
                for suggestion in suggestions:
                    lines.append(f"  • {suggestion}")

        return "\n".join(lines)

    def _get_error_suggestions(self, error: Exception) -> list:
        """Get actionable suggestions for an error.

        Args:
            error: Exception to get suggestions for

        Returns:
            List of suggestion strings
        """
        suggestions = []
        error_message = str(error).lower()

        if "network" in error_message or "connection" in error_message:
            suggestions.extend(
                [
                    "Check your internet connection",
                    "Verify the repository URL is accessible",
                    "Try again in a few moments",
                ]
            )
        elif "not found" in error_message or "404" in error_message:
            suggestions.extend(
                [
                    "Verify the repository URL is correct",
                    "Check if the repository is public",
                    "Ensure you have access to the repository",
                ]
            )
        elif "rate limit" in error_message:
            suggestions.extend(
                [
                    "Wait a few minutes before trying again",
                    "Check your GitHub API rate limits",
                    "Consider using a GitHub token for higher limits",
                ]
            )
        elif "authentication" in error_message or "unauthorized" in error_message:
            suggestions.extend(
                [
                    "Check your GitHub credentials",
                    "Verify your GitHub token is valid",
                    "Ensure you have access to the repository",
                ]
            )
        elif "validation" in error_message:
            suggestions.extend(
                [
                    "Check the input format and requirements",
                    "Verify all required fields are provided",
                    "Ensure the data meets validation rules",
                ]
            )
        elif "permission" in error_message or "access" in error_message:
            suggestions.extend(
                [
                    "Check file and directory permissions",
                    "Ensure you have write access to the target directory",
                    "Try running with elevated privileges if needed",
                ]
            )

        return suggestions

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

    def format_qa_session(
        self,
        question: str,
        answer: str,
        citations: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format a Q&A session as plain text.

        Args:
            question: User question
            answer: AI answer
            citations: Optional list of citations
            metadata: Optional metadata

        Returns:
            Formatted Q&A session string
        """
        lines = ["❓ Question:", question, "", "🤖 Answer:", answer]

        if citations:
            lines.extend(["", "📖 Sources:"])
            for i, citation in enumerate(citations, 1):
                file_path = citation.get("file", "Unknown")
                start_line = citation.get("start_line", "?")
                end_line = citation.get("end_line", "?")
                lines.append(f"  {i}. {file_path} (lines {start_line}-{end_line})")

        if metadata:
            lines.extend(["", "📊 Metadata:"])
            for key, value in metadata.items():
                lines.append(f"  {key.replace('_', ' ').title()}: {value}")

        return "\n".join(lines)

    def format_progress_update(
        self, operation: str, current: int, total: int, description: str = ""
    ) -> str:
        """Format a progress update as plain text.

        Args:
            operation: Operation name
            current: Current progress
            total: Total items
            description: Optional description

        Returns:
            Formatted progress update string
        """
        percentage = (current / total * 100) if total > 0 else 0
        text = f"{operation}"
        if description:
            text += f": {description}"
        text += f" - {current}/{total} ({percentage:.1f}%)"
        return text

    def format_ingestion_summary(
        self,
        repo_url: str,
        total_files: int,
        total_chunks: int,
        duration: float,
        collection_name: str,
        persist_directory: Optional[str] = None,
    ) -> str:
        """Format an ingestion summary as plain text.

        Args:
            repo_url: Repository URL that was ingested
            total_files: Number of files processed
            total_chunks: Number of chunks created
            duration: Ingestion duration in seconds
            collection_name: ChromaDB collection name
            persist_directory: Optional persistence directory

        Returns:
            Formatted ingestion summary string
        """
        lines = [
            "✅ Ingestion Summary:",
            f"  Repository: {repo_url}",
            f"  Files Processed: {total_files}",
            f"  Chunks Created: {total_chunks}",
            f"  Collection: {collection_name}",
            f"  Duration: {duration:.2f}s",
        ]

        if persist_directory:
            lines.append(f"  Persisted to: {persist_directory}")

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
            "duration": "seconds",
            "latency": "ms",
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
            "type": "message",
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
            "type": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_code": self._get_error_code(error),
        }

        if context:
            data["context"] = context

        if self.config.show_stack_traces:
            import traceback

            data["stack_trace"] = traceback.format_exc()

        # Add actionable suggestions
        if self.config.show_actionable_suggestions:
            suggestions = self._get_error_suggestions(error)
            if suggestions:
                data["suggestions"] = suggestions

        return json.dumps(data, indent=2)

    def _get_error_code(self, error: Exception) -> str:
        """Get a standardized error code for an exception.

        Args:
            error: Exception to get code for

        Returns:
            Error code string
        """
        _ = type(error).__name__
        error_message = str(error).lower()

        if any(pattern in error_message for pattern in ERROR_PATTERNS["network"]):
            return "NETWORK_ERROR"
        elif any(pattern in error_message for pattern in ERROR_PATTERNS["not_found"]):
            return "NOT_FOUND"
        elif any(pattern in error_message for pattern in ERROR_PATTERNS["rate_limit"]):
            return "RATE_LIMIT"
        elif any(pattern in error_message for pattern in ERROR_PATTERNS["auth"]):
            return "AUTH_ERROR"
        elif any(pattern in error_message for pattern in ERROR_PATTERNS["validation"]):
            return "VALIDATION_ERROR"
        elif any(pattern in error_message for pattern in ERROR_PATTERNS["permission"]):
            return "PERMISSION_ERROR"
        else:
            return "UNKNOWN_ERROR"

    def _get_error_suggestions(self, error: Exception) -> list:
        """Get actionable suggestions for an error.

        Args:
            error: Exception to get suggestions for

        Returns:
            List of suggestion strings
        """
        suggestions = []
        error_message = str(error).lower()

        if any(pattern in error_message for pattern in ERROR_PATTERNS["network"]):
            suggestions.extend(NETWORK_ERROR_SUGGESTIONS)
        elif any(pattern in error_message for pattern in ERROR_PATTERNS["not_found"]):
            suggestions.extend(NOT_FOUND_ERROR_SUGGESTIONS)
        elif any(pattern in error_message for pattern in ERROR_PATTERNS["rate_limit"]):
            suggestions.extend(RATE_LIMIT_ERROR_SUGGESTIONS)
        elif any(pattern in error_message for pattern in ERROR_PATTERNS["auth"]):
            suggestions.extend(AUTH_ERROR_SUGGESTIONS)
        elif any(pattern in error_message for pattern in ERROR_PATTERNS["validation"]):
            suggestions.extend(VALIDATION_ERROR_SUGGESTIONS)
        elif any(pattern in error_message for pattern in ERROR_PATTERNS["permission"]):
            suggestions.extend(PERMISSION_ERROR_SUGGESTIONS)

        return suggestions

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

    def format_qa_session(
        self,
        question: str,
        answer: str,
        citations: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format a Q&A session as JSON.

        Args:
            question: User question
            answer: AI answer
            citations: Optional list of citations
            metadata: Optional metadata

        Returns:
            JSON formatted Q&A session
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "qa_session",
            "question": question,
            "answer": answer,
        }

        if citations:
            data["citations"] = citations

        if metadata:
            data["metadata"] = metadata

        return json.dumps(data, indent=2)

    def format_progress_update(
        self, operation: str, current: int, total: int, description: str = ""
    ) -> str:
        """Format a progress update as JSON.

        Args:
            operation: Operation name
            current: Current progress
            total: Total items
            description: Optional description

        Returns:
            JSON formatted progress update
        """
        percentage = (current / total * 100) if total > 0 else 0

        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "progress_update",
            "operation": operation,
            "current": current,
            "total": total,
            "percentage": round(percentage, 1),
        }

        if description:
            data["description"] = description

        return json.dumps(data, indent=2)

    def format_ingestion_summary(
        self,
        repo_url: str,
        total_files: int,
        total_chunks: int,
        duration: float,
        collection_name: str,
        persist_directory: Optional[str] = None,
    ) -> str:
        """Format an ingestion summary as JSON.

        Args:
            repo_url: Repository URL that was ingested
            total_files: Number of files processed
            total_chunks: Number of chunks created
            duration: Ingestion duration in seconds
            collection_name: ChromaDB collection name
            persist_directory: Optional persistence directory

        Returns:
            JSON formatted ingestion summary
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "ingestion_summary",
            "repository_url": repo_url,
            "total_files": total_files,
            "total_chunks": total_chunks,
            "duration_seconds": round(duration, 2),
            "collection_name": collection_name,
            "status": "success",
        }

        if persist_directory:
            data["persist_directory"] = persist_directory

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
            "token_count": "tokens",
            "tool_calls": "calls",
            "memory_usage": "MB",
            "cpu_usage": "%",
            "duration": "s",
            "latency": "ms",
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
