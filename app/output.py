"""Enhanced output formatting for readme-mentor.

This module provides specialized output formatters for different types of
content including ingestion progress, Q&A sessions, performance metrics,
and error handling with support for Rich, Plain, and JSON formats.

This module now integrates with the enhanced output system from Phase 2,
providing backward compatibility while leveraging new features.
"""

import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console
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
from rich.table import Table

from .logging.config import LoggingConfig
from .logging.enums import OutputFormat

# Import will be handled dynamically to avoid circular import


class OutputManagerLegacy:
    """Legacy output manager with enhanced Phase 2 features.

    This class provides backward compatibility while integrating the new
    enhanced output formatting capabilities from Phase 2.
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

        # Initialize the enhanced output manager (imported dynamically to avoid circular import)
        from .output.manager import OutputManager

        self._enhanced_manager = OutputManager(config)

    def format_ingestion_progress(
        self,
        current_file: str,
        current: int,
        total: int,
        total_chunks: int,
        duration: float,
    ) -> str:
        """Format ingestion progress information with enhanced Phase 2 features.

        Args:
            current_file: Current file being processed
            current: Current file number
            total: Total number of files
            total_chunks: Total chunks created so far
            duration: Elapsed time

        Returns:
            Formatted progress string
        """
        if self.config.output_format == OutputFormat.RICH:
            return self._format_rich_ingestion_progress(
                current_file, current, total, total_chunks, duration
            )
        elif self.config.output_format == OutputFormat.JSON:
            return self._format_json_ingestion_progress(
                current_file, current, total, total_chunks, duration
            )
        else:
            return self._format_plain_ingestion_progress(
                current_file, current, total, total_chunks, duration
            )

    def format_qa_response(
        self,
        question: str,
        answer: str,
        citations: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format a Q&A response with enhanced Phase 2 features.

        Args:
            question: User question
            answer: AI answer
            citations: Optional citations
            metadata: Optional metadata

        Returns:
            Formatted Q&A response string
        """
        if self.config.output_format == OutputFormat.RICH:
            return self._format_rich_qa_response(question, answer, citations, metadata)
        elif self.config.output_format == OutputFormat.JSON:
            return self._format_json_qa_response(question, answer, citations, metadata)
        else:
            return self._format_plain_qa_response(question, answer, citations, metadata)

    def format_performance_summary(self, metrics: Dict[str, Any]) -> str:
        """Format performance summary with enhanced Phase 2 features.

        Args:
            metrics: Performance metrics

        Returns:
            Formatted performance summary string
        """
        if self.config.output_format == OutputFormat.RICH:
            return self._format_rich_performance_summary(metrics)
        elif self.config.output_format == OutputFormat.JSON:
            return self._format_json_performance_summary(metrics)
        else:
            return self._format_plain_performance_summary(metrics)

    def format_error_report(
        self, error: Exception, context: str, suggestions: Optional[List[str]] = None
    ) -> str:
        """Format an error report with enhanced Phase 2 features.

        Args:
            error: The exception
            context: Error context
            suggestions: Optional suggestions

        Returns:
            Formatted error report string
        """
        if self.config.output_format == OutputFormat.RICH:
            return self._format_rich_error_report(error, context, suggestions)
        elif self.config.output_format == OutputFormat.JSON:
            return self._format_json_error_report(error, context, suggestions)
        else:
            return self._format_plain_error_report(error, context, suggestions)

    def _format_rich_ingestion_progress(
        self,
        current_file: str,
        current: int,
        total: int,
        total_chunks: int,
        duration: float,
    ) -> str:
        """Format ingestion progress with enhanced Rich styling."""
        percentage = (current / total * 100) if total > 0 else 0

        content = f"""
[bold blue]Processing:[/bold blue] {current_file}
[bold green]Progress:[/bold green] {current}/{total} files ({percentage:.1f}%)
[bold yellow]Chunks:[/bold yellow] {total_chunks} created
[bold magenta]Duration:[/bold magenta] {duration:.1f}s
[bold cyan]Rate:[/bold cyan] {current / duration:.2f} files/sec
"""

        panel = Panel(
            content, title="📄 File Processing", border_style="blue", padding=(1, 2)
        )
        return str(panel)

    def _format_plain_ingestion_progress(
        self,
        current_file: str,
        current: int,
        total: int,
        total_chunks: int,
        duration: float,
    ) -> str:
        """Format ingestion progress as plain text with enhanced structure."""
        percentage = (current / total * 100) if total > 0 else 0
        rate = current / duration if duration > 0 else 0

        lines = [
            f"Processing: {current_file}",
            f"Progress: {current}/{total} files ({percentage:.1f}%)",
            f"Chunks: {total_chunks} created",
            f"Duration: {duration:.1f}s",
            f"Rate: {rate:.2f} files/sec",
        ]

        return "\n".join(lines)

    def _format_json_ingestion_progress(
        self,
        current_file: str,
        current: int,
        total: int,
        total_chunks: int,
        duration: float,
    ) -> str:
        """Format ingestion progress as JSON with enhanced metadata."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "ingestion_progress",
            "current_file": current_file,
            "current": current,
            "total": total,
            "percentage": round((current / total * 100) if total > 0 else 0, 1),
            "total_chunks": total_chunks,
            "duration_seconds": round(duration, 1),
            "processing_rate_files_per_sec": round(current / duration, 2)
            if duration > 0
            else 0,
            "status": "processing",
            "metadata": {
                "operation_type": "file_processing",
                "progress_type": "ingestion",
            },
        }

        return json.dumps(data, indent=2)

    def _format_rich_qa_response(
        self,
        question: str,
        answer: str,
        citations: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format Q&A response with enhanced Rich styling."""
        # Question panel with enhanced styling
        question_panel = Panel(
            f"[bold blue]❓ Question:[/bold blue]\n{question}",
            title="[bold blue]User Input[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )

        # Answer panel with enhanced styling
        answer_panel = Panel(
            f"[bold green]🤖 Answer:[/bold green]\n{answer}",
            title="[bold green]AI Response[/bold green]",
            border_style="green",
            padding=(1, 2),
        )

        result = str(question_panel) + "\n" + str(answer_panel)

        # Enhanced citations table
        if citations:
            citation_table = Table(
                title="📖 Sources & Citations",
                show_header=True,
                header_style="bold magenta",
                expand=True,
            )
            citation_table.add_column("File", style="cyan", no_wrap=True)
            citation_table.add_column("Lines", style="yellow")
            citation_table.add_column("Content", style="white")
            citation_table.add_column("Relevance", style="green")

            for citation in citations:
                file_path = citation.get("file", "Unknown")
                start_line = citation.get("start_line", "?")
                end_line = citation.get("end_line", "?")
                content = citation.get("content", "")
                relevance = citation.get("relevance", "N/A")

                # Truncate content for display
                display_content = (
                    content[:150] + "..." if len(content) > 150 else content
                )

                citation_table.add_row(
                    file_path,
                    f"{start_line}-{end_line}",
                    display_content,
                    str(relevance),
                )

            result += "\n" + str(citation_table)

        # Enhanced metadata table
        if metadata:
            meta_table = Table(
                title="📊 Response Metadata",
                show_header=True,
                header_style="bold cyan",
                expand=True,
            )
            meta_table.add_column("Metric", style="cyan")
            meta_table.add_column("Value", style="green")
            meta_table.add_column("Unit", style="dim")

            for key, value in metadata.items():
                unit = self._get_metric_unit(key)
                formatted_key = key.replace("_", " ").title()
                meta_table.add_row(formatted_key, str(value), unit)

            result += "\n" + str(meta_table)

        return result

    def _format_plain_qa_response(
        self,
        question: str,
        answer: str,
        citations: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format Q&A response as plain text with enhanced structure."""
        lines = ["❓ Question:", question, "", "🤖 Answer:", answer]

        if citations:
            lines.extend(["", "📖 Sources:"])
            for i, citation in enumerate(citations, 1):
                file_path = citation.get("file", "Unknown")
                start_line = citation.get("start_line", "?")
                end_line = citation.get("end_line", "?")
                relevance = citation.get("relevance", "N/A")
                lines.append(
                    f"  {i}. {file_path} (lines {start_line}-{end_line}, relevance: {relevance})"
                )

        if metadata:
            lines.extend(["", "📊 Metadata:"])
            for key, value in metadata.items():
                unit = self._get_metric_unit(key)
                formatted_key = key.replace("_", " ").title()
                lines.append(f"  {formatted_key}: {value} {unit}")

        return "\n".join(lines)

    def _format_json_qa_response(
        self,
        question: str,
        answer: str,
        citations: Optional[List[Dict]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format Q&A response as JSON with enhanced structure."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "qa_response",
            "question": question,
            "answer": answer,
            "status": "completed",
            "metadata": {
                "response_type": "ai_generated",
                "has_citations": bool(citations),
                "has_metadata": bool(metadata),
            },
        }

        if citations:
            data["citations"] = citations

        if metadata:
            data["performance_metadata"] = metadata

        return json.dumps(data, indent=2)

    def _format_rich_performance_summary(self, metrics: Dict[str, Any]) -> str:
        """Format performance summary with enhanced Rich styling."""
        table = Table(
            title="Performance Summary",
            show_header=True,
            header_style="bold magenta",
            expand=True,
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        table.add_column("Unit", style="dim")
        table.add_column("Category", style="yellow")

        for metric, value in metrics.items():
            if isinstance(value, dict):
                for sub_metric, sub_value in value.items():
                    unit = self._get_metric_unit(sub_metric)
                    category = metric.replace("_", " ").title()
                    table.add_row(
                        f"{metric}.{sub_metric}", str(sub_value), unit, category
                    )
            else:
                unit = self._get_metric_unit(metric)
                category = "General"
                table.add_row(metric, str(value), unit, category)

        return str(table)

    def _format_plain_performance_summary(self, metrics: Dict[str, Any]) -> str:
        """Format performance summary as plain text with enhanced structure."""
        lines = ["Performance Summary:"]

        for metric, value in metrics.items():
            if isinstance(value, dict):
                lines.append(f"  {metric}:")
                for sub_metric, sub_value in value.items():
                    unit = self._get_metric_unit(sub_metric)
                    lines.append(f"    {sub_metric}: {sub_value} {unit}")
            else:
                unit = self._get_metric_unit(metric)
                lines.append(f"  {metric}: {value} {unit}")

        return "\n".join(lines)

    def _format_json_performance_summary(self, metrics: Dict[str, Any]) -> str:
        """Format performance summary as JSON with enhanced metadata."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "performance_summary",
            "metrics": metrics,
            "metadata": {
                "summary_type": "comprehensive",
                "generated_at": datetime.now().isoformat(),
                "metric_count": len(metrics),
            },
        }

        return json.dumps(data, indent=2)

    def _format_rich_error_report(
        self, error: Exception, context: str, suggestions: Optional[List[str]] = None
    ) -> str:
        """Format error report with enhanced Rich styling."""
        error_content = f"""
[bold red]❌ Error Report[/bold red]
[bold red]Error Type:[/bold red] {type(error).__name__}
[bold red]Error Message:[/bold red] {str(error)}
[bold yellow]Context:[/bold yellow] {context}
"""

        if suggestions:
            error_content += "\n[bold cyan]💡 Suggestions:[/bold cyan]\n"
            for suggestion in suggestions:
                error_content += f"• {suggestion}\n"

        panel = Panel(
            error_content,
            title="[bold red]Error Report[/bold red]",
            border_style="red",
            padding=(1, 2),
        )
        return str(panel)

    def _format_plain_error_report(
        self, error: Exception, context: str, suggestions: Optional[List[str]] = None
    ) -> str:
        """Format error report as plain text with enhanced structure."""
        lines = [
            "❌ Error Report:",
            f"  Error Type: {type(error).__name__}",
            f"  Error Message: {str(error)}",
            f"  Context: {context}",
        ]

        if suggestions:
            lines.extend(["", "  Suggestions:"])
            for suggestion in suggestions:
                lines.append(f"    • {suggestion}")

        return "\n".join(lines)

    def _format_json_error_report(
        self, error: Exception, context: str, suggestions: Optional[List[str]] = None
    ) -> str:
        """Format error report as JSON with enhanced structure."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "type": "error_report",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "error_code": self._get_error_code(error),
            "status": "error",
            "metadata": {
                "report_type": "error_summary",
                "has_suggestions": bool(suggestions),
            },
        }

        if suggestions:
            data["suggestions"] = suggestions

        return json.dumps(data, indent=2)

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

    def _get_error_code(self, error: Exception) -> str:
        """Get a standardized error code for an exception."""
        error_message = str(error).lower()

        if "network" in error_message or "connection" in error_message:
            return "NETWORK_ERROR"
        elif "not found" in error_message or "404" in error_message:
            return "NOT_FOUND"
        elif "rate limit" in error_message:
            return "RATE_LIMIT"
        elif "authentication" in error_message or "unauthorized" in error_message:
            return "AUTH_ERROR"
        elif "validation" in error_message:
            return "VALIDATION_ERROR"
        elif "permission" in error_message or "access" in error_message:
            return "PERMISSION_ERROR"
        else:
            return "UNKNOWN_ERROR"

    def create_progress_bar(self, description: str = "") -> Progress:
        """Create a Rich progress bar with enhanced features."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )

    def print(self, content: str) -> None:
        """Print content using the configured format."""
        if not self.config.is_quiet():
            self.console.print(content)

    # Delegate to enhanced manager for new features
    def start_operation(self, operation_name: str) -> None:
        """Start tracking an operation with enhanced status indicators."""
        self._enhanced_manager.start_operation(operation_name)

    def end_operation(
        self, operation_name: str, additional_metrics: Optional[Dict[str, Any]] = None
    ) -> float:
        """End tracking an operation and display completion status."""
        return self._enhanced_manager.end_operation(operation_name, additional_metrics)

    def print_success(self, message: str, emoji: str = "✅") -> None:
        """Print a success message with enhanced formatting."""
        self._enhanced_manager.print_success(message, emoji)

    def print_warning(self, message: str, emoji: str = "⚠️") -> None:
        """Print a warning message with enhanced formatting."""
        self._enhanced_manager.print_warning(message, emoji)

    def print_error(
        self, message: str, error: Optional[Exception] = None, emoji: str = "❌"
    ) -> None:
        """Print an error message with enhanced formatting."""
        self._enhanced_manager.print_error(message, error, emoji)

    def print_info(self, message: str, emoji: str = "ℹ️") -> None:
        """Print an info message with enhanced formatting."""
        self._enhanced_manager.print_info(message, emoji)

    def print_table(
        self, data: List[Dict[str, Any]], title: str = "", show_header: bool = True
    ) -> None:
        """Print data in a structured table format."""
        self._enhanced_manager.print_table(data, title, show_header)

    def print_panel(self, content: str, title: str = "", style: str = "blue") -> None:
        """Print content in a panel format."""
        self._enhanced_manager.print_panel(content, title, style)

    def print_separator(self, char: str = "=", length: int = 80) -> None:
        """Print a separator line."""
        self._enhanced_manager.print_separator(char, length)

    def add_performance_metric(self, key: str, value: Any) -> None:
        """Add a performance metric for tracking."""
        self._enhanced_manager.add_performance_metric(key, value)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get all collected performance metrics."""
        return self._enhanced_manager.get_performance_metrics()

    def clear_performance_metrics(self) -> None:
        """Clear all performance metrics."""
        self._enhanced_manager.clear_performance_metrics()

    def print_performance_summary(self) -> None:
        """Print a comprehensive performance summary."""
        self._enhanced_manager.print_performance_summary()


# Backward compatibility alias
OutputManager = OutputManagerLegacy
